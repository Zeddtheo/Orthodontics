from pathlib import Path
import argparse, random, time
from shutil import copy2
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg

POINTNETREG_ROOT = Path("outputs/pointnetreg")
DEFAULT_CHECKPOINT_ROOT = POINTNETREG_ROOT / "checkpoints"
FINAL_PT_DIR = POINTNETREG_ROOT / "final_pt"

DEFAULT_TOOTH_IDS = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]


def _ensure_xyz(pos: torch.Tensor) -> torch.Tensor:
    if pos.dim() != 3:
        raise ValueError(f"Expected pos to have 3 dims, got {pos.shape}")
    if pos.shape[1] == 3 and pos.shape[2] != 3:
        return pos.transpose(1, 2).contiguous()
    return pos.contiguous()


def heatmap_expectation(
    logits: torch.Tensor,
    pos: torch.Tensor,
    temperature: float = 1.0,
    topk: int = 0,
) -> torch.Tensor:
    if temperature <= 0:
        temperature = 1.0
    pos_xyz = _ensure_xyz(pos)
    if pos_xyz.dtype != logits.dtype:
        pos_xyz = pos_xyz.to(dtype=logits.dtype)
    if topk and topk > 0:
        k = min(max(1, topk), logits.shape[-1])
        top_vals, top_idx = logits.topk(k, dim=-1)  # (B,L,K)
        weights = F.softmax(top_vals * (1.0 / temperature), dim=-1)  # (B,L,K)
        pos_t = pos_xyz.transpose(1, 2)  # (B,3,N)
        pos_exp = pos_t.unsqueeze(1).expand(-1, logits.shape[1], -1, -1)  # (B,L,3,N)
        gather_idx = top_idx.unsqueeze(2).expand(-1, -1, pos_exp.shape[2], -1)  # (B,L,3,K)
        selected = torch.gather(pos_exp, dim=3, index=gather_idx)  # (B,L,3,K)
        coords = (weights.unsqueeze(2) * selected).sum(dim=-1)  # (B,L,3)
        return coords
    weights = F.softmax(logits * (1.0 / temperature), dim=-1)  # (B,L,N)
    if pos_xyz.dtype != weights.dtype:
        pos_xyz = pos_xyz.to(dtype=weights.dtype)
    coords = torch.matmul(weights, pos_xyz.transpose(1, 2))        # (B,L,3)
    return coords


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def parse_schedule_spec(spec: str, param_name: str, default: float = 0.0) -> Tuple[float, Optional[Tuple[float, float, int]], str]:
    raw = (spec or "").strip()
    if not raw:
        return default, None, f"{default:.6f}".rstrip("0").rstrip(".")
    if ":" in raw and "@" in raw:
        try:
            weights, milestone = raw.split("@", 1)
            start_str, end_str = weights.split(":", 1)
            start_val = float(start_str)
            end_val = float(end_str)
            steps = max(1, int(milestone))
        except ValueError as exc:
            raise ValueError(
                f"Invalid schedule '{spec}' for {param_name}. Expected format 'start:end@T'."
            ) from exc
        return start_val, (start_val, end_val, steps), raw
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid value '{spec}' for {param_name}.") from exc
    return value, None, raw


def parse_coord_loss_weight(spec: str) -> Tuple[float, Optional[Tuple[float, float, int]], str]:
    return parse_schedule_spec(spec, "coord_loss_weight", 0.0)


def compute_scheduled_value(epoch: int, base_value: float, schedule: Optional[Tuple[float, float, int]]) -> float:
    if not schedule:
        return base_value
    start, end, steps = schedule
    progress = min(max((epoch - 1) / max(1, steps), 0.0), 1.0)
    return start * (1.0 - progress) + end * progress


def _limit_subset(subset, max_count: int):
    if max_count is None or max_count <= 0:
        return subset
    if len(subset) <= max_count:
        return subset
    # torch.utils.data.Subset exposes .dataset and .indices
    base_dataset = subset.dataset
    indices = subset.indices[:max_count]
    return torch.utils.data.Subset(base_dataset, indices)


def _gather_points(xyz: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    gather_idx = indices.unsqueeze(1).expand(-1, 3, -1)
    pts = torch.gather(xyz, 2, gather_idx)
    return pts.permute(0, 2, 1).contiguous()


def build_loaders(
    root: str,
    tooth_id: str,
    features: str,
    batch_size: int,
    workers: int,
    val_ratio: float,
    augment: bool,
    max_train_samples: int,
    max_val_samples: int,
    case_filter: Optional[str],
):
    patterns = []
    if case_filter:
        case_str = case_filter.strip()
        if case_str:
            patterns.append(f"{case_str}_*_{tooth_id}.npz")
            patterns.append(f"{case_str}_*_{tooth_id.upper()}.npz")
    if not patterns:
        patterns = [f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"]

    cfg = DatasetConfig(
        root=root,
        file_patterns=tuple(patterns),
        features=features,
        select_landmarks="all",
        augment=augment,
        ensure_constant_L=False,
        tooth_id=tooth_id,
    )
    dataset = P0PointNetRegDataset(cfg)
    n = len(dataset)
    if n == 0:
        raise ValueError(f"No samples found for tooth {tooth_id} in {root}")
    val_len = int(round(n * val_ratio))
    val_len = min(max(0, val_len), max(0, n - 1))
    train_len = n - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(2025))
    train_set = _limit_subset(train_set, max_train_samples)
    val_set = _limit_subset(val_set, max_val_samples)

    # pin_memory 只在 CUDA 可用时启用
    use_pin_memory = torch.cuda.is_available()
    
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=use_pin_memory,
        collate_fn=collate_p0,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=use_pin_memory,
        collate_fn=collate_p0,
    )
    return dataset, train_loader, val_loader


def train_all_teeth(args, device: torch.device) -> None:
    loaders = {}
    heads_config = {}
    in_channels = None

    for tooth in args.tooth:
        dataset, train_loader, val_loader = build_loaders(
            root=args.root,
            tooth_id=tooth,
            features=args.features,
            batch_size=args.batch_size,
            workers=args.workers,
            val_ratio=args.val_ratio,
            augment=args.augment,
            max_train_samples=args.max_train_samples,
            max_val_samples=args.max_val_samples,
            case_filter=args.case_filter,
        )
        key = tooth.lower()
        loaders[key] = {
            "train": train_loader,
            "val": val_loader,
            "len_train": len(train_loader.dataset),
            "len_val": len(val_loader.dataset),
        }
        heads_config[key] = dataset[0]["y"].shape[0]
        if in_channels is None:
            in_channels = dataset[0]["x"].shape[0]

    if not loaders:
        raise ValueError("No loaders constructed. Check --tooth and dataset root.")

    model = PointNetReg(
        in_channels=in_channels,
        num_landmarks=max(heads_config.values()),
        heads_config=heads_config,
        use_tnet=args.use_tnet,
        return_logits=True,
    ).to(device)
    print(f"heads_config={heads_config}")

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)
    scaler = GradScaler("cuda", enabled=device.type == "cuda")
    mse_loss = torch.nn.MSELoss(reduction="none")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    best_loss_epoch = 0
    best_mae_metric = float("inf")
    best_mae_epoch = 0
    log_path = out_dir / "log.txt"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(
            f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} multi-tooth ===="
            f" teeth={args.tooth} in_channels={in_channels}"
            f" coord_spec={args.coord_loss_weight_spec} temp_spec={args.coord_temperature_spec} peak_spec={args.peak_ce_spec} margin_spec={args.margin_loss_weight_spec}/{args.margin_floor}\n"
        )

    tooth_list = list(loaders.keys())

    for epoch in range(1, args.epochs + 1):
        coord_weight = compute_scheduled_value(epoch, args.coord_loss_weight, args.coord_weight_schedule)
        coord_temp = compute_scheduled_value(epoch, args.coord_temperature, args.coord_temperature_schedule)
        peak_ce_weight = compute_scheduled_value(epoch, args.peak_ce, args.peak_ce_schedule)
        margin_weight = compute_scheduled_value(epoch, args.margin_loss_weight, args.margin_weight_schedule)
        model.train()
        train_losses = {}
        train_stats = {}
        for tooth in tooth_list:
            train_loader = loaders[tooth]["train"]
            running = 0.0
            mae_sum = 0.0
            mae_count = 0
            match_count = 0
            active_total = 0
            hit05 = 0
            hit10 = 0
            margin_sum = 0.0
            soft_margin_sum = 0.0
            refined_sum = 0.0
            refined_hit05 = 0
            refined_hit10 = 0
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(device, non_blocking=True).float()
                else:
                    mask = torch.ones(y.shape[0], y.shape[1], device=device)
                mask_exp = mask.unsqueeze(-1)
                pos = batch.get("pos")
                if pos is not None:
                    pos = pos.to(device, non_blocking=True)
                else:
                    pos = x[:, :3, :]
                landmarks_gt = batch.get("landmarks")
                if landmarks_gt is not None:
                    landmarks_gt = landmarks_gt.to(device, non_blocking=True)
                    landmarks_gt = torch.nan_to_num(landmarks_gt, nan=0.0)

                optim.zero_grad(set_to_none=True)
                with autocast(device.type, enabled=device.type == "cuda"):
                    logits = model(x, tooth_id=tooth)
                    probs = torch.sigmoid(logits)
                    loss_target = y
                    if args.label_gamma != 1.0:
                        loss_target = torch.clamp(y.pow(args.label_gamma), 0.0, 1.0)
                    weight = torch.pow(loss_target, args.loss_power) + args.loss_eps
                    weight = weight * mask_exp
                    loss_map = mse_loss(probs, loss_target)
                    denom = torch.clamp(weight.sum(), min=1e-12)
                    heatmap_loss = (loss_map * weight).sum() / denom
                    loss = args.heatmap_loss_weight * heatmap_loss
                    gt_idx = torch.argmax(y, dim=-1)
                    if landmarks_gt is None:
                        src_pos = pos if pos is not None else x[:, :3, :]
                        landmarks_gt = _gather_points(src_pos, gt_idx)

                    coord_loss_val = torch.tensor(0.0, device=device)
                    if coord_weight > 0.0 and landmarks_gt is not None:
                        pred_coords = heatmap_expectation(logits, pos, temperature=coord_temp, topk=args.coord_topk)
                        coord_diff = pred_coords - landmarks_gt
                        coord_loss_val = (coord_diff.abs() * mask_exp).sum() / torch.clamp(mask_exp.sum(), min=1e-12)
                        loss = loss + coord_weight * coord_loss_val

                    if peak_ce_weight > 0.0:
                        active_flat = mask_exp.view(-1) > 0.5
                        if active_flat.any():
                            logits_flat = logits.view(-1, logits.shape[-1])[active_flat]
                            gt_flat = gt_idx.view(-1)[active_flat]
                            loss = loss + peak_ce_weight * torch.nn.functional.cross_entropy(logits_flat, gt_flat)

                    if margin_weight > 0.0 and args.margin_floor > 0.0:
                        top2_vals = logits.topk(2, dim=-1).values
                        delta = top2_vals[..., 0] - top2_vals[..., 1]
                        soft_margin = torch.sigmoid(delta)
                        margin_mask = mask > 0.5
                        if margin_mask.any():
                            margin_deficit = F.relu(args.margin_floor - soft_margin[margin_mask])
                            if margin_deficit.numel() > 0:
                                margin_loss_val = margin_deficit.mean()
                                loss = loss + margin_weight * margin_loss_val
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
                running += loss.item()

                with torch.no_grad():
                    mask_bool = mask > 0.5
                    pred_idx = torch.argmax(logits, dim=-1)
                    gt_idx = torch.argmax(y, dim=-1)
                    pred_pts = _gather_points(x[:, :3, :], pred_idx)
                    gt_pts = _gather_points(x[:, :3, :], gt_idx)
                    errors = torch.norm(pred_pts - gt_pts, dim=-1)
                    active = mask_bool
                    if active.any():
                        errors_active = errors[active]
                        mae_sum += float(errors_active.sum().item())
                        mae_count += int(errors_active.numel())
                        match_count += int(((pred_idx == gt_idx) & active).sum().item())
                        hit05 += int(((errors <= 0.5) & active).sum().item())
                        hit10 += int(((errors <= 1.0) & active).sum().item())
                        topk_logits = logits.topk(2, dim=-1).values
                        margin_vals = torch.sigmoid(topk_logits[..., 0] - topk_logits[..., 1])
                        margin_sum += float(margin_vals[active].sum().item())
                        soft_probs = torch.softmax(logits, dim=-1)
                        soft_topk = soft_probs.topk(2, dim=-1).values
                        soft_margin_vals = soft_topk[..., 0] - soft_topk[..., 1]
                        soft_margin_sum += float(soft_margin_vals[active].sum().item())
                        active_total += int(active.sum().item())
                        if coord_weight > 0.0 and landmarks_gt is not None:
                            refined_coords = heatmap_expectation(logits, pos, temperature=coord_temp, topk=args.coord_topk)
                            coord_errors = torch.norm(refined_coords - landmarks_gt, dim=-1)
                            refined_active = coord_errors[active]
                            refined_sum += float(refined_active.sum().item())
                            refined_hit05 += int(((coord_errors <= 0.5) & active).sum().item())
                            refined_hit10 += int(((coord_errors <= 1.0) & active).sum().item())
                        else:
                            refined_sum += float(errors_active.sum().item())
                            refined_hit05 += int(((errors <= 0.5) & active).sum().item())
                            refined_hit10 += int(((errors <= 1.0) & active).sum().item())
            train_losses[tooth] = running / max(1, len(train_loader))
            train_stats[tooth] = {
                "mae": (mae_sum / mae_count) if mae_count > 0 else 0.0,
                "refined_mae": (refined_sum / mae_count) if mae_count > 0 else 0.0,
                "matches": match_count,
                "count": active_total,
                "hit05": hit05,
                "hit10": hit10,
                "refined_hit05": refined_hit05,
                "refined_hit10": refined_hit10,
                "margin": (margin_sum / active_total) if active_total > 0 else 0.0,
                "soft_margin": (soft_margin_sum / active_total) if active_total > 0 else 0.0,
            }

        mean_train = sum(train_losses.values()) / max(1, len(train_losses))
        mean_train_refined = sum(stat["refined_mae"] for stat in train_stats.values()) / max(1, len(train_stats))

        model.eval()
        val_losses = {}
        val_stats = {}
        with torch.no_grad(), autocast(device.type, enabled=device.type == "cuda"):
            for tooth in tooth_list:
                val_loader = loaders[tooth]["val"]
                if len(val_loader.dataset) == 0:
                    val_losses[tooth] = train_losses[tooth]
                    val_stats[tooth] = train_stats[tooth]
                    continue
                running = 0.0
                mae_sum = 0.0
                mae_count = 0
                match_count = 0
                active_total = 0
                hit05 = 0
                hit10 = 0
                margin_sum = 0.0
                soft_margin_sum = 0.0
                refined_sum = 0.0
                refined_hit05 = 0
                refined_hit10 = 0
                for batch in val_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)
                    mask = batch.get("mask")
                    if mask is not None:
                        mask = mask.to(device, non_blocking=True).float()
                    else:
                        mask = torch.ones(y.shape[0], y.shape[1], device=device)
                    mask_exp = mask.unsqueeze(-1)
                    pos = batch.get("pos")
                    if pos is not None:
                        pos = pos.to(device, non_blocking=True)
                    else:
                        pos = x[:, :3, :]
                    landmarks_gt = batch.get("landmarks")
                    if landmarks_gt is not None:
                        landmarks_gt = landmarks_gt.to(device, non_blocking=True)
                        landmarks_gt = torch.nan_to_num(landmarks_gt, nan=0.0)
                    logits = model(x, tooth_id=tooth)
                    probs = torch.sigmoid(logits)
                    loss_target = y
                    if args.label_gamma != 1.0:
                        loss_target = torch.clamp(y.pow(args.label_gamma), 0.0, 1.0)
                    weight = torch.pow(loss_target, args.loss_power) + args.loss_eps
                    weight = weight * mask_exp
                    loss_map = mse_loss(probs, loss_target)
                    running += (
                        args.heatmap_loss_weight
                        * (loss_map * weight).sum()
                        / weight.sum().clamp(min=1e-12)
                    ).item()
                    gt_idx = torch.argmax(y, dim=-1)
                    if landmarks_gt is None:
                        src_pos = pos if pos is not None else x[:, :3, :]
                        landmarks_gt = _gather_points(src_pos, gt_idx)

                    mask_bool = mask > 0.5
                    pred_idx = torch.argmax(logits, dim=-1)
                    pred_pts = _gather_points(x[:, :3, :], pred_idx)
                    gt_pts = _gather_points(x[:, :3, :], gt_idx)
                    errors = torch.norm(pred_pts - gt_pts, dim=-1)
                    active = mask_bool
                    if active.any():
                        errors_active = errors[active]
                        mae_sum += float(errors_active.sum().item())
                        mae_count += int(errors_active.numel())
                        match_count += int(((pred_idx == gt_idx) & active).sum().item())
                        hit05 += int(((errors <= 0.5) & active).sum().item())
                        hit10 += int(((errors <= 1.0) & active).sum().item())
                        topk_logits = logits.topk(2, dim=-1).values
                        margin_vals = torch.sigmoid(topk_logits[..., 0] - topk_logits[..., 1])
                        margin_sum += float(margin_vals[active].sum().item())
                        soft_probs = torch.softmax(logits, dim=-1)
                        soft_topk = soft_probs.topk(2, dim=-1).values
                        soft_margin_vals = soft_topk[..., 0] - soft_topk[..., 1]
                        soft_margin_sum += float(soft_margin_vals[active].sum().item())
                        active_total += int(active.sum().item())
                        if coord_weight > 0.0 and landmarks_gt is not None:
                            refined_coords = heatmap_expectation(logits, pos, temperature=coord_temp, topk=args.coord_topk)
                            coord_errors = torch.norm(refined_coords - landmarks_gt, dim=-1)
                            coord_active = coord_errors[active]
                            refined_sum += float(coord_active.sum().item())
                            refined_hit05 += int(((coord_errors <= 0.5) & active).sum().item())
                            refined_hit10 += int(((coord_errors <= 1.0) & active).sum().item())
                        else:
                            refined_sum += float(errors_active.sum().item())
                            refined_hit05 += int(((errors <= 0.5) & active).sum().item())
                            refined_hit10 += int(((errors <= 1.0) & active).sum().item())
                val_losses[tooth] = running / max(1, len(val_loader))
                val_stats[tooth] = {
                    "mae": (mae_sum / mae_count) if mae_count > 0 else 0.0,
                    "refined_mae": (refined_sum / mae_count) if mae_count > 0 else 0.0,
                    "matches": match_count,
                    "count": active_total,
                    "hit05": hit05,
                    "hit10": hit10,
                    "refined_hit05": refined_hit05,
                    "refined_hit10": refined_hit10,
                    "margin": (margin_sum / active_total) if active_total > 0 else 0.0,
                    "soft_margin": (soft_margin_sum / active_total) if active_total > 0 else 0.0,
                }

        mean_val = sum(val_losses.values()) / max(1, len(val_losses))
        mean_mae = sum(stat["mae"] for stat in val_stats.values()) / max(1, len(val_stats))
        mean_refined_mae = sum(stat["refined_mae"] for stat in val_stats.values()) / max(1, len(val_stats))
        scheduler.step(mean_mae)

        if mean_val < best_loss:
            best_loss = mean_val
            best_loss_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "heads_config": heads_config,
                    "features": args.features,
                    "use_tnet": args.use_tnet,
                    "coord_loss_weight": coord_weight,
                    "coord_loss_weight_spec": args.coord_loss_weight_spec,
                    "coord_weight_schedule": args.coord_weight_schedule,
                    "coord_temperature": coord_temp,
                    "coord_temperature_spec": args.coord_temperature_spec,
                    "coord_temperature_schedule": args.coord_temperature_schedule,
                    "coord_topk": args.coord_topk,
                    "peak_ce": peak_ce_weight,
                    "peak_ce_spec": args.peak_ce_spec,
                    "peak_ce_schedule": args.peak_ce_schedule,
                    "margin_loss_weight": margin_weight,
                    "margin_loss_weight_spec": args.margin_loss_weight_spec,
                    "margin_weight_schedule": args.margin_weight_schedule,
                    "margin_floor": args.margin_floor,
                },
                out_dir / "best_mse.pt",
            )

        if mean_mae < best_mae_metric:
            best_mae_metric = mean_mae
            best_mae_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "heads_config": heads_config,
                    "features": args.features,
                    "use_tnet": args.use_tnet,
                    "coord_loss_weight": coord_weight,
                    "coord_loss_weight_spec": args.coord_loss_weight_spec,
                    "coord_weight_schedule": args.coord_weight_schedule,
                    "coord_temperature": coord_temp,
                    "coord_temperature_spec": args.coord_temperature_spec,
                    "coord_temperature_schedule": args.coord_temperature_schedule,
                    "coord_topk": args.coord_topk,
                    "peak_ce": peak_ce_weight,
                    "peak_ce_spec": args.peak_ce_spec,
                    "peak_ce_schedule": args.peak_ce_schedule,
                    "margin_loss_weight": margin_weight,
                    "margin_loss_weight_spec": args.margin_loss_weight_spec,
                    "margin_weight_schedule": args.margin_weight_schedule,
                    "margin_floor": args.margin_floor,
                },
                out_dir / "best_mae.pt",
            )
            try:
                copy2(out_dir / "best_mae.pt", out_dir / "best.pt")
            except Exception as exc:
                print(f"[warn] failed to refresh best.pt: {exc}")

        if epoch % args.log_every == 0 or epoch in (1, args.epochs):
            parts = [
                f"[epoch {epoch:03d}/{args.epochs}] coord_w {coord_weight:.4f} temp {coord_temp:.3f} "
                f"peak {peak_ce_weight:.3f} margin_w {margin_weight:.3f} "
                f"train {mean_train:.6f} val {mean_val:.6f} "
                f"mean_mae {mean_mae:.6f} refined {mean_refined_mae:.6f} train_refined {mean_train_refined:.6f} "
                f"best_mse {best_loss:.6f} (ep {best_loss_epoch}) "
                f"best_mae {best_mae_metric:.6f} (ep {best_mae_epoch})"
            ]
            for tooth in tooth_list:
                tr = train_losses[tooth]
                vl = val_losses[tooth]
                stat_tr = train_stats[tooth]
                stat_vl = val_stats.get(tooth, stat_tr)
                parts.append(
                    f"{tooth}:trn={tr:.6f}(mae={stat_tr['mae']:.4f}mm refined={stat_tr['refined_mae']:.4f}mm match={stat_tr['matches']}/{stat_tr['count']} "
                    f"hit05={stat_tr['hit05']} hit10={stat_tr['hit10']} margin={stat_tr['margin']:.6f} soft_margin={stat_tr['soft_margin']:.6f})"
                    f"/val={vl:.6f}(mae={stat_vl['mae']:.4f}mm refined={stat_vl['refined_mae']:.4f}mm match={stat_vl['matches']}/{stat_vl['count']} "
                    f"hit05={stat_vl['hit05']} hit10={stat_vl['hit10']} margin={stat_vl['margin']:.6f} soft_margin={stat_vl['soft_margin']:.6f})"
                )
            msg = " | ".join(parts)
            print(msg)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

        if args.early_stop and (epoch - best_mae_epoch) >= args.early_stop:
            stop_msg = f"[early-stop] no MAE improvement for {args.early_stop} epochs; stopping at epoch {epoch}."
            print(stop_msg)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(stop_msg + "\n")
            break

    final_coord_weight = compute_scheduled_value(args.epochs, args.coord_loss_weight, args.coord_weight_schedule)
    final_coord_temp = compute_scheduled_value(args.epochs, args.coord_temperature, args.coord_temperature_schedule)
    final_peak_ce = compute_scheduled_value(args.epochs, args.peak_ce, args.peak_ce_schedule)
    final_margin_weight = compute_scheduled_value(args.epochs, args.margin_loss_weight, args.margin_weight_schedule)

    torch.save(
        {
            "model": model.state_dict(),
            "in_channels": in_channels,
            "heads_config": heads_config,
            "features": args.features,
            "use_tnet": args.use_tnet,
            "coord_loss_weight": final_coord_weight,
            "coord_loss_weight_spec": args.coord_loss_weight_spec,
            "coord_weight_schedule": args.coord_weight_schedule,
            "coord_temperature": final_coord_temp,
            "coord_temperature_spec": args.coord_temperature_spec,
            "coord_temperature_schedule": args.coord_temperature_schedule,
            "coord_topk": args.coord_topk,
            "peak_ce": final_peak_ce,
            "peak_ce_spec": args.peak_ce_spec,
            "peak_ce_schedule": args.peak_ce_schedule,
            "margin_loss_weight": final_margin_weight,
            "margin_loss_weight_spec": args.margin_loss_weight_spec,
            "margin_weight_schedule": args.margin_weight_schedule,
            "margin_floor": args.margin_floor,
        },
        out_dir / "last.pt",
    )
    if (out_dir / "best.pt").exists() is False:
        try:
            copy2(out_dir / "best_mae.pt", out_dir / "best.pt")
        except Exception:
            try:
                copy2(out_dir / "last.pt", out_dir / "best.pt")
            except Exception as exc:
                print(f"[warn] unable to create best.pt fallback: {exc}")

    print(
        f"Training done; best_mse {best_loss:.6f} epoch {best_loss_epoch} -> {out_dir/'best_mse.pt'} | "
        f"best_mae {best_mae_metric:.6f} epoch {best_mae_epoch} -> {out_dir/'best_mae.pt'} | "
        f"last -> {out_dir/'last.pt'}"
    )

    FINAL_PT_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("best_mse.pt", "best_mae.pt", "last.pt"):
        src = out_dir / name
        if src.exists():
            copy2(src, FINAL_PT_DIR / name)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="datasets/p0_npz")
    parser.add_argument("--tooth", type=str, default=",".join(DEFAULT_TOOTH_IDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--features", type=str, choices=["pn", "all", "xyz"], default="pn")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--max_train_samples", type=int, default=None, help="Limit number of training samples per tooth (for overfit).")
    parser.add_argument("--max_val_samples", type=int, default=None, help="Limit number of validation samples per tooth.")
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--case", type=str, default=None, help="Restrict to a specific case ID (e.g. 001).")
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_CHECKPOINT_ROOT))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--disable_tnet", action="store_true", help="Disable TNet alignment.")
    parser.add_argument("--early_stop", type=int, default=0, help="Stop training if no improvement for N epochs (0 = disabled).")
    parser.add_argument("--label_gamma", type=float, default=1.0, help="Optional sharpening factor for targets (y^gamma).")
    parser.add_argument("--loss_power", type=float, default=2.0, help="Exponent applied to target heatmaps for weighted MSE (>=0).")
    parser.add_argument("--loss_eps", type=float, default=1e-3, help="Stability term added to the weighted MSE denominator.")
    parser.add_argument(
        "--peak_ce",
        type=str,
        default="0.0",
        help="Weight for auxiliary peak classification CE loss; accepts scalar or 'w0:w1@T'.",
    )
    parser.add_argument("--heatmap_loss_weight", type=float, default=1.0, help="Weight for primary heatmap regression loss.")
    parser.add_argument(
        "--coord_loss_weight",
        type=str,
        default="0.0",
        help="Weight for soft-argmax coordinate L1 loss; accepts scalar or 'w0:w1@T' for linear ramp.",
    )
    parser.add_argument(
        "--coord_temperature",
        type=str,
        default="1.0",
        help="Temperature for soft-argmax expectation; accepts scalar or 't0:t1@T'.",
    )
    parser.add_argument(
        "--coord_topk",
        type=int,
        default=0,
        help="Optional top-K truncation when computing soft-argmax expectation (0 disables).",
    )
    parser.add_argument(
        "--margin_loss_weight",
        type=str,
        default="0.0",
        help="Weight for logit margin regularizer (0 disables); accepts scalar or 'w0:w1@T'.",
    )
    parser.add_argument(
        "--margin_floor",
        type=float,
        default=0.0,
        help="Target floor for sigmoid(top1-top2) margin when margin regularizer is enabled.",
    )
    args = parser.parse_args()
    args.tooth = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
    args.use_tnet = not args.disable_tnet
    args.case_filter = args.case.strip() if args.case and args.case.strip() else None
    coord_base, coord_schedule, coord_spec = parse_coord_loss_weight(args.coord_loss_weight)
    args.coord_loss_weight = coord_base
    args.coord_weight_schedule = coord_schedule
    args.coord_loss_weight_spec = coord_spec
    peak_base, peak_schedule, peak_spec = parse_schedule_spec(args.peak_ce, "peak_ce", 0.0)
    args.peak_ce = peak_base
    args.peak_ce_schedule = peak_schedule
    args.peak_ce_spec = peak_spec
    temp_base, temp_schedule, temp_spec = parse_schedule_spec(args.coord_temperature, "coord_temperature", 1.0)
    if temp_base <= 0:
        temp_base = 1.0
    args.coord_temperature = temp_base
    args.coord_temperature_schedule = temp_schedule
    args.coord_temperature_spec = temp_spec
    args.coord_topk = max(0, int(args.coord_topk))
    margin_base, margin_schedule, margin_spec = parse_schedule_spec(args.margin_loss_weight, "margin_loss_weight", 0.0)
    args.margin_loss_weight = margin_base
    args.margin_weight_schedule = margin_schedule
    args.margin_loss_weight_spec = margin_spec
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} root={args.root} teeth={args.tooth}")
    train_all_teeth(args, device)


if __name__ == "__main__":
    main()

