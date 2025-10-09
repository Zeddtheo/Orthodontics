# p5_overfit.py
# Single-sample overfit test for PointNetReg.
# This script trains the model on a single data point to verify that it can
# achieve a very low loss, which is a good sanity check for the model and training pipeline.

import argparse
import random
from pathlib import Path

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg

DEFAULT_TOOTH_IDS = [
    "t11", "t12", "t13", "t14", "t15", "t16", "t17",
    "t21", "t22", "t23", "t24", "t25", "t26", "t27",
    "t31", "t32", "t33", "t34", "t35", "t36", "t37",
    "t41", "t42", "t43", "t44", "t45", "t46", "t47",
]


def set_seed(seed: int) -> None:
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_single_sample_loader(root: str, tooth_id: str, features: str, sample_idx: int):
    """Loads a single sample from the dataset and prepares a DataLoader for it."""
    cfg = DatasetConfig(
        root=root,
        file_patterns=(f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"),
        features=features,   # 建议用 'pn'（pos+nrm+cent_rel）
        select_landmarks="all",
        augment=False,
        ensure_constant_L=False,
    )
    dataset = P0PointNetRegDataset(cfg)
    if not dataset:
        raise ValueError(f"No data found for tooth '{tooth_id}' in '{root}'.")
    if sample_idx >= len(dataset):
        raise IndexError(f"sample_idx {sample_idx} is out of bounds for dataset of size {len(dataset)}.")

    # Get the single sample
    single_sample = dataset[sample_idx]
    
    # Create a loader that will repeatedly yield the same sample
    single_item_loader = DataLoader(
        [single_sample],
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_p0,
    )
    return single_item_loader, single_sample


def _gather_points(xyz: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    xyz: (B,3,N), indices: (B,L) -> (B,L,3)
    """
    gather_idx = indices.unsqueeze(1).expand(-1, 3, -1)
    pts = torch.gather(xyz, 2, gather_idx)  # (B,3,L)
    return pts.permute(0, 2, 1).contiguous()


def overfit_one_tooth(args, tooth_id: str, device: torch.device):
    """Runs the overfit test for a single tooth."""
    try:
        loader, sample = get_single_sample_loader(
            root=args.root,
            tooth_id=tooth_id,
            features=args.features,
            sample_idx=args.sample_idx,
        )
    except (ValueError, IndexError) as e:
        print(f"Could not load sample for tooth '{tooth_id}': {e}")
        return

    head_key = tooth_id.lower()
    model = PointNetReg(
        in_channels=sample["x"].shape[0],
        num_landmarks=sample["y"].shape[0],
        heads_config={head_key: sample["y"].shape[0]},
        use_tnet=args.use_tnet,
        return_logits=True,
        dropout_p=0.0,    # 过拟合时关闭
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scaler = GradScaler(enabled=False)  # 单样本更稳；若你想更快可再打开
    mse_loss = torch.nn.MSELoss(reduction="none")

    print(f"\n==== Overfitting test for tooth: {tooth_id} | Sample: {args.sample_idx} ====")
    print(f" C={sample['x'].shape[0]} L={sample['y'].shape[0]} N={sample['x'].shape[-1]}")
    print(f" Starting training for {args.epochs} epochs...")

    best_mae = float("inf")
    best_path = None
    if args.save_model:
        ckpt_dir = Path(args.out_dir) / tooth_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        best_path = ckpt_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True).float()
            else:
                mask = torch.ones(y.shape[0], y.shape[1], device=device)
            mask_exp = mask.unsqueeze(-1)
            xyz = x[:, :3, :]  # ROI 局部 (B,3,N)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=False):  # 全精度，确保单样本稳定收敛
                logits = model(x, tooth_id=head_key)
                probs = torch.sigmoid(logits)
                loss_target = y
                if args.label_gamma != 1.0:
                    loss_target = torch.clamp(y.pow(args.label_gamma), 0.0, 1.0)
                weight = torch.pow(loss_target, args.loss_power) + args.loss_eps
                weight = weight * mask_exp
                loss_map = mse_loss(probs, loss_target)
                denom = torch.clamp(weight.sum(), min=1e-12)
                loss = (loss_map * weight).sum() / denom

                peak_ce_val = torch.tensor(0.0, device=device)
                if args.peak_ce > 0.0:
                    active_flat = mask_exp.view(-1) > 0.5
                    if active_flat.any():
                        logits_flat = logits.view(-1, logits.shape[-1])[active_flat]
                        gt_flat = torch.argmax(y, dim=-1).view(-1)[active_flat]
                        peak_ce_val = torch.nn.functional.cross_entropy(logits_flat, gt_flat)
                        loss = loss + args.peak_ce * peak_ce_val
                    else:
                        peak_ce_val = torch.tensor(0.0, device=device)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()
            train_loss = loss.item()
            peak_ce_item = float(peak_ce_val.detach().cpu().item())

            with torch.no_grad():
                mask_bool = mask > 0.5
                # `probs` 已经在前向中计算；若 autocast 关闭，其仍可复用
                pred_idx = torch.argmax(probs, dim=-1)  # (B,L)
                gt_idx = torch.argmax(y, dim=-1)       # (B,L)
                pred_pts = _gather_points(xyz, pred_idx)
                gt_pts = _gather_points(xyz, gt_idx)
                errors = torch.norm(pred_pts - gt_pts, dim=-1)  # (B,L)
                errors_active = errors[mask_bool]
                if errors_active.numel() > 0:
                    mae_mm = float(errors_active.mean().item())
                    max_mm = float(errors_active.max().item())
                else:
                    mae_mm = 0.0
                    max_mm = 0.0
                matches = int(((pred_idx == gt_idx) & mask_bool).sum().item())
                total = int(mask_bool.sum().item())
                hit05 = int(((errors <= 0.5) & mask_bool).sum().item())
                hit10 = int(((errors <= 1.0) & mask_bool).sum().item())
                topk_vals = probs.topk(2, dim=-1).values
                margin_vals = topk_vals[..., 0] - topk_vals[..., 1]
                margin_active = margin_vals[mask_bool]
                mean_margin = float(margin_active.mean().item()) if margin_active.numel() > 0 else 0.0

            if args.save_model and mae_mm < best_mae:
                best_mae = mae_mm
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "peak_ce": peak_ce_item,
                        "use_tnet": model.use_tnet,
                        "heads_config": model.heads_config if model.multi_head else None,
                        "in_channels": sample["x"].shape[0],
                        "num_landmarks": sample["y"].shape[0],
                        "mae_mm": mae_mm,
                        "max_mm": max_mm,
                        "matches": matches,
                        "total": total,
                        "hit@0.5mm": hit05,
                        "hit@1.0mm": hit10,
                        "mean_margin": mean_margin,
                    },
                    best_path,
                )
                print(
                    f"[{tooth_id}] ★ best checkpoint | epoch {epoch} "
                    f"loss {train_loss:.6f} mae {mae_mm:.6f}mm matches {matches}/{total} "
                    f"hit@0.5 {hit05}/{total} hit@1.0 {hit10}/{total} margin {mean_margin:.6f}"
                )

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            print(
                f"[{tooth_id}] epoch {epoch:03d}/{args.epochs} "
                f"train_loss {train_loss:.8f} | mae {mae_mm:.6f}mm | "
                f"matches {matches}/{total} | hit@0.5 {hit05}/{total} | "
                f"hit@1.0 {hit10}/{total} | margin {mean_margin:.6f} | peak_ce {peak_ce_item:.6f}"
            )

    print(f"[{tooth_id}] Overfitting test finished. Final loss: {train_loss:.8f}")
    if args.save_model and best_mae == float("inf"):
        # 没有触发保存（例如训练早停前失败），兜底写一次
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": args.epochs,
                "train_loss": train_loss,
                "use_tnet": model.use_tnet,
                "heads_config": model.heads_config if model.multi_head else None,
                "in_channels": sample["x"].shape[0],
                "num_landmarks": sample["y"].shape[0],
            },
            best_path,
        )
        print(f"[{tooth_id}] Model saved to {best_path} (fallback write)")

    return model, sample


def overfit_shared(args, device: torch.device):
    """Overfit multiple teeth with a shared backbone + multiple heads."""
    tooth_ids = [t.strip() for t in args.tooth if t.strip()]
    loaders = {}
    samples = {}
    heads_config = {}
    in_channels = None

    for tooth in tooth_ids:
        try:
            loader, sample = get_single_sample_loader(
                root=args.root,
                tooth_id=tooth,
                features=args.features,
                sample_idx=args.sample_idx,
            )
        except (ValueError, IndexError) as e:
            print(f"[shared] skip tooth '{tooth}': {e}")
            continue
        key = tooth.lower()
        loaders[key] = loader
        samples[key] = sample
        heads_config[key] = sample["y"].shape[0]
        if in_channels is None:
            in_channels = sample["x"].shape[0]

    if not loaders:
        print("[shared] No valid teeth to overfit. Abort.")
        return

    model = PointNetReg(
        in_channels=in_channels,
        num_landmarks=max(heads_config.values()),
        heads_config=heads_config,
        use_tnet=args.use_tnet,
        return_logits=True,
        dropout_p=0.0,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scaler = GradScaler(enabled=False)
    mse_loss = torch.nn.MSELoss(reduction="none")

    out_dir = Path(args.out_dir) / "shared"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_mae = float("inf")
    best_path = out_dir / "best.pt" if args.save_model else None

    print(f"\n==== Shared overfit for teeth: {', '.join(heads_config.keys())} | Sample_idx {args.sample_idx} ====")
    print(f" in_channels={in_channels} heads={heads_config}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_metrics = {}
        total_weight = 0.0
        accum_mae = 0.0
        accum_peak = 0.0

    for tooth, loader in loaders.items():
        sample = samples[tooth]
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch.get("mask")
            if mask is not None:
                mask = mask.to(device, non_blocking=True).float()
            else:
                mask = torch.ones(y.shape[0], y.shape[1], device=device)
            mask_exp = mask.unsqueeze(-1)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=False):
                logits = model(x, tooth_id=tooth)
                probs = torch.sigmoid(logits)
                loss_target = y
                if args.label_gamma != 1.0:
                    loss_target = torch.clamp(y.pow(args.label_gamma), 0.0, 1.0)
                weight = torch.pow(loss_target, args.loss_power) + args.loss_eps
                weight = weight * mask_exp
                loss_map = mse_loss(probs, loss_target)
                denom = torch.clamp(weight.sum(), min=1e-12)
                loss = (loss_map * weight).sum() / denom

                peak_ce_val = torch.tensor(0.0, device=device)
                if args.peak_ce > 0.0:
                    active_flat = mask_exp.view(-1) > 0.5
                    if active_flat.any():
                        logits_flat = logits.view(-1, logits.shape[-1])[active_flat]
                        gt_flat = torch.argmax(y, dim=-1).view(-1)[active_flat]
                        peak_ce_val = torch.nn.functional.cross_entropy(logits_flat, gt_flat)
                        loss = loss + args.peak_ce * peak_ce_val

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()

            with torch.no_grad():
                mask_bool = mask > 0.5
                pred_idx = torch.argmax(probs, dim=-1)
                gt_idx = torch.argmax(y, dim=-1)
                pred_pts = _gather_points(x[:, :3, :], pred_idx)
                gt_pts = _gather_points(x[:, :3, :], gt_idx)
                errors = torch.norm(pred_pts - gt_pts, dim=-1)

                errors_active = errors[mask_bool]
                mae_mm = float(errors_active.mean().item()) if errors_active.numel() > 0 else 0.0
                matches = int(((pred_idx == gt_idx) & mask_bool).sum().item())
                active_count = int(mask_bool.sum().item())
                peak_ce_item = float(peak_ce_val.detach().cpu().item())
                topk_vals = probs.topk(2, dim=-1).values
                margin_vals = topk_vals[..., 0] - topk_vals[..., 1]
                margin_active = margin_vals[mask_bool]
                mean_margin = float(margin_active.mean().item()) if margin_active.numel() > 0 else 0.0

                epoch_metrics[tooth] = {
                    "mae": mae_mm,
                    "matches": matches,
                    "count": active_count,
                    "margin": mean_margin,
                    "peak_ce": peak_ce_item,
                }
                accum_mae += mae_mm * active_count
                total_weight += active_count
                accum_peak += peak_ce_item

        mean_mae = accum_mae / total_weight if total_weight > 0 else 0.0
        mean_peak = accum_peak / max(1, len(epoch_metrics))

        if args.save_model and best_path and mean_mae < best_mae:
            best_mae = mean_mae
            torch.save(
                {
                    "model": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": float(mean_mae),
                    "use_tnet": model.use_tnet,
                    "heads_config": heads_config,
                    "in_channels": in_channels,
                    "metrics": epoch_metrics,
                },
                best_path,
            )
            print(f"[shared] ★ best checkpoint | epoch {epoch} mean_mae {mean_mae:.6f} mean_peak {mean_peak:.6f}")

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            parts = [f"[shared] epoch {epoch:03d}/{args.epochs} mean_mae {mean_mae:.6f} mean_peak {mean_peak:.6f}"]
            parts.extend(
                f"{tooth}: mae={m['mae']:.6f} matches={m['matches']}/{m['count']} margin={m['margin']:.6f}"
                for tooth, m in epoch_metrics.items()
            )
            print(" | ".join(parts))

    print(f"[shared] Overfitting finished. Best mean_mae: {best_mae:.6f}")

    if args.save_model and best_mae == float("inf") and best_path:
        torch.save(
            {
                "model": model.state_dict(),
                "epoch": args.epochs,
                "train_loss": 0.0,
                "use_tnet": model.use_tnet,
                "heads_config": heads_config,
                "in_channels": in_channels,
            },
            best_path,
        )
        print(f"[shared] Model saved to {best_path} (fallback write)")


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="PointNetReg Overfitting Test")
    parser.add_argument("--root", type=str, default="datasets/landmarks_dataset/cooked/samples", help="Root directory of the NPZ dataset.")
    parser.add_argument(
        "--tooth",
        type=str,
        nargs="+",
        default=["t11"],
        help="Tooth IDs to test (e.g. t27) or 'all'. Multiple values or comma-separated tokens accepted.",
    )
    parser.add_argument("--epochs", type=int, default=600, help="Number of epochs to run the test.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--features", type=str, choices=["pn", "xyz"], default="pn", help="Features to use.")
    parser.add_argument("--sample_idx", type=int, default=0, help="The index of the sample to use for overfitting.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--log_every", type=int, default=10, help="Log frequency.")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model.")
    parser.add_argument("--out_dir", type=str, default="outputs/landmarks/overfit", help="Output directory for saving models.")
    parser.add_argument("--use_tnet", action="store_true", help="Enable TNet alignment (default disabled).")
    parser.add_argument("--loss_power", type=float, default=2.0, help="Exponent applied to target heatmaps for weighted MSE (>=0).")
    parser.add_argument("--loss_eps", type=float, default=1e-3, help="Stability term added to the weighted MSE denominator.")
    parser.add_argument("--label_gamma", type=float, default=1.0, help="Optional sharpening factor for targets (y^gamma).")
    parser.add_argument("--peak_ce", type=float, default=0.0, help="Weight for auxiliary peak classification CE loss.")
    parser.add_argument("--shared_model", action="store_true", help="Overfit all specified teeth with a shared backbone + multiple heads.")

    args = parser.parse_args()
    tokens: list[str] = []
    for token in args.tooth:
        token = token.strip()
        if not token:
            continue
        if "," in token:
            tokens.extend([t.strip() for t in token.split(",") if t.strip()])
        else:
            tokens.append(token)
    if not tokens:
        tokens = ["t11"]
    if len(tokens) == 1 and tokens[0].lower() == "all":
        args.tooth = DEFAULT_TOOTH_IDS
    else:
        args.tooth = tokens
    return args


def main():
    """Main execution function."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Root: {args.root} | Teeth: {args.tooth}")
    if args.shared_model:
        overfit_shared(args, device)
    else:
        for tooth in args.tooth:
            overfit_one_tooth(args, tooth, device)


if __name__ == "__main__":
    main()
