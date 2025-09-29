from pathlib import Path
import argparse, random, time
import json
from typing import List, Optional, Dict, Any

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg
from tooth_groups import TOOTH_GROUPS, get_group_teeth, is_valid_tooth_id, get_all_tooth_ids

DEFAULT_TOOTH_IDS = get_all_tooth_ids()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_loaders(
    root: str, 
    target: str,  # 可以是tooth_id或group_name
    mode: str,    # "per_tooth" 或 "per_group"
    features: str, 
    batch_size: int, 
    workers: int, 
    val_ratio: float, 
    augment: bool,
    arch_align: bool = False,
    mirror_prob: float = 0.0,
    zscore: bool = False,
    stats_path: Optional[str] = None,
    rotz_deg: float = 15.0,
    trans_mm: float = 1.0
):
    """构建数据加载器，支持按牙位或按组训练"""
    
    if mode == "per_tooth":
        # 按牙位训练
        if not is_valid_tooth_id(target):
            raise ValueError(f"Invalid tooth_id: {target}")
        
        cfg = DatasetConfig(
            root=root,
            file_patterns=(f"*_{target}.npz", f"*_{target.upper()}.npz"),
            features=features,
            select_landmarks="active",
            augment=augment,
            arch_align=arch_align,
            mirror_prob=mirror_prob,
            zscore=zscore,
            stats_path=stats_path,
            rotz_deg=rotz_deg,
            trans_mm=trans_mm,
            ensure_constant_L=True
        )
        
    elif mode == "per_group":
        # 按组训练
        if target not in TOOTH_GROUPS:
            raise ValueError(f"Invalid group name: {target}")
        
        cfg = DatasetConfig(
            root=root,
            group=target,
            features=features,
            select_landmarks="active",
            augment=augment,
            arch_align=arch_align,
            mirror_prob=mirror_prob,
            zscore=zscore,
            stats_path=stats_path,
            rotz_deg=rotz_deg,
            trans_mm=trans_mm,
            ensure_constant_L=True
        )
    else:
        raise ValueError(f"Invalid mode: {mode}, must be 'per_tooth' or 'per_group'")
    
    dataset = P0PointNetRegDataset(cfg)
    val_len = max(1, int(round(len(dataset) * val_ratio)))
    train_len = max(1, len(dataset) - val_len)
    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(2025))

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,
        collate_fn=collate_p0,
        persistent_workers=workers > 0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=max(0, workers // 2),
        pin_memory=True,
        collate_fn=collate_p0,
    )
    return dataset, train_loader, val_loader, train_set, val_set


def train_one_target(args, target: str, mode: str, device: torch.device) -> None:
    """训练单个目标（牙位或组）"""
    dataset, train_loader, val_loader, train_set, val_set = build_loaders(
        root=args.root,
        target=target,
        mode=mode,
        features=args.features,
        batch_size=args.batch_size,
        workers=args.workers,
        val_ratio=args.val_ratio,
        augment=args.augment,
        arch_align=args.arch_align,
        mirror_prob=args.mirror_prob,
        zscore=args.zscore,
        stats_path=args.stats_path,
        rotz_deg=args.rotz_deg,
        trans_mm=args.trans_mm,
    )

    sample = dataset[0]
    model = PointNetReg(
        in_channels=sample["x"].shape[0],
        num_landmarks=sample["y"].shape[0],
        use_tnet=True,
        return_logits=False,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(enabled=device.type == "cuda")
    criterion = torch.nn.MSELoss()

    # 根据模式创建输出目录
    out_dir = Path(args.out_dir) / mode / target
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = 0
    log_path = out_dir / "log.txt"
    
    # 计算数据集大小（避免DataLoader中的Subset长度问题）
    train_size = len(train_set)
    val_size = len(val_set)
    
    # 保存训练配置
    config = {
        "target": target,
        "mode": mode,
        "features": args.features,
        "augment": args.augment,
        "arch_align": args.arch_align,
        "mirror_prob": args.mirror_prob,
        "zscore": args.zscore,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "epochs": args.epochs,
        "in_channels": sample["x"].shape[0],
        "num_landmarks": sample["y"].shape[0],
        "num_points": sample["x"].shape[-1],
        "train_size": train_size,
        "val_size": val_size,
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {config['timestamp']} {target} ({mode}) ===="
                 f" C={config['in_channels']} L={config['num_landmarks']} N={config['num_points']}"
                 f" train={train_size} val={val_size} augment={args.augment}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optim.zero_grad(set_to_none=True)
            with autocast(enabled=device.type == "cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()
            train_loss += loss.item()
        train_loss /= max(1, len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            for batch in val_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                val_loss += criterion(model(x), y).item()
        val_loss /= max(1, len(val_loader))
        scheduler.step(val_loss)

        if val_loss < best:
            best = val_loss
            best_epoch = epoch
            torch.save({"model": model.state_dict(), "in_channels": sample["x"].shape[0], "num_landmarks": sample["y"].shape[0]}, out_dir / "best.pt")

        if epoch % args.log_every == 0 or epoch in (1, args.epochs):
            msg = f"[{target}] epoch {epoch:03d}/{args.epochs} train {train_loss:.6f} val {val_loss:.6f} best {best:.6f} (ep {best_epoch})"
            print(msg)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    print(f"[{target}] done; best {best:.6f} epoch {best_epoch}")


def parse_args():
    parser = argparse.ArgumentParser(description="PointNetReg Training with Group Support")
    
    # 基础参数
    parser.add_argument("--root", type=str, default="datasets/landmarks_dataset/cooked/p0/samples_consistent",
                       help="Dataset root directory")
    parser.add_argument("--out_dir", type=str, default="runs_pointnetreg",
                       help="Output directory for models and logs")
    
    # 训练模式
    parser.add_argument("--mode", type=str, choices=["per_tooth", "per_group"], default="per_tooth",
                       help="Training mode: per_tooth or per_group")
    parser.add_argument("--tooth", type=str, default=",".join(DEFAULT_TOOTH_IDS),
                       help="Comma-separated tooth IDs (for per_tooth mode)")
    parser.add_argument("--group", type=str, choices=list(TOOTH_GROUPS.keys()), 
                       help="Group name (for per_group mode)")
    
    # 数据增强参数
    parser.add_argument("--augment", action="store_true", help="Enable data augmentation")
    parser.add_argument("--arch_align", action="store_true", help="Enable arch alignment")
    parser.add_argument("--mirror_prob", type=float, default=0.0, help="Mirror augmentation probability")
    parser.add_argument("--zscore", action="store_true", help="Enable z-score normalization")
    parser.add_argument("--stats_path", type=str, help="Path to statistics file for z-score normalization")
    parser.add_argument("--rotz_deg", type=float, default=15.0, help="Rotation augmentation range (degrees)")
    parser.add_argument("--trans_mm", type=float, default=1.0, help="Translation augmentation range (mm)")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation split ratio")
    
    # 其他参数
    parser.add_argument("--features", type=str, choices=["all", "xyz"], default="all",
                       help="Feature type")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--log_every", type=int, default=1, help="Log every N epochs")
    
    args = parser.parse_args()
    
    # 解析牙位列表
    if args.mode == "per_tooth":
        args.tooth = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
        # 验证牙位ID
        invalid_teeth = [t for t in args.tooth if not is_valid_tooth_id(t)]
        if invalid_teeth:
            raise ValueError(f"Invalid tooth IDs: {invalid_teeth}")
    elif args.mode == "per_group":
        if not args.group:
            raise ValueError("--group is required when mode is per_group")
        args.tooth = [args.group]  # 统一接口
    
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"🚀 PointNetReg Training")
    print(f"   Device: {device}")
    print(f"   Mode: {args.mode}")
    print(f"   Root: {args.root}")
    print(f"   Targets: {args.tooth}")
    print(f"   Augmentation: {args.augment}")
    if args.augment:
        print(f"   - Arch align: {args.arch_align}")
        print(f"   - Mirror prob: {args.mirror_prob}")
        print(f"   - Z-score: {args.zscore}")
        print(f"   - Rotation: ±{args.rotz_deg}°")
        print(f"   - Translation: ±{args.trans_mm}mm")
    print()
    
    for target in args.tooth:
        print(f"🦷 Starting training for {target} ({args.mode})")
        train_one_target(args, target, args.mode, device)
        print()


if __name__ == "__main__":
    main()
