from pathlib import Path
import argparse, random, time

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg

DEFAULT_TOOTH_IDS = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_loaders(root: str, tooth_id: str, features: str, batch_size: int, workers: int, val_ratio: float, augment: bool):
    cfg = DatasetConfig(
        root=root,
        file_patterns=(f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"),
        features=features,
        select_landmarks="active",
        augment=augment,
    )
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
    return dataset, train_loader, val_loader


def train_one_tooth(args, tooth_id: str, device: torch.device) -> None:
    dataset, train_loader, val_loader = build_loaders(
        root=args.root,
        tooth_id=tooth_id,
        features=args.features,
        batch_size=args.batch_size,
        workers=args.workers,
        val_ratio=args.val_ratio,
        augment=args.augment,
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

    out_dir = Path(args.out_dir) / tooth_id
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = 0
    log_path = out_dir / "log.txt"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} {tooth_id} ===="
                 f" C={sample['x'].shape[0]} L={sample['y'].shape[0]} N={sample['x'].shape[-1]}"
                 f" train={len(train_loader.dataset)} val={len(val_loader.dataset)}\n")

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
            msg = f"[{tooth_id}] epoch {epoch:03d}/{args.epochs} train {train_loss:.6f} val {val_loss:.6f} best {best:.6f} (ep {best_epoch})"
            print(msg)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    print(f"[{tooth_id}] done; best {best:.6f} epoch {best_epoch}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="datasets/p0_npz")
    parser.add_argument("--tooth", type=str, default=",".join(DEFAULT_TOOTH_IDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--features", type=str, choices=["all", "xyz"], default="all")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs_pointnetreg")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()
    args.tooth = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} root={args.root} teeth={args.tooth}")
    for tooth in args.tooth:
        train_one_tooth(args, tooth, device)


if __name__ == "__main__":
    main()
