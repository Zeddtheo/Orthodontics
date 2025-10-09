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
        select_landmarks="all",
        augment=augment,
    )
    dataset = P0PointNetRegDataset(cfg)
    n = len(dataset)
    if n == 0:
        raise ValueError(f"No samples found for tooth {tooth_id} in {root}")
    val_len = int(round(n * val_ratio))
    val_len = min(max(0, val_len), max(0, n - 1))
    train_len = n - val_len
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
        return_logits=False,
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode="min", factor=0.5, patience=5)
    scaler = GradScaler(enabled=device.type == "cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best = float("inf")
    best_epoch = 0
    log_path = out_dir / "log.txt"
    with open(log_path, "a", encoding="utf-8") as fh:
        fh.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} multi-tooth ===="
                 f" teeth={args.tooth} in_channels={in_channels}\n")

    tooth_list = list(loaders.keys())

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = {}
        for tooth in tooth_list:
            train_loader = loaders[tooth]["train"]
            running = 0.0
            for batch in train_loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True)
                mask = batch.get("mask")
                if mask is not None:
                    mask = mask.to(device, non_blocking=True).float()
                else:
                    mask = torch.ones(y.shape[0], y.shape[1], device=device)
                mask_exp = mask.unsqueeze(-1)
                optim.zero_grad(set_to_none=True)
                with autocast(enabled=device.type == "cuda"):
                    pred = model(x, tooth_id=tooth)
                    loss_map = (pred - y) ** 2
                    loss = (loss_map * mask_exp).sum() / mask_exp.sum().clamp(min=1e-12)
                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optim)
                scaler.update()
                running += loss.item()
            train_losses[tooth] = running / max(1, len(train_loader))

        mean_train = sum(train_losses.values()) / max(1, len(train_losses))

        model.eval()
        val_losses = {}
        with torch.no_grad(), autocast(enabled=device.type == "cuda"):
            for tooth in tooth_list:
                val_loader = loaders[tooth]["val"]
                if len(val_loader.dataset) == 0:
                    val_losses[tooth] = train_losses[tooth]
                    continue
                running = 0.0
                for batch in val_loader:
                    x = batch["x"].to(device, non_blocking=True)
                    y = batch["y"].to(device, non_blocking=True)
                    mask = batch.get("mask")
                    if mask is not None:
                        mask = mask.to(device, non_blocking=True).float()
                    else:
                        mask = torch.ones(y.shape[0], y.shape[1], device=device)
                    mask_exp = mask.unsqueeze(-1)
                    pred = model(x, tooth_id=tooth)
                    loss_map = (pred - y) ** 2
                    running += ((loss_map * mask_exp).sum() / mask_exp.sum().clamp(min=1e-12)).item()
                val_losses[tooth] = running / max(1, len(val_loader))

        mean_val = sum(val_losses.values()) / max(1, len(val_losses))
        scheduler.step(mean_val)

        if mean_val < best:
            best = mean_val
            best_epoch = epoch
            torch.save(
                {
                    "model": model.state_dict(),
                    "in_channels": in_channels,
                    "heads_config": heads_config,
                    "features": args.features,
                    "use_tnet": args.use_tnet,
                },
                out_dir / "best.pt",
            )

        if epoch % args.log_every == 0 or epoch in (1, args.epochs):
            parts = [f"[epoch {epoch:03d}/{args.epochs}] train {mean_train:.6f} val {mean_val:.6f} best {best:.6f} (ep {best_epoch})"]
            parts.extend(f"{tooth}:trn={train_losses[tooth]:.6f}/val={val_losses[tooth]:.6f}" for tooth in tooth_list)
            msg = " | ".join(parts)
            print(msg)
            with open(log_path, "a", encoding="utf-8") as fh:
                fh.write(msg + "\n")

    print(f"Training done; best {best:.6f} epoch {best_epoch} saved to {out_dir/'best.pt'}")


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
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--out_dir", type=str, default="runs_pointnetreg")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--disable_tnet", action="store_true", help="Disable TNet alignment.")
    args = parser.parse_args()
    args.tooth = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
    args.use_tnet = not args.disable_tnet
    return args


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device} root={args.root} teeth={args.tooth}")
    train_all_teeth(args, device)


if __name__ == "__main__":
    main()
