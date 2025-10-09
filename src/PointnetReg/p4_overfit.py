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
        select_landmarks="active",
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

    model = PointNetReg(
        in_channels=sample["x"].shape[0],
        num_landmarks=sample["y"].shape[0],
        use_tnet=args.use_tnet,
        return_logits=True,
        dropout_p=0.0,    # 过拟合时关闭
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scaler = GradScaler(enabled=False)  # 单样本更稳；若你想更快可再打开
    criterion = torch.nn.BCEWithLogitsLoss()

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
            xyz = x[:, :3, :]  # ROI 局部 (B,3,N)

            optim.zero_grad(set_to_none=True)
            with autocast(enabled=False):  # 全精度，确保单样本稳定收敛
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optim)
            scaler.update()
            train_loss = loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                pred_idx = torch.argmax(probs, dim=-1)  # (B,L)
                gt_idx = torch.argmax(y, dim=-1)       # (B,L)
                pred_pts = _gather_points(xyz, pred_idx)
                gt_pts = _gather_points(xyz, gt_idx)
                errors = torch.norm(pred_pts - gt_pts, dim=-1)  # (B,L)
                errors_cpu = errors.detach().cpu()
                mae_mm = float(errors_cpu.mean().item())
                max_mm = float(errors_cpu.max().item())
                matches = int((pred_idx == gt_idx).sum().item())
                total = int(pred_idx.numel())
                hit05 = int((errors_cpu <= 0.5).sum().item())
                hit10 = int((errors_cpu <= 1.0).sum().item())
                topk_vals = probs.topk(2, dim=-1).values
                margin_vals = topk_vals[..., 0] - topk_vals[..., 1]
                mean_margin = float(margin_vals.detach().cpu().mean().item())

            if args.save_model and mae_mm < best_mae:
                best_mae = mae_mm
                torch.save(
                    {
                        "model": model.state_dict(),
                        "epoch": epoch,
                        "train_loss": train_loss,
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
                f"hit@1.0 {hit10}/{total} | margin {mean_margin:.6f}"
            )

    print(f"[{tooth_id}] Overfitting test finished. Final loss: {train_loss:.8f}")
    if args.save_model and best_mae == float("inf"):
        # 没有触发保存（例如训练早停前失败），兜底写一次
        torch.save({"model": model.state_dict(), "epoch": args.epochs, "train_loss": train_loss}, best_path)
        print(f"[{tooth_id}] Model saved to {best_path} (fallback write)")

    return model, sample


def parse_args():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="PointNetReg Overfitting Test")
    parser.add_argument("--root", type=str, default="datasets/landmarks_dataset/cooked/samples", help="Root directory of the NPZ dataset.")
    parser.add_argument("--tooth", type=str, default="t11", help="Comma-separated list of tooth IDs to test, or 'all'.")
    parser.add_argument("--epochs", type=int, default=600, help="Number of epochs to run the test.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--features", type=str, choices=["pn", "xyz"], default="pn", help="Features to use.")
    parser.add_argument("--sample_idx", type=int, default=0, help="The index of the sample to use for overfitting.")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed.")
    parser.add_argument("--log_every", type=int, default=10, help="Log frequency.")
    parser.add_argument("--save_model", action="store_true", help="Save the trained model.")
    parser.add_argument("--out_dir", type=str, default="outputs/landmarks/overfit", help="Output directory for saving models.")
    parser.add_argument("--use_tnet", action="store_true", help="Enable TNet alignment (default disabled).")
    
    args = parser.parse_args()
    args.tooth = DEFAULT_TOOTH_IDS if args.tooth.strip().lower() == "all" else [t.strip() for t in args.tooth.split(",") if t.strip()]
    return args


def main():
    """Main execution function."""
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Root: {args.root} | Teeth: {args.tooth}")
    
    for tooth in args.tooth:
        overfit_one_tooth(args, tooth, device)


if __name__ == "__main__":
    main()
