import argparse
import numpy as np
from vedo import load


def read_labels(vtp_path: str, preferred_keys=None) -> np.ndarray:
    """Load cell labels from a VTP file, trying common attribute keys."""
    mesh = load(vtp_path)
    if preferred_keys is None:
        preferred_keys = ("Label", "labels", "PredictedID")
    for key in preferred_keys:
        data = mesh.celldata.get(key)
        if data is not None:
            return np.asarray(data).astype(np.int32).reshape(-1)
    raise KeyError(f"未找到标签字段 {preferred_keys} (file={vtp_path})")


def dice_sen_ppv(pred: np.ndarray, gt: np.ndarray, num_classes: int = 15):
    metrics = {}
    eps = 1e-9
    for k in range(num_classes):
        pred_mask = (pred == k)
        gt_mask = (gt == k)
        inter = np.count_nonzero(pred_mask & gt_mask)
        pred_sum = np.count_nonzero(pred_mask)
        gt_sum = np.count_nonzero(gt_mask)
        dice = (2.0 * inter) / (pred_sum + gt_sum + eps)
        sen = inter / (gt_sum + eps)
        ppv = inter / (pred_sum + eps)
        metrics[k] = dict(dice=dice, sen=sen, ppv=ppv, p=int(pred_sum), g=int(gt_sum))

    tooth_ids = list(range(1, num_classes))
    dice_macro = float(np.mean([m["dice"] for m in metrics.values()])) if metrics else 0.0
    dice_tooth_macro = float(np.mean([metrics[k]["dice"] for k in tooth_ids])) if tooth_ids else 0.0
    dice_gum = metrics[0]["dice"]
    return dict(
        dice_macro=dice_macro,
        dice_tooth_macro=dice_tooth_macro,
        dice_gum=dice_gum,
        per_class=metrics,
    )


def main():
    parser = argparse.ArgumentParser(description="Compute Dice/SEN/PPV for MeshSegNet outputs.")
    parser.add_argument("--pred", required=True, help="预测结果 VTP（*_fine_argmax / *_gc_only / *_predicted_refined）")
    parser.add_argument("--gt", required=True, help="GT 标签 VTP（与预测同一细网格）")
    parser.add_argument("--classes", type=int, default=15, help="标签类别总数（含牙龈）")
    args = parser.parse_args()

    pred_labels = read_labels(args.pred)
    gt_labels = read_labels(args.gt)
    if pred_labels.shape != gt_labels.shape:
        raise SystemExit(f"cell count mismatch: pred {pred_labels.shape} vs gt {gt_labels.shape}")

    result = dice_sen_ppv(pred_labels, gt_labels, num_classes=args.classes)
    print(f"Dice(macro)={result['dice_macro']:.4f} | Dice(tooth)={result['dice_tooth_macro']:.4f} | Dice(gum)={result['dice_gum']:.4f}")
    for cls_id in range(args.classes):
        metrics = result['per_class'][cls_id]
        print(
            f"  cls {cls_id:02d}: dice={metrics['dice']:.4f} "
            f"sen={metrics['sen']:.4f} ppv={metrics['ppv']:.4f} | "
            f"P={metrics['p']} G={metrics['g']}"
        )


if __name__ == "__main__":
    main()
