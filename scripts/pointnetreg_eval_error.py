#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def _normalise_case_id(case_id: str) -> str:
    cid = str(case_id).strip()
    if cid.isdigit():
        return f"{int(cid):03d}"
    return cid


def _raw_case_stem(case_id: str) -> str:
    cid = str(case_id).strip()
    if cid.isdigit():
        return str(int(cid))
    return cid


def load_markups(path: Path) -> Dict[str, List[float]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    points: Dict[str, List[float]] = {}
    for markup in data.get("markups", []):
        for cp in markup.get("controlPoints", []):
            label = cp.get("label")
            pos = cp.get("position")
            if not label or not isinstance(pos, list) or len(pos) != 3:
                continue
            key = str(label).strip().lower()
            if not key:
                continue
            points[key] = [float(pos[0]), float(pos[1]), float(pos[2])]
    return points


@dataclass
class CaseStats:
    case_id: str
    count: int
    mean_error: float
    missing_pred: int
    missing_gt: int


def compute_case_error(pred_points: Dict[str, List[float]], gt_points: Dict[str, List[float]]) -> Tuple[CaseStats, List[Tuple[str, float]]]:
    pairs: List[Tuple[str, float]] = []
    missing_pred = 0
    missing_gt = 0

    for label, gt_pos in gt_points.items():
        pred_pos = pred_points.get(label)
        if pred_pos is None:
            missing_pred += 1
            continue
        dist = math.dist(pred_pos, gt_pos)
        pairs.append((label, dist))

    for label in pred_points:
        if label not in gt_points:
            missing_gt += 1

    count = len(pairs)
    mean_error = sum(dist for _, dist in pairs) / count if count else float("nan")
    stats = CaseStats(case_id="", count=count, mean_error=mean_error, missing_pred=missing_pred, missing_gt=missing_gt)
    return stats, pairs


def iter_cases(raw_root: Path, cases: Iterable[str] | None) -> List[str]:
    if cases:
        normalised = []
        for cid in cases:
            cid_norm = _normalise_case_id(cid)
            normalised.append(cid_norm)
        return sorted(dict.fromkeys(normalised))
    discovered = []
    for path in raw_root.iterdir():
        if path.is_dir():
            discovered.append(_normalise_case_id(path.name))
    return sorted(dict.fromkeys(discovered))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="计算 PointNet-Reg 预测与标注之间的平均误差。")
    parser.add_argument("--pred-dir", type=Path, default=Path("outputs/pointnetreg/final_output/json"), help="预测 JSON 所在目录。")
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/landmarks_dataset/raw"), help="原始标注所在根目录。")
    parser.add_argument("--cases", nargs="*", help="指定病例编号（默认自动遍历 raw-root 下全部病例）。")
    parser.add_argument("--per-case", action="store_true", help="输出每个病例的平均误差。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pred_dir = args.pred_dir.resolve()
    raw_root = args.raw_root.resolve()

    case_ids = iter_cases(raw_root, args.cases)
    if not case_ids:
        raise SystemExit("未找到任何病例。")

    overall_pairs = 0
    overall_error = 0.0
    per_case_rows: List[CaseStats] = []

    for cid in case_ids:
        pred_path = pred_dir / f"{cid}.json"
        if not pred_path.exists():
            raise FileNotFoundError(f"预测文件缺失：{pred_path}")

        raw_stem = _raw_case_stem(cid)
        raw_dir = raw_root / raw_stem
        if not raw_dir.exists():
            raise FileNotFoundError(f"原始案例目录缺失：{raw_dir}")

        gt_points = {}
        for arch in ("U", "L"):
            gt_path = raw_dir / f"{raw_stem}_{arch}.json"
            gt_points.update(load_markups(gt_path))

        pred_points = load_markups(pred_path)

        case_stats, pairs = compute_case_error(pred_points, gt_points)
        case_stats.case_id = cid
        per_case_rows.append(case_stats)

        overall_pairs += case_stats.count
        overall_error += sum(dist for _, dist in pairs)

    if overall_pairs == 0:
        raise SystemExit("没有匹配到任何 landmark，对应病例可能缺少预测或标注。")

    mean_error = overall_error / overall_pairs
    print(f"病例数: {len(per_case_rows)}")
    print(f"匹配点总数: {overall_pairs}")
    print(f"全局平均误差 (mm): {mean_error:.6f}")

    total_missing_pred = sum(r.missing_pred for r in per_case_rows)
    total_missing_gt = sum(r.missing_gt for r in per_case_rows)
    print(f"预测缺失点总数: {total_missing_pred}")
    print(f"GT 缺失点总数: {total_missing_gt}")

    if args.per_case:
        print("\n病例\t点数\t平均误差(mm)\t预测缺失\tGT缺失")
        for row in per_case_rows:
            mean_val = f"{row.mean_error:.6f}" if math.isfinite(row.mean_error) else "nan"
            print(f"{row.case_id}\t{row.count}\t{mean_val}\t{row.missing_pred}\t{row.missing_gt}")


if __name__ == "__main__":
    main()

