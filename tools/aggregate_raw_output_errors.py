#!/usr/bin/env python3
"""
Aggregate landmark error statistics between ground-truth JSONs and
predictions stored under outputs/raw_output.

Writes a Markdown summary with overall distribution metrics and the
top-5 worst landmarks by mean error.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

Vec3 = Tuple[float, float, float]


def _load_landmarks(path: Path) -> Dict[str, Vec3]:
    data = json.loads(path.read_text(encoding="utf-8"))
    lm: Dict[str, Vec3] = {}
    for markup in data.get("markups", []):
        for point in markup.get("controlPoints", []):
            label = point.get("label")
            position = point.get("position")
            if (
                isinstance(label, str)
                and isinstance(position, list)
                and len(position) == 3
            ):
                try:
                    coords = (float(position[0]), float(position[1]), float(position[2]))
                except (TypeError, ValueError):
                    continue
                lm[label] = coords
    return lm


def _euclidean(a: Vec3, b: Vec3) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _percentile(sorted_values: Sequence[float], pct: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    k = pct * (len(sorted_values) - 1)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_values[int(k)]
    d0 = sorted_values[f] * (c - k)
    d1 = sorted_values[c] * (k - f)
    return d0 + d1


def _summaries(values: List[float]) -> Dict[str, float]:
    values_sorted = sorted(values)
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else float("nan"),
        "median": statistics.median(values) if values else float("nan"),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": values_sorted[0] if values_sorted else float("nan"),
        "max": values_sorted[-1] if values_sorted else float("nan"),
        "p90": _percentile(values_sorted, 0.90),
        "p95": _percentile(values_sorted, 0.95),
    }


def _format_float(value: float, digits: int = 3) -> str:
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate landmark prediction errors.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/landmarks_dataset/raw"),
        help="Ground-truth landmark directory.",
    )
    parser.add_argument(
        "--pred-root",
        type=Path,
        default=Path("outputs/raw_output"),
        help="Prediction directory produced by run_batch_ios_pointnet.py.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/raw_output_summary.md"),
        help="Markdown file to write summary statistics.",
    )
    args = parser.parse_args(argv)

    raw_root = args.raw_root.resolve()
    pred_root = args.pred_root.resolve()

    if not raw_root.exists():
        parser.error(f"raw root not found: {raw_root}")
    if not pred_root.exists():
        parser.error(f"prediction root not found: {pred_root}")

    per_label: Dict[str, List[float]] = defaultdict(list)
    all_distances: List[float] = []
    per_case_stats: Dict[str, Dict[str, float]] = {}
    missing_pred: Dict[str, List[str]] = defaultdict(list)
    missing_gt: Dict[str, List[str]] = defaultdict(list)

    case_dirs = sorted(
        (p for p in pred_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )

    for case_dir in case_dirs:
        case_id = case_dir.name
        raw_dir = raw_root / case_id
        if not raw_dir.exists():
            missing_gt[case_id].append("<case>")
            continue

        pred_paths = {
            "U": case_dir / f"{case_id}_U.json",
            "L": case_dir / f"{case_id}_L.json",
        }
        raw_paths = {
            "U": raw_dir / f"{case_id}_U.json",
            "L": raw_dir / f"{case_id}_L.json",
        }

        case_distances: List[float] = []

        for arch in ("U", "L"):
            pred_path = pred_paths[arch]
            raw_path = raw_paths[arch]
            if not pred_path.exists():
                missing_pred[case_id].append(f"{arch}_pred")
                continue
            if not raw_path.exists():
                missing_gt[case_id].append(f"{arch}_gt")
                continue

            pred = _load_landmarks(pred_path)
            gt = _load_landmarks(raw_path)

            shared = sorted(set(pred) & set(gt))
            if not shared:
                continue

            for label in shared:
                dist = _euclidean(pred[label], gt[label])
                per_label[label].append(dist)
                all_distances.append(dist)
                case_distances.append(dist)

            missing_pred_labels = sorted(set(gt) - set(pred))
            if missing_pred_labels:
                missing_pred[case_id].extend(f"{arch}:{lbl}" for lbl in missing_pred_labels)

            missing_gt_labels = sorted(set(pred) - set(gt))
            if missing_gt_labels:
                missing_gt[case_id].extend(f"{arch}:{lbl}" for lbl in missing_gt_labels)

        if case_distances:
            per_case_stats[case_id] = {
                "mean": statistics.fmean(case_distances),
                "max": max(case_distances),
                "p90": _percentile(sorted(case_distances), 0.90),
            }

    overall = _summaries(all_distances)
    label_means: List[Tuple[str, float, List[float]]] = [
        (label, statistics.fmean(values), values)
        for label, values in per_label.items()
        if values
    ]
    top_high = sorted(label_means, key=lambda item: item[1], reverse=True)[:10]
    top_low = sorted(label_means, key=lambda item: item[1])[:10]

    lines: List[str] = []
    lines.append("# 原始输出标注误差汇总")
    lines.append("")
    lines.append(f"- 已处理病例：{len(per_case_stats)} / {len(case_dirs)}")
    lines.append(f"- 对齐比较的标点总数：{len(all_distances)}")
    lines.append(f"- 整体平均误差：{_format_float(overall['mean'])} mm")
    lines.append(f"- 整体中位误差：{_format_float(overall['median'])} mm")
    lines.append(f"- 整体标准差：{_format_float(overall['std'])} mm")
    lines.append(f"- 90 分位：{_format_float(overall['p90'])} mm")
    lines.append(f"- 95 分位：{_format_float(overall['p95'])} mm")
    lines.append(f"- 最大单点误差：{_format_float(overall['max'])} mm")
    lines.append("")

    if missing_pred or missing_gt:
        lines.append("## 缺失数据统计")
        if missing_pred:
            total_missing = sum(len(v) for v in missing_pred.values())
            lines.append(f"- 预测缺失：{len(missing_pred)} 个病例共 {total_missing} 条")
        if missing_gt:
            total_missing = sum(len(v) for v in missing_gt.values())
            lines.append(f"- 真值缺失：{len(missing_gt)} 个病例共 {total_missing} 条")
        lines.append("")

    lines.append("## 平均误差最高的前 10 个标点")
    if not top_high:
        lines.append("_未找到可比较的标点。_")
    else:
        lines.append("| 排名 | 标点 | 样本数 | 平均 (mm) | 标准差 (mm) | 最小 (mm) | 最大 (mm) | P90 (mm) | P95 (mm) |")
        lines.append("| ---: | :--- | ----: | --------: | ----------: | --------: | --------: | -------: | -------: |")
        for idx, (label, _, values) in enumerate(top_high, start=1):
            stats = _summaries(values)
            lines.append(
                f"| {idx} | {label} | {stats['count']:4d} | "
                f"{_format_float(stats['mean'])} | {_format_float(stats['std'])} | "
                f"{_format_float(stats['min'])} | {_format_float(stats['max'])} | "
                f"{_format_float(stats['p90'])} | {_format_float(stats['p95'])} |"
            )
        lines.append("")

    lines.append("## 平均误差最低的前 10 个标点")
    if not top_low:
        lines.append("_未找到可比较的标点。_")
    else:
        lines.append("| 排名 | 标点 | 样本数 | 平均 (mm) | 标准差 (mm) | 最小 (mm) | 最大 (mm) | P90 (mm) | P95 (mm) |")
        lines.append("| ---: | :--- | ----: | --------: | ----------: | --------: | --------: | -------: | -------: |")
        for idx, (label, _, values) in enumerate(top_low, start=1):
            stats = _summaries(values)
            lines.append(
                f"| {idx} | {label} | {stats['count']:4d} | "
                f"{_format_float(stats['mean'])} | {_format_float(stats['std'])} | "
                f"{_format_float(stats['min'])} | {_format_float(stats['max'])} | "
                f"{_format_float(stats['p90'])} | {_format_float(stats['p95'])} |"
            )
        lines.append("")

    lines.append("## 分病例指标（单位：mm）")
    if not per_case_stats:
        lines.append("_暂无病例统计。_")
    else:
        lines.append("| 病例 | 平均 | 最大 | P90 |")
        lines.append("| ---: | ---: | ---: | ---: |")
        for case_id, stats in sorted(per_case_stats.items(), key=lambda item: int(item[0])):
            lines.append(
                f"| {case_id} | {_format_float(stats['mean'])} | "
                f"{_format_float(stats['max'])} | {_format_float(stats['p90'])} |"
            )
        lines.append("")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[done] summary written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
