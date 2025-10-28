#!/usr/bin/env python3
"""
Batch runner that applies calc_metrics.generate_metrics() to a series of
landmark JSON files and writes the combined results into a Markdown report.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


def _resolve_src_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / "src"


KEY_ORDER = [
    "Arch_Form",
    "Arch_Width",
    "Bolton_Ratio_Front",
    "Bolton_Ratio_All",
    "Canine_Relationship_Right",
    "Canine_Relationship_Left",
    "Crossbite",
    "Crowding_Up",
    "Crowding_Down",
    "Curve_of_Spee",
    "Midline_Alignment",
    "Molar_Relationship_Right",
    "Molar_Relationship_Left",
    "Overbite",
    "Overjet",
]


def _build_report_data(
    report_title: str,
    entries: Iterable[Tuple[str, List[Path], dict]],
) -> List[str]:
    blocks: List[str] = []
    for label, sources, metrics in entries:
        metrics_payload: Dict[str, Any] = {}
        for key in KEY_ORDER:
            if key == "Bolton_Ratio_Front":
                metrics_payload[key] = str(metrics.get("Bolton_Ratio", "缺失"))
            elif key == "Bolton_Ratio_All":
                overall_raw = metrics.get("Bolton_Overall_Ratio")
                if isinstance(overall_raw, (int, float)):
                    metrics_payload[key] = f"{overall_raw / 100:.2f}"
                elif overall_raw is None:
                    metrics_payload[key] = "缺失"
                else:
                    metrics_payload[key] = overall_raw
            else:
                metrics_payload[key] = metrics.get(key, "缺失")
        lines: List[str] = []
        for key in KEY_ORDER:
            val = metrics_payload.get(key)
            if isinstance(val, str):
                val_str = val
            else:
                val_str = json.dumps(val, ensure_ascii=False)
            lines.append(f"{key}: {val_str}")
        blocks.append("\n".join(lines))
    return blocks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate Markdown metrics report for landmark JSON files.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Paths to landmark JSON files (Slicer Markups / dots.json).",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="Output Markdown file path.",
    )
    parser.add_argument(
        "--title",
        default="Metrics Report",
        help="Title for the Markdown report.",
    )
    parser.add_argument(
        "--group-by-parent",
        action="store_true",
        help="Group inputs by their parent directory and merge each group.",
    )
    parser.add_argument(
        "--group-label-parent-depth",
        type=int,
        default=0,
        help="When grouping by parent, climb this many extra levels for the group label.",
    )
    parser.add_argument(
        "--no-extended",
        action="store_true",
        help="Do not request extended metric fields (Bolton anterior/overall breakdown, etc.).",
    )
    args = parser.parse_args()

    src_path = _resolve_src_path()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        import calc_metrics  # type: ignore
    except Exception as exc:  # pragma: no cover - defensive
        parser.error(f"无法导入 calc_metrics: {exc}")

    inputs: list[Path] = []
    for item in args.inputs:
        path = Path(item)
        if not path.exists():
            parser.error(f"输入文件不存在: {path}")
        inputs.append(path)

    cfg: Dict[str, Any] = {}
    if not args.no_extended:
        cfg["include_extended_fields"] = True

    entries: List[Tuple[str, List[Path], dict]] = []
    if args.group_by_parent:
        grouped: Dict[Path, List[Path]] = {}
        for path in inputs:
            grouped.setdefault(path.parent, []).append(path)
        for parent in sorted(grouped):
            group_paths = sorted(grouped[parent])
            metrics = calc_metrics.generate_metrics([str(p) for p in group_paths], cfg=cfg or None)
            label_path = parent
            steps = max(args.group_label_parent_depth, 0)
            for _ in range(steps):
                next_parent = label_path.parent
                if next_parent == label_path:
                    break
                label_path = next_parent
            entries.append((label_path.as_posix(), group_paths, metrics))
    else:
        for path in inputs:
            metrics = calc_metrics.generate_metrics(str(path), cfg=cfg or None)
            entries.append((path.as_posix(), [path], metrics))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = _build_report_data(args.title, entries)
    out_path.write_text("\n\n".join(blocks), encoding="utf-8")

    print(f"Wrote metrics for {len(entries)} files → {out_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
