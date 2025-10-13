"""Utility to diagnose missing teeth coverage for PointNetReg pipeline."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

DEFAULT_TEETH = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]


def normalise_case_id(raw: str) -> str:
    """Normalise case ids to zero-padded 3-digit strings where possible."""
    if not raw:
        return raw
    tokens = "".join(ch for ch in raw if ch.isdigit())
    if tokens:
        return f"{int(tokens):03d}"
    return raw.strip()


def discover_p0_cases(p0_root: Path, teeth: Iterable[str]) -> Dict[str, Set[str]]:
    coverage: Dict[str, Set[str]] = defaultdict(set)
    patterns = [f"*_{tooth.lower()}.npz" for tooth in teeth]
    patterns += [f"*_{tooth.upper()}.npz" for tooth in teeth]
    for pattern in patterns:
        for npz_path in p0_root.glob(pattern):
            parts = npz_path.stem.split("_")
            if len(parts) < 3:
                continue
            case_part, arch_part, tooth_part = parts[0], parts[1], parts[2]
            case_id = normalise_case_id(case_part)
            tooth_id = tooth_part.lower()
            coverage[case_id].add(tooth_id)
    return coverage


def discover_infer_cases(infer_root: Path, teeth: Iterable[str]) -> Dict[str, Set[str]]:
    coverage: Dict[str, Set[str]] = defaultdict(set)
    tooth_set = {t.lower() for t in teeth}
    for json_path in infer_root.rglob("*.json"):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        meta = data.get("meta") or {}
        case_id = normalise_case_id(meta.get("case_id") or data.get("case_id") or json_path.stem)
        preds = data.get("predictions")
        if isinstance(preds, dict):
            for tooth_id in preds:
                tid = tooth_id.lower()
                if tid in tooth_set:
                    coverage[case_id].add(tid)
        else:
            # legacy format: entire file is the tooth
            tooth = meta.get("tooth") or meta.get("tooth_id") or data.get("tooth") or data.get("tooth_id")
            if tooth:
                tid = str(tooth).lower()
                if tid in tooth_set:
                    coverage[case_id].add(tid)
    return coverage


def summarise_missing(
    expected: Dict[str, Set[str]],
    actual: Dict[str, Set[str]],
    teeth: Iterable[str],
) -> List[str]:
    teeth_list = [t.lower() for t in teeth]
    lines: List[str] = []
    header = "case_id,available_p0,available_infer,missing_p0,missing_infer"
    lines.append(header)
    for case_id in sorted(expected.keys() | actual.keys()):
        expected_teeth = set(teeth_list)
        p0_teeth = expected.get(case_id, set())
        infer_teeth = actual.get(case_id, set())
        missing_p0 = sorted(expected_teeth - p0_teeth)
        missing_infer = sorted(p0_teeth - infer_teeth)
        lines.append(
            f"{case_id},{len(p0_teeth)},{len(infer_teeth)},"
            f"{' '.join(missing_p0) if missing_p0 else '-'},"
            f"{' '.join(missing_infer) if missing_infer else '-'}"
        )
    return lines


def main(args: argparse.Namespace) -> None:
    teeth = [t.strip().lower() for t in args.teeth.split(",") if t.strip()] if args.teeth else DEFAULT_TEETH
    p0_root = Path(args.p0_root)
    infer_root = Path(args.infer_root)
    if not p0_root.exists():
        raise FileNotFoundError(f"P0 root not found: {p0_root}")
    if not infer_root.exists():
        raise FileNotFoundError(f"Inference root not found: {infer_root}")

    p0_cov = discover_p0_cases(p0_root, teeth)
    infer_cov = discover_infer_cases(infer_root, teeth)
    report_lines = summarise_missing(p0_cov, infer_cov, teeth)
    if args.output:
        Path(args.output).write_text("\n".join(report_lines), encoding="utf-8")
        print(f"[report] wrote coverage summary to {args.output}")
    else:
        print("\n".join(report_lines))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose missing tooth coverage for PointNetReg pipeline.")
    parser.add_argument("--p0-root", type=str, default="datasets/landmarks_dataset/cooked/p0/samples", help="Directory containing per-tooth NPZ samples.")
    parser.add_argument("--infer-root", type=str, required=True, help="Directory containing inference JSON outputs.")
    parser.add_argument("--teeth", type=str, default=",".join(DEFAULT_TEETH), help="Comma-separated list of teeth to track.")
    parser.add_argument("--output", type=str, default=None, help="Optional CSV file path to store the report.")
    args = parser.parse_args()
    main(args)

