"""Fix mislabeled lower-arch landmarks that use upper FDI indices.

Some raw markups (e.g. cases 121, 23, 132, 133) store mandibular labels as
11–27 instead of 31–47. This utility rewrites the leading FDI digits so that
PointNetReg can pick up the correct teeth.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable


FDI_REGEX = re.compile(r"^(\d{2})(.*)$")


def normalise_case_id(case_id: str) -> str:
    """Convert case id to directory name (strip leading zeros)."""
    case_id = case_id.strip()
    if case_id.isdigit():
        return str(int(case_id))
    return case_id


def fix_labels(path: Path) -> bool:
    """Rewrite landmark labels inside a single *_L.json file."""
    data = json.loads(path.read_text(encoding="utf-8"))
    cps = data.get("markups", [{}])[0].get("controlPoints", [])
    changed = False
    for cp in cps:
        label = cp.get("label")
        if not label:
            continue
        m = FDI_REGEX.match(label)
        if not m:
            continue
        digits, suffix = m.groups()
        try:
            fdi = int(digits)
        except ValueError:
            continue
        if fdi >= 30:  # already lower-arch numbering
            continue
        if fdi < 11 or fdi > 27:
            continue
        new_fdi = fdi + 20  # 11->31, 27->47, 21->41, etc.
        new_label = f"{new_fdi:02d}{suffix}"
        if new_label != label:
            cp["label"] = new_label
            changed = True
    if changed:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return changed


def process_cases(raw_root: Path, cases: Iterable[str]) -> None:
    for raw_case in cases:
        case_dir = raw_root / raw_case
        json_path = case_dir / f"{raw_case}_L.json"
        if not json_path.exists():
            print(f"[skip] {json_path} missing")
            continue
        if fix_labels(json_path):
            print(f"[fix] {json_path}")
        else:
            print(f"[ok] {json_path} (no changes needed)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fix lower-arch landmark labels (11/21 -> 31/41).")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/landmarks_dataset/raw"),
        help="Root directory containing raw case folders.",
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        default=["121", "23", "132", "133"],
        help="Case IDs to fix (default: 121 23 132 133).",
    )
    args = parser.parse_args()

    raw_root = args.raw_root
    if not raw_root.exists():
        raise FileNotFoundError(f"Raw root not found: {raw_root}")

    cases = [normalise_case_id(c) for c in args.cases]
    process_cases(raw_root, cases)


if __name__ == "__main__":
    main()
