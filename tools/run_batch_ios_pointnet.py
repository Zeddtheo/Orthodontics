#!/usr/bin/env python3
"""
Batch runner for ios-model segmentation + PointNet-Reg landmark prediction.

Usage:
    python tools/run_batch_ios_pointnet.py \
        --raw-root datasets/landmarks_dataset/raw \
        --out-root outputs/raw_output
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path
from typing import Iterable, List, Sequence

try:
    from API.utils.runner import run_pipeline
except ModuleNotFoundError:  # pragma: no cover
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from API.utils.runner import run_pipeline  # type: ignore  # noqa: WPS440


def _find_cases(raw_root: Path, include: Sequence[str] | None = None) -> List[Path]:
    if include:
        wanted = {item.strip() for item in include if item.strip()}
        paths = []
        for item in wanted:
            candidate = raw_root / item
            if candidate.is_dir():
                paths.append(candidate)
        return sorted(paths, key=lambda p: int(p.name))

    return sorted(
        (p for p in raw_root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def _copy_outputs(workdir: Path, case_id: str, out_dir: Path) -> None:
    arch_map = {"A": "U", "B": "L"}
    out_dir.mkdir(parents=True, exist_ok=True)
    for prefix, arch in arch_map.items():
        src_vtp = workdir / f"{prefix}.vtp"
        src_json = workdir / f"{prefix}.json"
        if src_vtp.exists():
            shutil.copy2(src_vtp, out_dir / f"{case_id}_{arch}.vtp")
        if src_json.exists():
            shutil.copy2(src_json, out_dir / f"{case_id}_{arch}.json")


def _has_outputs(out_dir: Path, case_id: str) -> bool:
    expected = [
        out_dir / f"{case_id}_U.vtp",
        out_dir / f"{case_id}_L.vtp",
        out_dir / f"{case_id}_U.json",
        out_dir / f"{case_id}_L.json",
    ]
    return all(path.exists() for path in expected)


def _summarise(stats: Iterable[tuple[str, float]]) -> str:
    parts = []
    width = max((len(case) for case, _ in stats), default=0)
    for case, elapsed in stats:
        parts.append(f"{case.rjust(width)}: {elapsed:6.2f}s")
    return "\n".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run ios-model + PointNet-Reg over multiple cases.")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=Path("datasets/landmarks_dataset/raw"),
        help="Root directory containing per-case subfolders with STL files.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("outputs/raw_output"),
        help="Directory where the resulting VTP/JSON files will be copied.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Optional list of case IDs to run (defaults to all numeric subfolders).",
    )
    parser.add_argument("--overwrite", action="store_true", help="Re-run even if outputs already exist.")
    parser.add_argument("--keep-workdir", action="store_true", help="Keep intermediate workdir folders.")
    args = parser.parse_args(argv)

    raw_root = args.raw_root.resolve()
    out_root = args.out_root.resolve()

    if not raw_root.exists():
        parser.error(f"Raw root not found: {raw_root}")

    cases = _find_cases(raw_root, args.cases)
    if not cases:
        parser.error("No cases found to process.")

    out_root.mkdir(parents=True, exist_ok=True)

    timings: List[tuple[str, float]] = []
    failures: List[str] = []

    for case_dir in cases:
        case_id = case_dir.name
        upper_stl = case_dir / f"{case_id}_U.stl"
        lower_stl = case_dir / f"{case_id}_L.stl"

        if not upper_stl.exists() or not lower_stl.exists():
            print(f"[skip] {case_id}: missing STL(s)", file=sys.stderr)
            continue

        dest_dir = out_root / case_id
        if not args.overwrite and _has_outputs(dest_dir, case_id):
            print(f"[skip] {case_id}: outputs already present")
            continue

        print(f"[run] {case_id}: starting…")
        start = time.perf_counter()
        try:
            result = run_pipeline(upper_stl, lower_stl)
        except Exception as exc:  # noqa: BLE001
            failures.append(case_id)
            print(f"[fail] {case_id}: {exc}", file=sys.stderr)
            continue

        elapsed = time.perf_counter() - start
        timings.append((case_id, elapsed))
        print(f"[done] {case_id}: {elapsed:.2f}s")

        workdir = Path(result["workdir"])
        try:
            _copy_outputs(workdir, case_id, dest_dir)
        except Exception as exc:  # noqa: BLE001
            failures.append(case_id)
            print(f"[fail] {case_id}: copy failed ({exc})", file=sys.stderr)

        if not args.keep_workdir and workdir.exists():
            shutil.rmtree(workdir, ignore_errors=True)

    if timings:
        print("\n[summary]")
        print(_summarise(timings))
        avg = sum(t for _, t in timings) / len(timings)
        print(f"Average: {avg:.2f}s over {len(timings)} cases")

    if failures:
        print(f"\n[warn] {len(failures)} case(s) failed: {', '.join(failures)}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
