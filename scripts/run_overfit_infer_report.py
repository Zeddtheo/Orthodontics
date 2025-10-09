#!/usr/bin/env python
"""
One-touch helper that runs the PointNet-Reg single-tooth overfit pipeline,
launches inference + aggregation, and writes a compact text report that
captures:
  • last 5 epochs of coordinate-level overfit metrics (loss / MAE / matches)
  • inference margin statistics for the chosen case
  • aggregated CSV coordinates for the specified tooth

Example:
  python scripts/run_overfit_infer_report.py --root datasets/landmarks_dataset/cooked/p0/samples --tooth t27 --case 001
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import List


def run_cmd(cmd: List[str], cwd: Path | None = None) -> str:
    """Run a command and return stdout (also prints the command)."""
    display = " ".join(cmd)
    print(f"\n>>> {display}")
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        cwd=str(cwd) if cwd else None,
        check=True,
    )
    if proc.stderr.strip():
        print(proc.stderr.strip())
    return proc.stdout


def ensure_clean_dir(path: Path, keep_existing: bool) -> None:
    if path.exists():
        if keep_existing:
            return
        for item in path.glob("**/*"):
            if item.is_file():
                item.unlink()
        for item in sorted(path.glob("**"), reverse=True):
            if item.is_dir():
                item.rmdir()
        try:
            path.rmdir()
        except FileNotFoundError:
            pass
    path.mkdir(parents=True, exist_ok=True)


def parse_training_metrics(stdout: str, max_items: int = 5) -> List[str]:
    lines = []
    for line in stdout.splitlines():
        line = line.strip()
        if "mae" in line and "matches" in line:
            lines.append(line)
    return lines[-max_items:] if lines else []


def summarize_margins(json_path: Path, tooth: str) -> str:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    preds = data["predictions"].get(tooth)
    if not preds or "margin" not in preds:
        return f"No margin data for tooth {tooth} in {json_path.name}"
    margins = [float(v) for v in preds["margin"].values()]
    if not margins:
        return f"Empty margin list for tooth {tooth}"
    summary = {
        "count": len(margins),
        "mean": statistics.mean(margins),
        "stdev": statistics.pstdev(margins) if len(margins) > 1 else 0.0,
        "min": min(margins),
        "max": max(margins),
    }
    return (
        f"Margins count={summary['count']} "
        f"mean={summary['mean']:.6f} "
        f"stdev={summary['stdev']:.6f} "
        f"min={summary['min']:.6f} "
        f"max={summary['max']:.6f}"
    )


def extract_csv_rows(csv_path: Path, tooth: str) -> List[str]:
    rows = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        rows.append(",".join(header))
        for row in reader:
            if len(row) >= 3 and row[1].strip().lower() == tooth.lower():
                rows.append(",".join(row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run PointNet-Reg overfit → infer → report pipeline.")
    parser.add_argument("--root", required=True, help="Root directory containing per-tooth NPZ samples (p0).")
    parser.add_argument("--tooth", required=True, help="Tooth ID, e.g. t27.")
    parser.add_argument("--case", required=True, help="Case ID to summarise, e.g. 001.")
    parser.add_argument("--epochs", type=int, default=600, help="Number of overfit epochs.")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for overfit.")
    parser.add_argument("--sample_idx", type=int, default=0, help="Sample index during overfit.")
    parser.add_argument("--landmark_json", default="datasets/landmarks_dataset/cooked/landmark_def.json", help="Path to landmark_def.json.")
    parser.add_argument("--features", choices=["pn", "xyz"], default="pn", help="Feature set to use.")
    parser.add_argument("--tag", default=None, help="Optional tag for output directories (defaults to tooth id).")
    parser.add_argument("--output_txt", default=None, help="Where to write the report (defaults to report_<tag>.txt).")
    parser.add_argument("--keep_dirs", action="store_true", help="Reuse existing output directories instead of cleaning them.")
    parser.add_argument("--use_tnet", action="store_true", help="Enable TNet alignment during overfit/infer.")
    args = parser.parse_args()

    tag = args.tag or args.tooth
    ckpt_root = Path(f"outputs/landmarks/overfit_{tag}")
    infer_root = Path(f"runs_infer_{tag}")
    post_root = Path(f"runs_postprocess_{tag}")
    report_path = Path(args.output_txt or f"overfit_infer_report_{tag}.txt")

    ensure_clean_dir(ckpt_root, args.keep_dirs)
    ensure_clean_dir(infer_root, args.keep_dirs)
    ensure_clean_dir(post_root, args.keep_dirs)

    # 1. Overfit training
    train_cmd = [
        "python",
        "src/PointnetReg/p4_overfit.py",
        "--root",
        args.root,
        "--tooth",
        args.tooth,
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.lr),
        "--features",
        args.features,
        "--sample_idx",
        str(args.sample_idx),
        "--out_dir",
        str(ckpt_root),
        "--save_model",
    ]
    if args.use_tnet:
        train_cmd.append("--use_tnet")
    train_stdout = run_cmd(train_cmd)
    train_metrics = parse_training_metrics(train_stdout)

    # 2. Inference
    infer_cmd = [
        "python",
        "src/PointnetReg/p2_infer.py",
        "--root",
        args.root,
        "--ckpt_root",
        str(ckpt_root),
        "--tooth",
        args.tooth,
        "--features",
        args.features,
        "--out_dir",
        str(infer_root),
        "--landmark_json",
        args.landmark_json,
        "--batch_size",
        "1",
        "--workers",
        "0",
        "--cases",
        args.case,
    ]
    if args.use_tnet:
        infer_cmd.append("--use_tnet")
    run_cmd(infer_cmd)

    # 3. Aggregation
    post_cmd = [
        "python",
        "src/PointnetReg/p3_postprocess.py",
        "--infer-root",
        str(infer_root),
        "--landmark-def",
        args.landmark_json,
        "--out-dir",
        str(post_root),
        "--export-csv",
        "--cases",
        args.case,
    ]
    run_cmd(post_cmd)

    # Summaries
    case_json_path = infer_root / f"{args.case}.json"
    if not case_json_path.exists():
        # fallback to first available json
        json_candidates = sorted(infer_root.glob("*.json"))
        if not json_candidates:
            raise FileNotFoundError(f"No inference JSON found in {infer_root}")
        case_json_path = json_candidates[0]
    margin_summary = summarize_margins(case_json_path, args.tooth)

    csv_path = post_root / "csv" / f"{args.case}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Aggregated CSV not found: {csv_path}")
    csv_rows = extract_csv_rows(csv_path, args.tooth)

    report_lines = []
    report_lines.append(f"Overfit/Inference Report (tag={tag}, tooth={args.tooth}, case={args.case})")
    report_lines.append("")
    report_lines.append("Training metrics (last 5 epochs with MAE):")
    report_lines.extend(train_metrics if train_metrics else ["[No MAE lines captured]"])
    report_lines.append("")
    report_lines.append(f"Inference margin summary ({case_json_path.name}):")
    report_lines.append(margin_summary)
    report_lines.append("")
    report_lines.append(f"Aggregated CSV rows for tooth {args.tooth} (case {args.case}):")
    report_lines.extend(csv_rows if len(csv_rows) > 1 else ["[No rows matched]"])

    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to {report_path.resolve()}")


if __name__ == "__main__":
    main()
