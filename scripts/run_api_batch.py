#!/usr/bin/env python3
"""
Batch runner for the API pipeline on local STL cases.

For each case ID, this script:
1. Invokes API.utils.runner.run_pipeline on the upper/lower STL pair.
2. Copies the resulting VTP/JSON artefacts into <case>/api_results/.
3. Runs calc_metrics via the pipeline and renders a markdown summary that
   also includes average landmark error against ground-truth JSON files.
"""

from __future__ import annotations

import json
import math
import shutil
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Ensure the API runner can locate micromamba and reusable environments.
REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "API"
MICROMAMBA_EXE = REPO_ROOT / "tools" / "micromamba.exe"
ENV_ROOT = REPO_ROOT / ".micromamba"

import os

import sys

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("MICROMAMBA_EXE", str(MICROMAMBA_EXE))
os.environ.setdefault("MAMBA_ROOT_PREFIX", str(ENV_ROOT))

from API.utils.runner import run_pipeline  # noqa: E402


@dataclass
class CaseArtefacts:
    case_id: str
    upper_vtp: Path
    lower_vtp: Path
    upper_json: Path
    lower_json: Path
    metrics_json: Path


def load_markups_json(path: Path) -> Dict[str, Tuple[float, float, float]]:
    """Read a Slicer markups JSON file into {label: (x, y, z)}."""
    data = json.loads(path.read_text(encoding="utf-8"))
    markups = data.get("markups") or []
    out: Dict[str, Tuple[float, float, float]] = {}
    for markup in markups:
        for point in markup.get("controlPoints", []):
            label = point.get("label")
            position = point.get("position")
            if (
                label
                and isinstance(position, (list, tuple))
                and len(position) == 3
                and all(isinstance(v, (int, float)) for v in position)
            ):
                out[str(label)] = (float(position[0]), float(position[1]), float(position[2]))
    return out


def compute_landmark_errors(
    preds: Dict[str, Tuple[float, float, float]],
    gts: Dict[str, Tuple[float, float, float]],
) -> List[float]:
    """Return Euclidean distance (mm) for matched labels."""
    errors: List[float] = []
    for label, gt_pos in gts.items():
        pred_pos = preds.get(label)
        if pred_pos is None:
            continue
        errors.append(float(math.dist(pred_pos, gt_pos)))
    return errors


def render_markdown(
    case_id: str,
    metrics: Dict[str, object],
    errors: Iterable[float],
    matched: int,
    total_gt: int,
    out_path: Path,
) -> None:
    errors = list(errors)
    lines: List[str] = [
        f"# Case {case_id} API 输出",
        "",
        "## 平均误差",
    ]
    if errors:
        lines.append(f"- 匹配点数：{matched} / {total_gt}")
        lines.append(f"- 平均误差 (MAE)：{statistics.mean(errors):.3f} mm")
        lines.append(f"- 中位误差：{statistics.median(errors):.3f} mm")
        lines.append(f"- 最大误差：{max(errors):.3f} mm")
        if len(errors) >= 20:
            p95 = statistics.quantiles(errors, n=20)[18]
            lines.append(f"- 95% 分位误差：{p95:.3f} mm")
    else:
        lines.append("- 无匹配点，无法计算误差")

    lines.append("")
    lines.append("## calc_metrics 指标")
    lines.append("")
    if metrics:
        lines.append("| 指标 | 结果 |")
        lines.append("| --- | --- |")
        for key, value in metrics.items():
            lines.append(f"| {key} | {value} |")
    else:
        lines.append("无指标输出。")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_case(case_id: str, raw_root: Path, staging_root: Path) -> CaseArtefacts:
    case_dir = raw_root / case_id
    if not case_dir.exists():
        raise FileNotFoundError(f"Case directory not found: {case_dir}")

    upper_stl = next(case_dir.glob("*_U.stl"))
    lower_stl = next(case_dir.glob("*_L.stl"))
    upper_gt_json = next(case_dir.glob("*_U.json"))
    lower_gt_json = next(case_dir.glob("*_L.json"))

    workdir = staging_root / f"case_{case_id}"
    if workdir.exists():
        shutil.rmtree(workdir)

    workdir.mkdir(parents=True, exist_ok=True)

    outputs = run_pipeline(
        stl_a=upper_stl,
        stl_b=lower_stl,
        workdir=workdir,
        run_metrics=True,
    )

    return CaseArtefacts(
        case_id=case_id,
        upper_vtp=Path(outputs["vtp_a"]),
        lower_vtp=Path(outputs["vtp_b"]),
        upper_json=Path(outputs["json_a"]),
        lower_json=Path(outputs["json_b"]),
        metrics_json=Path(outputs["metrics"]),
    )


def main() -> None:
    raw_root = REPO_ROOT / "datasets" / "tests"
    staging_root = API_ROOT / "runs"

    cases = ["316", "317", "318", "319", "320"]
    results: List[CaseArtefacts] = []
    for case_id in cases:
        artefacts = process_case(case_id, raw_root=raw_root, staging_root=staging_root)
        results.append(artefacts)

        case_dir = raw_root / case_id
        target_dir = case_dir / "api_results"
        if target_dir.exists():
            shutil.rmtree(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        # Copy artefacts with case-specific filenames.
        shutil.copy2(artefacts.upper_vtp, target_dir / f"{case_id}_U_seg.vtp")
        shutil.copy2(artefacts.lower_vtp, target_dir / f"{case_id}_L_seg.vtp")
        shutil.copy2(artefacts.upper_json, target_dir / f"{case_id}_U_pred.json")
        shutil.copy2(artefacts.lower_json, target_dir / f"{case_id}_L_pred.json")

        # Compute errors.
        gt_upper = load_markups_json(next(case_dir.glob("*_U.json")))
        gt_lower = load_markups_json(next(case_dir.glob("*_L.json")))
        pred_upper = load_markups_json(artefacts.upper_json)
        pred_lower = load_markups_json(artefacts.lower_json)

        gt_all = {**gt_upper, **gt_lower}
        pred_all = {**pred_upper, **pred_lower}

        errors = compute_landmark_errors(pred_all, gt_all)
        metrics_payload = json.loads(artefacts.metrics_json.read_text(encoding="utf-8"))

        render_markdown(
            case_id=case_id,
            metrics=metrics_payload,
            errors=errors,
            matched=len(errors),
            total_gt=len(gt_all),
            out_path=target_dir / f"{case_id}_report.md",
        )


if __name__ == "__main__":
    main()
