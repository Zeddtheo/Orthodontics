from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import pyvista as pv
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from calc_metrics import generate_metrics  # type: ignore  # noqa: E402

from .pointnet import PointNetRegRunner  # noqa: E402
from .segmentation import MeshSegNetRunner  # noqa: E402

DEFAULT_LANDMARK_DEF = Path("datasets/landmarks_dataset/cooked/landmark_def.json")
DEFAULT_MESHSEGNET_MODELS = Path("src/MeshSegNet/models")
DEFAULT_POINTNET_CKPT = Path("outputs/pointnetreg/final_pt/best_mse.pt")


@dataclass
class WorkflowConfig:
    upper_stl: Path
    lower_stl: Path
    output_dir: Path
    meshsegnet_model_dir: Path = DEFAULT_MESHSEGNET_MODELS
    pointnet_ckpt: Path = DEFAULT_POINTNET_CKPT
    landmark_def: Path = DEFAULT_LANDMARK_DEF
    device: Optional[str] = None

    def resolved_device(self) -> str:
        if self.device:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class CaseArtifacts:
    seg_vtp: Path
    payload_json: Path
    markup_json: Path


@dataclass
class WorkflowResult:
    upper_seg_vtp: Path
    lower_seg_vtp: Path
    upper_payload_json: Path
    lower_payload_json: Path
    upper_markup_json: Path
    lower_markup_json: Path
    metrics_json: Path
    segmentation_dir: Path
    pointnetreg_dir: Path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_cli_path(path: Path) -> Path:
    path = Path(path).expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _resolve_repo_path(path: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _payload_to_markups(payload_path: Path, dest_path: Path) -> Path:
    with open(payload_path, "r", encoding="utf-8") as fh:
        payload: Dict = json.load(fh)

    control_points = []
    predictions: Dict[str, Dict] = payload.get("predictions", {})
    for head_key, tooth_payload in predictions.items():
        meta = tooth_payload.get("meta", {})
        fdi = meta.get("fdi")
        if fdi is None:
            fdi = head_key.lstrip("t")
        label_prefix = str(fdi)

        for name, coords in (tooth_payload.get("landmarks_global") or {}).items():
            if not coords or not isinstance(coords, Iterable):
                continue
            coords_list = list(coords)
            if len(coords_list) != 3:
                continue
            try:
                x, y, z = (float(coords_list[0]), float(coords_list[1]), float(coords_list[2]))
            except (TypeError, ValueError):
                continue
            control_points.append(
                {
                    "label": f"{label_prefix}{name}",
                    "position": [x, y, z],
                }
            )

    control_points.sort(key=lambda item: item["label"])
    markups = {
        "markups": [
            {
                "label": payload.get("case_id", dest_path.stem),
                "controlPoints": control_points,
            }
        ]
    }

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "w", encoding="utf-8") as fh:
        json.dump(markups, fh, ensure_ascii=False, indent=2)
    return dest_path


def _run_case(
    seg_runner: MeshSegNetRunner,
    pointnet_runner: PointNetRegRunner,
    mesh_path: Path,
    seg_dir: Path,
    reg_dir: Path,
    markups_dir: Path,
) -> CaseArtifacts:
    seg_result = seg_runner.segment(mesh_path, seg_dir)
    poly = pv.read(str(seg_result.polydata_path))
    case_id = mesh_path.stem

    pointnet_outputs = pointnet_runner.run_case(
        mesh=poly,
        arch=seg_result.arch,
        case_id=case_id,
        output_dir=reg_dir,
    )

    markup_path = markups_dir / f"{case_id}_markups.json"
    _payload_to_markups(pointnet_outputs.payload_json, markup_path)

    return CaseArtifacts(
        seg_vtp=seg_result.seg_vtp,
        payload_json=pointnet_outputs.payload_json,
        markup_json=markup_path,
    )


def run_workflow(cfg: WorkflowConfig) -> WorkflowResult:
    upper_stl = _resolve_cli_path(cfg.upper_stl)
    lower_stl = _resolve_cli_path(cfg.lower_stl)
    output_root = _ensure_dir(_resolve_cli_path(cfg.output_dir))
    seg_dir = _ensure_dir(output_root / "segmentation")
    reg_dir = _ensure_dir(output_root / "pointnetreg")
    markups_dir = _ensure_dir(output_root / "markups")

    device_str = cfg.resolved_device()
    device = torch.device(device_str)

    seg_runner = MeshSegNetRunner(
        model_dir=_resolve_repo_path(cfg.meshsegnet_model_dir),
        device=device,
    )
    pointnet_runner = PointNetRegRunner(
        ckpt_path=_resolve_repo_path(cfg.pointnet_ckpt),
        landmark_def=_resolve_repo_path(cfg.landmark_def),
        device=device_str,
    )

    upper_artifacts = _run_case(seg_runner, pointnet_runner, upper_stl, seg_dir, reg_dir, markups_dir)
    lower_artifacts = _run_case(seg_runner, pointnet_runner, lower_stl, seg_dir, reg_dir, markups_dir)

    metrics_path = output_root / "metrics.json"
    generate_metrics(
        str(upper_stl),
        str(lower_stl),
        str(upper_artifacts.markup_json),
        str(lower_artifacts.markup_json),
        out_path=str(metrics_path),
    )

    return WorkflowResult(
        upper_seg_vtp=upper_artifacts.seg_vtp,
        lower_seg_vtp=lower_artifacts.seg_vtp,
        upper_payload_json=upper_artifacts.payload_json,
        lower_payload_json=lower_artifacts.payload_json,
        upper_markup_json=upper_artifacts.markup_json,
        lower_markup_json=lower_artifacts.markup_json,
        metrics_json=metrics_path,
        segmentation_dir=seg_dir,
        pointnetreg_dir=reg_dir,
    )
