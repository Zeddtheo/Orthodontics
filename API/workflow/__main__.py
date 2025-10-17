from __future__ import annotations

import argparse
from pathlib import Path

from .workflow import (
    DEFAULT_LANDMARK_DEF,
    DEFAULT_MESHSEGNET_MODELS,
    DEFAULT_POINTNET_CKPT,
    WorkflowConfig,
    run_workflow,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the MeshSegNet → PointNetReg → metric calculation workflow.",
    )
    parser.add_argument("--upper", required=True, help="Path to the upper jaw STL file.")
    parser.add_argument("--lower", required=True, help="Path to the lower jaw STL file.")
    parser.add_argument("--out", required=True, help="Output directory for artifacts.")
    parser.add_argument(
        "--meshsegnet-model-dir",
        default=str(DEFAULT_MESHSEGNET_MODELS),
        help=f"Directory containing MeshSegNet weights (default: {DEFAULT_MESHSEGNET_MODELS})",
    )
    parser.add_argument(
        "--pointnet-ckpt",
        default=str(DEFAULT_POINTNET_CKPT),
        help=f"PointNet-Reg checkpoint path (default: {DEFAULT_POINTNET_CKPT})",
    )
    parser.add_argument(
        "--landmark-def",
        default=str(DEFAULT_LANDMARK_DEF),
        help=f"Landmark definition JSON (default: {DEFAULT_LANDMARK_DEF})",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device to use (default: auto-detect).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    cfg = WorkflowConfig(
        upper_stl=Path(args.upper),
        lower_stl=Path(args.lower),
        output_dir=Path(args.out),
        meshsegnet_model_dir=Path(args.meshsegnet_model_dir),
        pointnet_ckpt=Path(args.pointnet_ckpt),
        device=args.device,
        landmark_def=Path(args.landmark_def),
    )

    result = run_workflow(cfg)
    print("✅ Workflow finished")
    print(f"• Segmentation dir: {result.segmentation_dir}")
    print(f"  - Upper mesh: {result.upper_seg_vtp}")
    print(f"  - Lower mesh: {result.lower_seg_vtp}")
    print(f"• PointNet-Reg dir: {result.pointnetreg_dir}")
    print(f"  - Upper JSON: {result.upper_payload_json}")
    print(f"  - Lower JSON: {result.lower_payload_json}")
    print(f"• Upper markup JSON: {result.upper_markup_json}")
    print(f"• Lower markup JSON: {result.lower_markup_json}")
    print(f"• Metrics JSON: {result.metrics_json}")


if __name__ == "__main__":
    main()
