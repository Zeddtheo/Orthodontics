import argparse
from pathlib import Path

import numpy as np
import torch
import vedo

import teeth_seg_torchscript as seg_module

MAX_LABEL_TO_FDI = np.array(
    [
        0,  # background
        17, 16, 15, 14, 13, 12, 11,  # right maxillary teeth
        21, 22, 23, 24, 25, 26, 27,  # left maxillary teeth
        18, 28,  # wisdom teeth
    ],
    dtype=np.int32,
)
MAN_LABEL_TO_FDI = np.array(
    [
        0,  # background
        37, 36, 35, 34, 33, 32, 31,  # left mandibular teeth
        41, 42, 43, 44, 45, 46, 47,  # right mandibular teeth
        38, 48,  # wisdom teeth
    ],
    dtype=np.int32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run TorchScript teeth segmentation on an STL mesh and export a labeled VTP."
        )
    )
    parser.add_argument("--input", required=True, help="Path to the input STL mesh.")
    parser.add_argument(
        "--jaw-type",
        choices=("man", "max"),
        required=True,
        help="Select mandibular (man) or maxillary (max) model weights.",
    )
    parser.add_argument(
        "--output",
        help=(
            "Path to the output VTP file. Defaults to <input_basename>_predicted.vtp "
            "in the same folder as the input mesh."
        ),
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string, e.g. cuda:0 or cpu. Default: cuda:0.",
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=50000,
        help="Target face count for optional mesh decimation before inference.",
    )
    parser.add_argument(
        "--graph-k",
        type=int,
        default=8,
        help="K for k-NN graph construction in feature preprocessing.",
    )
    parser.add_argument(
        "--downsample-backend",
        choices=("vedo", "open3d"),
        default="vedo",
        help="Select decimation library. Use open3d if vedo fails on a mesh.",
    )
    return parser.parse_args()


def resolve_output_path(input_path: Path, output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).resolve()
    return input_path.with_name(f"{input_path.stem}_predicted.vtp")


def load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray, vedo.Mesh]:
    mesh = vedo.load(str(path))
    if mesh is None:
        raise ValueError(f"Failed to load mesh from {path}")
    points = np.asarray(mesh.points())
    faces = np.asarray(mesh.faces(), dtype=np.int64)
    return points, faces, mesh


def remap_labels_to_fdi(labels: np.ndarray, jaw_type: str) -> np.ndarray:
    """
    Map ios-model label space (0-16) to FDI tooth numbering expected by the PointNetReg pipeline.
    """
    lookup = MAN_LABEL_TO_FDI if jaw_type == "man" else MAX_LABEL_TO_FDI
    labels = labels.astype(np.int32, copy=False)
    remapped = np.zeros_like(labels)
    valid_mask = labels < lookup.size
    remapped[valid_mask] = lookup[labels[valid_mask]]
    return remapped


def _resolve_model_path(script_dir: Path, filename: str) -> Path:
    candidates = [
        script_dir / filename,
        script_dir.parent.parent / "models" / "ios-model" / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Model weights {filename} not found in {candidates}")


def run_inference(
    input_path: Path,
    output_path: Path,
    jaw_type: str,
    device: str = "cuda:0",
    downsample_num: int = 50000,
    graph_k: int = 8,
    downsample_backend: str = "vedo",
) -> Path:
    script_dir = Path(__file__).resolve().parent

    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device requested but torch.cuda.is_available() returned False."
            )
        torch.cuda.set_device(torch_device)

    weight_name = "max_teeth_seg_model_script.pt" if jaw_type == "max" else "man_teeth_seg_model_script.pt"
    model_path = _resolve_model_path(script_dir, weight_name)

    model = torch.jit.load(str(model_path), map_location=torch_device)
    model.to(torch_device)
    model.eval()

    points_origin, faces_origin, mesh = load_mesh(input_path)

    if downsample_backend == "open3d":
        seg_module.downsample_mesh_vedo = seg_module.downsample_mesh_open3d

    try:
        labels_down, labels_origin, _, _ = seg_module.predict(
            device=torch_device,
            model=model,
            points_origin=points_origin,
            faces_origin=faces_origin,
            downsample_num=downsample_num,
            graph_k=graph_k,
        )
    except ValueError as exc:
        if "No input was provided" in str(exc) and downsample_backend == "vedo":
            print("Vedo decimation failed, retrying with open3d backend...")
            seg_module.downsample_mesh_vedo = seg_module.downsample_mesh_open3d
            labels_down, labels_origin, _, _ = seg_module.predict(
                device=torch_device,
                model=model,
                points_origin=points_origin,
                faces_origin=faces_origin,
                downsample_num=downsample_num,
                graph_k=graph_k,
            )
        else:
            raise

    labels_origin = labels_origin.astype(np.int32, copy=False)
    mesh.celldata["Label_ios"] = labels_origin
    mesh.celldata["Label"] = remap_labels_to_fdi(labels_origin, jaw_type)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.write(str(output_path))

    print(f"Inference complete. Downsampled faces: {len(labels_down)}")
    print(f"Saved labeled mesh to: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).resolve()
    output_path = resolve_output_path(input_path, args.output)
    run_inference(
        input_path=input_path,
        output_path=output_path,
        jaw_type=args.jaw_type,
        device=args.device,
        downsample_num=args.downsample,
        graph_k=args.graph_k,
        downsample_backend=args.downsample_backend,
    )


if __name__ == "__main__":
    main()
