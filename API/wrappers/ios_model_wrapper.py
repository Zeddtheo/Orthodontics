from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple


def _ensure_ios_module() -> Tuple[Path, Path]:
    """
    Ensure the ios-model TorchScript package is importable.

    Returns:
        repo_root: project root directory
        ios_module_dir: directory containing run_teeth_seg_from_stl.py
    """
    repo_root = Path(__file__).resolve().parents[1]
    candidates = [
        repo_root / "vendor" / "ios-model",
        repo_root / "models" / "ios-model",
        repo_root.parent / "src" / "IOS-Model" / "teeth_seg",
    ]
    ios_module_dir = None
    for path in candidates:
        if path.exists():
            ios_module_dir = path
            break
    if ios_module_dir is None:
        raise FileNotFoundError("Missing ios-model module (expected under API/vendor/ios-model).")
    if str(ios_module_dir) not in sys.path:
        sys.path.insert(0, str(ios_module_dir))
    return repo_root, ios_module_dir


def _infer_arch_from_name(filename: str) -> str:
    stem = Path(filename).stem.lower()
    tokens = stem.replace("-", "_").split("_")
    for token in reversed(tokens):
        if token in {"u", "upper", "max", "maxilla"}:
            return "max"
        if token in {"l", "lower", "man", "mandible"}:
            return "man"
    if stem.endswith("u"):
        return "max"
    if stem.endswith("l"):
        return "man"
    raise ValueError(
        f"无法从文件名 {filename} 推断颌别（需包含 U/L 或 upper/lower 标记）"
    )


def run_ios_segmentation(
    stl_path: Path,
    out_vtp: Path,
    jaw_type: str,
    device: str = "cuda:0",
    downsample: int = 50000,
    graph_k: int = 8,
    downsample_backend: str = "vedo",
) -> Path:
    from run_teeth_seg_from_stl import run_inference

    stl_path = stl_path.resolve()
    out_vtp = out_vtp.resolve()
    return run_inference(
        input_path=stl_path,
        output_path=out_vtp,
        jaw_type=jaw_type,
        device=device,
        downsample_num=downsample,
        graph_k=graph_k,
        downsample_backend=downsample_backend,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ios-model TorchScript segmentation wrapper"
    )
    parser.add_argument("--upper_stl", help="上颌 STL 输入路径")
    parser.add_argument("--upper_out", help="上颌 VTP 输出路径")
    parser.add_argument("--lower_stl", help="下颌 STL 输入路径")
    parser.add_argument("--lower_out", help="下颌 VTP 输出路径")
    parser.add_argument(
        "--upper_arch",
        choices=["man", "max"],
        help="上颌颌别覆盖（默认基于文件名推断）",
    )
    parser.add_argument(
        "--lower_arch",
        choices=["man", "max"],
        help="下颌颌别覆盖（默认基于文件名推断）",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Torch 设备字符串（例如 cuda:0 或 cpu）",
    )
    parser.add_argument(
        "--downsample", type=int, default=50000, help="可选：下采样目标面片数"
    )
    parser.add_argument(
        "--graph_k", type=int, default=8, help="k-NN 图构建的 k 值"
    )
    parser.add_argument(
        "--downsample_backend",
        choices=["vedo", "open3d"],
        default="vedo",
        help="下采样后端；若 vedo 失败将自动退回 open3d",
    )
    # backward compatibility (single arch call)
    parser.add_argument("--stl", help="待兼容旧接口的 STL 输入")
    parser.add_argument("--out_vtp", help="待兼容旧接口的 VTP 输出")
    parser.add_argument(
        "--arch",
        choices=["man", "max"],
        help="兼容旧接口的颌别（默认基于文件名推断）",
    )
    return parser.parse_args()


def main() -> None:
    repo_root, ios_module_dir = _ensure_ios_module()
    args = parse_args()

    tasks = []
    if args.upper_stl and args.upper_out:
        tasks.append(
            (
                args.upper_arch or "max",
                Path(args.upper_stl),
                Path(args.upper_out),
            )
        )
    if args.lower_stl and args.lower_out:
        tasks.append(
            (
                args.lower_arch or "man",
                Path(args.lower_stl),
                Path(args.lower_out),
            )
        )
    if not tasks and args.stl and args.out_vtp:
        inferred = args.arch or _infer_arch_from_name(args.stl)
        tasks.append((inferred, Path(args.stl), Path(args.out_vtp)))

    if not tasks:
        raise ValueError(
            "至少需要提供一对 --upper_stl/--upper_out 或 --lower_stl/--lower_out"
        )

    for jaw_hint, stl_path, out_vtp in tasks:
        if jaw_hint not in {"man", "max"}:
            jaw = _infer_arch_from_name(stl_path.name)
        else:
            jaw = jaw_hint
        if not stl_path.exists():
            raise FileNotFoundError(f"STL not found: {stl_path}")

        runtime_start = time.time()
        try:
            run_ios_segmentation(
                stl_path=stl_path,
                out_vtp=out_vtp,
                jaw_type=jaw,
                device=args.device,
                downsample=args.downsample,
                graph_k=args.graph_k,
                downsample_backend=args.downsample_backend,
            )
        finally:
            duration = time.time() - runtime_start
            print(
                f"✅ ios-model 完成 {stl_path.name} (jaw={jaw}) -> {out_vtp} 用时 {duration:.2f}s"
            )


if __name__ == "__main__":
    main()
