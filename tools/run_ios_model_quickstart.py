#!/usr/bin/env python3
"""One-click ios-model TorchScript segmentation helper.

This script takes STL inputs, ensures the `ios_model` micromamba environment
exists (creating it from `API/envs/ios_model.yml` when absent), installs the
extra PyG wheels if needed, and then runs
`API/wrappers/ios_model_wrapper.py` inside that environment.

Example:
    python tools/run_ios_model_quickstart.py \
        --upper-stl datasets/tests/316/316_U.stl \
        --lower-stl datasets/tests/316/316_L.stl \
        --out-dir outputs/ios_model_quickstart
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "API"
ENV_SPEC = API_ROOT / "envs" / "ios_model.yml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "ios_model_quickstart"
PYG_INDEX_URL = "https://data.pyg.org/whl/torch-2.2.2+cu121.html"
IOS_ENV_NAME = "ios_model"


def _quote_cmd(cmd: list[str | os.PathLike[str]]) -> str:
    return " ".join(shlex.quote(str(token)) for token in cmd)


def _run(cmd: list[str | os.PathLike[str]], env: dict[str, str] | None = None) -> None:
    print(f"[cmd] {_quote_cmd(cmd)}")
    subprocess.run(cmd, check=True, env=env)  # noqa: S603


def _find_micromamba() -> Path:
    candidates = [
        os.environ.get("MICROMAMBA_EXE"),
        "micromamba",
        REPO_ROOT / "tools" / "micromamba.exe",
        API_ROOT / "build-tools" / "micromamba-linux",
        REPO_ROOT / "tools" / "micromamba",
    ]
    for cand in candidates:
        if not cand:
            continue
        if isinstance(cand, Path):
            path = cand
        else:
            found = shutil.which(str(cand))
            if found:
                return Path(found)
            path = Path(cand)
        if path.exists():
            return path
    raise RuntimeError("micromamba executable not found. Set MICROMAMBA_EXE or install micromamba.")


def _resolve_root_prefix(override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    env_override = os.environ.get("MAMBA_ROOT_PREFIX")
    if env_override:
        return Path(env_override).expanduser().resolve()
    return (API_ROOT / ".micromamba").resolve()


def _ensure_env(micromamba: Path, env_spec: Path, root_prefix: Path) -> None:
    env_dir = root_prefix / "envs" / IOS_ENV_NAME
    if env_dir.exists():
        print(f"[info] micromamba env '{IOS_ENV_NAME}' already present under {env_dir}")
        return
    print(f"[info] creating micromamba env '{IOS_ENV_NAME}' from {env_spec}")
    env_spec = env_spec.resolve()
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(micromamba),
            "create",
            "-y",
            "-n",
            IOS_ENV_NAME,
            "-f",
            str(env_spec),
            "--root-prefix",
            str(root_prefix),
        ]
    )


def _ensure_ios_extras(micromamba: Path, root_prefix: Path) -> None:
    env_dir = root_prefix / "envs" / IOS_ENV_NAME
    marker = env_dir / ".torch_geometric_installed"
    if marker.exists():
        return
    print("[info] installing additional PyG wheels inside ios_model environment")
    _run(
        [
            str(micromamba),
            "run",
            "--root-prefix",
            str(root_prefix),
            "-n",
            IOS_ENV_NAME,
            "python",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-f",
            PYG_INDEX_URL,
            "torch_scatter",
            "torch_sparse",
            "torch_cluster",
            "torch_spline_conv",
        ]
    )
    _run(
        [
            str(micromamba),
            "run",
            "--root-prefix",
            str(root_prefix),
            "-n",
            IOS_ENV_NAME,
            "python",
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "torch_geometric==2.5.3",
        ]
    )
    marker.touch()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensure ios-model environment is ready and run TorchScript segmentation."
    )
    parser.add_argument("--upper-stl", help="Path to the upper STL mesh.")
    parser.add_argument("--lower-stl", help="Path to the lower STL mesh.")
    parser.add_argument(
        "--out-dir",
        help=f"Directory to store VTP outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--upper-out",
        help="Explicit path for the upper VTP output (defaults to <out-dir>/upper_seg.vtp).",
    )
    parser.add_argument(
        "--lower-out",
        help="Explicit path for the lower VTP output (defaults to <out-dir>/lower_seg.vtp).",
    )
    parser.add_argument("--device", default="cuda:0", help="torch device string passed to the wrapper.")
    parser.add_argument("--downsample", type=int, default=50000, help="target face count for downsampling.")
    parser.add_argument("--graph-k", type=int, default=8, help="k for k-NN graph construction.")
    parser.add_argument(
        "--downsample-backend",
        choices=["vedo", "open3d"],
        default="vedo",
        help="backend for downsampling (wrapper falls back automatically).",
    )
    parser.add_argument(
        "--root-prefix",
        help="Optional micromamba root prefix override (defaults to $MAMBA_ROOT_PREFIX or API/.micromamba).",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.upper_stl and not args.lower_stl:
        parser.error("at least one of --upper-stl or --lower-stl must be provided.")

    root_prefix = _resolve_root_prefix(args.root_prefix)
    root_prefix.mkdir(parents=True, exist_ok=True)

    micromamba = _find_micromamba()
    os.environ.setdefault("MICROMAMBA_EXE", str(micromamba))
    os.environ.setdefault("MAMBA_ROOT_PREFIX", str(root_prefix))

    _ensure_env(micromamba, ENV_SPEC, root_prefix)
    _ensure_ios_extras(micromamba, root_prefix)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    upper_out = (
        Path(args.upper_out).expanduser().resolve()
        if args.upper_out
        else (out_dir / "upper_seg.vtp")
    )
    lower_out = (
        Path(args.lower_out).expanduser().resolve()
        if args.lower_out
        else (out_dir / "lower_seg.vtp")
    )

    wrapper_script = API_ROOT / "wrappers" / "ios_model_wrapper.py"
    if not wrapper_script.exists():
        raise FileNotFoundError(f"wrapper script missing: {wrapper_script}")

    cmd: list[str] = [
        str(micromamba),
        "run",
        "--root-prefix",
        str(root_prefix),
        "-n",
        IOS_ENV_NAME,
        "python",
        str(wrapper_script),
        "--device",
        args.device,
        "--downsample",
        str(args.downsample),
        "--graph_k",
        str(args.graph_k),
        "--downsample_backend",
        args.downsample_backend,
    ]

    if args.upper_stl:
        cmd.extend(
            [
                "--upper_stl",
                str(Path(args.upper_stl).expanduser().resolve()),
                "--upper_out",
                str(upper_out),
            ]
        )
    if args.lower_stl:
        cmd.extend(
            [
                "--lower_stl",
                str(Path(args.lower_stl).expanduser().resolve()),
                "--lower_out",
                str(lower_out),
            ]
        )

    _run(cmd)

    outputs_created: list[str] = []
    if args.upper_stl:
        if upper_out.exists():
            outputs_created.append(str(upper_out))
        else:
            raise FileNotFoundError(f"expected upper VTP not created: {upper_out}")
    if args.lower_stl:
        if lower_out.exists():
            outputs_created.append(str(lower_out))
        else:
            raise FileNotFoundError(f"expected lower VTP not created: {lower_out}")

    print("[done] ios-model segmentation finished.")
    for path in outputs_created:
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

