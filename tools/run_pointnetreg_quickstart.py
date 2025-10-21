#!/usr/bin/env python3
"""One-click PointNet-Reg inference helper.

This script:
1. Ensures the `pointnetreg` micromamba environment exists (creating it from
   `API/envs/pointnetreg.yml` when missing).
2. Invokes `API/wrappers/pointnetreg_wrapper.py` inside that environment to
   generate landmark JSON outputs from labelled VTP meshes.

Example:
    python tools/run_pointnetreg_quickstart.py \
        --upper-vtp datasets/tests/316/316_U_seg.vtp \
        --lower-vtp datasets/tests/316/316_L_seg.vtp \
        --out-dir outputs/pointnetreg_quickstart
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
API_ROOT = REPO_ROOT / "API"
ENV_SPEC = API_ROOT / "envs" / "pointnetreg.yml"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "pointnetreg_quickstart"
DEFAULT_WORKDIR_NAME = "tmp_pointnetreg"


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


def _ensure_env(micromamba: Path, env_name: str, env_spec: Path, root_prefix: Path) -> None:
    env_dir = root_prefix / "envs" / env_name
    if env_dir.exists():
        print(f"[info] micromamba env '{env_name}' already present under {env_dir}")
        return
    print(f"[info] creating micromamba env '{env_name}' from {env_spec}")
    env_spec = env_spec.resolve()
    env_dir.parent.mkdir(parents=True, exist_ok=True)
    _run(
        [
            str(micromamba),
            "create",
            "-y",
            "-n",
            env_name,
            "-f",
            str(env_spec),
            "--root-prefix",
            str(root_prefix),
        ]
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ensure PointNet-Reg environment is ready and run inference via the project wrapper."
    )
    parser.add_argument("--upper-vtp", help="Path to the upper-arch labelled VTP mesh.")
    parser.add_argument("--lower-vtp", help="Path to the lower-arch labelled VTP mesh.")
    parser.add_argument(
        "--out-dir",
        help=f"Directory to store landmark JSON outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--upper-output",
        help="Explicit path for the upper JSON output (defaults to <out-dir>/upper_landmarks.json).",
    )
    parser.add_argument(
        "--lower-output",
        help="Explicit path for the lower JSON output (defaults to <out-dir>/lower_landmarks.json).",
    )
    parser.add_argument(
        "--workdir",
        help="Temporary working directory for intermediate files (default: <out-dir>/tmp_pointnetreg).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary working directory instead of cleaning it up.",
    )
    parser.add_argument(
        "--root-prefix",
        help="Optional custom micromamba root prefix (defaults to $MAMBA_ROOT_PREFIX or API/.micromamba).",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Forwarded to the wrapper: skip preprocessing if cached samples are missing.",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    if not args.upper_vtp and not args.lower_vtp:
        parser.error("at least one of --upper-vtp or --lower-vtp must be provided.")

    root_prefix = _resolve_root_prefix(args.root_prefix)
    root_prefix.mkdir(parents=True, exist_ok=True)

    micromamba = _find_micromamba()
    os.environ.setdefault("MICROMAMBA_EXE", str(micromamba))
    os.environ.setdefault("MAMBA_ROOT_PREFIX", str(root_prefix))

    _ensure_env(micromamba, "pointnetreg", ENV_SPEC, root_prefix)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else DEFAULT_OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    upper_output = (
        Path(args.upper_output).expanduser().resolve()
        if args.upper_output
        else (out_dir / "upper_landmarks.json")
    )
    lower_output = (
        Path(args.lower_output).expanduser().resolve()
        if args.lower_output
        else (out_dir / "lower_landmarks.json")
    )

    workdir = (
        Path(args.workdir).expanduser().resolve()
        if args.workdir
        else (out_dir / DEFAULT_WORKDIR_NAME)
    )
    workdir.mkdir(parents=True, exist_ok=True)

    wrapper_script = API_ROOT / "wrappers" / "pointnetreg_wrapper.py"
    if not wrapper_script.exists():
        raise FileNotFoundError(f"wrapper script missing: {wrapper_script}")

    cmd: list[str] = [
        str(micromamba),
        "run",
        "--root-prefix",
        str(root_prefix),
        "-n",
        "pointnetreg",
        "python",
        str(wrapper_script),
        "--json_a",
        str(upper_output),
        "--json_b",
        str(lower_output),
        "--model_dir",
        str((API_ROOT / "models" / "pointnetreg").resolve()),
        "--workdir",
        str(workdir),
    ]

    if args.upper_vtp:
        cmd.extend(["--upper_vtp", str(Path(args.upper_vtp).expanduser().resolve())])
    if args.lower_vtp:
        cmd.extend(["--lower_vtp", str(Path(args.lower_vtp).expanduser().resolve())])
    if args.skip_preprocess:
        cmd.append("--skip_preprocess")

    try:
        _run(cmd)
    finally:
        if not args.keep_temp and workdir.exists():
            print(f"[info] cleaning up temporary directory {workdir}")
            shutil.rmtree(workdir, ignore_errors=True)

    outputs_created: list[str] = []
    if args.upper_vtp:
        if upper_output.exists():
            outputs_created.append(str(upper_output))
        else:
            raise FileNotFoundError(f"expected upper JSON not created: {upper_output}")
    if args.lower_vtp:
        if lower_output.exists():
            outputs_created.append(str(lower_output))
        else:
            raise FileNotFoundError(f"expected lower JSON not created: {lower_output}")

    print("[done] PointNet-Reg inference finished.")
    for path in outputs_created:
        print(f"  -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

