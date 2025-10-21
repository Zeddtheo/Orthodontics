from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List

DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("API_ROOT", str(DEFAULT_ROOT))).resolve()
WRAPPERS = ROOT / "wrappers"
_default_prefix = DEFAULT_ROOT / ".micromamba"
if not _default_prefix.exists():
    _default_prefix = Path.home() / ".micromamba"
ROOT_PREFIX = Path(os.environ.get("MAMBA_ROOT_PREFIX", str(_default_prefix))).resolve()
_MICROMAMBA_CANDIDATES = [
    os.environ.get("MICROMAMBA_EXE"),
    "micromamba",
    str(Path.cwd() / "tools" / "micromamba.exe"),
    str(DEFAULT_ROOT / "build-tools" / "micromamba-linux"),
    str(Path(__file__).resolve().parents[2] / "tools" / "micromamba.exe"),
]


def _resolve_micromamba() -> str:
    for candidate in _MICROMAMBA_CANDIDATES:
        if not candidate:
            continue
        if shutil.which(candidate):
            return candidate
        path = Path(candidate)
        if path.exists():
            return str(path)
    raise RuntimeError("未找到 micromamba 可执行文件，请设置 MICROMAMBA_EXE 环境变量或放置在 PATH 中。")


MICROMAMBA = _resolve_micromamba()


def _run(cmd: List[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    return proc.stdout


def run_pipeline(
    stl_a: Path,
    stl_b: Path,
    workdir: Path | None = None,
    run_metrics: bool = False,
) -> Dict[str, str]:
    workdir = workdir or (ROOT / "runs" / str(uuid.uuid4()))
    workdir.mkdir(parents=True, exist_ok=True)

    vtp_a = workdir / "A.vtp"
    vtp_b = workdir / "B.vtp"
    ios_wrapper = WRAPPERS / "ios_model_wrapper.py"
    base_cmd = [MICROMAMBA, "run", "--root-prefix", str(ROOT_PREFIX)]

    ios_cmd = (
        base_cmd
        + [
            "-n",
            "ios_model",
            "python",
            str(ios_wrapper),
        ]
    )
    ios_cmd += [
        "--upper_stl",
        str(stl_a),
        "--upper_out",
        str(vtp_a),
        "--lower_stl",
        str(stl_b),
        "--lower_out",
        str(vtp_b),
    ]
    _run(ios_cmd)

    json_a = workdir / "A.json"
    json_b = workdir / "B.json"
    pointnet_wrapper = WRAPPERS / "pointnetreg_wrapper.py"
    pointnet_workdir = workdir / "pointnetreg"
    pointnet_workdir.mkdir(parents=True, exist_ok=True)

    _run(
        base_cmd
        + [
            "-n",
            "pointnetreg",
            "python",
            str(pointnet_wrapper),
            "--upper_vtp",
            str(vtp_a),
            "--lower_vtp",
            str(vtp_b),
            "--json_a",
            str(json_a),
            "--json_b",
            str(json_b),
            "--model_dir",
            str(ROOT / "models" / "pointnetreg"),
            "--workdir",
            str(pointnet_workdir),
        ]
    )

    result = {
        "workdir": str(workdir),
        "vtp_a": str(vtp_a),
        "vtp_b": str(vtp_b),
        "json_a": str(json_a),
        "json_b": str(json_b),
    }

    if run_metrics:
        metrics_path = workdir / "metrics.json"
        calc_wrapper = WRAPPERS / "calc_metrics_wrapper.py"
        _run(
            base_cmd
            + [
                "-n",
                "calc",
                "python",
                str(calc_wrapper),
                "--json_a",
                str(json_a),
                "--json_b",
                str(json_b),
                "--stl_a",
                str(stl_a),
                "--stl_b",
                str(stl_b),
                "--out_json",
                str(metrics_path),
            ]
        )
        result["metrics"] = str(metrics_path)

    return result


__all__ = ["run_pipeline"]
