from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import orjson

from .runner import run_pipeline


def _extract_landmarks(payload: Dict) -> Dict[str, List[float]]:
    """Convert Slicer markups JSON to {label: [x, y, z]} dict."""
    landmarks: Dict[str, List[float]] = {}
    for markup in payload.get("markups", []):
        for point in markup.get("controlPoints", []):
            label = point.get("label")
            position = point.get("position")
            if not label or not isinstance(position, list) or len(position) != 3:
                continue
            landmarks[label] = position
    return landmarks


def core(path_a: str, path_b: str, include_intermediate: bool = False) -> Dict:
    a = Path(path_a)
    b = Path(path_b)
    if not a.exists():
        raise FileNotFoundError(f"STL not found: {a}")
    if not b.exists():
        raise FileNotFoundError(f"STL not found: {b}")

    outputs = run_pipeline(a, b)

    workdir_path = Path(outputs["workdir"])
    json_a_path = Path(outputs["json_a"])
    json_b_path = Path(outputs["json_b"])
    up_payload = orjson.loads(json_a_path.read_bytes())
    low_payload = orjson.loads(json_b_path.read_bytes())

    response: Dict[str, object] = {
        "result": {
            "up": _extract_landmarks(up_payload),
            "low": _extract_landmarks(low_payload),
        },
    }

    if include_intermediate:
        response["artifacts"] = {
            "workdir": str(workdir_path),
            "vtp_a": outputs["vtp_a"],
            "vtp_b": outputs["vtp_b"],
            "json_a": outputs["json_a"],
            "json_b": outputs["json_b"],
        }
    else:
        if workdir_path.exists():
            shutil.rmtree(workdir_path, ignore_errors=True)
    return response


__all__ = ["core"]
