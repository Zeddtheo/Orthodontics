from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List

import json

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


def _group_landmarks(landmarks: Dict[str, List[float]]) -> Dict[str, Dict[str, List[float]]]:
    """Group landmarks into upper/lower buckets based on label prefix."""

    grouped: Dict[str, Dict[str, List[float]]] = {"upper": {}, "lower": {}}
    for label, coords in landmarks.items():
        if label.startswith(("1", "2")):
            grouped["upper"][label] = coords
        elif label.startswith(("3", "4")):
            grouped["lower"][label] = coords
    return grouped


def core(path_a: str, path_b: str, include_intermediate: bool = False) -> Dict[str, Dict[str, List[float]]]:
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
    up_payload = json.loads(json_a_path.read_text(encoding="utf-8"))
    low_payload = json.loads(json_b_path.read_text(encoding="utf-8"))

    up_landmarks = _extract_landmarks(up_payload)
    low_landmarks = _extract_landmarks(low_payload)
    landmarks: Dict[str, List[float]] = {}
    landmarks.update(up_landmarks)
    landmarks.update(low_landmarks)

    grouped = _group_landmarks(landmarks)

    if not include_intermediate and workdir_path.exists():
        shutil.rmtree(workdir_path, ignore_errors=True)

    return grouped


__all__ = ["core"]
