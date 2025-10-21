#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_landmarks(path: Path) -> dict[str, list[float]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unexpected JSON payload type: {type(payload)}")
    landmarks: dict[str, list[float]] = {}
    for key, value in payload.items():
        if not isinstance(key, str):
            raise TypeError(f"Invalid landmark key type: {type(key)}")
        if not isinstance(value, list) or len(value) != 3:
            raise ValueError(f"Landmark {key} must be a 3D coordinate list.")
        coords = []
        for idx, item in enumerate(value):
            if not isinstance(item, (int, float)):
                raise TypeError(f"Landmark {key} coordinate {idx} is not numeric: {item!r}")
            coords.append(float(item))
        landmarks[key] = coords  # type: ignore[assignment]
    return landmarks


def _validate_landmarks(landmarks: dict[str, list[float]], min_count: int) -> None:
    if len(landmarks) < min_count:
        raise ValueError(f"Expected at least {min_count} landmarks, got {len(landmarks)}.")
    missing = [name for name in ("11", "21", "31", "41") if name not in landmarks]
    if missing:
        raise ValueError(f"Missing key landmarks: {', '.join(missing)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate pipeline JSON output structure.")
    parser.add_argument("--json", required=True, type=Path, help="Path to pipeline output JSON.")
    parser.add_argument(
        "--min-count",
        type=int,
        default=24,
        help="Minimum number of landmarks expected in the result (default: 24).",
    )
    args = parser.parse_args()

    json_path = args.json.resolve()
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    landmarks = _load_landmarks(json_path)
    _validate_landmarks(landmarks, args.min_count)
    print(f"[OK] {json_path} 包含 {len(landmarks)} 个牙位坐标，结构校验通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
