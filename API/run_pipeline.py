#!/usr/bin/env python
"""Command-line entrypoint for the ios-model → PointNet-Reg pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from utils.pano_core import core


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ios-model segmentation followed by PointNet-Reg landmark prediction.\n"
            "Requires the micromamba environments defined in envs/ to be available."
        )
    )
    parser.add_argument("--upper", required=True, help="Absolute path to the upper STL (path_a).")
    parser.add_argument("--lower", required=True, help="Absolute path to the lower STL (path_b).")
    parser.add_argument(
        "-o",
        "--output",
        help="Optional path to write the JSON result (defaults to stdout).",
    )
    parser.add_argument(
        "--keep-intermediate",
        "--keep",
        action="store_true",
        help="Keep intermediate artifacts on disk instead of auto-cleaning.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the JSON output with indentation.",
    )
    return parser


def _serialize_payload(payload: dict[str, object], pretty: bool) -> str:
    data = payload
    indent = 2 if pretty else None
    return json.dumps(data, ensure_ascii=False, indent=indent)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        payload = core(
            path_a=args.upper,
            path_b=args.lower,
            include_intermediate=args.keep_intermediate,
        )
    except FileNotFoundError as exc:
        parser.error(str(exc))
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Pipeline failed: {exc}", file=sys.stderr)
        return 1

    json_text = _serialize_payload(payload, args.pretty)

    if args.output:
        output_path = Path(args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + ("\n" if args.pretty else ""), encoding="utf-8")
    else:
        sys.stdout.write(json_text)
        if not json_text.endswith("\n"):
            sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
