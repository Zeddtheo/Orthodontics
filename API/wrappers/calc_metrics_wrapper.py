from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_repo_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / 'vendor'
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    calc_path = vendor_root / 'calc_metrics.py'
    if calc_path.exists():
        sys.path.insert(0, str(vendor_root))
    return repo_root
def main() -> None:
    parser = argparse.ArgumentParser(description="calc_metrics 包装：两个 JSON -> metrics.json")
    parser.add_argument("--json_a", required=True, help="上颌预测 JSON")
    parser.add_argument("--json_b", required=True, help="下颌预测 JSON")
    parser.add_argument("--stl_a", required=True, help="上颌 STL（可用于采集 meta）")
    parser.add_argument("--stl_b", required=True, help="下颌 STL")
    parser.add_argument("--out_json", required=True, help="输出 metrics.json 路径")
    args = parser.parse_args()

    out_path = Path(args.out_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _ensure_repo_root()

    from calc_metrics import generate_metrics  # type: ignore  # noqa: WPS433,E402

    generate_metrics(
        upper_stl_path=str(args.stl_a),
        lower_stl_path=str(args.stl_b),
        upper_json_path=str(args.json_a),
        lower_json_path=str(args.json_b),
        out_path=str(out_path),
    )


if __name__ == "__main__":
    main()
