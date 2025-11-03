import importlib.util
import json
import os
import sys


CASES = [321, 322, 323, 324, 325, 326]
CFG = {
    "arch_form_side_diff_deg": 37.5,
    "arch_form_sharp_deg": 125.8,
    "arch_form_square_deg": 142.4,
}
BASE_DIR = os.path.join("outputs", "test1028")
OUT_DIR = os.path.join("outputs", "test1028_adjusted")
CALC_METRICS_PATH = os.path.join("src", "calc_metrics.py")


def load_calc_metrics():
    spec = importlib.util.spec_from_file_location("calc_metrics", CALC_METRICS_PATH)
    module = importlib.util.module_from_spec(spec)
    if spec.loader is None:
        raise RuntimeError("Failed to load calc_metrics module.")
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = load_calc_metrics()
    os.makedirs(OUT_DIR, exist_ok=True)

    summary = []
    def pick_source(case_dir: str, stem: str) -> str:
        candidates = [
            f"{stem}_landmarks.json",
            f"{stem}_pred.json",
            f"{stem}.json",
        ]
        for name in candidates:
            path = os.path.join(case_dir, name)
            if os.path.isfile(path):
                return path
        raise FileNotFoundError(f"missing JSON source for {stem} in {case_dir}")

    for idx in CASES:
        case = str(idx)
        case_dir = os.path.join(BASE_DIR, case)
        sources = [
            pick_source(case_dir, f"{case}_L"),
            pick_source(case_dir, f"{case}_U"),
        ]

        metrics = module.generate_metrics(
            sources,
            out_path=os.path.join(OUT_DIR, f"{case}_metrics.json"),
            cfg=CFG,
        )

        summary.append(
            {
                "case": case,
                "Arch_Form": metrics.get("Arch_Form"),
                "Arch_Width": metrics.get("Arch_Width"),
                "Bolton_Ratio": metrics.get("Bolton_Ratio"),
            }
        )

    json.dump(summary, sys.stdout, ensure_ascii=False, indent=2)
    if summary:
        sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
