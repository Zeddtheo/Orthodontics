#!/usr/bin/env python3
"""
Batch-run calc_metrics.generate_metrics for cases 70..80.
For each case N, tries to use:
  datasets/landmarks_dataset/raw/N/N_L.json
  datasets/landmarks_dataset/raw/N/N_U.json
  datasets/landmarks_dataset/raw/N/N_L.vtp  (optional; passed as vtp_lower_path if present)
Results written to outputs/test1031/N_metrics.json
"""
import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# load calc_metrics via importlib to ensure the src file is used
CALC_PATH = SRC / "calc_metrics.py"
if not CALC_PATH.exists():
    print(f"ERROR: calc_metrics.py not found at {CALC_PATH}")
    raise SystemExit(2)

spec = importlib.util.spec_from_file_location("calc_metrics", str(CALC_PATH))
calc_metrics = importlib.util.module_from_spec(spec)
if spec.loader is None:
    print("ERROR: failed to load calc_metrics module")
    raise SystemExit(3)
spec.loader.exec_module(calc_metrics)

OUT_DIR = ROOT / "outputs" / "test1031"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_ROOT = ROOT / "datasets" / "landmarks_dataset" / "raw"

results = {}
errors = {}
for n in range(70, 81):
    case = str(n)
    base = DATA_ROOT / case
    if not base.exists() or not base.is_dir():
        errors[case] = f"case directory missing: {base}"
        print(f"[SKIP] {case}: directory not found")
        continue

    L_json = base / f"{case}_L.json"
    U_json = base / f"{case}_U.json"
    L_vtp = base / f"{case}_L.vtp"
    U_vtp = base / f"{case}_U.vtp"

    missing = []
    if not L_json.exists():
        missing.append(str(L_json))
    if not U_json.exists():
        missing.append(str(U_json))
    if missing:
        errors[case] = {"missing_files": missing}
        print(f"[SKIP] {case}: missing JSON(s): {missing}")
        continue

    sources = [str(L_json), str(U_json)]
    cfg = {}
    # prefer the lower vtp if present (used by Overjet point-to-surface)
    if L_vtp.exists():
        cfg['vtp_lower_path'] = str(L_vtp)
    elif U_vtp.exists():
        # if lower vtp missing but upper vtp present, still pass it (best-effort)
        cfg['vtp_lower_path'] = str(U_vtp)

    out_path = str(OUT_DIR / f"{case}_metrics.json")
    try:
        print(f"[RUN] {case}: sources={sources} cfg={cfg}")
        metrics = calc_metrics.generate_metrics(sources, out_path=out_path, cfg=cfg or None)
        # ensure something written/returned
        results[case] = {
            "out_path": out_path,
            "keys": sorted(list(metrics.keys())) if isinstance(metrics, dict) else None,
        }
        print(f"[OK] {case}: wrote {out_path} (items={len(metrics) if isinstance(metrics, dict) else 'N/A'})")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        errors[case] = {"error": str(e), "trace": tb}
        print(f"[ERROR] {case}: {e}")

# write a summary
summary_path = OUT_DIR / "summary_run.json"
with open(summary_path, 'w', encoding='utf-8') as f:
    json.dump({"results": results, "errors": errors}, f, ensure_ascii=False, indent=2)

print(f"Batch finished. {len(results)} succeeded, {len(errors)} had issues. Summary: {summary_path}")
if errors:
    print("Some cases had issues; see summary_run.json for details.")
