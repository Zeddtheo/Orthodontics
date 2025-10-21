from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
API_ROOT = ROOT / "API"
VENDOR_ROOT = API_ROOT / "vendor"
POINTNET_ROOT = VENDOR_ROOT / "PointnetReg"
for path in [ROOT, API_ROOT, VENDOR_ROOT, POINTNET_ROOT]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from API.wrappers.pointnetreg_wrapper import _ensure_repo_root
import PointnetReg.p5_generate as p5_generate  # type: ignore

CASE_ID = "317"
WORK_ROOT = (ROOT / "datasets/tests/testout/317_debug").resolve()
RAW_ROOT = VENDOR_ROOT / "datasets" / "landmarks_dataset" / "raw"
RAW_CASE_DIR = RAW_ROOT / CASE_ID
RAW_CASE_DIR.mkdir(parents=True, exist_ok=True)

upper_vtp_src = ROOT / "datasets/tests/testout/317_U.vtp"
lower_vtp_src = ROOT / "datasets/tests/testout/317_L.vtp"
shutil.copy2(upper_vtp_src, RAW_CASE_DIR / f"{CASE_ID}_U.vtp")
shutil.copy2(lower_vtp_src, RAW_CASE_DIR / f"{CASE_ID}_L.vtp")

paths = _ensure_repo_root()
vendor_root = paths["vendor"]
datasets_root = paths["datasets"]

p5_generate.REPO_ROOT = vendor_root
p5_generate.DEFAULT_SAMPLES_ROOT = datasets_root / "cooked" / "p0" / "samples"
p5_generate.DEFAULT_LANDMARK_DEF = datasets_root / "cooked" / "landmark_def.json"
p5_generate.DEFAULT_RAW_ROOT = datasets_root / "raw"
p5_generate.DEFAULT_CKPT_ROOT = ROOT / "API/models/pointnetreg"
p5_generate.DEFAULT_OUTPUT_ROOT = WORK_ROOT / "pointnetreg" / "out"

samples_root = datasets_root / "cooked" / "p0" / "samples"
ckpt_root = ROOT / "API/models/pointnetreg"
infer_root = WORK_ROOT / "pointnetreg" / "infer_tmp"
out_root = WORK_ROOT / "pointnetreg" / "out"
landmark_def = datasets_root / "cooked" / "landmark_def.json"

out_root.mkdir(parents=True, exist_ok=True)
infer_root.mkdir(parents=True, exist_ok=True)

result = p5_generate.run_case(
    case_id=CASE_ID,
    raw_dir=RAW_CASE_DIR,
    samples_root=samples_root,
    ckpt_root=ckpt_root,
    infer_root=infer_root,
    out_root=out_root,
    landmark_def=landmark_def,
    threshold_table=None,
    batch_size=4,
    workers=0,
    features="pn",
    use_tnet=False,
    export_roi_ply=True,
    force_keep_all_teeth=True,
    export_csv=True,
    export_ply=True,
    export_mrk=True,
    export_debug=True,
    skip_preprocess=False,
)

print(result.summary)
print("JSON:", result.json_paths)
