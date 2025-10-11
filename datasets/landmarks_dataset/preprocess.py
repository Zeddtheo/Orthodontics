# p0_one_case_slicer.py  —— raw/1  ->  cooked/p0/samples/*.npz
# 依赖: pip install vedo numpy
import argparse
import json, csv, re
import numpy as np
from pathlib import Path
from typing import List, Optional
import vedo as v

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent
RAW_BASE = ROOT/"raw"  # raw case folders (e.g. contains 1/1_L.vtp)
COOK = ROOT/"cooked"/"p0"
(COOK/"samples").mkdir(parents=True, exist_ok=True)
(COOK/"reports").mkdir(parents=True, exist_ok=True)

toothmap_path    = ROOT/"cooked"/"toothmap.json"       # 你已放好
landmarkdef_path = ROOT/"cooked"/"landmark_def.json"   # 你已放好

# ---------- 配置 ----------
TOOTHMAP = json.loads(toothmap_path.read_text(encoding="utf-8"))
LM_DEF   = json.loads(landmarkdef_path.read_text(encoding="utf-8"))
TEMPLATES = LM_DEF["templates"]; PER_TOOTH = LM_DEF["per_tooth"]
L_MAX = int(LM_DEF.get("L_max", 12))
SIGMA_MM = 5.0
CUTOFF_SIGMA = 3.0
N_PRIME = 3000
MIN_FACES = 800

# ---------- 工具 ----------
def tooth_names(tooth_key:str):
    return TEMPLATES[PER_TOOTH[tooth_key]]

def _case_dir_name(case_id) -> str:
    case_str = str(case_id).strip()
    try:
        return str(int(case_str))
    except (ValueError, TypeError):
        return case_str

def parse_markups_json(json_path:Path):
    """只取 label/position，返回 dict: {'t37':{'db':xyz,...}, ...}, unit"""
    jd = json.loads(json_path.read_text(encoding="utf-8"))
    mks = jd["markups"][0]
    unit = mks.get("coordinateUnits","mm").lower()
    out = {}
    for cp in mks["controlPoints"]:
        lbl = cp.get("label","")
        pos = np.asarray(cp.get("position",[np.nan]*3), dtype=np.float32)
        # label 形如 "37db"、"36bgb"、"32ma"...
        m = re.match(r"^(\d{2})([A-Za-z]+)$", lbl)
        if not m: 
            continue
        fdi, name = m.group(1), m.group(2).lower()
        key = f"t{fdi}"
        if key not in out: out[key] = {}
        out[key][name] = pos
    return out, unit

def extract_triangles(poly: v.Mesh):
    try:
        faces = poly.faces()
    except AttributeError:
        faces = None
    if faces is None:
        faces = np.asarray(poly.cells)
    else:
        faces = np.asarray(faces)
    if faces.dtype == object:
        faces = np.stack(faces)
    if faces.ndim != 2 or faces.shape[1] < 3:
        raise ValueError('mesh cells must be triangles')
    if faces.shape[1] == 4 and np.all(faces[:, 0] == 3):
        tri = faces[:, 1:].astype(np.int64, copy=False)
    else:
        tri = faces[:, :3].astype(np.int64, copy=False)
    pts_attr = getattr(poly, 'points', None)
    if pts_attr is None:
        pts = np.asarray(poly.points())
    else:
        pts = np.asarray(pts_attr() if callable(pts_attr) else pts_attr)
    v0, v1, v2 = pts[tri[:,0]], pts[tri[:,1]], pts[tri[:,2]]
    return v0, v1, v2

def compute_features(v0, v1, v2):
    cent = (v0+v1+v2)/3.0
    nrm  = np.cross(v1-v0, v2-v0)
    nrm /= (np.linalg.norm(nrm, axis=1, keepdims=True)+1e-9)
    cent_rel = cent - cent.mean(axis=0, keepdims=True)
    x15 = np.concatenate([v0, v1, v2, nrm, cent_rel], axis=1).astype(np.float32)  # (N,15)
    pos = cent.astype(np.float32)                                                  # (N,3)
    return x15, pos

def fps(points: np.ndarray, m: int, start_idx: int = 0):
    N = points.shape[0]; m = min(m, N)
    sel = np.empty(m, dtype=np.int64); sel[0] = start_idx % N
    dists = np.full(N, 1e18, dtype=np.float64)
    for i in range(1, m):
        p = points[sel[i-1]]
        d = np.sum((points - p)**2, axis=1)
        dists = np.minimum(dists, d)
        sel[i] = int(np.argmax(dists))
    return sel

def make_heatmaps(centroids:(np.ndarray), lm_xyz:(np.ndarray), valid:(np.ndarray),
                  sigma_mm:float=5.0, cutoff_sigma:float=3.0):
    N = centroids.shape[0]; L_t = lm_xyz.shape[0]
    y = np.zeros((L_t, N), dtype=np.float32)
    mask = (valid>0.5).astype(np.float32)
    idx = np.where(mask>0.5)[0]
    if idx.size:
        C = centroids[None,:,:].astype(np.float32)
        P = lm_xyz[idx][:,None,:].astype(np.float32)
        d2 = ((C-P)**2).sum(axis=-1)
        s2 = sigma_mm**2
        y_eff = np.exp(-0.5*d2/s2).astype(np.float32)
        if cutoff_sigma and cutoff_sigma>0:
            thr2 = (cutoff_sigma**2)*s2
            y_eff[d2>thr2] = 0.0
        y[idx] = y_eff
    return y, mask

def pad_to_Lmax(y, lm_xyz, mask, L_max):
    L_t, N = y.shape
    y_pad   = np.zeros((L_max, N), dtype=np.float32)
    lm_pad  = np.full((L_max, 3), np.nan, dtype=np.float32)
    mask_pad= np.zeros((L_max,), dtype=np.float32)
    y_pad[:L_t]    = y
    lm_pad[:L_t]   = lm_xyz
    mask_pad[:L_t] = mask
    return y_pad, lm_pad, mask_pad


def discover_case_ids() -> List[str]:
    """List available case IDs (zero-padded for numeric names)."""
    if not RAW_BASE.exists():
        return []
    numeric = {}
    others = []
    for entry in RAW_BASE.iterdir():
        if not entry.is_dir():
            continue
        name = entry.name
        if name.isdigit():
            numeric[int(name)] = f"{int(name):03d}"
        else:
            others.append(name)
    ordered = [numeric[k] for k in sorted(numeric)]
    ordered.extend(sorted(others))
    return ordered

# ---------- 单例主流程 ----------
def detect_arches(raw_dir: Path) -> List[str]:
    arches = set()
    for pattern in ("*_*.vtp", "*_*.stl"):
        for mesh_path in raw_dir.glob(pattern):
            parts = mesh_path.stem.split("_")
            if len(parts) >= 2:
                arch = parts[-1].upper()
                if arch:
                    arches.add(arch)
    return sorted(a for a in arches if a in TOOTHMAP)

def process_arch(case_id="001", arch="L", for_infer: bool = False):
    arch = arch.upper()
    raw_dir = RAW_BASE / _case_dir_name(case_id)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw case dir not found: {raw_dir}")
    mesh_path = next(raw_dir.glob(f"*_{arch}.vtp"), None)
    if mesh_path is None:
        mesh_path = next(raw_dir.glob(f"*_{arch}.stl"), None)
    if mesh_path is None:
        print(f"[skip] {case_id} arch={arch} missing mesh file (.vtp/.stl) in {raw_dir}")
        return 0
    jsn = next(raw_dir.glob(f"*_{arch}.json"), None)
    if jsn is None and not for_infer:
        print(f"[skip] {case_id} arch={arch} missing landmark json in {raw_dir}")
        return 0

    mesh = v.load(str(mesh_path))
    if "Label" not in mesh.celldata.keys():
        print(f"[skip] {case_id} arch={arch} missing 'Label' cell data")
        return 0
    labels = np.asarray(mesh.celldata["Label"])

    # 单位（基本就是 mm；若意外很小则×1000）
    b = np.array(mesh.bounds()).reshape(3,2)
    unit_scale = 1000.0 if np.linalg.norm(b[:,1]-b[:,0]) < 1.0 else 1.0

    # 读 JSON（只用 label/position）；推理模式下跳过 GT，写占位
    if not for_infer and jsn is not None:
        lm_all, unit_json = parse_markups_json(jsn)
        if unit_json == "m":  # 一般不会发生
            for t in lm_all:
                for k in lm_all[t]:
                    lm_all[t][k] = lm_all[t][k]*1000.0
    else:
        lm_all = {}

    # arch 的 FDI -> tooth_id 映射，及忽略集
    arch_map = {int(k):int(v) for k,v in TOOTHMAP[arch].items()}
    ignore   = set(int(x) for x in TOOTHMAP.get("ignore", []))
    inv_map  = {tid:fdi for fdi,tid in arch_map.items()}  # 1..14 -> FDI

    made = 0
    for tid in range(1,15):
        fdi = inv_map.get(tid)
        if fdi is None or fdi in ignore:
            continue
        npz_name = COOK/"samples"/f"{case_id}_{arch}_t{fdi}.npz"

        # 裁 ROI（按 Label==FDI）
        sub = mesh.clone().threshold(scalars="Label", above=fdi-0.5, below=fdi+0.5, on="cells")
        if sub.ncells < MIN_FACES:
            print(f"[skip] {case_id} {arch} FDI={fdi} faces={sub.ncells}<{MIN_FACES}")
            continue

        # 单位对齐
        if unit_scale != 1.0:
            sub.points(sub.points()*unit_scale)

        # 特征
        v0, v1, v2 = extract_triangles(sub)
        x15, pos = compute_features(v0, v1, v2)

        # 零均值平移到 ROI 局部
        center = pos.mean(axis=0, keepdims=True)
        center_vec = center.squeeze(0).astype(np.float32)
        raw_bounds = np.asarray(sub.bounds(), dtype=np.float32)
        extent_vec = (raw_bounds[1::2] - raw_bounds[0::2]).astype(np.float32)
        pos = (pos - center).astype(np.float32)
        x15[:, -3:] = (x15[:, -3:] - x15[:, -3:].mean(axis=0, keepdims=True)).astype(np.float32)

        # 该牙应有的点位顺序 + 从 JSON 取对应点（转到局部坐标）
        tooth_key = f"t{fdi}"
        names = tooth_names(tooth_key)
        L_t = len(names)
        lm_dict = {} if for_infer else lm_all.get(tooth_key, {})
        lm_xyz = np.full((L_t,3), np.nan, dtype=np.float32)
        valid  = np.zeros((L_t,), dtype=np.float32)
        for i, nm in enumerate(names):
            p = lm_dict.get(nm)
            if p is not None and len(p)==3:
                lm_xyz[i] = p*unit_scale - center.squeeze(0)
                valid[i]  = 1.0

          # 热图 + mask
          # 与论文一致：σ=5mm，默认截断 3σ；可通过 CLI 调整。
        if for_infer:
            y_full = np.zeros((L_t, pos.shape[0]), dtype=np.float32)
            mask = np.zeros((L_t,), dtype=np.float32)
        else:
            y_full, mask = make_heatmaps(pos, lm_xyz, valid, sigma_mm=SIGMA_MM, cutoff_sigma=CUTOFF_SIGMA)
            if np.count_nonzero(mask) != L_t:
                missing = L_t - int(np.count_nonzero(mask))
                print(f"[skip] {case_id} arch={arch} FDI={fdi} 缺少 {missing} 个 landmark (期望 {L_t})")
                if npz_name.exists():
                    npz_name.unlink(missing_ok=True)
                continue

        # 采样 N'
        sel = fps(pos, N_PRIME, start_idx=0)
        if sel.shape[0] < N_PRIME:
            need = N_PRIME - sel.shape[0]
            if sel.shape[0] == 0:
                print(f"[skip] {case_id} arch={arch} FDI={fdi} 无有效点")
                if npz_name.exists():
                    npz_name.unlink(missing_ok=True)
                continue
            extra = np.random.choice(sel, size=need, replace=True)
            sel = np.concatenate([sel, extra])
        x_s, pos_s, y_s = x15[sel], pos[sel], y_full[:, sel]

        # pad 到 L_max
        y_pad, lm_pad, mask_pad = pad_to_Lmax(y_s, lm_xyz, mask, L_MAX)

        # 落盘
        npz_name = COOK/"samples"/f"{case_id}_{arch}_t{fdi}.npz"
        x_pn9 = np.concatenate([pos_s, x_s[:, 9:15]], axis=1).astype(np.float32)  # [pos, nrm, cent_rel]
        np.savez_compressed(npz_name,
            x=x_pn9,
            pos=pos_s.astype(np.float32),
            y=y_pad.astype(np.float32),
            loss_mask=mask_pad.astype(np.float32),
            landmarks=lm_pad.astype(np.float32),
            sample_indices=sel.astype(np.int64),
            meta=dict(case_id=case_id, arch=arch, fdi=int(fdi), tooth_id=int(tid),
                      L_t=int(L_t), L_max=int(L_MAX), sigma_mm=float(SIGMA_MM), cutoff_sigma=float(CUTOFF_SIGMA), unit="mm",
                      has_gt=not for_infer,
                      center_mm=center_vec.tolist(),
                      bounds_mm=extent_vec.tolist())
        )
        made += 1
        print(f"[ok] -> {npz_name.name}  x:{x_s.shape}  y:{y_pad.shape}  mask:{mask_pad.shape}")
    return made


def process_case(case_id="001", arches: Optional[List[str]] = None, for_infer: bool = False):
    raw_dir = RAW_BASE / _case_dir_name(case_id)
    if not raw_dir.exists():
        raise FileNotFoundError(f"raw case dir not found: {raw_dir}")

    if not arches:
        arch_candidates = detect_arches(raw_dir)
    else:
        arch_candidates = []
        for arch in arches:
            arch_upper = arch.upper()
            if arch_upper not in TOOTHMAP:
                print(f"[warn] skip arch '{arch}' not defined in toothmap")
                continue
            arch_candidates.append(arch_upper)

    if not arch_candidates:
        raise SystemExit(f"No arches to process for case {case_id}")

    total_made = 0
    processed_arches: List[str] = []
    for arch in arch_candidates:
        made = process_arch(case_id=case_id, arch=arch, for_infer=for_infer)
        upsert_manifest(case_id=case_id, arch=arch)
        print(f"[case] {case_id} arch={arch} -> {made} samples")
        total_made += made
        processed_arches.append(arch)
    return total_made, processed_arches

def upsert_manifest(case_id="001", arch="L"):
    man = COOK/"manifest.csv"
    rows = []
    if man.exists():
        reader = csv.DictReader(man.read_text(encoding="utf-8").splitlines())
        rows = [r for r in reader if not (r["case_id"] == case_id and r["arch"] == arch)]
    samples = sorted((COOK/"samples").glob(f"{case_id}_{arch}_t*.npz"))
    for f in samples:
        fdi = f.stem.split("_t")[-1]
        rows.append({
            "case_id": case_id, "arch": arch, "fdi": fdi,
            "tooth_id": TOOTHMAP[arch].get(fdi, ""),
            "npz_path": str(f.relative_to(COOK)),
            "split": "train"
        })
    fieldnames = ["case_id","arch","fdi","tooth_id","npz_path","split"]
    with man.open("w", newline="", encoding="utf-8") as fw:
        w = csv.DictWriter(fw, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[manifest] {man}  total={len(rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Slice tooth landmarks into per-tooth training samples.")
    parser.add_argument("--cases", nargs="+", default=["001"],
                        help="Case IDs to process (default: 001). Accepts numeric or zero-padded values. Use 'ALL' to process every detected case.")
    parser.add_argument("--arches", nargs="*", default=None,
                        help="Subset of arches to process (e.g. L U). Default: auto-detect from raw data.")
    parser.add_argument("--all", action="store_true",
                        help="Process every case discovered under the raw directory (same as --cases ALL).")
    parser.add_argument("--for-infer", action="store_true",
                        help="Skip landmark markups and emit zeroed y/mask placeholders for inference.")
    parser.add_argument("--sigma-mm", type=float, default=5.0,
                        help="Gaussian sigma in mm for heatmap generation (default: 5.0).")
    parser.add_argument("--cutoff-sigma", type=float, default=3.0,
                        help="Cutoff radius in multiples of sigma; 0 disables truncation (default: 3.0).")
    args = parser.parse_args()

    case_args = args.cases or []
    all_flag = args.all or any(str(c).strip().lower() in {"all", "*"} for c in case_args)
    if all_flag:
        case_ids = discover_case_ids()
        if not case_ids:
            print(f"No raw cases discovered under {RAW_BASE}")
            raise SystemExit(0)
    else:
        case_ids = case_args

    if not case_ids:
        print("No cases specified.")
        raise SystemExit(0)

    seen = set()
    ordered_case_ids = []
    for case in case_ids:
        case_str = str(case).strip()
        if case_str.lower() in {"all", "*"}:
            continue
        if case_str.isdigit():
            case_str = f"{int(case_str):03d}"
        if case_str in seen:
            continue
        seen.add(case_str)
        ordered_case_ids.append(case_str)

    if not ordered_case_ids:
        print("No valid case IDs to process.")
        raise SystemExit(0)

    SIGMA_MM = float(max(1e-6, args.sigma_mm))
    CUTOFF_SIGMA = float(max(0.0, args.cutoff_sigma))

    total_samples = 0
    for case_str in ordered_case_ids:
        made, processed_arches = process_case(case_id=case_str, arches=args.arches, for_infer=args.for_infer)
        arch_desc = ",".join(processed_arches) if processed_arches else "(none)"
        print(f"[summary] case {case_str} arches={arch_desc} samples={made}")
        total_samples += made

    print(f"done, samples made: {total_samples}")
