from __future__ import annotations

import argparse
import json
import shutil
import sys
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pyvista as pv


def _ensure_repo_root() -> Dict[str, Path]:
    repo_root = Path(__file__).resolve().parents[1]
    vendor_root = repo_root / "vendor"
    pointnet_pkg = vendor_root / "PointnetReg"
    if str(vendor_root) not in sys.path:
        sys.path.insert(0, str(vendor_root))
    if str(pointnet_pkg) not in sys.path:
        sys.path.insert(0, str(pointnet_pkg))
    datasets_root = vendor_root / "datasets" / "landmarks_dataset"
    return {
        "repo": repo_root,
        "vendor": vendor_root,
        "datasets": datasets_root,
    }


def _default_paths(paths: Dict[str, Path]) -> Dict[str, Path]:
    datasets = paths["datasets"]
    samples = datasets / "cooked" / "p0" / "samples"
    landmark_def = datasets / "cooked" / "landmark_def.json"
    toothmap = datasets / "cooked" / "toothmap.json"
    raw_root = datasets / "raw"
    return {
        "samples_root": samples,
        "landmark_def": landmark_def,
        "toothmap": toothmap,
        "raw_root": raw_root,
    }


def _case_id_from_args(case_id: Optional[str]) -> str:
    if case_id:
        return str(case_id).strip()
    return uuid.uuid4().hex[:12]


def _copy_arch_vtp(src: Optional[Path], dst: Path) -> bool:
    if src is None or not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _list_case_npz(samples_root: Path, case_id: str) -> List[Path]:
    patterns = [f"{case_id}_*_t*.npz", f"{case_id.lower()}_*_t*.npz", f"{case_id.upper()}_*_t*.npz"]
    collected: List[Path] = []
    for pat in patterns:
        collected.extend(samples_root.glob(pat))
    return collected


def _cleanup_manifest(manifest: Path, case_id: str) -> None:
    if not manifest.exists():
        return
    lines = manifest.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    header, *rows = lines
    fieldnames = [h.strip() for h in header.split(",")]
    if "case_id" not in fieldnames:
        return
    idx_case = fieldnames.index("case_id")
    kept = [header]
    for row in rows:
        parts = row.split(",")
        if idx_case < len(parts) and parts[idx_case].strip() == case_id:
            continue
        kept.append(row)
    manifest.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")


def _cleanup_reports(reports_root: Path, case_id: str) -> None:
    if not reports_root.exists():
        return
    for path in reports_root.glob(f"{case_id}*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        else:
            path.unlink(missing_ok=True)


def _cleanup_samples(samples_root: Path, case_id: str) -> None:
    for npz_path in _list_case_npz(samples_root, case_id):
        npz_path.unlink(missing_ok=True)


def _load_min_faces() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    repo_path = str(repo_root)
    cleanup = False
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
        cleanup = True
    try:
        import datasets.landmarks_dataset.preprocess as preprocess_module  # type: ignore

        return int(getattr(preprocess_module, "MIN_FACES", 800))
    finally:
        if cleanup:
            sys.path.remove(repo_path)


MIN_FACES = _load_min_faces()


def _load_toothmap(path: Path) -> Dict[str, Dict[str, int]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    toothmap: Dict[str, Dict[str, int]] = {}
    for arch, mapping in data.items():
        if isinstance(mapping, dict):
            toothmap[arch.upper()] = {str(k): int(v) for k, v in mapping.items()}
    return toothmap


def _count_faces_by_label(mesh: pv.PolyData) -> Dict[int, int]:
    if "Label" in mesh.cell_data:
        labels = np.asarray(mesh.cell_data["Label"], dtype=np.int32)
    elif "PredictedID" in mesh.cell_data:
        labels = np.asarray(mesh.cell_data["PredictedID"], dtype=np.int32)
    else:
        raise RuntimeError("VTP 缺少 Label/PredictedID 数据，无法校验面片数")
    unique, counts = np.unique(labels, return_counts=True)
    return {int(label): int(count) for label, count in zip(unique, counts)}


def _validate_arch_vtp(
    vtp_path: Path,
    arch: str,
    toothmap: Dict[str, Dict[str, int]],
    min_faces: int,
) -> List[Tuple[int, int]]:
    if not vtp_path.exists():
        return []
    mesh = pv.read(str(vtp_path))
    counts = _count_faces_by_label(mesh)
    arch_key = arch.upper()
    mapping = toothmap.get(arch_key)
    if not mapping:
        return []
    labels_are_fdi = any(label >= 20 for label in counts if label not in (0,))
    missing: List[Tuple[int, int]] = []
    for fdi_str, class_id in mapping.items():
        fdi = int(fdi_str)
        if labels_are_fdi:
            face_count = counts.get(fdi, 0)
        else:
            face_count = counts.get(int(class_id), 0)
        if face_count < min_faces:
            missing.append((fdi, face_count))
    return missing


def run_pointnet_pipeline(
    case_id: str,
    upper_vtp: Optional[Path],
    lower_vtp: Optional[Path],
    json_a: Path,
    json_b: Path,
    model_root: Path,
    samples_root: Path,
    raw_root: Path,
    infer_root: Path,
    out_root: Path,
    landmark_def: Path,
    threshold_table: Optional[Path],
    skip_preprocess: bool = False,
) -> None:
    paths = _ensure_repo_root()
    vendor_root = paths["vendor"]
    datasets_root = paths["datasets"]
    import PointnetReg.p5_generate as p5_generate  # type: ignore  # noqa: WPS433
    p5_generate.REPO_ROOT = vendor_root
    p5_generate.DEFAULT_SAMPLES_ROOT = datasets_root / "cooked" / "p0" / "samples"
    p5_generate.DEFAULT_LANDMARK_DEF = datasets_root / "cooked" / "landmark_def.json"
    p5_generate.DEFAULT_RAW_ROOT = datasets_root / "raw"
    p5_generate.DEFAULT_CKPT_ROOT = model_root
    p5_generate.DEFAULT_OUTPUT_ROOT = out_root
    run_case = p5_generate.run_case
    raw_case_dir = raw_root / case_id
    if raw_case_dir.exists():
        shutil.rmtree(raw_case_dir)
    raw_case_dir.mkdir(parents=True, exist_ok=True)

    upper_present = _copy_arch_vtp(upper_vtp, raw_case_dir / f"{case_id}_U.vtp")
    lower_present = _copy_arch_vtp(lower_vtp, raw_case_dir / f"{case_id}_L.vtp")
    if not upper_present and not lower_present:
        raise FileNotFoundError("未提供任何 VTP 输入")

    defaults = _default_paths(paths)
    toothmap = _load_toothmap(defaults["toothmap"])
    issues: List[str] = []
    if upper_present:
        upper_missing = _validate_arch_vtp(
            raw_case_dir / f"{case_id}_U.vtp",
            "U",
            toothmap,
            MIN_FACES,
        )
        if upper_missing:
            issues.append(
                "上颌缺少面片：" + ", ".join(f"FDI {fdi} faces={faces}" for fdi, faces in upper_missing)
            )
    if lower_present:
        lower_missing = _validate_arch_vtp(
            raw_case_dir / f"{case_id}_L.vtp",
            "L",
            toothmap,
            MIN_FACES,
        )
        if lower_missing:
            issues.append(
                "下颌缺少面片：" + ", ".join(f"FDI {fdi} faces={faces}" for fdi, faces in lower_missing)
            )
    if issues:
        raise RuntimeError(
            "输入 VTP 面片数不足（每个牙位至少需要 "
            f"{MIN_FACES} 个面片）：{'；'.join(issues)}"
        )

    samples_root.mkdir(parents=True, exist_ok=True)
    infer_root.mkdir(parents=True, exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = samples_root.parent / "manifest.csv"
    reports_root = samples_root.parent / "reports"

    existing_samples = _list_case_npz(samples_root, case_id)
    case_infer_root = infer_root / case_id
    if case_infer_root.exists():
        shutil.rmtree(case_infer_root)

    result = run_case(
        case_id=case_id,
        raw_dir=raw_case_dir,
        samples_root=samples_root,
        ckpt_root=model_root,
        infer_root=infer_root,
        out_root=out_root,
        landmark_def=landmark_def,
        threshold_table=threshold_table,
        batch_size=4,
        workers=0,
        features="pn",
        use_tnet=False,
        export_roi_ply=False,
        force_keep_all_teeth=True,
        export_csv=False,
        export_ply=False,
        export_mrk=False,
        export_debug=False,
        skip_preprocess=skip_preprocess,
    )

    # Map produced JSONs to outputs
    json_out_dir = out_root / "json"
    arch_map = {"U": json_a, "L": json_b}
    for json_path in result.json_paths:
        stem = json_path.stem
        arch_suffix = stem.split("_")[-1].upper()
        target = arch_map.get(arch_suffix)
        if target is None:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(json_path, target)

    # Cleanup temporary artifacts
    shutil.rmtree(raw_case_dir, ignore_errors=True)
    _cleanup_reports(reports_root, case_id)
    _cleanup_samples(samples_root, case_id)
    _cleanup_manifest(manifest, case_id)
    shutil.rmtree(case_infer_root, ignore_errors=True)
    # remove out_root except JSON (already copied)
    shutil.rmtree(out_root, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="PointNet-Reg 推理包装器（复用 p5_generate.py）")
    parser.add_argument("--upper_vtp", help="上颌带标签 VTP")
    parser.add_argument("--lower_vtp", help="下颌带标签 VTP")
    parser.add_argument("--json_a", required=True, help="输出上颌 JSON 路径")
    parser.add_argument("--json_b", required=True, help="输出下颌 JSON 路径")
    parser.add_argument("--model_dir", default="models/pointnetreg", help="PointNet-Reg checkpoint 目录")
    parser.add_argument("--samples_root", help="p0 样本目录（默认 datasets/.../cooked/p0/samples）")
    parser.add_argument("--raw_root", help="原始数据目录（默认 datasets/.../raw）")
    parser.add_argument("--infer_root", help="推理临时目录（默认 <workdir>/pointnetreg/infer_tmp）")
    parser.add_argument("--out_root", help="最终输出目录（默认 <workdir>/pointnetreg/out）")
    parser.add_argument("--landmark_def", help="landmark_def.json 路径")
    parser.add_argument("--threshold_table", help="可选阈值表 JSON 路径")
    parser.add_argument("--workdir", required=True, help="工作目录（用于放置临时文件）")
    parser.add_argument("--case_id", help="案例 ID（默认随机）")
    parser.add_argument("--skip_preprocess", action="store_true", help="若样本缺失则直接报错，不调用预处理")
    args = parser.parse_args()

    paths = _ensure_repo_root()
    defaults = _default_paths(paths)

    case_id = _case_id_from_args(args.case_id)
    workdir = Path(args.workdir).resolve()
    infer_root = Path(args.infer_root).resolve() if args.infer_root else (workdir / "pointnetreg" / "infer_tmp")
    out_root = Path(args.out_root).resolve() if args.out_root else (workdir / "pointnetreg" / "out")
    samples_root = Path(args.samples_root).resolve() if args.samples_root else defaults["samples_root"]
    raw_root = Path(args.raw_root).resolve() if args.raw_root else defaults["raw_root"]
    landmark_def = Path(args.landmark_def).resolve() if args.landmark_def else defaults["landmark_def"]
    threshold_table = Path(args.threshold_table).resolve() if args.threshold_table else None
    model_root = Path(args.model_dir).resolve()
    json_a = Path(args.json_a).resolve()
    json_b = Path(args.json_b).resolve()

    run_pointnet_pipeline(
        case_id=case_id,
        upper_vtp=Path(args.upper_vtp).resolve() if args.upper_vtp else None,
        lower_vtp=Path(args.lower_vtp).resolve() if args.lower_vtp else None,
        json_a=json_a,
        json_b=json_b,
        model_root=model_root,
        samples_root=samples_root,
        raw_root=raw_root,
        infer_root=infer_root,
        out_root=out_root,
        landmark_def=landmark_def,
        threshold_table=threshold_table,
        skip_preprocess=args.skip_preprocess,
    )


if __name__ == "__main__":
    main()
