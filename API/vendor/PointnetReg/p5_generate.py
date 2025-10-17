#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from p2_infer import DEFAULT_TOOTH_IDS, infer_one_tooth
from p3_postprocess import (
    aggregate_case,
    collect_inference_records,
    export_case_csv,
    export_case_json,
    export_case_mrk,
    export_case_ply,
    export_debug_metrics,
    load_landmark_def,
    load_threshold_table,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAMPLES_ROOT = REPO_ROOT / "datasets" / "landmarks_dataset" / "cooked" / "p0" / "samples"
DEFAULT_LANDMARK_DEF = REPO_ROOT / "datasets" / "landmarks_dataset" / "cooked" / "landmark_def.json"
DEFAULT_RAW_ROOT = REPO_ROOT / "datasets" / "landmarks_dataset" / "raw"
DEFAULT_CKPT_ROOT = REPO_ROOT / "outputs" / "pointnetreg" / "final_pt"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs" / "pointnetreg" / "final_output"


@dataclass
class CaseResult:
    case_id: str
    summary: str
    json_paths: List[Path]
    debug_csv: Optional[Path]


def _load_preprocess_module() -> object:
    script_path = REPO_ROOT / "datasets" / "landmarks_dataset" / "preprocess.py"
    spec = importlib.util.spec_from_file_location("ld_preprocess", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载预处理脚本：{script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("ld_preprocess", module)
    spec.loader.exec_module(module)
    return module


def _looks_like_case_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    case_id = path.name
    patterns = [
        f"{case_id}_L.*",
        f"{case_id}_U.*",
        f"{case_id.lower()}_L.*",
        f"{case_id.lower()}_U.*",
        f"{case_id.upper()}_L.*",
        f"{case_id.upper()}_U.*",
    ]
    for pat in patterns:
        if any(path.glob(pat)):
            return True
    return False


def _normalise_case_id(case_id: str) -> str:
    cid = str(case_id).strip()
    if not cid:
        return cid
    if cid.isdigit():
        return f"{int(cid):03d}"
    return cid


def _discover_case_paths(input_path: Path, cases: Optional[Sequence[str]]) -> Dict[str, Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"输入路径不存在：{input_path}")

    if _looks_like_case_dir(input_path):
        canonical = _normalise_case_id(input_path.name)
        return {canonical or input_path.name: input_path.resolve()}

    if input_path.is_file():
        raise ValueError(f"输入路径 {input_path} 是文件，请指定包含病例的目录")

    all_dirs = sorted([p for p in input_path.iterdir() if p.is_dir()])
    if not all_dirs:
        raise FileNotFoundError(f"{input_path} 下没有案例目录")

    index: Dict[str, Path] = {}
    for case_dir in all_dirs:
        name = case_dir.name
        resolved = case_dir.resolve()
        index.setdefault(name, resolved)
        normalised = _normalise_case_id(name)
        if normalised:
            index.setdefault(normalised, resolved)

    def _pick_case(cid: str) -> tuple[str, Path]:
        key = str(cid).strip()
        if not key:
            raise ValueError("空的病例编号无效")
        if key in index:
            canonical = _normalise_case_id(key) or key
            return canonical, index[key]
        normalised = _normalise_case_id(key)
        if normalised and normalised in index:
            return normalised, index[normalised]
        if key.lstrip("0") and key.lstrip("0") in index:
            canonical = _normalise_case_id(key)
            return canonical, index[key.lstrip("0")]
        raise FileNotFoundError(f"未在 {input_path} 下找到案例 {cid}")

    selected: Dict[str, Path] = {}
    if cases:
        for cid in cases:
            key_lower = str(cid).strip().lower()
            if key_lower in {"all", "*"}:
                for case_dir in all_dirs:
                    canonical = _normalise_case_id(case_dir.name) or case_dir.name
                    selected[canonical] = case_dir.resolve()
                continue
            canonical, path = _pick_case(str(cid))
            selected[canonical] = path
    else:
        for case_dir in all_dirs:
            canonical = _normalise_case_id(case_dir.name) or case_dir.name
            selected[canonical] = case_dir.resolve()

    if not selected:
        raise ValueError("未选择任何案例")

    return dict(sorted(selected.items()))


def _ensure_preprocessed(case_id: str, samples_root: Path, skip: bool) -> List[Path]:
    candidates: set[str] = {case_id}
    if case_id.isdigit():
        stripped = case_id.lstrip("0")
        if stripped:
            candidates.add(stripped)
        candidates.add(f"{int(case_id):03d}")
    patterns = [f"{cid}_*_t*.npz" for cid in candidates]
    seen: Dict[Path, None] = {}
    def _refresh_existing() -> List[Path]:
        seen.clear()
        for pat in patterns:
            for path in samples_root.glob(pat):
                seen[path] = None
        return sorted(seen.keys())

    existing = _refresh_existing()
    if existing:
        teeth_found = set(_discover_teeth(existing))
        missing_teeth = [tid for tid in DEFAULT_TOOTH_IDS if tid not in teeth_found]
        if not missing_teeth:
            return existing
        if skip:
            raise RuntimeError(
                f"{case_id} 样本缺少牙位: {', '.join(missing_teeth)}; "
                "如需自动生成完整样本，请移除 --skip-preprocess"
            )

    if skip:
        raise FileNotFoundError(
            f"未找到 {case_id} 的 NPZ：{samples_root}，尝试模式 {', '.join(patterns)}"
        )
    preprocess = _load_preprocess_module()
    made, arches = preprocess.process_case(case_id=case_id, arches=None, for_infer=True)  # type: ignore[attr-defined]
    if made <= 0:
        raise RuntimeError(f"预处理未产生任何样本：case={case_id} arches={arches}")
    existing = _refresh_existing()
    if existing:
        teeth_found = set(_discover_teeth(existing))
        missing_teeth = [tid for tid in DEFAULT_TOOTH_IDS if tid not in teeth_found]
        if missing_teeth:
            print(
                f"[warn] 预处理后仍缺少牙位: {', '.join(missing_teeth)}；"
                "将继续推理，其余牙位结果正常生成。"
            )
    if not existing:
        raise RuntimeError(
            f"预处理后仍未找到样本：{samples_root}，已尝试模式 {', '.join(patterns)}"
        )
    return existing


def _discover_teeth(samples: List[Path]) -> List[str]:
    teeth: set[str] = set()
    for npz_path in samples:
        stem = npz_path.stem
        if "_t" not in stem:
            continue
        suffix = stem.split("_t", 1)[-1].strip()
        if not suffix:
            continue
        tooth_id = f"t{suffix.lower()}"
        teeth.add(tooth_id)
    ordered = [tid for tid in DEFAULT_TOOTH_IDS if tid in teeth]
    extras = sorted(tid for tid in teeth if tid not in DEFAULT_TOOTH_IDS)
    return ordered + extras


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def run_case(
    case_id: str,
    raw_dir: Path,
    samples_root: Path,
    ckpt_root: Path,
    infer_root: Path,
    out_root: Path,
    landmark_def: Path,
    threshold_table: Optional[Path],
    batch_size: int,
    workers: int,
    features: str,
    use_tnet: bool,
    export_roi_ply: bool,
    force_keep_all_teeth: bool,
    export_csv: bool,
    export_ply: bool,
    export_mrk: bool,
    export_debug: bool,
    skip_preprocess: bool,
) -> CaseResult:
    samples = _ensure_preprocessed(case_id, samples_root, skip_preprocess)
    teeth = _discover_teeth(samples)
    if not teeth:
        raise RuntimeError(f"{case_id} 未发现可用牙位样本")

    case_infer_root = infer_root / case_id
    _clean_dir(case_infer_root)

    for tooth_id in teeth:
        infer_one_tooth(
            root=samples_root,
            ckpt_root=ckpt_root,
            tooth_id=tooth_id,
            features=features,
            batch_size=batch_size,
            workers=workers,
            out_dir=case_infer_root,
            landmark_json=landmark_def,
            use_tnet=use_tnet,
            export_roi_ply=export_roi_ply,
            cases=[case_id],
        )

    defs, order = load_landmark_def(landmark_def)
    records = collect_inference_records(case_infer_root)
    if case_id not in records:
        raise RuntimeError(f"后处理入口未找到 {case_id} 的推理结果（{case_infer_root}）")

    thresholds = load_threshold_table(threshold_table)
    debug_rows: Optional[List[Dict[str, object]]] = [] if export_debug else None

    case = aggregate_case(
        case_id,
        records[case_id],
        defs,
        order,
        thresholds=thresholds,
        force_keep=force_keep_all_teeth,
        debug_rows=debug_rows,
    )

    out_root.mkdir(parents=True, exist_ok=True)
    json_paths = export_case_json(case, out_root / "json")
    if export_csv:
        export_case_csv(case, out_root / "csv")
    if export_ply:
        export_case_ply(case, out_root / "ply")
    if export_mrk:
        export_case_mrk(case, out_root / "markups")

    debug_csv = None
    if export_debug and debug_rows is not None:
        debug_csv = export_debug_metrics(debug_rows, out_root)

    stats = case.stats
    summary = (
        f"Case {case.case_id}: {stats['total_present']}/{stats['total_expected']} landmarks "
        f"({stats['coverage'] * 100:.1f}% coverage). Missing teeth: {stats['missing_teeth']}"
    )

    log_path = out_root / "pipeline.log"
    log_lines = [
        summary,
        f"raw_dir={raw_dir}",
        f"infer_dir={case_infer_root}",
        f"samples_root={samples_root}",
        f"checkpoint_root={ckpt_root}",
        f"teeth={' '.join(teeth)}",
    ]
    if debug_csv:
        log_lines.append(f"debug_metrics={debug_csv}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(log_lines) + "\n\n")

    return CaseResult(case_id=case.case_id, summary=summary, json_paths=[p.resolve() for p in json_paths], debug_csv=debug_csv)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="一键运行 PointNet-Reg 推理 + 后处理并生成最终 JSON 输出。"
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_RAW_ROOT, help="原始数据根目录或单个案例目录。")
    parser.add_argument("--cases", nargs="*", help="指定案例编号（默认自动从 --input 检测）。")
    parser.add_argument("--samples-root", type=Path, default=DEFAULT_SAMPLES_ROOT, help="NPZ 样本所在目录。")
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT, help="PointNet-Reg 模型 checkpoint 目录。")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="最终输出根目录。")
    parser.add_argument("--infer-dir", type=Path, default=None, help="推理中间结果目录（默认 out-dir/infer_tmp）。")
    parser.add_argument("--landmark-def", type=Path, default=DEFAULT_LANDMARK_DEF, help="landmark_def.json 路径。")
    parser.add_argument("--threshold-table", type=Path, default=None, help="可选阈值表 JSON。")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--features", type=str, default="pn", choices=["pn", "xyz"])
    parser.add_argument("--use-tnet", action="store_true", help="显式启用 TNet（若 checkpoint 已定义则以其中设置为准）。")
    parser.add_argument("--export-roi-ply", action="store_true", help="导出 ROI 点云用于调试。")
    parser.add_argument("--skip-preprocess", action="store_true", help="跳过缺失时的自动预处理。")
    parser.add_argument("--no-force-keep", dest="force_keep", action="store_false", help="允许阈值过滤掉整颗牙。")
    parser.add_argument("--force-keep", dest="force_keep", action="store_true", help="始终保留每颗牙（默认）。")
    parser.set_defaults(force_keep=True)
    parser.add_argument("--export-csv", action="store_true", help="导出 CSV 汇总。")
    parser.add_argument("--export-ply", action="store_true", help="导出单牙 PLY 点云。")
    parser.add_argument("--export-mrk", action="store_true", help="导出 3D Slicer Markups。")
    parser.add_argument("--export-debug", action="store_true", help="导出调试用指标 CSV。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = args.input.resolve()
    samples_root = args.samples_root.resolve()
    ckpt_root = args.ckpt_root.resolve()
    out_root = args.out_dir.resolve()
    infer_root = (args.infer_dir.resolve() if args.infer_dir else out_root / "infer_tmp")

    case_map = _discover_case_paths(input_path, args.cases)

    results: List[CaseResult] = []
    for case_id, raw_dir in case_map.items():
        print(f"=== Processing case {case_id} ===")
        result = run_case(
            case_id=case_id,
            raw_dir=raw_dir,
            samples_root=samples_root,
            ckpt_root=ckpt_root,
            infer_root=infer_root,
            out_root=out_root,
            landmark_def=args.landmark_def.resolve(),
            threshold_table=(args.threshold_table.resolve() if args.threshold_table else None),
            batch_size=args.batch_size,
            workers=args.workers,
            features=args.features,
            use_tnet=args.use_tnet,
            export_roi_ply=args.export_roi_ply,
            force_keep_all_teeth=args.force_keep,
            export_csv=args.export_csv,
            export_ply=args.export_ply,
            export_mrk=args.export_mrk,
            export_debug=args.export_debug,
            skip_preprocess=args.skip_preprocess,
        )
        results.append(result)
        print(result.summary)
        print("JSON:", ", ".join(str(p) for p in result.json_paths))
        if result.debug_csv:
            print(f"Debug metrics: {result.debug_csv}")

    if not results:
        print("No cases processed.")
        return

    summary_path = out_root / "summary.json"
    summary_payload = [
        {
            "case_id": r.case_id,
            "summary": r.summary,
            "json_paths": [str(p) for p in r.json_paths],
            "debug_csv": str(r.debug_csv) if r.debug_csv else None,
        }
        for r in results
    ]
    summary_path.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
