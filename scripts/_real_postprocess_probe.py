import json
import sys
from pathlib import Path

import numpy as np
import pyvista as pv
import torch

# Ensure local imports resolve
ROOT = Path(__file__).resolve().parent.parent
for candidate in (ROOT / "scripts", ROOT / "src"):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

import ceiling_stage1 as cs  # type: ignore[import]
from iMeshSegNet.m0_dataset import (  # type: ignore[import]
    SINGLE_ARCH_NUM_CLASSES,
    _build_single_arch_label_maps,
    _decim_cache_path,
    remap_labels_single_arch,
)
from iMeshSegNet.m2_infer import (  # type: ignore[import]
    _build_model,
    _features_and_pos_for_contract,
    _load_pipeline,
    _read_decimated,
    _select_6k_ids,
)
from iMeshSegNet.m3_postprocess import PPConfig, build_cell_adjacency  # type: ignore[import]


def run_cases(cases: list[Path], ckpt: Path) -> dict:
    ckpt_obj, meta = _load_pipeline(ckpt)
    device = torch.device("cpu")
    model = _build_model(ckpt, device, meta)
    model.eval()

    results: dict[str, dict] = {}
    for case_path in cases:
        meta_case = dict(meta)
        # Force auto cache resolution per case
        meta_case["decim_cache_vtp"] = None

        mesh_full = pv.read(str(case_path))
        try:
            mesh_full_tri = mesh_full.triangulate()
        except Exception:
            mesh_full_tri = mesh_full
        full_adj = build_cell_adjacency(mesh_full_tri)
        pos_full_mm = mesh_full.cell_centers().points.astype(np.float32)

        raw_labels = np.asarray(mesh_full.cell_data["Label"], dtype=np.int64)
        single_maps = _build_single_arch_label_maps(0, 15, False)
        gt_full = remap_labels_single_arch(raw_labels, case_path, single_maps)

        mesh10k = _read_decimated(case_path.stem, meta_case, case_path)
        try:
            mesh10k_tri = mesh10k.triangulate()
        except Exception:
            mesh10k_tri = mesh10k
        adj10 = build_cell_adjacency(mesh10k_tri)

        feats10k, pos_norm10k, pos_mm10k, _, normals10k = _features_and_pos_for_contract(mesh10k, meta_case)
        ids6k = _select_6k_ids(pos_mm10k, meta_case)
        pos6_mm = pos_mm10k[ids6k]
        normals6 = normals10k[ids6k] if normals10k is not None else None

        x = torch.from_numpy(feats10k[ids6k].T).unsqueeze(0).to(device)
        p = torch.from_numpy(pos_norm10k[ids6k].T).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(x, p)
        logits6 = logits[0].permute(1, 0).cpu().numpy()

        orig_ids = cs._extract_orig_ids(mesh10k)
        num_classes = int(meta_case.get("num_classes", SINGLE_ARCH_NUM_CLASSES))

        base_path = _decim_cache_path(case_path, int(meta_case.get("target_cells", pos_mm10k.shape[0])))
        assign_ids, assign_indices, assign_weights = cs._load_assignments(base_path)
        soft_rows = int(assign_indices.shape[0]) if assign_indices is not None else 0
        soft_k = int(assign_indices.shape[1]) if assign_indices is not None and assign_indices.ndim == 2 else 0
        soft_cover = float(soft_rows) / float(pos_full_mm.shape[0]) if pos_full_mm.shape[0] > 0 else 0.0

        cfg_base = PPConfig()
        variants = [
            ("full_real", False, None, None),
            ("s1_gc_off_real", False, {"gc_beta": 0.0, "gc_iterations": 0}, None),
            ("s2_fullgc_off_real", False, {"full_gc_enabled": False, "full_gc_lambda": 0.0, "full_gc_iterations": 0}, None),
            ("s3_dihedral_flat_real", False, None, "coef1"),
            ("s4_clean_off_real", False, {"min_component_size": 0, "min_component_size_full": 0, "enforce_single_component": False}, None),
            ("s5_gingiva_off_real", False, {"gingiva_dilate_iters": 0, "gingiva_protect_conf": 0.98}, None),
        ]

        case_out: dict[str, dict] = {}
        for variant, use_exact, overrides, pairwise_mode in variants:
            lab10k, lab_full, logs = cs.run_variant(
                variant=variant,
                use_exact=use_exact,
                meta=meta_case,
                cfg_base=cfg_base,
                pos6_mm=pos6_mm,
                logits6=logits6,
                pos10_mm=pos_mm10k,
                pos_full_mm=pos_full_mm,
                normals6=normals6,
                normals10=normals10k,
                full_adj=full_adj,
                adj10=adj10,
                orig_ids=orig_ids,
                assign_ids=assign_ids,
                assign_indices=assign_indices,
                assign_weights=assign_weights,
                cfg_override=overrides,
                pairwise_mode=pairwise_mode,
            )
            dsc = cs._compute_dsc(gt_full, lab_full, range(1, num_classes))
            mean_dsc = cs._summarise_dsc(dsc, range(1, num_classes - 1))
            case_out[variant] = {
                "mean_dsc": float(mean_dsc),
                "conf10_mean": float(logs.get("conf10_mean", -1.0)),
                "seed_ratio": float(logs.get("seed_ratio", -1.0)),
                "soft_seed_ratio": float(logs.get("soft_seed_ratio", logs.get("seed_ratio", -1.0))),
                "low_conf_ratio": float(logs.get("low_conf_ratio", logs.get("low_conf_ratio_full", np.nan))),
            }

        results[case_path.name] = {
            "assign_info": {
                "soft_rows": soft_rows,
                "soft_k": soft_k,
                "soft_cover": soft_cover,
                "has_weights": bool(assign_weights is not None),
                "has_assign_ids": bool(assign_ids is not None),
            },
            "variants": case_out,
        }
    return results


if __name__ == "__main__":
    CASES = [
        Path("datasets/segmentation_dataset/1_U.vtp"),
        Path("datasets/segmentation_dataset/8_U.vtp"),
    ]
    CKPT = Path("outputs/segmentation/final_pt/best.pt")

    summary = run_cases(CASES, CKPT)
    print(json.dumps(summary, indent=2))
