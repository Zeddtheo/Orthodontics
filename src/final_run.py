#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import pyvista as pv

from iMeshSegNet.m0_dataset import ARCH_LABEL_ORDERS, _infer_jaw_from_stem
from iMeshSegNet.m2_infer import (
    _build_model,
    _features_and_pos_for_contract,
    _load_pipeline,
    _read_decimated,
    _select_6k_ids,
)
from iMeshSegNet.m3_postprocess import PPConfig, postprocess_6k_10k_full
from PointnetReg.pointnetreg import PointNetReg

LMK_PER_TOOTH_7 = [4, 4, 3, 5, 5, 6, 6]
LMK_PER_TOOTH_14 = LMK_PER_TOOTH_7 + LMK_PER_TOOTH_7


def default_heads_config_14() -> Dict[str, int]:
    return {f"t{i+1}": LMK_PER_TOOTH_14[i] for i in range(14)}


def class_to_head_key(cls_id: int, jaw: str) -> str:
    order = ARCH_LABEL_ORDERS.get(jaw.upper())
    if order is None:
        raise ValueError(f"Unknown jaw identifier: {jaw}")
    if not (1 <= cls_id <= len(order)):
        raise ValueError(f"invalid tooth class {cls_id} for jaw {jaw}")
    fdi = order[cls_id - 1]
    return f"t{fdi}"


def _load_landmark_names(def_path: Optional[Path], tooth_id: str, L: int) -> List[str]:
    default = [f"lm_{i:02d}" for i in range(L)]
    if def_path is None:
        return default
    try:
        with open(def_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception:
        return default

    per_tooth = spec.get("per_tooth", {})
    templates = spec.get("templates", {})
    entry = None
    for key in (tooth_id, tooth_id.lower(), tooth_id.upper()):
        if key in per_tooth:
            entry = per_tooth[key]
            break

    if isinstance(entry, dict):
        if isinstance(entry.get("order"), list):
            names = list(entry["order"])
            return names[:L] if len(names) >= L else names + default[len(names):L]
        if isinstance(entry.get("template"), str):
            entry = entry["template"]

    if isinstance(entry, str):
        tpl_names = templates.get(entry)
        if isinstance(tpl_names, list) and tpl_names:
            names = list(tpl_names)
            return names[:L] if len(names) >= L else names + default[len(names):L]

    return default


@dataclass
class TSConfig:
    seg_ckpt: Path
    reg_ckpt: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    min_roi_cells: int = 120
    use_sigmoid: bool = True
    landmark_def: Optional[Path] = Path("datasets/landmarks_dataset/cooked/landmark_def.json")


class TSMDL:
    def __init__(self, cfg: TSConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.seg_ckpt, self.seg_meta = _load_pipeline(cfg.seg_ckpt)
        self.seg_model = _build_model(cfg.seg_ckpt, self.device, self.seg_meta)

        reg_raw = torch.load(str(cfg.reg_ckpt), map_location="cpu")
        if isinstance(reg_raw, dict):
            reg_meta = dict(reg_raw)
            load_state = reg_raw
            for key in ("state_dict", "model_state_dict", "model"):
                if key in reg_raw:
                    load_state = reg_raw[key]
                    break
        else:
            load_state = reg_raw
            reg_meta = {"model": reg_raw}

        in_channels = int(reg_meta.get("in_channels", 3))
        heads_cfg = reg_meta.get("heads_config")
        has_heads_cfg = isinstance(heads_cfg, dict) and bool(heads_cfg)
        if has_heads_cfg:
            heads_config = {str(key): int(value) for key, value in heads_cfg.items()}
        else:
            heads_config = default_heads_config_14()
        if has_heads_cfg:
            num_landmarks = max(int(v) for v in heads_config.values())
        else:
            num_landmarks = int(reg_meta.get("num_landmarks", 0) or max(heads_config.values()))
        use_tnet = bool(reg_meta.get("use_tnet", True))
        enable_presence_head = bool(reg_meta.get("enable_presence_head", True))
        presence_hidden = int(reg_meta.get("presence_hidden", 128))

        self.reg_in_channels = in_channels
        self.reg_features = str(reg_meta.get("features", "pn"))
        self.reg_enable_presence = enable_presence_head

        self.reg_model = PointNetReg(
            in_channels=in_channels,
            num_landmarks=num_landmarks,
            heads_config=heads_config if has_heads_cfg else None,
            use_tnet=use_tnet,
            norm="gn",
            dropout_p=0.0,
            return_logits=True,
            enable_presence_head=enable_presence_head,
            presence_hidden=presence_hidden,
        ).to(self.device).eval()
        self.reg_model.load_state_dict(load_state, strict=False)

    @torch.no_grad()
    def segment_full(self, mesh_path: Path) -> Tuple[pv.PolyData, np.ndarray]:
        full = pv.read(str(mesh_path))
        pos_full_mm = full.cell_centers().points.astype(np.float32)

        mesh10k = _read_decimated(mesh_path.stem, self.seg_meta, mesh_path)
        feats10k, pos_norm10k, pos_mm10k, _, normals10k = _features_and_pos_for_contract(mesh10k, self.seg_meta)
        ids6k = _select_6k_ids(pos_mm10k, self.seg_meta)

        x = torch.from_numpy(feats10k[ids6k].T).unsqueeze(0).to(self.device)
        p = torch.from_numpy(pos_norm10k[ids6k].T).unsqueeze(0).to(self.device)
        logits = self.seg_model(x, p)[0].permute(1, 0).cpu().numpy()
        pos6_mm = pos_mm10k[ids6k]

        cfg = PPConfig()
        orig_ids = None
        for key in ("vtkOriginalCellIds", "orig_cell_ids"):
            if key in mesh10k.cell_data:
                orig_ids = np.asarray(mesh10k.cell_data[key]).astype(np.int64, copy=False)
                break
        lab10k, lab_full, _ = postprocess_6k_10k_full(
            pos6=pos6_mm,
            logits6=logits,
            pos10=pos_mm10k,
            pos_full=pos_full_mm,
            normals6=None if normals10k is None else normals10k[ids6k],
            orig_cell_ids=orig_ids,
            cfg=cfg,
        )
        full.cell_data["PredLabel"] = lab_full.astype(np.int32, copy=False)
        return full, lab_full

    @staticmethod
    def _extract_rois(mesh_full: pv.PolyData, labels_full: np.ndarray, min_cells: int) -> Dict[int, np.ndarray]:
        rois: Dict[int, np.ndarray] = {}
        for cls_id in range(1, 15):
            idx = np.where(labels_full == cls_id)[0]
            if idx.size >= min_cells:
                rois[cls_id] = idx
        return rois

    @torch.no_grad()
    def regress_landmarks(
        self,
        mesh_full: pv.PolyData,
        labels_full: np.ndarray,
        jaw: str,
    ) -> Tuple[List[Dict], Dict[str, Dict]]:
        results: List[Dict] = []
        json_payload: Dict[str, Dict] = {"predictions": {}}
        if self.cfg.landmark_def is not None:
            landmark_def_path = Path(self.cfg.landmark_def)
        else:
            landmark_def_path = None

        pos_full_mm = mesh_full.cell_centers().points.astype(np.float32)
        rois = self._extract_rois(mesh_full, labels_full, self.cfg.min_roi_cells)

        normals_full: Optional[np.ndarray] = None
        if getattr(self, "reg_in_channels", 3) >= 6:
            if "Normals" in mesh_full.cell_data:
                normals_full = np.asarray(mesh_full.cell_data["Normals"], dtype=np.float32)
            else:
                try:
                    normals_full = np.asarray(mesh_full.cell_normals, dtype=np.float32)
                except Exception:
                    normals_full = None
            if normals_full is not None and normals_full.shape[0] != mesh_full.n_cells:
                normals_full = None

        for cls_id, idx in rois.items():
            head = class_to_head_key(cls_id, jaw)
            xyz = pos_full_mm[idx]
            if xyz.shape[0] == 0:
                continue

            features: List[np.ndarray] = [xyz.T]
            in_channels = getattr(self, "reg_in_channels", 3)
            if in_channels >= 6:
                if normals_full is not None:
                    normals = normals_full[idx]
                else:
                    normals = np.zeros_like(xyz, dtype=np.float32)
                features.append(normals.T)
            if in_channels >= 9:
                centroid = xyz.mean(axis=0, keepdims=True)
                cent_rel = xyz - centroid
                features.append(cent_rel.T.astype(np.float32))
            feat_np = np.concatenate(features, axis=0).astype(np.float32, copy=False)
            if feat_np.shape[0] < in_channels:
                pad = np.zeros((in_channels - feat_np.shape[0], feat_np.shape[1]), dtype=np.float32)
                feat_np = np.concatenate([feat_np, pad], axis=0)
            x_tensor = torch.from_numpy(feat_np).unsqueeze(0).to(self.device)

            model_out = self.reg_model(
                x_tensor,
                tooth_id=head,
                return_presence=getattr(self, "reg_enable_presence", False),
            )
            if isinstance(model_out, tuple):
                logits_tensor, presence_logits = model_out
            else:
                logits_tensor = model_out
                presence_logits = None

            logits_np = logits_tensor.squeeze(0).cpu().numpy()
            if logits_np.ndim != 2:
                continue
            L, N = logits_np.shape
            if N == 0:
                continue

            order = np.argsort(logits_np, axis=1)
            top_idx = order[:, -1]
            if N >= 2:
                second_idx = order[:, -2]
            else:
                second_idx = top_idx.copy()

            top_vals = logits_np[np.arange(L), top_idx]
            second_vals = logits_np[np.arange(L), second_idx]
            scores = 1.0 / (1.0 + np.exp(-top_vals))
            second_scores = 1.0 / (1.0 + np.exp(-second_vals))
            margin_vals = 1.0 / (1.0 + np.exp(-(top_vals - second_vals)))

            coords = xyz[top_idx]
            names = _load_landmark_names(landmark_def_path, head, L)

            tooth_payload = {
                "landmarks_local": {},
                "landmarks_global": {},
                "indices": {},
                "scores": {},
                "top1": {},
                "peak_scores": {},
                "top2": {},
                "margin": {},
                "meta": {
                    "tooth_class": int(cls_id),
                    "roi_size": int(xyz.shape[0]),
                },
            }

            if presence_logits is not None:
                presence_logit = float(presence_logits.squeeze().cpu().item())
                presence_prob = float(torch.sigmoid(presence_logits).squeeze().cpu().item())
                tooth_payload["presence_logit"] = presence_logit
                tooth_payload["presence_prob"] = presence_prob

            for idx_l in range(L):
                name = names[idx_l]
                pt = coords[idx_l]
                tooth_payload["landmarks_local"][name] = pt.tolist()
                tooth_payload["landmarks_global"][name] = pt.tolist()
                tooth_payload["indices"][name] = int(top_idx[idx_l])
                tooth_payload["scores"][name] = float(scores[idx_l])
                tooth_payload["top1"][name] = float(scores[idx_l])
                tooth_payload["peak_scores"][name] = float(scores[idx_l])
                tooth_payload["top2"][name] = float(second_scores[idx_l])
                tooth_payload["margin"][name] = float(margin_vals[idx_l])

                results.append(
                    {
                        "tooth_class": int(cls_id),
                        "tooth_head": head,
                        "lm_index": int(idx_l),
                        "name": name,
                        "x_mm": float(pt[0]),
                        "y_mm": float(pt[1]),
                        "z_mm": float(pt[2]),
                        "confidence": float(scores[idx_l]),
                        "margin": float(margin_vals[idx_l]),
                        "roi_size": int(xyz.shape[0]),
                    }
                )

            json_payload["predictions"][head] = tooth_payload

        return results, json_payload

    def run_one(self, mesh_path: Path, seg_dir: Path, reg_dir: Path) -> None:
        seg_dir.mkdir(parents=True, exist_ok=True)
        reg_dir.mkdir(parents=True, exist_ok=True)
        full, lab_full = self.segment_full(mesh_path)

        full_col = full.copy(deep=True)
        full_col.cell_data["PredLabel"] = lab_full.astype(np.int32, copy=False)
        full_col.save(seg_dir / f"{mesh_path.stem}_seg_full.vtp", binary=True)

        jaw = _infer_jaw_from_stem(mesh_path.stem)
        preds, payload = self.regress_landmarks(full, lab_full, jaw)
        payload["case_id"] = mesh_path.stem
        payload.setdefault("meta", {})["jaw"] = jaw

        import csv

        csv_path = reg_dir / f"{mesh_path.stem}_landmarks.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["tooth_class", "tooth_head", "lm_index", "name", "x_mm", "y_mm", "z_mm", "confidence", "margin", "roi_size"]
            )
            for r in preds:
                writer.writerow(
                    [
                        r["tooth_class"],
                        r["tooth_head"],
                        r["lm_index"],
                        r.get("name", f"lm_{r['lm_index']:02d}"),
                        r["x_mm"],
                        r["y_mm"],
                        r["z_mm"],
                        r["confidence"],
                        r["margin"],
                        r["roi_size"],
                    ]
                )

        if preds:
            pts = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in preds], dtype=np.float32)
            pcloud = pv.PolyData(pts)
            pcloud.point_data["tooth_class"] = np.array([r["tooth_class"] for r in preds], np.int32)
            pcloud.point_data["lm_index"] = np.array([r["lm_index"] for r in preds], np.int32)
            pcloud.point_data["confidence"] = np.array([r["confidence"] for r in preds], np.float32)
            pcloud.save(reg_dir / f"{mesh_path.stem}_landmarks.vtp", binary=True)

        json_path = reg_dir / f"{mesh_path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

        print(f"✅ Done: {mesh_path.name} -> seg={seg_dir} | lmks={reg_dir}")


SEG_OUTPUT_ROOT = Path("outputs/segmentation")
REG_OUTPUT_ROOT = Path("outputs/pointnetreg")
DEFAULT_SEG_FINAL = SEG_OUTPUT_ROOT / "final_pt" / "best.pt"
DEFAULT_REG_FINAL = REG_OUTPUT_ROOT / "final_pt" / "best_mse.pt"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser("TS-MDL End-to-End Inference")
    ap.add_argument(
        "--seg-ckpt",
        type=Path,
        default=DEFAULT_SEG_FINAL,
        help=f"Stage-1 checkpoint路径（默认: {DEFAULT_SEG_FINAL.as_posix()})",
    )
    ap.add_argument(
        "--reg-ckpt",
        type=Path,
        default=DEFAULT_REG_FINAL,
        help=f"Stage-2 PointNet-Reg checkpoint路径（默认: {DEFAULT_REG_FINAL.as_posix()})",
    )
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path, help="输出目录或tag；相对路径会映射到 outputs/segmentation 与 outputs/pointnetreg")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--landmark-def",
        type=Path,
        default=Path("datasets/landmarks_dataset/cooked/landmark_def.json"),
        help="Landmark 定义文件，用于命名输出点（默认: datasets/landmarks_dataset/cooked/landmark_def.json）",
    )
    return ap.parse_args()


def _resolve_output_dirs(spec: Path) -> Tuple[Path, Path]:
    if spec.is_absolute():
        seg_dir = spec / "segmentation"
        reg_dir = spec / "pointnetreg"
        return seg_dir, reg_dir

    parts = list(spec.parts)
    if parts and parts[0].lower() == "outputs":
        parts = parts[1:]
    seg_dir = SEG_OUTPUT_ROOT.joinpath(*parts)
    reg_dir = REG_OUTPUT_ROOT.joinpath(*parts)
    return seg_dir, reg_dir


def main() -> None:
    args = parse_args()
    cfg = TSConfig(seg_ckpt=args.seg_ckpt, reg_ckpt=args.reg_ckpt, device=args.device)
    pipe = TSMDL(cfg)

    if args.input.is_file():
        files = [args.input]
    else:
        files = sorted(list(args.input.glob("*.vtp")) + list(args.input.glob("*.stl")))

    seg_dir, reg_dir = _resolve_output_dirs(args.out)

    for fp in files:
        pipe.run_one(fp, seg_dir, reg_dir)


if __name__ == "__main__":
    main()
