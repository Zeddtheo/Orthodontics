#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import pyvista as pv

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


def class_to_head_key(cls_id: int) -> str:
    if not (1 <= cls_id <= 14):
        raise ValueError(f"invalid tooth class: {cls_id}")
    return f"t{cls_id}"


@dataclass
class TSConfig:
    seg_ckpt: Path
    reg_ckpt: Path
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    min_roi_cells: int = 120
    use_sigmoid: bool = True


class TSMDL:
    def __init__(self, cfg: TSConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device)

        self.seg_ckpt, self.seg_meta = _load_pipeline(cfg.seg_ckpt)
        self.seg_model = _build_model(cfg.seg_ckpt, self.device, self.seg_meta)

        heads = default_heads_config_14()
        self.reg_model = PointNetReg(
            in_channels=3,
            num_landmarks=-1,
            heads_config=heads,
            use_tnet=True,
            norm="gn",
            dropout_p=0.0,
            return_logits=True,
        ).to(self.device).eval()
        state = torch.load(str(cfg.reg_ckpt), map_location="cpu")
        if isinstance(state, dict):
            load_state = state
            for key in ("state_dict", "model_state_dict", "model"):
                if key in state:
                    load_state = state[key]
                    break
        else:
            load_state = state
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
    def regress_landmarks(self, mesh_full: pv.PolyData, labels_full: np.ndarray) -> List[Dict]:
        results: List[Dict] = []
        pos_full_mm = mesh_full.cell_centers().points.astype(np.float32)
        rois = self._extract_rois(mesh_full, labels_full, self.cfg.min_roi_cells)

        for cls_id, idx in rois.items():
            head = class_to_head_key(cls_id)
            xyz = pos_full_mm[idx]
            x = torch.from_numpy(xyz.T).unsqueeze(0).to(self.device)
            logits = self.reg_model(x, tooth_id=head)

            if self.cfg.use_sigmoid:
                heat = torch.sigmoid(logits).squeeze(0).cpu().numpy().T
            else:
                heat = logits.squeeze(0).cpu().numpy().T

            L = heat.shape[1]
            for l in range(L):
                ridx = int(np.argmax(heat[:, l]))
                pt = xyz[ridx]
                conf = float(heat[ridx, l])
                results.append(
                    {
                        "tooth_class": int(cls_id),
                        "tooth_head": head,
                        "lm_index": int(l),
                        "x_mm": float(pt[0]),
                        "y_mm": float(pt[1]),
                        "z_mm": float(pt[2]),
                        "confidence": conf,
                        "roi_size": int(xyz.shape[0]),
                    }
                )
        return results

    def run_one(self, mesh_path: Path, seg_dir: Path, reg_dir: Path) -> None:
        seg_dir.mkdir(parents=True, exist_ok=True)
        reg_dir.mkdir(parents=True, exist_ok=True)
        full, lab_full = self.segment_full(mesh_path)

        full_col = full.copy(deep=True)
        full_col.cell_data["PredLabel"] = lab_full.astype(np.int32, copy=False)
        full_col.save(seg_dir / f"{mesh_path.stem}_seg_full.vtp", binary=True)

        preds = self.regress_landmarks(full, lab_full)

        import csv

        csv_path = reg_dir / f"{mesh_path.stem}_landmarks.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["tooth_class", "tooth_head", "lm_index", "x_mm", "y_mm", "z_mm", "confidence", "roi_size"]
            )
            for r in preds:
                writer.writerow(
                    [
                        r["tooth_class"],
                        r["tooth_head"],
                        r["lm_index"],
                        r["x_mm"],
                        r["y_mm"],
                        r["z_mm"],
                        r["confidence"],
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
