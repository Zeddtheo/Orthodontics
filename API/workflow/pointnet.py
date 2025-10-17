from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv
import torch

from PointnetReg.pointnetreg import PointNetReg
from iMeshSegNet.m0_dataset import ARCH_LABEL_ORDERS

LMK_PER_TOOTH_7 = [4, 4, 3, 5, 5, 6, 6]
LMK_PER_TOOTH_14 = LMK_PER_TOOTH_7 + LMK_PER_TOOTH_7


def default_heads_config_14() -> Dict[str, int]:
    return {f"t{i+1}": LMK_PER_TOOTH_14[i] for i in range(14)}


def class_to_head_key(cls_id: int, arch: str) -> str:
    order = ARCH_LABEL_ORDERS.get(arch.upper())
    if order is None:
        raise ValueError(f"Unknown arch identifier: {arch}")
    if not (1 <= cls_id <= len(order)):
        raise ValueError(f"Invalid tooth class {cls_id} for arch {arch}")
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


def _extract_rois(labels_full: np.ndarray, min_cells: int = 120) -> Dict[int, np.ndarray]:
    rois: Dict[int, np.ndarray] = {}
    for cls_id in range(1, 15):
        idx = np.where(labels_full == cls_id)[0]
        if idx.size >= min_cells:
            rois[cls_id] = idx
    return rois


@dataclass
class PointNetOutputs:
    payload_json: Path
    csv_path: Optional[Path]
    landmarks_vtp: Optional[Path]


class PointNetRegRunner:
    def __init__(
        self,
        ckpt_path: Path,
        landmark_def: Optional[Path] = None,
        device: Optional[str] = None,
    ) -> None:
        self.ckpt_path = Path(ckpt_path).resolve()
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.landmark_def = landmark_def
        self._load_model()

    def _load_model(self) -> None:
        ckpt_raw = torch.load(str(self.ckpt_path), map_location="cpu")
        if isinstance(ckpt_raw, dict):
            reg_meta = dict(ckpt_raw)
            load_state = ckpt_raw
            for key in ("state_dict", "model_state_dict", "model"):
                if key in ckpt_raw:
                    load_state = ckpt_raw[key]
                    break
        else:
            load_state = ckpt_raw
            reg_meta = {"model": ckpt_raw}

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
        self.reg_enable_presence = enable_presence_head

        self.reg_model = PointNetReg(
            in_channels=in_channels,
            num_landmarks=num_landmarks,
            heads_config=heads_config,
            use_tnet=use_tnet,
            enable_presence_head=enable_presence_head,
            presence_hidden=presence_hidden,
        )
        if isinstance(load_state, dict):
            self.reg_model.load_state_dict(load_state, strict=False)
        else:
            self.reg_model.load_state_dict(load_state.state_dict())  # type: ignore[attr-defined]
        self.reg_model = self.reg_model.to(self.device)
        self.reg_model.eval()

    def _ensure_normals(self, mesh: pv.PolyData) -> np.ndarray | None:
        if "Normals" in mesh.cell_data:
            normals = np.asarray(mesh.cell_data["Normals"], dtype=np.float32)
            if normals.shape[0] == mesh.n_cells:
                return normals
        try:
            normals_mesh = mesh.compute_normals(
                cell_normals=True,
                point_normals=False,
                inplace=False,
            )
        except TypeError:
            normals_mesh = mesh.compute_normals(inplace=False)
        if "Normals" in normals_mesh.cell_data:
            return np.asarray(normals_mesh.cell_data["Normals"], dtype=np.float32)
        return None

    def run_case(
        self,
        mesh: pv.PolyData,
        arch: str,
        case_id: str,
        output_dir: Path,
        min_roi_cells: int = 120,
    ) -> PointNetOutputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        labels_full = None
        for key in ("Label", "PredLabel", "PredictedID"):
            if key in mesh.cell_data:
                labels_full = np.asarray(mesh.cell_data[key], dtype=np.int32).reshape(-1)
                break
        if labels_full is None:
            raise ValueError("Provided mesh缺少 cell_data['Label']，无法执行 PointNet-Reg 推理。")

        rois = _extract_rois(labels_full, min_cells=min_roi_cells)
        pos_full_mm = mesh.cell_centers().points.astype(np.float32)
        normals_full = self._ensure_normals(mesh)

        character = arch.upper()
        json_payload: Dict[str, Dict] = {"predictions": {}}
        results: List[Dict] = []
        def_path = self.landmark_def

        for cls_id, idx in rois.items():
            head = class_to_head_key(cls_id, character)
            xyz = pos_full_mm[idx]
            if xyz.shape[0] == 0:
                continue

            features: List[np.ndarray] = [xyz.T]
            if self.reg_in_channels >= 6:
                if normals_full is not None:
                    normals = normals_full[idx]
                else:
                    normals = np.zeros_like(xyz, dtype=np.float32)
                features.append(normals.T)
            if self.reg_in_channels >= 9:
                centroid = xyz.mean(axis=0, keepdims=True)
                cent_rel = xyz - centroid
                features.append(cent_rel.T.astype(np.float32))
            feat_np = np.concatenate(features, axis=0).astype(np.float32, copy=False)
            if feat_np.shape[0] < self.reg_in_channels:
                pad = np.zeros((self.reg_in_channels - feat_np.shape[0], feat_np.shape[1]), dtype=np.float32)
                feat_np = np.concatenate([feat_np, pad], axis=0)
            x_tensor = torch.from_numpy(feat_np).unsqueeze(0).to(self.device)

            with torch.no_grad():
                model_out = self.reg_model(
                    x_tensor,
                    tooth_id=head,
                    return_presence=self.reg_enable_presence,
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
            names = _load_landmark_names(def_path, head, L)

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
                    "arch": character,
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

        json_payload["case_id"] = case_id
        json_payload.setdefault("meta", {})["arch"] = character

        json_path = output_dir / f"{case_id}.json"
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(json_payload, fh, ensure_ascii=False, indent=2)

        csv_path = None
        if results:
            csv_path = output_dir / f"{case_id}_landmarks.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["tooth_class", "tooth_head", "lm_index", "name", "x_mm", "y_mm", "z_mm", "confidence", "margin", "roi_size"]
                )
                for row in results:
                    writer.writerow(
                        [
                            row["tooth_class"],
                            row["tooth_head"],
                            row["lm_index"],
                            row["name"],
                            row["x_mm"],
                            row["y_mm"],
                            row["z_mm"],
                            row["confidence"],
                            row["margin"],
                            row["roi_size"],
                        ]
                    )

        landmarks_vtp = None
        if results:
            pts = np.array([[r["x_mm"], r["y_mm"], r["z_mm"]] for r in results], dtype=np.float32)
            cloud = pv.PolyData(pts)
            cloud.point_data["tooth_class"] = np.array([r["tooth_class"] for r in results], np.int32)
            cloud.point_data["lm_index"] = np.array([r["lm_index"] for r in results], np.int32)
            cloud.point_data["confidence"] = np.array([r["confidence"] for r in results], np.float32)
            landmarks_vtp = output_dir / f"{case_id}_landmarks.vtp"
            cloud.save(landmarks_vtp, binary=True)

        return PointNetOutputs(
            payload_json=json_path,
            csv_path=csv_path,
            landmarks_vtp=landmarks_vtp,
        )
