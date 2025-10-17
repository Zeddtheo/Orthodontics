from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg

_POINTNETREG_ROOT = Path("outputs/pointnetreg")
DEFAULT_CKPT_ROOT = _POINTNETREG_ROOT / "checkpoints"
DEFAULT_INFER_OUT = _POINTNETREG_ROOT / "infer"

DEFAULT_TOOTH_IDS = [
    "t11","t12","t13","t14","t15","t16","t17",
    "t21","t22","t23","t24","t25","t26","t27",
    "t31","t32","t33","t34","t35","t36","t37",
    "t41","t42","t43","t44","t45","t46","t47",
]


def _load_landmark_names(def_path: str | None, tooth_id: str, L: int) -> list[str]:
    default = [f"lm_{i:02d}" for i in range(L)]
    if not def_path:
        return default
    try:
        with open(def_path, "r", encoding="utf-8") as fh:
            spec = json.load(fh)
    except Exception:
        return default

    per_tooth = spec.get("per_tooth", {})
    templates = spec.get("templates", {})
    entry = None
    for key in (tooth_id, tooth_id.upper(), tooth_id.lower()):
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


def _case_id(meta) -> str:
    if isinstance(meta, dict):
        for key in ("case_id", "case", "id", "name", "basename"):
            if key in meta:
                return str(meta[key])
        if "path" in meta:
            return Path(meta["path"]).stem
    if isinstance(meta, str):
        return Path(meta).stem
    return "unknown_case"


def _offset(meta) -> np.ndarray | None:
    if not isinstance(meta, dict):
        return None
    for key in ("origin_mm", "center_mm", "offset_mm", "origin", "center", "offset", "shift_mm", "shift"):
        if key in meta:
            arr = np.asarray(meta[key], dtype=np.float32).reshape(-1)
            if arr.size == 3:
                return arr
    return None


def infer_one_tooth(
    root,
    ckpt_root,
    tooth_id,
    features="pn",
    batch_size=8,
    workers=2,
    out_dir=DEFAULT_INFER_OUT,
    landmark_json=None,
    use_tnet=False,
    export_roi_ply=False,
    cases: list[str] | None = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DatasetConfig(
        root=root,
        file_patterns=(f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"),
        features=features,
        select_landmarks="all",
        ensure_constant_L=False,
        tooth_id=tooth_id,
        health_check=False,
    )
    dataset = P0PointNetRegDataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_p0)
    sample = None
    ckpt_root = Path(ckpt_root)
    candidate_names = ["best.pt", "best_mae.pt", "best_mse.pt", "last.pt"]
    search_paths = [(ckpt_root / name) for name in candidate_names]
    tooth_variants = {tooth_id, tooth_id.lower(), tooth_id.upper()}
    for variant in tooth_variants:
        tooth_dir = ckpt_root / variant
        search_paths.extend(tooth_dir / name for name in candidate_names)
    ckpt_path = next((path for path in search_paths if path.exists()), None)
    if ckpt_path is None:
        print(f"[{tooth_id}] missing checkpoint under {ckpt_root}")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    payload = state if isinstance(state, dict) else {"model": state}

    heads_config = payload.get("heads_config")
    in_channels = payload.get("in_channels")
    num_landmarks = payload.get("num_landmarks")

    if isinstance(heads_config, dict):
        resolved_heads: dict[str, int] = {}
        for key, value in heads_config.items():
            try:
                norm_key = PointNetReg._normalize_head_key(key)
            except Exception:
                norm_key = str(key).strip().lower()
            resolved_heads[norm_key] = int(value)
        try:
            tooth_key = PointNetReg._normalize_head_key(tooth_id)
        except Exception:
            tooth_key = str(tooth_id).strip().lower()
        num_landmarks = resolved_heads.get(tooth_key, num_landmarks)

    if in_channels is None or num_landmarks is None:
        sample = dataset[0]
        if in_channels is None:
            in_channels = int(sample["x"].shape[0])
        if num_landmarks is None:
            num_landmarks = int(sample["y"].shape[0])

    if num_landmarks is None:
        raise RuntimeError(f"Unable to resolve landmark count for tooth '{tooth_id}'.")

    in_channels = int(in_channels)
    num_landmarks = int(num_landmarks)
    names = _load_landmark_names(landmark_json, tooth_id, num_landmarks)

    ckpt_use_tnet = payload.get("use_tnet", use_tnet)
    enable_presence_head = bool(payload.get("enable_presence_head", False))
    presence_hidden = int(payload.get("presence_hidden", 128))

    model = PointNetReg(
        in_channels=in_channels,
        num_landmarks=num_landmarks,
        heads_config=heads_config,
        use_tnet=ckpt_use_tnet,
        return_logits=True,
        enable_presence_head=enable_presence_head,
        presence_hidden=presence_hidden,
    ).to(device)
    model.load_state_dict(payload["model"])
    model.eval()

    allowed_cases: set[str] | None = None
    if cases:
        allowed_cases = {str(c).strip() for c in cases if str(c).strip()}
        normalised = set()
        for cid in allowed_cases:
            if cid.isdigit():
                normalised.add(f"{int(cid):03d}")
            else:
                normalised.add(cid)
        allowed_cases = normalised if normalised else None

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    ply_root = None
    if export_roi_ply:
        ply_root = out_root / "roi_ply" / tooth_id
        ply_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            x = batch["x"].to(device, non_blocking=True)
            model_out = model(
                x,
                tooth_id=tooth_id if model.multi_head else None,
                return_presence=enable_presence_head,
            )
            if isinstance(model_out, tuple):
                logits, presence_logits = model_out
            else:
                logits = model_out
                presence_logits = None
            probs = torch.sigmoid(logits)
            topk_logits, topk_idx = torch.topk(logits, k=2, dim=-1)
            top1_vals = torch.sigmoid(topk_logits[..., 0])
            top2_vals_sig = torch.sigmoid(topk_logits[..., 1])
            top1_idx = topk_idx[..., 0]
            idx = top1_idx.cpu().numpy()
            scores = top1_vals.cpu().numpy()
            seconds = top2_vals_sig.cpu().numpy()
            if presence_logits is not None:
                presence_logits_np = presence_logits.detach().cpu().numpy()
                presence_probs_np = torch.sigmoid(presence_logits).detach().cpu().numpy()
            else:
                presence_logits_np = None
                presence_probs_np = None
            mask_batch = batch.get("mask")
            if mask_batch is not None:
                mask_np = (mask_batch.cpu().numpy() > 0.5)
            else:
                mask_np = None
            for b, meta in enumerate(batch["meta"]):
                case_id = _case_id(meta)
                if allowed_cases and case_id not in allowed_cases:
                    continue
                npz_path = meta.get("path") if isinstance(meta, dict) else None
                pos = None
                if npz_path:
                    with np.load(npz_path, allow_pickle=True) as data:
                        if "pos" in data:
                            pos = data["pos"].astype(np.float32)
                            if pos.shape[0] < pos.shape[1]:
                                pos = pos.T
                        else:
                            x_raw = data["x"]
                            if x_raw.shape[0] == logits.shape[-1]:
                                x_raw = x_raw.T
                            pos = x_raw[:, :3].astype(np.float32)
                if pos is None:
                    pos = batch["x"][b, :3].permute(1, 0).cpu().numpy()

                lm_local = pos[idx[b]]
                offset = _offset(meta)
                lm_global = lm_local + offset if offset is not None else lm_local.copy()
                active_mask = None
                if mask_np is not None:
                    candidate = mask_np[b]
                    if candidate.dtype != np.bool_:
                        candidate = candidate > 0.5
                    if np.any(candidate):
                        active_mask = candidate.astype(bool, copy=False)
                        inactive = ~active_mask
                        lm_local = lm_local.copy()
                        lm_local[inactive] = np.nan
                        lm_global = lm_global.copy()
                        lm_global[inactive] = np.nan
                    else:
                        active_mask = None

                out_path = out_root / f"{case_id}.json"
                payload = {}
                if out_path.exists():
                    try:
                        payload = json.load(open(out_path, "r", encoding="utf-8"))
                    except Exception:
                        payload = {}
                payload.setdefault("predictions", {})
                tooth_payload = payload["predictions"].get(tooth_id, {})
                tooth_payload["landmarks_local"] = {names[i]: lm_local[i].tolist() if np.all(np.isfinite(lm_local[i])) else None for i in range(len(names))}
                tooth_payload["landmarks_global"] = {names[i]: lm_global[i].tolist() if np.all(np.isfinite(lm_global[i])) else None for i in range(len(names))}
                tooth_payload["indices"] = {names[i]: (int(idx[b, i]) if active_mask is None or active_mask[i] else -1) for i in range(len(names))}
                # 附带峰值强度、第二峰与置信边际，方便后处理过滤
                margin_logits = torch.sigmoid(
                    (topk_logits[..., 0] - topk_logits[..., 1])
                ).cpu().numpy()
                score_map = {names[i]: (float(scores[b, i]) if active_mask is None or active_mask[i] else 0.0) for i in range(len(names))}
                second_map = {names[i]: (float(seconds[b, i]) if active_mask is None or active_mask[i] else 0.0) for i in range(len(names))}
                margin_map = {names[i]: (float(margin_logits[b, i]) if active_mask is None or active_mask[i] else 0.0) for i in range(len(names))}
                tooth_payload["scores"] = score_map.copy()
                tooth_payload.setdefault("top1", score_map.copy())
                tooth_payload.setdefault("peak_scores", score_map.copy())
                tooth_payload["top2"] = second_map.copy()
                tooth_payload.setdefault("second_scores", second_map.copy())
                tooth_payload["margin"] = margin_map
                if presence_logits_np is not None and presence_probs_np is not None:
                    tooth_payload["presence_logit"] = float(np.asarray(presence_logits_np[b]).reshape(-1)[0])
                    tooth_payload["presence_prob"] = float(np.asarray(presence_probs_np[b]).reshape(-1)[0])
                if active_mask is not None:
                    tooth_payload["active_mask"] = {names[i]: bool(active_mask[i]) for i in range(len(names))}
                tooth_meta_out = {}
                if isinstance(meta, dict):
                    for key in ("case_id", "arch", "fdi", "tooth_id", "sigma_mm", "unit"):
                        if key in meta:
                            tooth_meta_out[key] = meta[key]
                if tooth_meta_out:
                    tooth_payload["meta"] = tooth_meta_out
                payload["predictions"][tooth_id] = tooth_payload
                payload_meta = payload.setdefault("meta", {})
                payload_meta.setdefault("root", str(Path(root).resolve()))
                if isinstance(meta, dict):
                    if "bounds_mm" in meta and "bounds_mm" not in payload_meta:
                        payload_meta["bounds_mm"] = meta["bounds_mm"]
                if offset is not None:
                    tooth_payload.setdefault("meta", {})["offset_mm"] = offset.tolist()
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                print(f"[{tooth_id}] {case_id} -> {out_path}")

                if export_roi_ply and ply_root is not None:
                    ply_path = ply_root / f"{case_id}_sample{batch_idx:04d}.ply"
                    with ply_path.open("w", encoding="utf-8") as fh:
                        fh.write("ply\nformat ascii 1.0\n")
                        fh.write(f"element vertex {lm_local.shape[0]}\n")
                        fh.write("property float x\nproperty float y\nproperty float z\n")
                        fh.write("end_header\n")
                        for point in lm_local:
                            fh.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="datasets/p0_npz")
    parser.add_argument("--ckpt_root", type=str, default=str(DEFAULT_CKPT_ROOT))
    parser.add_argument("--tooth", type=str, default="t31")
    parser.add_argument("--features", type=str, choices=["pn", "xyz"], default="pn")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_INFER_OUT))
    parser.add_argument("--landmark_json", type=str, default="landmark_def.json")
    parser.add_argument(
        "--use_tnet",
        action="store_true",
        help="Enable TNet alignment (default disabled). Note: if the checkpoint stores a use_tnet flag, that setting overrides this switch.",
    )
    parser.add_argument("--cases", type=str, default=None, help="仅导出指定 case（逗号分隔）。默认导出全部。")
    parser.add_argument("--export_roi_ply", action="store_true", help="导出 ROI PLY 点云（默认不导出）。")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tooth.strip().lower() in {"all", "auto"}:
        candidates = sorted(p.name for p in Path(args.ckpt_root).iterdir() if p.is_dir())
        teeth = [t for t in candidates if t in DEFAULT_TOOTH_IDS or t.lower() in DEFAULT_TOOTH_IDS] or DEFAULT_TOOTH_IDS
    else:
        teeth = [t.strip() for t in args.tooth.split(",") if t.strip()]
    for tooth in teeth:
        infer_one_tooth(
            args.root,
            args.ckpt_root,
            tooth,
            args.features,
            args.batch_size,
            args.workers,
            args.out_dir,
            args.landmark_json,
            use_tnet=args.use_tnet,
            export_roi_ply=args.export_roi_ply,
            cases=[c.strip() for c in args.cases.split(",")] if args.cases else None,
        )


if __name__ == "__main__":
    main()
