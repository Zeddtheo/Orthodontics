from pathlib import Path
import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from p0_dataset import DatasetConfig, P0PointNetRegDataset, collate_p0
from pointnetreg import PointNetReg

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


def infer_one_tooth(root, ckpt_root, tooth_id, features="all", batch_size=8, workers=2, out_dir="runs_infer", landmark_json=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = DatasetConfig(
        root=root,
        file_patterns=(f"*_{tooth_id}.npz", f"*_{tooth_id.upper()}.npz"),
        features=features,
        select_landmarks="active",
        ensure_constant_L=False,
    )
    dataset = P0PointNetRegDataset(cfg)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, collate_fn=collate_p0)

    sample = dataset[0]
    names = _load_landmark_names(landmark_json, tooth_id, sample["y"].shape[0])
    model = PointNetReg(
        in_channels=sample["x"].shape[0],
        num_landmarks=sample["y"].shape[0],
        use_tnet=True,
        return_logits=False,
    ).to(device)
    ckpt_path = Path(ckpt_root) / tooth_id / "best.pt"
    if not ckpt_path.exists():
        print(f"[{tooth_id}] missing checkpoint: {ckpt_path}")
        return
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"] if isinstance(state, dict) and "model" in state else state)
    model.eval()

    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device, non_blocking=True)
            pred = model(x)  # (B, L, N) in [0,1]
            topk_vals, topk_idx = torch.topk(pred, k=2, dim=-1)
            top1_vals = topk_vals[..., 0]
            top2_vals = topk_vals[..., 1]
            top1_idx = topk_idx[..., 0]
            idx = top1_idx.cpu().numpy()
            scores = top1_vals.cpu().numpy()
            seconds = top2_vals.cpu().numpy()
            for b, meta in enumerate(batch["meta"]):
                case_id = _case_id(meta)
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
                            if x_raw.shape[0] == pred.shape[-1]:
                                x_raw = x_raw.T
                            pos = x_raw[:, :3].astype(np.float32)
                if pos is None:
                    pos = batch["x"][b, :3].permute(1, 0).cpu().numpy()

                lm_local = pos[idx[b]]
                offset = _offset(meta)
                lm_global = lm_local + offset if offset is not None else lm_local.copy()

                out_path = out_root / f"{case_id}.json"
                payload = {}
                if out_path.exists():
                    try:
                        payload = json.load(open(out_path, "r", encoding="utf-8"))
                    except Exception:
                        payload = {}
                payload.setdefault("predictions", {})
                tooth_payload = payload["predictions"].get(tooth_id, {})
                tooth_payload["landmarks_local"] = {names[i]: lm_local[i].tolist() for i in range(len(names))}
                tooth_payload["landmarks_global"] = {names[i]: lm_global[i].tolist() for i in range(len(names))}
                tooth_payload["indices"] = {names[i]: int(idx[b, i]) for i in range(len(names))}
                # 附带峰值强度、第二峰与置信边际，方便后处理过滤
                score_map = {names[i]: float(scores[b, i]) for i in range(len(names))}
                second_map = {names[i]: float(seconds[b, i]) for i in range(len(names))}
                margin_map = {names[i]: float(scores[b, i] - seconds[b, i]) for i in range(len(names))}
                tooth_payload["scores"] = score_map.copy()
                tooth_payload.setdefault("top1", score_map.copy())
                tooth_payload.setdefault("peak_scores", score_map.copy())
                tooth_payload["top2"] = second_map.copy()
                tooth_payload.setdefault("second_scores", second_map.copy())
                tooth_payload["margin"] = margin_map
                tooth_meta_out = {}
                if isinstance(meta, dict):
                    for key in ("case_id", "arch", "fdi", "tooth_id", "sigma_mm", "unit"):
                        if key in meta:
                            tooth_meta_out[key] = meta[key]
                if tooth_meta_out:
                    tooth_payload["meta"] = tooth_meta_out
                payload["predictions"][tooth_id] = tooth_payload
                payload.setdefault("meta", {}).setdefault("root", str(Path(root).resolve()))
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, ensure_ascii=False, indent=2)
                print(f"[{tooth_id}] {case_id} -> {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="datasets/p0_npz")
    parser.add_argument("--ckpt_root", type=str, default="outputs/landmarks/overfit")
    parser.add_argument("--tooth", type=str, default="t31")
    parser.add_argument("--features", type=str, choices=["all", "xyz"], default="all")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out_dir", type=str, default="runs_infer")
    parser.add_argument("--landmark_json", type=str, default="landmark_def.json")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.tooth.strip().lower() in {"all", "auto"}:
        candidates = sorted(p.name for p in Path(args.ckpt_root).iterdir() if p.is_dir())
        teeth = [t for t in candidates if t in DEFAULT_TOOTH_IDS or t.lower() in DEFAULT_TOOTH_IDS] or DEFAULT_TOOTH_IDS
    else:
        teeth = [t.strip() for t in args.tooth.split(",") if t.strip()]
    for tooth in teeth:
        infer_one_tooth(args.root, args.ckpt_root, tooth, args.features, args.batch_size, args.workers, args.out_dir, args.landmark_json)


if __name__ == "__main__":
    main()
