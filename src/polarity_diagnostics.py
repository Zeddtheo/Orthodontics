#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Polarity diagnostics for mc/dc landmarks.

Implements the two-layer validation requested by the oral scanner team:
Layer A: geometry-only sanity checks (midline polarity, contact pair consistency, tangent polarity).
Layer B: ground-truth assisted swap tests (swap-vs-correct cost, per-point swap preference, vector angle).

The script aggregates results over a batch of cases and emits a JSON/CSV summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from calc_metrics import build_module0, _load_landmarks_json

EPS = 1e-9

# Tooth ordering per quadrant (FDI)
QUADRANTS: List[List[str]] = [
    ['11', '12', '13', '14', '15', '16', '17'],
    ['21', '22', '23', '24', '25', '26', '27'],
    ['31', '32', '33', '34', '35', '36', '37'],
    ['41', '42', '43', '44', '45', '46', '47'],
]

ALL_TEETH: List[str] = [tooth for quad in QUADRANTS for tooth in quad]

MIDLINE_PAIRS = {frozenset({'11', '21'}), frozenset({'31', '41'})}

ADJACENT_PAIRS: List[Tuple[str, str]] = []
DISTAL_MAP: Dict[str, Optional[str]] = {}
for quad in QUADRANTS:
    for idx, tooth in enumerate(quad):
        next_tooth = quad[idx + 1] if idx + 1 < len(quad) else None
        DISTAL_MAP[tooth] = next_tooth
        if next_tooth is not None:
            ADJACENT_PAIRS.append((tooth, next_tooth))
ADJACENT_PAIRS.extend([('11', '21'), ('41', '31')])  # cross-midline contacts


@dataclass
class ToothStats:
    midline_total: int = 0
    midline_violations: int = 0
    midline_violation_sum: float = 0.0

    cpc_total: int = 0
    cpc_violations: int = 0
    cpc_improve_sum: float = 0.0

    tangent_total: int = 0
    tangent_negative: int = 0

    swap_total: int = 0
    swap_better: int = 0
    swap_improve_sum: float = 0.0
    swap_mc_count: int = 0
    swap_dc_count: int = 0
    swap_better_cases: List[str] = field(default_factory=list)

    angle_total: int = 0
    angle_opposite: int = 0


@dataclass
class PairStats:
    total: int = 0
    reverse_better: int = 0
    reverse_gain_sum: float = 0.0
    cpi_sum: float = 0.0


def _safe_point(landmarks: Dict[str, Sequence[float]], label: str) -> Optional[np.ndarray]:
    coords = landmarks.get(label)
    if coords is None:
        return None
    arr = np.asarray(coords, dtype=float)
    if not np.isfinite(arr).all() or arr.shape != (3,):
        return None
    return arr


def _vector_angle_deg(vec_a: np.ndarray, vec_b: np.ndarray) -> Optional[float]:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= EPS or norm_b <= EPS:
        return None
    cos_val = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    cos_val = max(-1.0, min(1.0, cos_val))
    return float(np.degrees(np.arccos(cos_val)))


def _rate(numerator: int, denominator: int) -> Optional[float]:
    if denominator <= 0:
        return None
    return numerator / denominator


def _avg(total: float, count: int) -> Optional[float]:
    if count <= 0:
        return None
    return total / count


def gather_sample_ids(pred_root: str, explicit_ids: Optional[Iterable[str]] = None) -> List[str]:
    if explicit_ids:
        return sorted({str(sid) for sid in explicit_ids}, key=lambda x: (len(x), x))
    entries = []
    if not os.path.isdir(pred_root):
        return entries
    for name in os.listdir(pred_root):
        if name.isdigit():
            entries.append(name)
    return sorted(entries, key=lambda x: (len(x), x))


def load_landmarks(root: str, sample_id: str, arch: str) -> Optional[Dict[str, List[float]]]:
    fname = f"{sample_id}_{arch}.json"
    path = os.path.join(root, sample_id, fname)
    if not os.path.exists(path):
        return None
    return _load_landmarks_json(path)


def run_analysis(
    pred_root: str,
    gt_root: str,
    sample_ids: Iterable[str],
    delta_mm: float,
    midline_threshold_mm: float,
    cpc_gain_threshold_mm: float,
    angle_flip_deg: float,
) -> Dict[str, Any]:
    tooth_stats: Dict[str, ToothStats] = {tooth: ToothStats() for tooth in ALL_TEETH}
    pair_stats: Dict[Tuple[str, str], PairStats] = {tuple(pair): PairStats() for pair in ADJACENT_PAIRS}

    samples_total = 0
    samples_with_frame = 0
    skipped_samples: Dict[str, str] = {}

    for sample_id in sample_ids:
        samples_total += 1
        pred_upper = load_landmarks(pred_root, sample_id, 'U')
        pred_lower = load_landmarks(pred_root, sample_id, 'L')
        if not pred_upper or not pred_lower:
            skipped_samples[sample_id] = 'prediction_json_missing'
            continue
        pred_landmarks: Dict[str, List[float]] = {}
        pred_landmarks.update(pred_upper)
        pred_landmarks.update(pred_lower)

        gt_upper = load_landmarks(gt_root, sample_id, 'U')
        gt_lower = load_landmarks(gt_root, sample_id, 'L')
        gt_landmarks: Dict[str, List[float]] = {}
        if gt_upper:
            gt_landmarks.update(gt_upper)
        if gt_lower:
            gt_landmarks.update(gt_lower)

        frame_info = build_module0(pred_landmarks)
        ops = frame_info.get('ops')
        if not ops:
            skipped_samples[sample_id] = 'frame_failed'
            continue
        samples_with_frame += 1

        projP = ops['projP']
        x_of = ops['x']
        H = ops['H']

        centers: Dict[str, np.ndarray] = {}
        pred_vectors: Dict[str, np.ndarray] = {}

        # --- Layer A.1 Midline ---
        for tooth in ALL_TEETH:
            mc = _safe_point(pred_landmarks, f"{tooth}mc")
            dc = _safe_point(pred_landmarks, f"{tooth}dc")
            if mc is None or dc is None:
                continue
            try:
                x_mc = abs(x_of(mc))
                x_dc = abs(x_of(dc))
                delta = x_mc - x_dc
            except Exception:
                continue
            stats = tooth_stats[tooth]
            stats.midline_total += 1
            if delta > midline_threshold_mm:
                stats.midline_violations += 1
                stats.midline_violation_sum += delta

            mc_proj = projP(mc)
            dc_proj = projP(dc)
            centers[tooth] = 0.5 * (mc_proj + dc_proj)
            pred_vec = dc_proj - mc_proj
            pred_vectors[tooth] = pred_vec

        # --- Layer A.2 Contact Pair Consistency ---
        for pair in ADJACENT_PAIRS:
            a, b = pair
            pair_stat = pair_stats[pair]

            if frozenset(pair) in MIDLINE_PAIRS:
                labels_correct = (f"{a}mc", f"{b}mc")
                labels_reverse = (f"{a}dc", f"{b}dc")
            else:
                labels_correct = (f"{a}dc", f"{b}mc")
                labels_reverse = (f"{a}mc", f"{b}dc")

            pts = [(_safe_point(pred_landmarks, lbl_correct), _safe_point(pred_landmarks, lbl_reverse))
                   for lbl_correct, lbl_reverse in zip(labels_correct, labels_reverse)]
            if any(pt[0] is None or pt[1] is None for pt in pts):
                continue
            p1_correct, p1_reverse = pts[0]
            p2_correct, p2_reverse = pts[1]
            try:
                correct_dist = H(p1_correct, p2_correct)
                reverse_dist = H(p1_reverse, p2_reverse)
            except Exception:
                continue
            gain = correct_dist - reverse_dist

            pair_stat.total += 1
            pair_stat.cpi_sum += gain

            for tooth in (a, b):
                stats = tooth_stats[tooth]
                stats.cpc_total += 1
                if gain > 0:
                    stats.cpc_violations += 1
                    stats.cpc_improve_sum += gain

            if gain > 0:
                pair_stat.reverse_better += 1
                pair_stat.reverse_gain_sum += gain

        # --- Layer A.3 Tangent polarity ---
        for tooth, next_tooth in DISTAL_MAP.items():
            if next_tooth is None:
                continue
            pred_vec = pred_vectors.get(tooth)
            center = centers.get(tooth)
            next_center = centers.get(next_tooth)
            if pred_vec is None or center is None or next_center is None:
                continue
            tangent_vec = next_center - center
            norm_tangent = float(np.linalg.norm(tangent_vec))
            norm_pred = float(np.linalg.norm(pred_vec))
            if norm_tangent <= EPS or norm_pred <= EPS:
                continue
            dot_val = float(np.dot(pred_vec, tangent_vec)) / (norm_pred * norm_tangent)
            stats = tooth_stats[tooth]
            stats.tangent_total += 1
            if dot_val < 0:
                stats.tangent_negative += 1

        # --- Layer B Swap diagnostics ---
        for tooth in ALL_TEETH:
            stats = tooth_stats[tooth]

            mc_pred = _safe_point(pred_landmarks, f"{tooth}mc")
            dc_pred = _safe_point(pred_landmarks, f"{tooth}dc")
            mc_gt = _safe_point(gt_landmarks, f"{tooth}mc")
            dc_gt = _safe_point(gt_landmarks, f"{tooth}dc")

            if mc_pred is None or dc_pred is None or mc_gt is None or dc_gt is None:
                continue

            stats.swap_total += 1
            dist_mc_correct = float(np.linalg.norm(mc_pred - mc_gt))
            dist_dc_correct = float(np.linalg.norm(dc_pred - dc_gt))
            dist_mc_swap = float(np.linalg.norm(mc_pred - dc_gt))
            dist_dc_swap = float(np.linalg.norm(dc_pred - mc_gt))

            e_correct = dist_mc_correct + dist_dc_correct
            e_swap = dist_mc_swap + dist_dc_swap

            if e_swap + delta_mm < e_correct:
                stats.swap_better += 1
                stats.swap_improve_sum += (e_correct - e_swap)
                stats.swap_better_cases.append(sample_id)
                if dist_mc_swap < dist_mc_correct:
                    stats.swap_mc_count += 1
                if dist_dc_swap < dist_dc_correct:
                    stats.swap_dc_count += 1

            pred_vec = dc_pred - mc_pred
            gt_vec = dc_gt - mc_gt
            angle = _vector_angle_deg(pred_vec, gt_vec)
            if angle is not None:
                stats.angle_total += 1
                if angle >= angle_flip_deg:
                    stats.angle_opposite += 1

    summary_per_tooth: Dict[str, Any] = {}
    for tooth, stats in tooth_stats.items():
        midline_rate = _rate(stats.midline_violations, stats.midline_total)
        cpc_rate = _rate(stats.cpc_violations, stats.cpc_total)
        tangent_rate = _rate(stats.tangent_negative, stats.tangent_total)
        swap_rate = _rate(stats.swap_better, stats.swap_total)
        swap_avg_improve = _avg(stats.swap_improve_sum, stats.swap_better)
        swap_rate_mc = _rate(stats.swap_mc_count, stats.swap_total)
        swap_rate_dc = _rate(stats.swap_dc_count, stats.swap_total)
        angle_rate = _rate(stats.angle_opposite, stats.angle_total)
        midline_avg_delta = _avg(stats.midline_violation_sum, stats.midline_violations)
        cpc_avg_improve = _avg(stats.cpc_improve_sum, stats.cpc_violations)

        summary_per_tooth[tooth] = {
            'midline_total': stats.midline_total,
            'midline_violation_count': stats.midline_violations,
            'midline_violation_rate': midline_rate,
            'midline_violation_avg_mm': midline_avg_delta,
            'cpc_total': stats.cpc_total,
            'cpc_violation_count': stats.cpc_violations,
            'cpc_violation_rate': cpc_rate,
            'cpc_reverse_gain_avg_mm': cpc_avg_improve,
            'tangent_total': stats.tangent_total,
            'tangent_negative_rate': tangent_rate,
            'swap_total': stats.swap_total,
            'swap_better_count': stats.swap_better,
            'swap_rate': swap_rate,
            'swap_avg_improve_mm': swap_avg_improve,
            'swap_rate_mc': swap_rate_mc,
            'swap_rate_dc': swap_rate_dc,
            'swap_better_cases': stats.swap_better_cases,
            'angle_total': stats.angle_total,
            'angle_opposite_rate': angle_rate,
        }

    summary_per_pair: Dict[str, Any] = {}
    for pair, stats in pair_stats.items():
        key = f"{pair[0]}-{pair[1]}"
        reverse_rate = _rate(stats.reverse_better, stats.total)
        reverse_gain_avg = _avg(stats.reverse_gain_sum, stats.reverse_better)
        summary_per_pair[key] = {
            'pair': list(pair),
            'type': 'midline' if frozenset(pair) in MIDLINE_PAIRS else 'adjacent',
            'total': stats.total,
            'reverse_better_count': stats.reverse_better,
            'reverse_better_rate': reverse_rate,
            'reverse_gain_avg_mm': reverse_gain_avg,
            'cpi_avg_mm': _avg(stats.cpi_sum, stats.total),
        }

    return {
        'config': {
            'pred_root': pred_root,
            'gt_root': gt_root,
            'delta_mm': delta_mm,
            'midline_threshold_mm': midline_threshold_mm,
            'cpc_gain_threshold_mm': cpc_gain_threshold_mm,
            'angle_flip_deg': angle_flip_deg,
        },
        'samples_total': samples_total,
        'samples_with_frame': samples_with_frame,
        'skipped_samples': skipped_samples,
        'per_tooth': summary_per_tooth,
        'per_pair': summary_per_pair,
    }


def classify_tooth(
    metrics: Dict[str, Any],
    thresholds: Dict[str, float],
) -> Tuple[str, str]:
    """Return classification label and short note."""
    def _val(name: str) -> Optional[float]:
        val = metrics.get(name)
        return None if val is None else float(val)

    midline_rate = _val('midline_violation_rate')
    cpc_rate = _val('cpc_violation_rate')
    tangent_rate = _val('tangent_negative_rate')
    cpc_gain = _val('cpc_reverse_gain_avg_mm')
    swap_rate = _val('swap_rate')
    swap_gain = _val('swap_avg_improve_mm')
    swap_rate_mc = _val('swap_rate_mc')
    swap_rate_dc = _val('swap_rate_dc')
    angle_rate = _val('angle_opposite_rate')

    strong_a = (
        cpc_gain is not None and cpc_gain >= thresholds['cpc_gain_strong'] and
        max(
            val for val in (midline_rate, cpc_rate, tangent_rate)
            if val is not None
        ) >= thresholds['A_strong_rate']
    ) if any(val is not None for val in (midline_rate, cpc_rate, tangent_rate)) else False

    suspect_a = any(
        val is not None and val >= thresholds['A_suspect_rate']
        for val in (midline_rate, cpc_rate, tangent_rate)
    )

    strong_b = (
        swap_rate is not None and swap_rate >= thresholds['B_swap_rate_strong'] and
        swap_gain is not None and swap_gain >= thresholds['B_swap_improve_strong']
    )

    suspect_b = (
        swap_rate is not None and swap_rate >= thresholds['B_swap_rate_suspect'] and
        swap_gain is not None and swap_gain >= thresholds['B_swap_improve_suspect']
    )

    if strong_a and strong_b:
        label = 'systemic_flip'
    elif strong_b and suspect_a:
        label = 'swap_supports_flip'
    elif strong_a and not strong_b:
        label = 'geometry_only_flag'
    elif suspect_b:
        label = 'swap_only_flag'
    elif suspect_a:
        label = 'geometry_suspect'
    else:
        label = 'no_flag'

    notes: List[str] = []
    if midline_rate is not None:
        notes.append(f"Midline {midline_rate:.0%}/{metrics['midline_total']} samples")
    if cpc_rate is not None:
        notes.append(f"CPC {cpc_rate:.0%}")
    if cpc_gain is not None and cpc_gain > 0:
        notes.append(f"CPC gain {cpc_gain:.2f}mm")
    if tangent_rate is not None and metrics['tangent_total'] > 0:
        notes.append(f"Tangent {tangent_rate:.0%}")
    if swap_rate is not None:
        notes.append(f"Swap {swap_rate:.0%}")
    if swap_gain is not None and swap_gain > 0:
        notes.append(f"Δ {swap_gain:.2f}mm")
    if swap_rate_mc is not None:
        notes.append(f"mc→dc {swap_rate_mc:.0%}")
    if swap_rate_dc is not None:
        notes.append(f"dc→mc {swap_rate_dc:.0%}")
    if angle_rate is not None:
        notes.append(f"θ≈180° {angle_rate:.0%}")
    note_str = '; '.join(notes)
    return label, note_str


def attach_classification(report: Dict[str, Any]) -> None:
    thresholds = {
        'A_suspect_rate': 0.30,
        'A_strong_rate': 0.60,
        'cpc_gain_strong': 0.30,
        'B_swap_rate_strong': 0.60,
        'B_swap_improve_strong': 0.40,
        'B_swap_rate_suspect': 0.45,
        'B_swap_improve_suspect': 0.25,
    }

    for tooth, metrics in report['per_tooth'].items():
        label, note = classify_tooth(metrics, thresholds)
        metrics['classification'] = label
        metrics['classification_note'] = note


def write_csv(report: Dict[str, Any], path: str) -> None:
    fieldnames = [
        'tooth',
        'classification',
        'midline_violation_rate',
        'midline_violation_avg_mm',
        'cpc_violation_rate',
        'cpc_reverse_gain_avg_mm',
        'tangent_negative_rate',
        'swap_rate',
        'swap_avg_improve_mm',
        'swap_rate_mc',
        'swap_rate_dc',
        'angle_opposite_rate',
        'midline_total',
        'cpc_total',
        'tangent_total',
        'swap_total',
        'angle_total',
    ]
    rows = []
    for tooth in sorted(report['per_tooth'].keys()):
        metrics = report['per_tooth'][tooth]
        row = {'tooth': tooth, 'classification': metrics.get('classification')}
        for key in fieldnames[2:]:
            row[key] = metrics.get(key)
        rows.append(row)

    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose mc/dc polarity issues over a batch of cases.")
    parser.add_argument('--pred-root', default=os.path.join('outputs', 'raw_output'), help='Directory containing predicted landmarks per case.')
    parser.add_argument('--gt-root', default=os.path.join('datasets', 'landmarks_dataset', 'raw'), help='Directory containing ground-truth landmarks per case.')
    parser.add_argument('--samples', nargs='*', help='Optional explicit sample IDs to process.')
    parser.add_argument('--delta-mm', type=float, default=0.30, help='Swap penalty delta in mm.')
    parser.add_argument('--midline-threshold-mm', type=float, default=0.20, help='Threshold for midline violation (Δx).')
    parser.add_argument('--cpc-gain-threshold-mm', type=float, default=0.30, help='Minimum average gain to treat CPC reversal as strong.')
    parser.add_argument('--angle-threshold-deg', type=float, default=135.0, help='Angle threshold to treat vectors as opposite.')
    parser.add_argument('--out-json', default=os.path.join('outputs', 'test1028', 'polarity_report.json'), help='Path to write JSON summary.')
    parser.add_argument('--out-csv', default=os.path.join('outputs', 'test1028', 'polarity_report.csv'), help='Path to write CSV summary.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_ids = gather_sample_ids(args.pred_root, args.samples)
    report = run_analysis(
        pred_root=args.pred_root,
        gt_root=args.gt_root,
        sample_ids=sample_ids,
        delta_mm=args.delta_mm,
        midline_threshold_mm=args.midline_threshold_mm,
        cpc_gain_threshold_mm=args.cpc_gain_threshold_mm,
        angle_flip_deg=args.angle_threshold_deg,
    )
    attach_classification(report)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    write_csv(report, args.out_csv)

    print(f"Processed {report['samples_with_frame']}/{report['samples_total']} samples.")
    print(f"JSON report: {args.out_json}")
    print(f"CSV report:  {args.out_csv}")


if __name__ == '__main__':
    main()
