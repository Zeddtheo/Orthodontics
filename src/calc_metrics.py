# -*- coding: utf-8 -*-
"""
calc_metrics.py — 北大口腔·口扫模型 11 指标一体化实现（统一口径版）
修正：Crossbite·第一磨牙代表点采用“四点均值法”（mb/db 与 ml/dl 取均值，缺失时回退单点）。
接口：
    generate_metrics(landmarks_payload, out_path="", cfg=None) -> Dict

说明：
- 所有指标均在统一咬合平面 P 内计算（右+、上+、前+）。
- 依赖：仅标准库 + numpy。输入为 {label: [x,y,z]} 或 Slicer Markups 格式的字典；可传入单个或列表。
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Set, Iterable
import numpy as np
import json, os, re

# =========================
# 工具 & 常量
# =========================
EPS = 1e-9

def _is_xyz(p: Any) -> bool:
    return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()

def _v(a) -> Optional[np.ndarray]:
    return np.asarray(a, float) if _is_xyz(a) else None

def _len(v: np.ndarray) -> float:
    return float(np.linalg.norm(v))

def _nrm(v: np.ndarray) -> Optional[np.ndarray]:
    n = _len(v);  return (v / n) if n > EPS else None

def _dot(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)

def _pick(landmarks: Dict[str, List[float]], names: List[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
    for nm in names:
        p = _v(landmarks.get(nm))
        if p is not None:
            return nm, p
    return None, None

def _star_mid(landmarks: Dict[str, List[float]], tooth: str) -> Tuple[List[str], Optional[np.ndarray]]:
    """
    切缘“*”中点：(ma+da)/2；若缺则回退 m；都缺则 ([], None)。
    返回 (使用到的标签列表, 坐标)
    """
    ma = _v(landmarks.get(f"{tooth}ma"))
    da = _v(landmarks.get(f"{tooth}da"))
    if ma is not None and da is not None:
        return [f"{tooth}ma", f"{tooth}da"], 0.5 * (ma + da)
    m = _v(landmarks.get(f"{tooth}m"))
    if m is not None:
        return [f"{tooth}m"], m
    return [], None

_JAW_PREFIX_RE = re.compile(r"^(upper|lower)[_\-]?(\d.*)$", flags=re.IGNORECASE)

def _robust_ap_offset(landmarks, ops, upper_labels, lower_labels, hard_clip_mm=12.0):
    """
    计算“上参考点 vs 下参考点”的前后(Ŷ)位移，使用多个候选点名的组合，
    - 过滤缺失
    - 去除绝对异常值（>|hard_clip_mm|）
    - 以中位数作为偏移量
    返回 (dy_mm, debug_info)
    """
    cand_pairs = []
    for u_nm in upper_labels:
        u = _v(landmarks.get(u_nm))
        if u is None: continue
        for l_nm in lower_labels:
            l = _v(landmarks.get(l_nm))
            if l is None: continue
            dy = ops['y'](u) - ops['y'](l)
            cand_pairs.append((u_nm, l_nm, float(dy)))
    debug = {'pairs': cand_pairs.copy(), 'used_pairs': [], 'IQR_mm': None, 'n': 0}
    if not cand_pairs:
        return None, debug
    # 硬截断
    cand_pairs = [p for p in cand_pairs if abs(p[2]) <= hard_clip_mm]
    if not cand_pairs:
        return None, debug
    vals = np.array([p[2] for p in cand_pairs], dtype=float)
    # IQR 去异常
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1)
    lo, hi = q1 - 1.5*iqr, q3 + 1.5*iqr
    keep = [(u,l,v) for (u,l,v) in cand_pairs if lo <= v <= hi]
    if keep:
        vals2 = np.array([p[2] for p in keep], dtype=float)
        med = float(np.median(vals2))
        debug.update({'used_pairs': keep, 'IQR_mm': float(iqr), 'n': len(keep)})
        return med, debug
    # IQR全剔则回退：均值±2.5mm
    mu = float(np.mean(vals))
    keep = [(u,l,v) for (u,l,v) in cand_pairs if abs(v - mu) <= 2.5]
    if keep:
        vals2 = np.array([p[2] for p in keep], dtype=float)
        med = float(np.median(vals2))
        debug.update({'used_pairs': keep, 'IQR_mm': None, 'n': len(keep)})
        return med, debug
    return float(np.median(vals)), debug

def _normalise_label(label: Any) -> Optional[str]:
    if not isinstance(label, str):
        return str(label) if label is not None else None
    cleaned = label.strip()
    if not cleaned:
        return None
    match = _JAW_PREFIX_RE.match(cleaned)
    if match:
        candidate = match.group(2)
        if candidate:
            return candidate
    return cleaned


def _coerce_xyz(value: Any) -> Optional[List[float]]:
    if isinstance(value, (list, tuple)) and len(value) == 3:
        try:
            return [float(value[0]), float(value[1]), float(value[2])]
        except (TypeError, ValueError):
            return None
    return None

_TOOTH_TYPE_MAP = {
    '1': 'incisor',
    '2': 'incisor',
    '3': 'canine',
    '4': 'premolar',
    '5': 'premolar',
    '6': 'molar',
    '7': 'molar',
    '8': 'molar'
}

_MIN_TOOTH_WIDTH_MM = {
    'incisor': 4.5,
    'canine': 5.0,
    'premolar': 3.5,
    'molar': 7.0
}

def _tooth_category(tooth: str) -> Optional[str]:
    if not isinstance(tooth, str):
        return None
    digits = ''.join(ch for ch in tooth if ch.isdigit())
    if len(digits) < 2:
        return None
    return _TOOTH_TYPE_MAP.get(digits[-1])

def _min_width_for_tooth(tooth: str) -> Optional[float]:
    cat = _tooth_category(tooth)
    if cat is None:
        return None
    return _MIN_TOOTH_WIDTH_MM.get(cat)

# =========================
# Module 0 — 前置：咬合平面与坐标系
# =========================
def build_module0(landmarks: Dict[str, List[float]]) -> Dict[str, Any]:
    """
    按 I、L、R 三点定义咬合平面 P；坐标轴：X̂=左右(右+), Ŷ=前后(前+), Ẑ=上下(上+)。
    并提供常用算子：projP/x/y/z/H 以及单牙宽度 wt（mc↔dc 的 P 内距离）。
    """
    warnings: List[str] = []
    used: Dict[str, Any] = {}

    # --- I（上下中切“*”的中点） ---
    u_used, U_CI = _star_mid(landmarks, '11')
    u2, p = _star_mid(landmarks, '21'); u_used += u2
    U_CI = 0.5*(U_CI + p) if (U_CI is not None and p is not None) else None

    l_used, L_CI = _star_mid(landmarks, '31')
    l2, p = _star_mid(landmarks, '41'); l_used += l2
    L_CI = 0.5*(L_CI + p) if (L_CI is not None and p is not None) else None

    if U_CI is None or L_CI is None:
        warnings.append("incisal star midpoints missing; fallback origin")
    I = 0.5*(U_CI + L_CI) if (U_CI is not None and L_CI is not None) else None
    used['U_CI_sources'] = u_used; used['L_CI_sources'] = l_used

    # --- L、R（左右咬合中点）---
    _, R1 = _pick(landmarks, ['16mb'])
    _, R2 = _pick(landmarks, ['46mb'])
    _, L1 = _pick(landmarks, ['26mb'])
    _, L2 = _pick(landmarks, ['36mb'])
    if any(p is None for p in (R1, R2, L1, L2)):
        return {'frame': None, 'ops': None, 'wt': None, 'quality': 'missing',
                'warnings': warnings + ['insufficient molar landmarks (16mb/46mb/26mb/36mb)'], 'used': used}
    R = 0.5*(R1 + R2); L = 0.5*(L1 + L2); midLR = 0.5*(L + R)

    # --- 坐标轴 ---
    X = _nrm(R - L)  # 右+
    if I is None:
        I = 0.5*(midLR + 0.5*(R+L))
    Z = _nrm(_cross(R - L, I - midLR))   # 竖直
    if X is None or Z is None:
        return {'frame': None, 'ops': None, 'wt': None, 'quality': 'missing',
                'warnings': warnings + ['failed to form basis'], 'used': used}
    Y = _nrm(_cross(Z, X))               # 前+

    # 极性（上+、前+）
    if (U_CI is not None and L_CI is not None) and (_dot(U_CI - L_CI, Z) < 0):
        Z = -Z; Y = _nrm(_cross(Z, X))
    if _dot(I - midLR, Y) < 0:
        Y = -Y

    Pc = (I + L + R) / 3.0
    def projP(p: np.ndarray) -> np.ndarray: return p - Z * _dot(Z, p - Pc)
    def x(p: np.ndarray) -> float: return _dot(X, projP(p) - I)
    def y(p: np.ndarray) -> float: return _dot(Y, projP(p) - I)
    def z(p: np.ndarray) -> float: return _dot(Z, p - I)
    def H(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(projP(b) - projP(a)))

    width_records: Dict[Tuple[str, bool], Dict[str, Any]] = {}

    def wt(tooth: str, allow_fallback: bool = True) -> Optional[float]:
        key = (tooth, bool(allow_fallback))
        cached = width_records.get(key)
        if cached is not None:
            return cached.get('final_width_mm')

        pts = {suffix: _v(landmarks.get(f"{tooth}{suffix}")) for suffix in ('mc', 'mr', 'm', 'dc', 'dr')}
        min_width = _min_width_for_tooth(tooth)
        record: Dict[str, Any] = {
            'tooth': tooth,
            'allow_fallback': bool(allow_fallback),
            'threshold_mm': min_width,
            'status': 'missing',
            'initial_labels': [],
            'initial_width_mm': None,
            'final_labels': [],
            'final_width_mm': None,
            'fallback_stage': None,
            'removed': False,
            'notes': []
        }

        def _pick_first(labels: List[str]) -> Optional[str]:
            for lb in labels:
                if pts.get(lb) is not None:
                    return lb
            return None

        mesial_order = ['mc']
        distal_order = ['dc']
        if allow_fallback:
            mesial_order.extend(['mr', 'm'])
            distal_order.append('dr')

        m_label = _pick_first(mesial_order)
        d_label = _pick_first(distal_order)
        if (m_label is None) or (d_label is None):
            record['notes'].append('missing_landmarks')
            width_records[key] = record
            return None

        m_pt = pts[m_label]
        d_pt = pts[d_label]
        width_val = H(m_pt, d_pt)
        record.update({
            'status': 'ok',
            'initial_labels': [m_label, d_label],
            'initial_width_mm': float(width_val),
            'final_labels': [m_label, d_label],
            'final_width_mm': float(width_val)
        })

        # Robust guard: fallback for implausibly narrow widths
        if (min_width is not None) and (width_val < min_width):
            fallback_success = False
            attempted: List[Tuple[str, Tuple[str, str], float]] = []
            for stage, pair in (('mr_dr', ('mr', 'dr')), ('m_dr', ('m', 'dr'))):
                if list(pair) == record['final_labels']:
                    continue
                pm, pd = pts.get(pair[0]), pts.get(pair[1])
                if pm is None or pd is None:
                    continue
                new_width = H(pm, pd)
                attempted.append((stage, pair, new_width))
                if new_width >= min_width:
                    record['status'] = 'fallback'
                    record['fallback_stage'] = stage
                    record['final_labels'] = [pair[0], pair[1]]
                    record['final_width_mm'] = float(new_width)
                    record['notes'].append(f"fallback:{stage}")
                    fallback_success = True
                    break
            if not fallback_success:
                if attempted:
                    stage, pair, new_width = max(attempted, key=lambda item: item[2])
                    record['final_labels'] = [pair[0], pair[1]]
                    record['final_width_mm'] = float(new_width)
                    record['fallback_stage'] = stage
                record['status'] = 'warning'
                record['threshold_breach'] = True
                record['notes'].append('below_threshold')

        width_records[key] = record
        return record['final_width_mm']

    frame = {'origin': I, 'ex': X, 'ey': Y, 'ez': Z}
    ops = {'projP': projP, 'x': x, 'y': y, 'z': z, 'H': H}
    return {'frame': frame, 'ops': ops, 'wt': wt, 'quality': 'ok', 'warnings': warnings,
            'used': used, 'wt_records': width_records}

# =========================
# 1) Arch Form — 仅犬牙转折角 α_avg
# =========================
def compute_arch_form(landmarks: Dict[str, List[float]], frame: Dict, ops: Dict, dec:int=1) -> Dict:
    def _get(nm: str) -> Optional[np.ndarray]:
        return _v(landmarks.get(nm))

    def _prefer(*names: str) -> Optional[np.ndarray]:
        for nm in names:
            pt = _v(landmarks.get(nm))
            if pt is not None:
                return pt
        return None

    # 侧切缘中点
    _, U12 = _star_mid(landmarks, '12')
    _, U22 = _star_mid(landmarks, '22')
    A_R, A_L = U12, U22
    # 尖牙尖点
    B_R = _get('13m')
    B_L = _get('23m')
    # C 点：优先 mr → mc → b
    C_R = _prefer('14mr', '14mc', '14b')
    C_L = _prefer('24mr', '24mc', '24b')

    if any(pt is None for pt in (A_R, A_L, B_R, B_L, C_R, C_L)):
        return {
            'form': '缺失',
            'indices': {},
            'summary_text': 'Arch_Form_牙弓形态*: 缺失',
            'quality': 'missing',
        }

    def _angle(A, B, C):
        # 在 P 面计算 ∠ABC
        Ap = ops['projP'](A); Bp = ops['projP'](B); Cp = ops['projP'](C)
        u = Ap - Bp; v = Cp - Bp
        nu, nv = _len(u), _len(v)
        if nu<EPS or nv<EPS: return None
        c = np.clip(_dot(u, v)/(nu*nv), -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    def _range_distance(val: float) -> float:
        if val is None:
            return float('inf')
        if 140.0 <= val <= 175.0:
            return 0.0
        if val < 140.0:
            return 140.0 - val
        return val - 175.0

    def _robust_average(angle_map: Dict[str, Optional[float]]) -> Tuple[Optional[float], Dict[str, float]]:
        valid = {side: ang for side, ang in angle_map.items() if isinstance(ang, (int, float))}
        if not valid:
            return None, {}
        if len(valid) == 1:
            side, val = next(iter(valid.items()))
            return float(val), {side: float(val)}
        vals = list(valid.values())
        median = float(np.median(vals))
        filtered = {side: ang for side, ang in valid.items() if abs(ang - median) <= 3.0}
        if not filtered:
            filtered = valid
        avg = float(np.mean(list(filtered.values())))
        return avg, filtered

    angles_primary = {
        'right': _angle(A_R, B_R, C_R),
        'left': _angle(A_L, B_L, C_L),
    }
    if not any(isinstance(v, (int, float)) for v in angles_primary.values()):
        return {
            'form': '缺失',
            'indices': {},
            'summary_text': 'Arch_Form_牙弓形态*: 缺失',
            'quality': 'missing',
        }

    valid_primary_vals = [v for v in angles_primary.values() if isinstance(v, (int, float))]
    raw_avg_primary = float(np.mean(valid_primary_vals))

    if valid_primary_vals and (raw_avg_primary >= 175.0 or raw_avg_primary <= 140.0):
        # 尝试非常规回退（剔除 mr，改用 mc/b）
        C_R_alt = _prefer('14mc', '14b')
        C_L_alt = _prefer('24mc', '24b')
        alt_angles = {
            'right': _angle(A_R, B_R, C_R_alt) if C_R_alt is not None else None,
            'left': _angle(A_L, B_L, C_L_alt) if C_L_alt is not None else None,
        }
        alt_valid = [v for v in alt_angles.values() if isinstance(v, (int, float))]
        if alt_valid:
            alt_avg = float(np.mean(alt_valid))
            if _range_distance(alt_avg) + 1e-6 < _range_distance(raw_avg_primary):
                angles_primary = alt_angles
                raw_avg_primary = alt_avg
            elif abs(_range_distance(alt_avg) - _range_distance(raw_avg_primary)) <= 1e-6:
                # tie-break：选择更接近中间值（约156.5°）
                if abs(alt_avg - 156.5) < abs(raw_avg_primary - 156.5):
                    angles_primary = alt_angles
                    raw_avg_primary = alt_avg

    alpha_avg, used_angles = _robust_average(angles_primary)
    if alpha_avg is None:
        return {
            'form': '缺失',
            'indices': {},
            'summary_text': 'Arch_Form_牙弓形态*: 缺失',
            'quality': 'missing',
        }

    quality = 'ok'
    valid_count = len([v for v in angles_primary.values() if isinstance(v, (int, float))])
    if len(used_angles) == 0:
        quality = 'missing'
    elif len(used_angles) < valid_count:
        quality = 'fallback'

    # 分型阈值仍沿用 160/168° 经验口径
    if alpha_avg <= 160.0:
        form = '尖圆形'
    elif alpha_avg >= 168.0:
        form = '方圆形'
    else:
        form = '卵圆形'

    indices = {'alpha_deg': round(alpha_avg, dec)}
    if 'right' in used_angles:
        indices['alpha_right_deg'] = round(used_angles['right'], dec)
    if 'left' in used_angles:
        indices['alpha_left_deg'] = round(used_angles['left'], dec)

    return {
        'form': form,
        'indices': indices,
        'summary_text': f"Arch_Form_牙弓形态*: {form}",
        'quality': quality
    }

# =========================
# 2) Arch Width — 前/中/后三段
# =========================
def compute_arch_width(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    dec: int = 1,
    cfg: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    Arch_Width_牙弓宽度（性别分层，前/中/后分段，按均值±k·SD 判定）
    - 取点：
      上颌：前 13m–23m，中 14b–24b，后 16mb–26mb
      下颌：前 33m–43m，中 34b–44b，后 36mb–46mb
    - 度量：统一咬合平面 P 内 H 距离
    - 阈值：男女分层的均值±k·SD，默认 k=1；可用 cfg['arch_width_sigma'] 调整
      性别来源：cfg['sex'] / cfg['gender'] / cfg['patient_sex']，可接受 'male'/'female'/'男'/'女'/'M'/'F'
      未提供性别时采用“男女合并区间”（两性的区间并集），避免误判
    返回：
      - upper/lower 三段实测（mm）
      - judge: 上/下颌 前/中/后 → '偏窄'/'正常'/'偏宽'
      - ranges_mm: 实际使用的阈值上下界
      - 兼容字段：diff_UL_mm、upper_is_narrow、summary_text、quality
    """
    def _vget(nm: str) -> Optional[np.ndarray]:
        return _v(landmarks.get(nm))

    need = ['13m','23m','14b','24b','16mb','26mb',
            '33m','43m','34b','44b','36mb','46mb']
    if any(_vget(n) is None for n in need):
        return {
            'upper': None, 'lower': None, 'diff_UL_mm': None, 'upper_is_narrow': None,
            'judge': None, 'ranges_mm': None, 'sex_used': None,
            'summary_text': 'Arch_Width_牙弓宽度*: 缺失', 'quality': 'missing'
        }

    H = ops['H']
    # --- 实测 ---
    u_raw = {
        'anterior': H(_vget('13m'),  _vget('23m')),
        'middle'  : H(_vget('14b'),  _vget('24b')),
        'posterior': H(_vget('16mb'), _vget('26mb')),
    }
    l_raw = {
        'anterior': H(_vget('33m'),  _vget('43m')),
        'middle'  : H(_vget('34b'),  _vget('44b')),
        'posterior': H(_vget('36mb'), _vget('46mb')),
    }
    u = {k: round(v, dec) for k, v in u_raw.items()}
    l = {k: round(v, dec) for k, v in l_raw.items()}

    # --- 阈值库（均值, SD） ---
    BASE = {
        'upper': {
            'anterior': {'male': (37.51, 2.00), 'female': (35.85, 1.39)},
            'middle'  : {'male': (38.10, 2.47), 'female': (36.81, 1.67)},
            'posterior': {'male': (48.89, 1.96), 'female': (47.44, 1.61)},
        },
        'lower': {
            'anterior': {'male': (29.04, 1.80), 'female': (27.46, 1.29)},
            'middle'  : {'male': (31.59, 1.89), 'female': (31.20, 1.40)},
            'posterior': {'male': (42.66, 2.07), 'female': (41.98, 2.35)},
        }
    }

    # --- 读入 cfg ---
    cfg = cfg or {}
    sigma = float(cfg.get('arch_width_sigma', 1.0))
    # 限制合理范围，避免误配
    sigma = max(0.5, min(sigma, 3.0))

    def _norm_sex(val: Any) -> Optional[str]:
        if not isinstance(val, str):
            return None
        s = val.strip().lower()
        if s in ('male', 'm', 'man', '男', '♂'):
            return 'male'
        if s in ('female', 'f', 'woman', '女', '♀'):
            return 'female'
        return None

    sex = _norm_sex(cfg.get('sex') or cfg.get('gender') or cfg.get('patient_sex'))

    # --- 生成阈值上下界 ---
    def _range_for(jaw: str, seg: str) -> Dict[str, float]:
        male_mu, male_sd = BASE[jaw][seg]['male']
        female_mu, female_sd = BASE[jaw][seg]['female']
        if sex in ('male', 'female'):
            mu, sd = (male_mu, male_sd) if sex == 'male' else (female_mu, female_sd)
            lo, hi = mu - sigma*sd, mu + sigma*sd
            return {'mean': mu, 'sd': sd, 'lo': lo, 'hi': hi}
        # 未知性别：取两者并集，降低误判风险
        lo = min(male_mu - sigma*male_sd, female_mu - sigma*female_sd)
        hi = max(male_mu + sigma*male_sd, female_mu + sigma*female_sd)
        # 同时给出“合并均值/SD”（仅作展示，无统计学含义）
        return {'mean': (male_mu + female_mu)/2.0, 'sd': None, 'lo': lo, 'hi': hi}

    ranges = {
        'upper': {seg: _range_for('upper', seg) for seg in ('anterior', 'middle', 'posterior')},
        'lower': {seg: _range_for('lower', seg) for seg in ('anterior', 'middle', 'posterior')},
    }

    # --- 逐段判定 ---
    def _judge(val: float, lo: float, hi: float) -> str:
        if val < lo - 1e-6:
            return '偏窄'
        if val > hi + 1e-6:
            return '偏宽'
        return '正常'

    judge = {
        'upper': {seg: _judge(u_raw[seg], ranges['upper'][seg]['lo'], ranges['upper'][seg]['hi'])
                  for seg in ('anterior', 'middle', 'posterior')},
        'lower': {seg: _judge(l_raw[seg], ranges['lower'][seg]['lo'], ranges['lower'][seg]['hi'])
                  for seg in ('anterior', 'middle', 'posterior')},
    }

    # --- 兼容旧字段（U-L 差与“上颌较窄”投票）---
    diff = {f"{seg}_mm": round(u_raw[seg] - l_raw[seg], dec) for seg in ('anterior', 'middle', 'posterior')}
    votes_upper_narrow = sum(1 for seg in ('anterior', 'middle', 'posterior') if (u_raw[seg] - l_raw[seg]) < 0.0)
    upper_is_narrow = votes_upper_narrow >= 2

    # --- 汇总文本 ---
    name_seg = {'anterior': '前段', 'middle': '中段', 'posterior': '后段'}
    up_txt = '、'.join([f"{name_seg[s]}{judge['upper'][s]}" for s in ('anterior', 'middle', 'posterior')])
    lo_txt = '、'.join([f"{name_seg[s]}{judge['lower'][s]}" for s in ('anterior', 'middle', 'posterior')])
    sex_tag = {'male': '男性', 'female': '女性', None: '（性别未知→合并区间）'}[sex]
    sigma_tag = f"±{sigma:.1f}SD"
    summary = f"Arch_Width_牙弓宽度*: 上颌{up_txt}；下颌{lo_txt}（{sex_tag}，阈值{sigma_tag}）"

    return {
        'upper': {f"{k}_mm": v for k, v in u.items()},
        'lower': {f"{k}_mm": v for k, v in l.items()},
        'diff_UL_mm': diff,                 # 兼容：上-下差值
        'upper_is_narrow': bool(upper_is_narrow),   # 兼容：上颌是否相对较窄（投票≥2）
        'judge': judge,                     # 新：逐段宽/窄/正常
        'ranges_mm': ranges,                # 新：使用的阈值上下界
        'sex_used': (sex or 'combined'),    # 新：阈值采用的性别模式
        'summary_text': summary,
        'quality': 'ok'
    }

# =========================
# 3) Bolton 比 — 6牙 / 12牙
# =========================
def compute_bolton(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    wt_callable,
    wt_records: Optional[Dict[Tuple[str, bool], Dict[str, Any]]] = None,
    dec: int = 2
) -> Dict:
    U_overall = ['16','15','14','13','12','11','21','22','23','24','25','26']
    L_overall = ['36','35','34','33','32','31','41','42','43','44','45','46']
    U_ant = ['13','12','11','21','22','23']
    L_ant = ['33','32','31','41','42','43']

    # 严格：每颗牙必须有 mc+dc
    def _has_md(t):
        return _is_xyz(landmarks.get(f"{t}mc")) and _is_xyz(landmarks.get(f"{t}dc"))

    missing = [t for t in sorted(set(U_overall+L_overall)) if not _has_md(t)]
    if missing:
        return {
            'anterior': None, 'overall': None, 'quality':'error',
            'summary_text': 'Bolton输入缺失：' + '、'.join(missing)
        }

    def _sum_width(lst):
        s = 0.0
        invalid: List[str] = []
        for t in lst:
            w = wt_callable(t, allow_fallback=False)
            if w is None:
                invalid.append(t)
            else:
                s += w
        return (None if invalid else s), invalid

    UA, invalid_UA = _sum_width(U_ant)
    LA, invalid_LA = _sum_width(L_ant)
    UO, invalid_UO = _sum_width(U_overall)
    LO, invalid_LO = _sum_width(L_overall)

    removed_teeth = sorted(set(invalid_UA + invalid_LA + invalid_UO + invalid_LO))
    if removed_teeth:
        return {
            'anterior': None,
            'overall': None,
            'quality': 'error',
            'removed_teeth': removed_teeth,
            'summary_text': 'Bolton_Ratio_Bolton比*: 数据异常（剔除牙位：' + '、'.join(removed_teeth) + '）'
        }

    if UA is None or LA is None or UO is None or LO is None:
        return {
            'anterior': None,
            'overall': None,
            'quality': 'error',
            'summary_text': 'Bolton_Ratio_Bolton比*: 关键牙宽缺失'
        }

    ratio_ant = (LA/UA)*100.0 if UA>EPS else None
    ratio_ovr = (LO/UO)*100.0 if UO>EPS else None

    # 目标与容差
    targA, tolA = 78.8, 1.72
    targO, tolO = 91.5, 1.51

    def _judge(ratio, targ, tol):
        if ratio is None: return '缺失'
        if abs(ratio - targ) <= tol: return '正常'
        return '下颌牙量过大' if ratio > targ+tol else '上颌牙量过大'

    jA, jO = _judge(ratio_ant, targA, tolA), _judge(ratio_ovr, targO, tolO)

    def _discrep(sumU, sumL, targ):
        return sumL - sumU*(targ/100.0)

    dA = _discrep(UA, LA, targA)
    dO = _discrep(UO, LO, targO)

    summary = ('正常' if (jA=='正常' and jO=='正常')
               else f"前牙比{jA}（{'下颌' if dA>0 else '上颌'}差 {abs(dA):.2f}mm）；全牙比{jO}（{'下颌' if dO>0 else '上颌'}差 {abs(dO):.2f}mm）")

    fallback_teeth: List[str] = []
    if wt_records:
        seen: Set[str] = set()
        for tooth in sorted(set(U_overall + L_overall)):
            rec = wt_records.get((tooth, False)) or wt_records.get((tooth, True))
            if rec and rec.get('status') == 'fallback':
                if tooth not in seen:
                    fallback_teeth.append(tooth)
                    seen.add(tooth)
    quality = 'ok' if not fallback_teeth else 'fallback'

    return {
        'anterior': {'ratio': None if ratio_ant is None else round(ratio_ant, dec),
                     'status': jA, 'discrep_mm': round(dA,2)},
        'overall' : {'ratio': None if ratio_ovr is None else round(ratio_ovr, dec),
                     'status': jO, 'discrep_mm': round(dO,2)},
        'quality': quality,
        'summary_text': f"Bolton_Ratio_Bolton比*: {summary}",
        'fallback_teeth': fallback_teeth
    }

# =========================
# 4) 尖牙关系 — 用 13m/23m 对 43dc/33dc 的 y 差
# =========================
def compute_canine_relationship(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    dec: int = 2,
    cfg: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    尖牙关系（统一咬合平面 P）：
      右侧 dy = y(13m) - y(43dc)；左侧 dy = y(23m) - y(33dc)
      - 上颌：仅 13m/23m；缺失则该侧缺失
      - 下颌：dc 缺失回退 m（不再使用 dr）
    分度（|dy| 单位 mm）：
      |dy| ≤ neutral_tol         → 中性（尖对尖）
      0.5 < |dy| < complete_tol 且 dy>0 → 远中尖对尖
      0.5 < |dy| < complete_tol 且 dy<0 → 近中尖对尖
      |dy| ≥ complete_tol 且 dy>0 → 完全远中
      |dy| ≥ complete_tol 且 dy<0 → 完全近中
    备注：build_module0 已保证 Ŷ=前+，因此 dy>0 表示下颌参照点相对上颌更“后/远中”。
    """
    cfg = cfg or {}
    neutral_tol = float(cfg.get("canine_neutral_tol_mm", 0.5))
    complete_tol = float(cfg.get("canine_complete_tol_mm", 2.0))

    def _get(label: str) -> Optional[np.ndarray]:
        return _v(landmarks.get(label))

    def _side(upper_m: str, lower_dc: str, lower_m: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        dbg = {"upper": upper_m, "lower_candidates": [lower_dc, lower_m]}
        U = _get(upper_m)
        if U is None:
            dbg["status"] = "missing_upper"
            return None, dbg
        L = _get(lower_dc)
        used_lower = lower_dc
        if L is None:
            L = _get(lower_m)
            used_lower = lower_m if L is not None else None
        if L is None:
            dbg["status"] = "missing_lower"
            return None, dbg

        dy = float(ops['y'](U) - ops['y'](L))
        ady = abs(dy)

        if ady <= neutral_tol:
            label = "中性（尖对尖）"
        elif dy > 0:
            label = "远中尖对尖" if ady < complete_tol else "完全远中"
        else:
            label = "近中尖对尖" if ady < complete_tol else "完全近中"

        detail = {
            "class": label,
            "offset_mm": round(dy, dec),
            "raw_offset_mm": dy,
            "upper_label": upper_m,
            "lower_label": used_lower,
        }
        dbg.update({"status": "ok", "dy": dy, "neutral_tol_mm": neutral_tol, "complete_tol_mm": complete_tol})
        return detail, dbg

    right, dbgR = _side("13m", "43dc", "43m")
    left,  dbgL = _side("23m", "33dc", "33m")

    parts = [f"右侧{right['class']}" if right else "右侧缺失",
             f"左侧{left['class']}" if left else "左侧缺失"]
    summary_text = "Canine_Relationship_尖牙关系*: " + " ".join(parts)
    quality = "ok" if (right and left) else ("fallback" if (right or left) else "missing")

    return {
        "right": right, "left": left,
        "summary_text": summary_text,
        "quality": quality,
        "debug": {"right": dbgR, "left": dbgL},
        "thresholds": {"neutral_tol_mm": neutral_tol, "complete_tol_mm": complete_tol},
    }

# =========================
# 5) 锁𬌗 — 前磨 & 第一磨牙，侧别判定（第一磨牙用“四点均值法”）
# =========================
def compute_crossbite(landmarks: Dict[str, List[float]], ops: Dict, tau:float=0.5) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))

    def _mean_of(names: List[str]) -> Optional[np.ndarray]:
        pts = [ _get(n) for n in names if _get(n) is not None ]
        if not pts:
            return None
        return sum(pts) / float(len(pts))

    def _pair_pts(U_bu, U_pal, L_bu, L_pal):
        """
        传入为“点名列表”，内部自动取均值；若同侧一对仅有单点，则回退单点；均缺则返回 '缺失'。
        """
        U_b = _mean_of(U_bu);  U_l = _mean_of(U_pal)
        L_b = _mean_of(L_bu);  L_l = _mean_of(L_pal)
        if any(p is None for p in [U_b, U_l, L_b, L_l]): return '缺失'
        xU_b = ops['x'](U_b); xU_l = ops['x'](U_l)
        xL_b = ops['x'](L_b); xL_l = ops['x'](L_l)
        S1 = abs(xU_l) - abs(xL_b)   # 正锁检测
        S2 = abs(xU_b) - abs(xL_l)   # 反锁检测
        if S1 > tau: return '正锁'
        if S2 < -tau: return '反锁'
        return '无'

    # —— 前磨牙（单点口径不变）
    pm_R = _pair_pts(['14b'], ['14l'], ['44b'], ['44l'])
    pm_L = _pair_pts(['24b'], ['24l'], ['34b'], ['34l'])

    # —— 第一磨牙（四点均值口径，缺失时自动回退单点）
    m1_R = _pair_pts(['16mb','16db'], ['16ml','16dl'], ['46mb','46db'], ['46ml','46dl'])
    m1_L = _pair_pts(['26mb','26db'], ['26ml','26dl'], ['36mb','36db'], ['36ml','36dl'])

    def _side_txt(pm, m1, side):
        if pm=='缺失' and m1=='缺失': return f"{side}侧缺失"
        if pm=='无' and m1=='无': return None
        parts = []
        if pm in ('正锁','反锁'): parts.append('第一前磨牙')
        if m1 in ('正锁','反锁'): parts.append('第一磨牙')
        typ = (m1 if m1 in ('正锁','反锁') else pm)
        return f"{side}侧{typ}（累及" + ('、'.join(parts)) + '）'

    items = []
    rtxt = _side_txt(pm_R, m1_R, '右');  ltxt = _side_txt(pm_L, m1_L, '左')
    if rtxt: items.append(rtxt)
    if ltxt: items.append(ltxt)
    summary = ('Crossbite_锁牙合: 无' if not items else 'Crossbite_锁牙合: ' + '；'.join(items))
    quality = 'ok' if ('缺失' not in (pm_R, m1_R, pm_L, m1_L)) else 'fallback'
    return {'right': {'premolar': pm_R, 'm1': m1_R},
            'left' : {'premolar': pm_L, 'm1': m1_L},
            'summary_text': summary, 'quality': quality}

# =========================
# 6) 拥挤度 — 新版（圆弧拟合 + 6-6/7-7 自动回退）
# =========================
def _pt_for_crowding(landmarks: Dict[str, List[float]], label: str) -> Optional[np.ndarray]:
    """
    拥挤度专用取点：
    - dc → dr → m
    - mc → mr → m
    - mb → db → m
    其余标签直接返回。
    """
    if not isinstance(label, str):
        return None
    if label.endswith('dc'):
        tooth = label[:-2]
        for suffix in ('dc', 'dr', 'm'):
            pt = _v(landmarks.get(f"{tooth}{suffix}"))
            if pt is not None:
                return pt
        return None
    if label.endswith('mc'):
        tooth = label[:-2]
        for suffix in ('mc', 'mr', 'm'):
            pt = _v(landmarks.get(f"{tooth}{suffix}"))
            if pt is not None:
                return pt
        return None
    if label.endswith('mb'):
        tooth = label[:-2]
        for suffix in ('mb', 'db', 'm'):
            pt = _v(landmarks.get(f"{tooth}{suffix}"))
            if pt is not None:
                return pt
        return None
    return _v(landmarks.get(label))


def _get_crowding_nodes(
    landmarks: Dict[str, List[float]],
    order: List[str],
    endpoint: str = 'dc'
) -> Tuple[Optional[List[np.ndarray]], List[str]]:
    """
    生成用于拟合弓长的节点：
    - endpoint='mb' → 起终点采用 mb（缺→db→m）
    - endpoint='dc' → 起终点采用 dc（缺→dr→m）
    - 中间节点：牙间中点（a.dc 与 b.mc），跨中线使用 mc↔mc（缺则回退切缘*）
    返回 (节点列表 或 None, 缺失标签列表)。
    """
    missing: List[str] = []
    endpoint = endpoint if endpoint in ('mb', 'dc') else 'dc'

    def _need(label: str) -> Optional[np.ndarray]:
        pt = _pt_for_crowding(landmarks, label)
        if pt is None:
            missing.append(label)
        return pt

    if not order:
        return None, ['order_empty']

    start_label = f"{order[0]}{endpoint}"
    end_label = f"{order[-1]}{endpoint}"
    start = _need(start_label)
    end = _need(end_label)
    if start is None or end is None:
        return None, missing

    nodes: List[np.ndarray] = [start]
    midline_pairs = {('11', '21'), ('41', '31')}
    for a, b in zip(order[:-1], order[1:]):
        if (a, b) in midline_pairs:
            left = _pt_for_crowding(landmarks, a + 'mc')
            right = _pt_for_crowding(landmarks, b + 'mc')
            if left is not None and right is not None:
                nodes.append(0.5 * (left + right))
                continue
            _, star_left = _star_mid(landmarks, a)
            _, star_right = _star_mid(landmarks, b)
            if star_left is not None and star_right is not None:
                nodes.append(0.5 * (star_left + star_right))
            else:
                if left is None:
                    missing.append(f"{a}mc")
                if right is None:
                    missing.append(f"{b}mc")
                if star_left is None:
                    missing.append(f"{a}*")
                if star_right is None:
                    missing.append(f"{b}*")
                return None, missing
        else:
            p_dc = _pt_for_crowding(landmarks, a + 'dc')
            p_mc = _pt_for_crowding(landmarks, b + 'mc')
            if p_dc is None or p_mc is None:
                if p_dc is None:
                    missing.append(f"{a}dc")
                if p_mc is None:
                    missing.append(f"{b}mc")
                return None, missing
            nodes.append(0.5 * (p_dc + p_mc))

    nodes.append(end)
    if len(nodes) < 3:
        return None, missing or ['insufficient_nodes']
    return nodes, []


def _sum_strict_widths_v3(landmarks: Dict[str, List[float]], order: List[str], wt_callable) -> Optional[float]:
    """严格 mc↔dc 宽度求和。"""
    total = 0.0
    for tooth in order:
        w = wt_callable(tooth, allow_fallback=False)
        if w is None:
            return None
        total += float(w)
    return total


def _arc_len_spline_v1(
    points_list: List[np.ndarray],
    ops: Dict[str, Any],
    samples_per_seg: int = 40,
    tension: float = 0.5
) -> float:
    """
    在 P 面对医生链点做参数样条（C1 连续的 Hermite/Catmull-Rom 近似），
    逐段等参采样并累加弧长；若点数不足则回退折线长度。
    - samples_per_seg: 每段采样密度（>=10；弧长随之收敛）
    - tension: 0~1，越大曲线越贴折线。0.5 对口扫数据较稳健。
    """
    projP = ops['projP']
    x_of, y_of = ops['x'], ops['y']
    H = ops['H']

    P2: List[np.ndarray] = []
    for p in points_list or []:
        pP = projP(p)
        P2.append(np.array([x_of(pP), y_of(pP)], dtype=float))
    P2 = np.asarray(P2, float)

    if P2.shape[0] < 3:
        length = 0.0
        if points_list:
            for a, b in zip(points_list[:-1], points_list[1:]):
                length += H(a, b)
        return float(length)

    n = P2.shape[0]
    T = np.zeros_like(P2)
    T[0] = (P2[1] - P2[0])
    T[-1] = (P2[-1] - P2[-2])
    if n > 2:
        T[1:-1] = 0.5 * (P2[2:] - P2[:-2])
    T *= float(tension)

    def hermite(P0, P1, M0, M1, t):
        t2 = t * t
        t3 = t2 * t
        h00 = 2 * t3 - 3 * t2 + 1
        h10 = t3 - 2 * t2 + t
        h01 = -2 * t3 + 3 * t2
        h11 = t3 - t2
        return h00 * P0 + h10 * M0 + h01 * P1 + h11 * M1

    length = 0.0
    prev = P2[0]
    m = max(10, int(samples_per_seg))
    for i in range(n - 1):
        P0, P1 = P2[i], P2[i + 1]
        M0, M1 = T[i], T[i + 1]
        for k in range(1, m + 1):
            t = k / m
            cur = hermite(P0, P1, M0, M1, t)
            length += float(np.linalg.norm(cur - prev))
            prev = cur
    return float(length)



def compute_crowding(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    wt_callable,   # 严格宽度：mc↔dc
    dec: int = 1,
) -> Dict:
    """
    Crowding（ALD）= Σ(mc↔dc) - 牙弓弧长
    - 弓长：医生链功能点 → P 面拟合曲线（二维多项式，弦长参数化），失败/异常回退折线；
            6–6 端点使用 mb（缺→db→m），7–7 端点使用 dc（缺→dr→m），切缘 * 用 (ma+da)/2，缺回退 m。
    - 双口径：同时计算 6–6 与 7–7；优先报告 6–6（缺失再退 7–7），两套结果都保留在 detail。
    - 严格牙宽：仅 mc↔dc；缺失即该口径不出 ALD 值。
    """

    # ————— 医生链：功能点顺序（按您提供的口径） —————
    U_CHAIN_66 = ['16mb','15b','14b','13m','12*','11*','21*','22*','23m','24b','25b','26mb']
    U_CHAIN_77 = ['17dc'] + U_CHAIN_66 + ['27dc']
    L_CHAIN_66 = ['46mb','45b','44b','43m','42*','41*','31*','32*','33m','34b','35b','36mb']
    L_CHAIN_77 = ['47dc'] + L_CHAIN_66 + ['37dc']

    U_TEETH_66 = ['16','15','14','13','12','11','21','22','23','24','25','26']
    U_TEETH_77 = ['17'] + U_TEETH_66 + ['27']
    L_TEETH_66 = ['46','45','44','43','42','41','31','32','33','34','35','36']
    L_TEETH_77 = ['47'] + L_TEETH_66 + ['37']

    # ————— 取点：带回退规则（dc→dr→m，mc→mr→m，mb→db→m，*→(ma+da)/2→m） —————
    def _pick(label: str) -> Optional[np.ndarray]:
        if not isinstance(label, str):
            return None
        if label.endswith('*'):
            base = label[:-1]
            _, star = _star_mid(landmarks, base)  # (ma+da)/2，内部已做 ma/da 缺失回退
            if star is not None: return star
            # 兜底回退：若星点失败，直接用 m
            return _v(landmarks.get(base + 'm'))

        if label.endswith('dc'):
            t = label[:-2]
            for suf in ('dc', 'dr', 'm'):
                pt = _v(landmarks.get(t + suf))
                if pt is not None: return pt
            return None

        if label.endswith('mc'):
            t = label[:-2]
            for suf in ('mc', 'mr', 'm'):
                pt = _v(landmarks.get(t + suf))
                if pt is not None: return pt
            return None

        if label.endswith('mb'):
            t = label[:-2]
            for suf in ('mb', 'db', 'm'):
                pt = _v(landmarks.get(t + suf))
                if pt is not None: return pt
            return None

        return _v(landmarks.get(label))

    def _chain_points(tags: List[str]) -> Tuple[Optional[List[np.ndarray]], List[str]]:
        pts, missing = [], []
        for tag in tags:
            p = _pick(tag)
            if p is None:
                missing.append(tag)
            else:
                pts.append(p)
        if missing or len(pts) < 4:
            return None, missing
        return pts, []

    # ————— P 面折线长度（作为回退与守护的参考） —————
    def _polyline_len(pts: List[np.ndarray]) -> float:
        H = ops['H']
        return float(sum(H(a, b) for a, b in zip(pts[:-1], pts[1:])))

    # ————— 弦长参数化二维拟合弧长（稳） —————
    def _arc_len_parametric(pts: List[np.ndarray], deg: int = 5, num_samples: int = 400) -> Optional[float]:
        if not pts or len(pts) < max(4, deg + 1):
            return None
        projP, x_of, y_of = ops['projP'], ops['x'], ops['y']
        try:
            xs = np.array([x_of(projP(p)) for p in pts], dtype=float)
            ys = np.array([y_of(projP(p)) for p in pts], dtype=float)
        except Exception:
            return None
        seg = np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)
        s = np.concatenate([[0.0], np.cumsum(seg)])
        total = float(s[-1])
        if total <= EPS:
            return None
        t = s / total
        deg = int(max(3, min(deg, len(xs) - 1)))
        try:
            cx = np.polyfit(t, xs, deg);  cy = np.polyfit(t, ys, deg)
            dcx = np.polyder(cx);        dcy = np.polyder(cy)
        except (np.linalg.LinAlgError, ValueError):
            return None
        ts = np.linspace(0.0, 1.0, int(max(50, num_samples)))
        dx = np.polyval(dcx, ts);  dy = np.polyval(dcy, ts)
        speed = np.sqrt(dx*dx + dy*dy)
        try:
            return float(np.trapz(speed, ts))
        except Exception:
            return None

    # ————— 计算某条功能链的弓长：优先拟合，守护回退折线 —————
    def _arc_len_by_fit(tags: List[str]) -> Tuple[Optional[float], List[str], Dict[str, Any]]:
        pts, miss = _chain_points(tags)
        if pts is None:
            return None, miss, {'arc_mode': 'missing', 'missing_nodes': miss}

        L_poly = _polyline_len(pts)
        L_fit  = _arc_len_parametric(pts, deg=5, num_samples=400)

        arc_mode, chosen = 'param', L_fit
        reasons, ratio = [], None
        if L_fit is None:
            arc_mode, chosen = ('polyline' if L_poly > EPS else 'missing'), L_poly
            reasons.append('fit_failed')
        else:
            if L_poly > EPS:
                ratio = float(L_fit / L_poly)
                # 守护：拟合不应比折线短；也不应 > 1.15×折线（过度外插）
                if (ratio < 1.0 - 1e-6) or (ratio > 1.15):
                    arc_mode, chosen = 'polyline', L_poly
                    reasons.append('fit_guard_fallback')

        detail = {
            'arc_mode': arc_mode,
            'fit_len_mm': float(L_fit) if L_fit is not None else None,
            'polyline_len_mm': float(L_poly),
            'fit_to_poly_ratio': (float(ratio) if ratio is not None else None),
        }
        if reasons: detail['reasons'] = reasons
        if chosen is not None: detail['chosen_len_mm'] = float(chosen)
        return (float(chosen) if chosen is not None else None, [], detail)

    # ————— 严格口径的牙冠宽度和（mc↔dc；带缺失列表） —————
    def _sum_width_strict(teeth: List[str]) -> Tuple[Optional[float], List[str]]:
        total, missing = 0.0, []
        for t in teeth:
            w = wt_callable(t, allow_fallback=False)
            if w is None:
                missing.append(t)
            else:
                total += float(w)
        return (None, missing) if missing else (float(total), [])

    # ————— 单弓汇总：同时产出 6–6 与 7–7，优先 6–6 —————
    def _one_arch(label: str, chain66, chain77, teeth66, teeth77):
        modes, msgs, cache = {}, [], {}
        for mode, chain_tags, tooth_list in (('6-6', chain66, teeth66), ('7-7', chain77, teeth77)):
            L, miss_nodes, arc_detail = _arc_len_by_fit(chain_tags)
            S, miss_width = _sum_width_strict(tooth_list)
            cache[mode] = {'arc_len': L, 'sum_width': S, 'arc_detail': arc_detail}

            if L is None or S is None:
                if miss_nodes: msgs.append(f"{label}{mode}弓长缺失({','.join(miss_nodes)})")
                if miss_width: msgs.append(f"{label}{mode}牙宽缺失({','.join(miss_width)})")
                continue

            ald_raw = S - L
            entry = {
                'mode': mode,
                'ald_mm': float(np.round(ald_raw, dec)),
                'raw_ald_mm': float(ald_raw),
                'sum_width_mm': float(np.round(S, dec)),
                'raw_sum_width_mm': float(S),
                'arc_len_mm': float(np.round(L, dec)),
                'raw_arc_len_mm': float(L),
                'arc_mode': arc_detail.get('arc_mode'),
                'arc_detail': arc_detail,
                'width_policy': 'strict',
            }
            r = arc_detail.get('fit_to_poly_ratio')
            ratio_txt = (f", ratio:{r:.3f}" if isinstance(r, (int, float)) else "")
            crowd_txt = (
                f"拥挤{entry['ald_mm']:.{dec}f}mm" if entry['ald_mm'] > 0 else
                f"间隙{abs(entry['ald_mm']):.{dec}f}mm" if entry['ald_mm'] < 0 else
                f"拥挤/间隙{0:.{dec}f}mm"
            )
            entry['text'] = (
                f"{mode}{crowd_txt}(Σ{entry['sum_width_mm']:.{dec}f}mm, "
                f"L{entry['arc_len_mm']:.{dec}f}mm, arc:{entry['arc_mode']}{ratio_txt}, strict)"
            )
            modes[mode] = entry

        primary = modes.get('6-6') or modes.get('7-7')
        return primary, modes, msgs, cache

    up_primary, up_modes, up_msgs, _ = _one_arch('上牙列', U_CHAIN_66, U_CHAIN_77, U_TEETH_66, U_TEETH_77)
    lo_primary, lo_modes, lo_msgs, _ = _one_arch('下牙列', L_CHAIN_66, L_CHAIN_77, L_TEETH_66, L_TEETH_77)

    if not up_modes and not lo_modes:
        return {
            'upper': None, 'lower': None,
            'summary_text': 'Crowding_拥挤度*: 缺失',
            'quality': 'missing',
            'detail': {'upper': {'messages': up_msgs}, 'lower': {'messages': lo_msgs}}
        }

    def _summary(label: str, modes: Dict[str, Dict[str, Any]], msgs: List[str]) -> str:
        parts = [modes[m]['text'] for m in ('6-6', '7-7') if m in modes]
        if not parts:
            return f"{label}: " + ('；'.join(msgs) if msgs else '缺失')
        if msgs: parts.extend(msgs)
        return f"{label}: " + '；'.join(parts)

    summary_text = 'Crowding_拥挤度*: ' + '；'.join([
        _summary('上牙列', up_modes, up_msgs),
        _summary('下牙列', lo_modes, lo_msgs),
    ])

    quality = 'ok'
    if (up_primary is None) or (lo_primary is None):
        quality = 'fallback' if (up_modes or lo_modes) else 'missing'

    return {
        'upper': up_primary,
        'lower': lo_primary,
        'summary_text': summary_text,
        'quality': quality,
        'detail': {
            'upper': {'primary_mode': (up_primary or {}).get('mode'), 'modes': up_modes, 'messages': up_msgs},
            'lower': {'primary_mode': (lo_primary or {}).get('mode'), 'modes': lo_modes, 'messages': lo_msgs},
        },
    }

# =========================
# 7) Curve of Spee — 竖直深度判定（6–6 / 7–7）
# =========================
def compute_spee(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    dec: int = 1,
    cfg: Optional[Dict] = None
) -> Dict:
    """
    Curve_of_Spee（竖直深度版）：
    A = (31ma+41ma)/2（缺 → 31m → 41m）；右 B = 47db（缺 → 46db），左 B = 37db（缺 → 36db）
    采样链（可选 6–6 / 7–7，默认 6–6），先投到 P 做范围筛选，再用相对基线的 z 差取深度（mm）。
    """
    cfg = cfg or {}
    span = str(cfg.get('spee_span', '66'))  # '66' or '77'

    def _v_local(nm: str) -> Optional[np.ndarray]:
        p = landmarks.get(nm)
        if isinstance(p, (list, tuple)) and len(p) == 3:
            try:
                return np.asarray(p, float)
            except (TypeError, ValueError):
                return None
        return None

    def _A_point() -> Tuple[Optional[np.ndarray], List[str]]:
        a, b = _v_local('31ma'), _v_local('41ma')
        if a is not None and b is not None:
            return 0.5 * (a + b), ['31ma', '41ma']
        for lbl in ('31m', '41m'):
            p = _v_local(lbl)
            if p is not None:
                return p, [lbl]
        return None, []

    span_map = {
        '66': {
            'right': (['43m', '44b', '45b', '46mb', '46db'], ['46db']),
            'left':  (['33m', '34b', '35b', '36mb', '36db'], ['36db']),
        },
        '77': {
            'right': (['43m', '44b', '45b', '46mb', '46db', '47mb', '47db'], ['47db', '46db']),
            'left':  (['33m', '34b', '35b', '36mb', '36db', '37mb', '37db'], ['37db', '36db']),
        },
    }
    chains = span_map.get(span)
    if chains is None:
        span = '66'
        chains = span_map['66']

    def _pick_first(names: List[str]) -> Tuple[Optional[str], Optional[np.ndarray]]:
        for nm in names:
            p = _v_local(nm)
            if p is not None:
                return nm, p
        return None, None

    def _side_depth(
        A_pt: Optional[np.ndarray],
        labels: List[str],
        B_candidates: List[str]
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        B_name, B_pt = _pick_first(B_candidates)
        if A_pt is None or B_pt is None:
            return None, {'status': 'missing', 'B': B_name}

        A0, B0 = ops['projP'](A_pt), ops['projP'](B_pt)
        AB = B0 - A0
        L = float(np.linalg.norm(AB))
        if L <= 1e-9:
            return None, {'status': 'degenerate_baseline'}

        u = AB / L
        base_z = float(ops['z'](A0))

        min_z_rel = float('inf')
        min_lbl: Optional[str] = None
        used: List[str] = []
        for lbl in labels:
            p = _v_local(lbl)
            if p is None:
                continue
            pP = ops['projP'](p)
            s = float(np.dot(pP - A0, u) / L)
            if s < -0.05 or s > 1.05:
                continue
            z_rel = float(ops['z'](p) - base_z)
            if z_rel < min_z_rel:
                min_z_rel = z_rel
                min_lbl = lbl
            used.append(lbl)

        if not used:
            return None, {'status': 'insufficient_samples', 'B': B_name, 'AB_len_mm': round(L, 2)}

        depth = max(0.0, -min_z_rel)
        return depth, {
            'status': 'ok',
            'B': B_name,
            'AB_len_mm': round(L, 2),
            'n_samples': len(used),
            'used_labels': used,
            'min_label': min_lbl,
            'min_z_mm': round(min_z_rel, dec),
        }

    A_pt, A_sources = _A_point()

    right_chain, right_B = chains['right']
    left_chain, left_B = chains['left']

    right_val, right_detail = _side_depth(A_pt, right_chain, right_B)
    left_val, left_detail = _side_depth(A_pt, left_chain, left_B)

    if right_val is None and left_val is None:
        return {
            'value_mm': None,
            'primary_side': None,
            'summary_text': 'Curve_of_Spee_Spee曲线*: 缺失',
            'quality': 'missing',
            'detail': {'span': span, 'right': right_detail, 'left': left_detail}
        }

    if (left_val or 0.0) > (right_val or 0.0):
        val, side = left_val, 'left'
    else:
        val, side = right_val, 'right'

    val_rounded = round(val, dec) if isinstance(val, (int, float)) else None
    quality = 'ok' if isinstance(val_rounded, (int, float)) else 'missing'

    return {
        'value_mm': val_rounded,
        'primary_side': side if isinstance(val_rounded, (int, float)) else None,
        'summary_text': (f'Curve_of_Spee_Spee曲线*: 深{val_rounded:.{dec}f}mm' if isinstance(val_rounded, (int, float))
                         else 'Curve_of_Spee_Spee曲线*: 缺失'),
        'quality': quality,
        'detail': {
            'span': span,
            'A_sources': A_sources,
            'right': right_detail,
            'left': left_detail,
        }
    }

# =========================
# 8) Midline — 上下牙列中线偏差
# =========================
def compute_midline_alignment(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    dec: int = 1,
    tol_center: float = 0.1
) -> Dict:
    """
    Midline_牙列中线（以下对上，单位 mm）
    - 上：11mc–21mc 中点；回退：11*–21* 中点；再回退 11m–21m 中点
    - 下：31mc–41mc 中点；回退：31*–41* 中点；再回退 31m–41m 中点
    - 结果：summary_text 形如“上牙列正 下牙列右偏0.2mm”（右为正）；|Δ| < tol_center → “下牙列居中”
    """
    def _is_xyz(p):
        return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()
    def _v(p): return np.asarray(p, float) if _is_xyz(p) else None

    def _mid(a: str, b: str) -> Optional[np.ndarray]:
        pa, pb = _v(landmarks.get(a)), _v(landmarks.get(b))
        return 0.5*(pa + pb) if (pa is not None and pb is not None) else None

    def _star_midpoint(tooth: str) -> Optional[np.ndarray]:
        _, pt = _star_mid(landmarks, tooth)  # (ma+da)/2；缺则回退 m
        return pt

    def _choose_mid(tooth_left: str, tooth_right: str) -> Tuple[Optional[np.ndarray], List[str]]:
        sources: List[str] = []
        # 1) 首选：两个近中接触点 mc 的中点
        primary = _mid(f"{tooth_left}mc", f"{tooth_right}mc")
        if primary is not None:
            sources.extend([f"{tooth_left}mc", f"{tooth_right}mc"])
            return primary, sources
        # 2) 回退：切缘* 的中点（* 自身再回退 m）
        star_left = _star_midpoint(tooth_left)
        star_right = _star_midpoint(tooth_right)
        if star_left is not None and star_right is not None:
            sources.extend([f"{tooth_left}*", f"{tooth_right}*"])
            return 0.5 * (star_left + star_right), sources
        # 3) 兜底：牙尖 m
        fallback = _mid(f"{tooth_left}m", f"{tooth_right}m")
        if fallback is not None:
            sources.extend([f"{tooth_left}m", f"{tooth_right}m"])
            return fallback, sources
        return None, sources

    # —— 取上下中线点（上固定作基准）——
    U_mid, U_sources = _choose_mid('11', '21')
    L_mid, L_sources = _choose_mid('31', '41')

    if U_mid is None or L_mid is None:
        return {
            'summary_text': 'Midline_Alignment_牙列中线*: 缺失',
            'quality': 'missing',
            'detail': {
                'upper_sources': U_sources,
                'lower_sources': L_sources,
            }
        }

    # —— 只看“以下对上”的横向差 —— 
    delta = float(ops['x'](L_mid) - ops['x'](U_mid))  # 右+ 为正
    if abs(delta) < tol_center:
        lower_txt = '下牙列居中'
    else:
        direction = '右偏' if delta > 0 else '左偏'
        lower_txt = f"下牙列{direction}{abs(round(delta, dec)):.{dec}f}mm"

    summary = f"Midline_Alignment_牙列中线*: 上牙列正 {lower_txt}"
    return {
        'delta_mm': round(delta, dec),
        'summary_text': summary,
        'quality': 'ok',
        'detail': {
            'delta_raw_mm': float(delta),
            'upper_sources': U_sources,
            'lower_sources': L_sources,
        }
    }


# =========================
# 9) 第一磨牙关系 — 16/26 对 46bg/36bg（个体化阈值）
# =========================
def compute_molar_relationship(landmarks: Dict[str, List[float]], ops: Dict, wt_callable, dec:int=1, cfg: Optional[Dict[str, Any]]=None) -> Dict:
    """
    第一磨牙关系（与尖牙口径一致）：
    - 方向：上对下。d = dot(上颌近中颊尖 - 下颌参考点, 下颌近中→远中方向单位向量)
      d > 0 → 近中；d < 0 → 远中。
    - 上颌代表点：仅用 16mb/26mb；缺失回退 m（质量降级）。
    - 下颌参考点：优先用 46bg/36bg；bg 缺失才回退 (mb,db) 中点。
    - 分度：|d| ≤ 0.5 → 中性；
            0.5 < |d| < 2.5 → 中性偏近/远中；
            | |d| - 2.5 | ≤ 0.5 → 近中/远中尖对尖；
            | |d| - 5.0 | ≤ 0.5 或 |d| > 5.0 → 完全近/远中；
            其余 → 介于分度之间（趋向近/远中）。
    """
    cfg = cfg or {}
    neutral_tol  = float(cfg.get('molar_neutral_tol_mm', 0.5))
    cusp_mm      = float(cfg.get('molar_cusp_mm', 2.5))
    complete_mm  = float(cfg.get('molar_complete_mm', 5.0))
    match_tol    = float(cfg.get('molar_match_tol_mm', 0.5))

    def _get(nm): return _v(landmarks.get(nm))
    projP = ops['projP']

    def _project(pt: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return projP(pt) if pt is not None else None

    # —— 下颌颊沟（严格 bg，缺失才回退中点）——
    def _lower_groove_strict(tooth: str):
        bg  = _get(f"{tooth}bg")
        if bg is not None:
            return bg, {'status': 'bg_only'}
        mb, db = _get(f"{tooth}mb"), _get(f"{tooth}db")
        if (mb is not None) and (db is not None):
            return 0.5*(mb+db), {'status': 'midpoint_fallback'}
        return None, {'status': 'missing'}

    # —— 上颌仅用近中颊尖（缺失才回退 m）——
    def _upper_mb_strict(upper_tooth: str):
        mb = _get(f"{upper_tooth}mb")
        if mb is not None:
            return mb, 'mb'
        m = _get(f"{upper_tooth}m")
        if m is not None:
            return m, 'm_fallback'
        return None, 'missing'

    def _scalar(pt: Optional[np.ndarray], origin: np.ndarray, axis: np.ndarray) -> Optional[float]:
        if pt is None:
            return None
        p = _project(pt)
        if p is None:
            return None
        return float(_dot(p - origin, axis))

    def _classify_fixed(d: float) -> str:
        a = abs(d)
        if a <= neutral_tol:
            return '中性'
        if a < cusp_mm - match_tol:
            return '中性偏近中' if d > 0 else '中性偏远中'
        if abs(a - cusp_mm) <= match_tol:
            return '近中尖对尖' if d > 0 else '远中尖对尖'
        if abs(a - complete_mm) <= match_tol or a > complete_mm:
            return '完全近中' if d > 0 else '完全远中'
        # 介于 2.5 与 5.0 之间但不在匹配带内
        return '介于分度之间（趋向近中）' if d > 0 else '介于分度之间（趋向远中）'

    def _one(upper: str, lower: str):
        groove_raw, groove_info = _lower_groove_strict(lower)
        if groove_raw is None:
            return None
        grooveP = _project(groove_raw)
        if grooveP is None:
            return None

        lower_mc, lower_dc = _get(f"{lower}mc"), _get(f"{lower}dc")
        if lower_mc is None or lower_dc is None:
            return None
        mcP, dcP = _project(lower_mc), _project(lower_dc)
        if mcP is None or dcP is None:
            return None

        # 轴：近中方向为正
        axis_vec  = mcP - dcP
        axis_len  = _len(axis_vec)
        if axis_len < EPS:
            return None
        axis_unit = axis_vec / axis_len

        upper_mb, upper_src = _upper_mb_strict(upper)
        if upper_mb is None:
            return None
        upperP = _project(upper_mb)
        if upperP is None:
            return None

        # 上对下偏移（沿下颌 mc→dc 方向）：正值→近中，负值→远中
        offset_up = float(_dot(upperP - grooveP, axis_unit))
        offset_down = -offset_up

        label = _classify_fixed(offset_up)

        # 为了便于对齐老版调试字段，把两种标量都保留
        lower_mb = _get(f"{lower}mb"); lower_db = _get(f"{lower}db")
        upper_m  = _get(f"{upper}m")
        def _round(v): 
            return None if v is None or not np.isfinite(v) else float(np.round(v, 3))

        result = {
            'offset_mm': float(np.round(offset_up, dec)),   # 上对下（近中为正）
            'raw_offset_mm': float(offset_up),
            'offset_lower_vs_upper_mm': float(np.round(offset_down, dec)),  # 下对上（近中为正）
            'label': label,
            'width_mm': _round(_len(dcP - mcP)),
            'neutral_tol_mm': neutral_tol,
            'cusp_mm': cusp_mm,
            'complete_mm': complete_mm,
            'match_tol_mm': match_tol,
            'lower_axis_mm': {
                'bg': 0.0,
                'mc': _round(_scalar(lower_mc, grooveP, axis_unit)),
                'dc': _round(_scalar(lower_dc, grooveP, axis_unit)),
                'mb': _round(_scalar(lower_mb, grooveP, axis_unit)) if lower_mb is not None else None,
                'db': _round(_scalar(lower_db, grooveP, axis_unit)) if lower_db is not None else None,
            },
            'upper_axis_mm': {
                'mb': _round(_scalar(upper_mb, grooveP, axis_unit)),
                'm':  _round(_scalar(upper_m,  grooveP, axis_unit)) if upper_m is not None else None,
            },
            'groove_status': groove_info.get('status'),
            'upper_pick': upper_src,
        }
        return result

    R = _one('16', '46')
    L = _one('26', '36')
    if R is None and L is None:
        return {'summary_text': 'Molar_Relationship_磨牙关系*: 缺失', 'quality': 'missing'}
    return {
        'right': R, 'left': L,
        'summary_text': f"Molar_Relationship_磨牙关系*: 右侧{R['label'] if R else '缺失'} 左侧{L['label'] if L else '缺失'}",
        'quality': 'ok' if (R and L) else 'fallback',
    }


# =========================
# 10) Overbite — 垂直覆𬌗（比值分度）
# =========================
def _incisal_incisor_point(landmarks: Dict[str, List[float]], tooth: str, ops: Dict) -> Optional[np.ndarray]:
    """Select the most anterior incisal point for the given tooth."""
    candidates: List[np.ndarray] = []
    for suffix in ('ma', 'da', 'm'):
        pt = _v(landmarks.get(f"{tooth}{suffix}"))
        if pt is not None:
            candidates.append(pt)
    if candidates:
        candidates.sort(key=lambda p: (ops['y'](p), ops['z'](p)))
        return candidates[-1]
    _, star = _star_mid(landmarks, tooth)
    return star


def _incisor_pairs(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    upper_teeth: List[str],
    lower_teeth: List[str],
) -> List[Dict[str, Any]]:
    """Build candidate incisor pairs with geometric metadata."""
    def _side_from_x(x_val: float, tol: float = 0.5) -> str:
        if x_val > tol:
            return 'right'
        if x_val < -tol:
            return 'left'
        return 'mid'

    pairs: List[Dict[str, Any]] = []
    uppers: List[Dict[str, Any]] = []
    lowers: List[Dict[str, Any]] = []

    for tooth in upper_teeth:
        pt = _incisal_incisor_point(landmarks, tooth, ops)
        if pt is None:
            continue
        x_val = ops['x'](pt)
        uppers.append(
            {
                'tooth': tooth,
                'pt': pt,
                'x': x_val,
                'y': ops['y'](pt),
                'z': ops['z'](pt),
                'side': _side_from_x(x_val),
            }
        )

    for tooth in lower_teeth:
        pt = _incisal_incisor_point(landmarks, tooth, ops)
        if pt is None:
            continue
        x_val = ops['x'](pt)
        lowers.append(
            {
                'tooth': tooth,
                'pt': pt,
                'x': x_val,
                'y': ops['y'](pt),
                'z': ops['z'](pt),
                'side': _side_from_x(x_val),
            }
        )

    for up in uppers:
        for low in lowers:
            same_side = (
                (up['side'] == low['side'])
                or (up['side'] == 'mid')
                or (low['side'] == 'mid')
            )
            pairs.append(
                {
                    'upper': up,
                    'lower': low,
                    'y': float(up['y'] - low['y']),
                    'z': float(up['z'] - low['z']),
                    'x_gap': float(abs(up['x'] - low['x'])),
                    'same_side': same_side,
                }
            )
    return pairs


def _filter_pairs_by_iqr(pairs: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Remove outlier pairs based on IQR of the selected metric."""
    if len(pairs) < 4:
        return pairs
    values = np.array([p[key] for p in pairs], dtype=float)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = float(q3 - q1)
    if iqr <= EPS:
        return pairs
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    filtered: List[Dict[str, Any]] = []
    for p, value in zip(pairs, values):
        if lo <= value <= hi:
            filtered.append(p)
    return filtered if filtered else pairs


def compute_overbite(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    # —— 辅助：拿 * 切缘（(ma+da)/2；缺则 m）——
    def _star(tooth: str) -> Optional[np.ndarray]:
        _, p = _star_mid(landmarks, tooth)
        return p

    # 右：11*–41*；左：21*–31*
    U11, L41 = _star('11'), _star('41')
    U21, L31 = _star('21'), _star('31')

    # 两侧都缺 → missing；有一侧缺 → 用另一侧
    side_vals = {}
    if (U11 is not None) and (L41 is not None):
        side_vals['right'] = float(ops['z'](U11) - ops['z'](L41))
    if (U21 is not None) and (L31 is not None):
        side_vals['left']  = float(ops['z'](U21) - ops['z'](L31))

    if not side_vals:
        return {'summary_text': 'Overbite_前牙覆𬌗*: 缺失', 'quality': 'missing'}

    # 选 |OB| 较大侧（并规定平局取右侧）
    side_of_max = max(sorted(side_vals.keys(), key=lambda s: 0 if s=='right' else 1),
                      key=lambda s: abs(side_vals[s]))
    OB = side_vals[side_of_max]

    # 该侧下中切冠高 CH（* ↔ bgb）
    if side_of_max == 'right':
        L_star = L41
        L_bgb  = _v(landmarks.get('41bgb'))
    else:
        L_star = L31
        L_bgb  = _v(landmarks.get('31bgb'))

    CH = None
    if (L_star is not None) and (L_bgb is not None):
        ch_raw = float(abs(ops['z'](L_star) - ops['z'](L_bgb)))
        if ch_raw > EPS:
            CH = ch_raw

    # —— 分度（等号前不取、后取）——
    zero_tol = 0.5
    if abs(OB) <= zero_tol:
        category = '对刃'
    elif OB < 0:
        mag = abs(OB)
        if mag <= 3:
            category = 'Ⅰ度开𬌗'
        elif mag <= 5:
            category = 'Ⅱ度开𬌗'
        else:
            category = 'Ⅲ度开𬌗'
    else:  # OB > 0
        if CH is not None:
            r = OB / CH
            if r <= (1.0/3.0):
                category = '正常覆𬌗'
            elif r <= 0.5:
                category = 'Ⅰ度深覆𬌗'
            elif r <= (2.0/3.0):
                category = 'Ⅱ度深覆𬌗'
            else:
                category = 'Ⅲ度深覆𬌗'
        else:
            if OB < 3:
                category = '轻度覆𬌗'
            elif OB <= 5:
                category = 'Ⅰ度深覆𬌗'
            elif OB <= 8:
                category = 'Ⅱ度深覆𬌗'
            else:
                category = 'Ⅲ度深覆𬌗'

    return {
        'value_mm': round(OB, dec),
        'side_of_max': side_of_max,
        'category': category,
        'summary_text': f"Overbite_前牙覆𬌗*: {category}",
        'quality': 'ok',
        'detail': {
            'OB_right_mm': round(side_vals.get('right', float('nan')), dec) if 'right' in side_vals else None,
            'OB_left_mm':  round(side_vals.get('left',  float('nan')), dec) if 'left'  in side_vals else None,
            'CH_mm': round(CH, dec) if CH is not None else None,
        }
    }

# =========================
# 11) Overjet — 水平覆盖（y 前后）
# =========================
def compute_overjet(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    """
    Overjet_前牙覆盖（严格口径）：
      - 点位：“* 切缘点”= (ma+da)/2，若缺→回退 m
      - 用到：上中切 11*、21*；下中切 41*、31*
      - 坐标：统一咬合平面 P；y 为前后向（前为正）
      - 计算：
          右侧 OJ_R = y(11*) − y(41*)
          左侧 OJ_L = y(21*) − y(31*)
          取 |OJ| 较大侧为 primary_side，OJ 为该侧值
      - 分度（“等号前不取后取”）：
          |OJ| ≤ 0.5 → 对刃
          OJ < -0.5 → 反覆盖
          OJ > 0.5:
            OJ ≤ 3         → 轻度覆盖
            3 < OJ ≤ 5     → Ⅰ度深覆盖
            5 < OJ ≤ 8     → Ⅱ度深覆盖
            OJ > 8         → Ⅲ度深覆盖
    """
    def _star_or_m(tooth: str) -> Tuple[List[str], Optional[np.ndarray]]:
        labels, pt = _star_mid(landmarks, tooth)  # (ma+da)/2；若缺回退 m
        return (labels if labels else [f"{tooth}m"]), pt

    # —— 取四个“*”点（或 m 回退）——
    src11, P11 = _star_or_m('11')
    src21, P21 = _star_or_m('21')
    src41, P41 = _star_or_m('41')
    src31, P31 = _star_or_m('31')

    # —— 按侧计算 —— 
    right_val = None
    left_val  = None
    detail = {'right': None, 'left': None}

    if P11 is not None and P41 is not None:
        OJ_R = float(ops['y'](P11) - ops['y'](P41))
        right_val = OJ_R
        detail['right'] = {'OJ_R_mm': round(OJ_R, dec), 'sources': {'upper':'11*','lower':'41*','upper_labels':src11,'lower_labels':src41}}

    if P21 is not None and P31 is not None:
        OJ_L = float(ops['y'](P21) - ops['y'](P31))
        left_val = OJ_L
        detail['left'] = {'OJ_L_mm': round(OJ_L, dec), 'sources': {'upper':'21*','lower':'31*','upper_labels':src21,'lower_labels':src31}}

    if right_val is None and left_val is None:
        return {'summary_text': 'Overjet_前牙覆盖*: 缺失', 'quality': 'missing'}

    # —— 选 |OJ| 较大侧 —— 
    choose_left = (left_val is not None) and (right_val is None or abs(left_val) > abs(right_val) + 1e-9)
    primary_side = 'left' if choose_left else 'right'
    OJ = left_val if choose_left else right_val
    value_mm = round(OJ, dec)

    # —— 分度（严格“前不取后取”）——
    absOJ = abs(OJ)
    if absOJ <= 0.5:
        category = '对刃'
    elif OJ < -0.5:
        category = '反覆盖'
    else:
        # OJ > 0.5 → 正覆盖
        if OJ <= 3:
            category = '轻度覆盖'
        elif OJ <= 5:        # 3 < OJ ≤ 5
            category = 'Ⅰ度深覆盖'
        elif OJ <= 8:        # 5 < OJ ≤ 8
            category = 'Ⅱ度深覆盖'
        else:                 # OJ > 8
            category = 'Ⅲ度深覆盖'

    summary = f"Overjet_前牙覆盖*: {abs(round(value_mm, dec))}mm_{category}"
    quality = 'ok' if (right_val is not None and left_val is not None) else 'fallback'

    return {
        'value_mm': value_mm,
        'side_of_max': primary_side,
        'category': category,
        'summary_text': summary,
        'quality': quality,
        'detail': detail
    }

# =========================
# 报告行（brief） & 公开接口
# =========================
def _fmt_suffix(ok: bool) -> str: return '' if ok else ' ⚠️'

def make_brief_report(landmarks: Dict[str, List[float]], frame_ops: Dict[str, Any]) -> List[str]:
    frame, ops, wt = frame_ops['frame'], frame_ops['ops'], frame_ops['wt']
    wt_records = frame_ops.get('wt_records') or {}

    arch_form = compute_arch_form(landmarks, frame, ops)
    arch_width = compute_arch_width(landmarks, ops, cfg=cfg)
    bolton = compute_bolton(landmarks, ops, wt, wt_records=wt_records)
    canine = compute_canine_relationship(landmarks, ops)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_info = compute_spee(landmarks, ops)
    midline = compute_midline_alignment(landmarks, ops)
    molar = compute_molar_relationship(landmarks, ops, wt)
    ob = compute_overbite(landmarks, ops)
    oj = compute_overjet(landmarks, ops)

    spee_val = spee_info.get('value_mm')
    spee_text = ('%.1fmm' % spee_val) if isinstance(spee_val, (int, float)) else '缺失'
    spee_ok = (spee_info.get('quality') != 'missing')
    return [
        f"{arch_form['summary_text']}{_fmt_suffix(arch_form['quality']!='missing')}",
        f"{arch_width['summary_text']}{_fmt_suffix(arch_width['quality']!='missing')}",
        f"{bolton['summary_text']}{_fmt_suffix(bolton['quality']!='missing')}",
        f"{canine['summary_text']}{_fmt_suffix(canine['quality']!='missing')}",
        f"{crossbite['summary_text']}{_fmt_suffix(crossbite['quality']!='missing')}",
        f"{crowding['summary_text']}{_fmt_suffix(crowding['quality']!='missing')}",
        f"Curve_of_Spee_Spee曲线*: {spee_text}{_fmt_suffix(spee_ok)}",
        f"{midline['summary_text']}{_fmt_suffix(midline['quality']!='missing')}",
        f"{molar['summary_text']}{_fmt_suffix(molar['quality']!='missing')}",
        f"{ob['summary_text']}{_fmt_suffix(ob['quality']!='missing')}",
        f"{oj['summary_text']}{_fmt_suffix(oj['quality']!='missing')}",
    ]

def _brief_lines_to_kv(lines: List[str]) -> Dict[str, str]:
    kv = {}
    for line in (lines or []):
        line = (line or "").strip()
        if not line: continue
        if "*:" in line:
            k, v = line.split("*:", 1)
            key = k.strip() + "*"; val = v.strip()
        elif ":" in line:
            k, v = line.split(":", 1)
            key = k.strip(); val = v.strip()
        else:
            key, val = line, ""
        kv[key] = val
    return kv

# ---- I/O helpers（仅 JSON，STL 可忽略）----
def _load_landmarks_json(path: Optional[Union[str, os.PathLike, Dict[str, Any]]]) -> Dict[str, List[float]]:
    if path is None:
        return {}
    if isinstance(path, (str, os.PathLike)):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    elif isinstance(path, dict):
        data = path
    else:
        raise TypeError("Unsupported landmarks source type")
    out: Dict[str, List[float]] = {}

    def _ingest_mapping(mapping: Dict) -> None:
        for raw_label, pos in (mapping or {}).items():
            coords = _coerce_xyz(pos)
            if coords is None:
                continue
            label = _normalise_label(raw_label)
            if not label:
                continue
            out[label] = coords

    if isinstance(data, dict):
        if 'markups' in data:
            for mk in data.get('markups', []):
                for cp in mk.get('controlPoints', []):
                    label = _normalise_label(cp.get('label'))
                    coords = _coerce_xyz(cp.get('position'))
                    if label and coords is not None:
                        out[label] = coords
        else:
            result_obj = data.get('result')
            if isinstance(result_obj, dict):
                _ingest_mapping(result_obj)
            result_plain = data.get('result_plain')
            if isinstance(result_plain, dict):
                _ingest_mapping(result_plain)
            elif isinstance(result_plain, str):
                try:
                    parsed = json.loads(result_plain)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    _ingest_mapping(parsed)
            result_json = data.get('result_json')
            if isinstance(result_json, str):
                try:
                    parsed = json.loads(result_json)
                except (TypeError, ValueError):
                    parsed = None
                if isinstance(parsed, dict):
                    _ingest_mapping(parsed)
            _ingest_mapping(data)
    return out

def _merge_landmarks(*dicts: Dict[str, List[float]]) -> Dict[str, List[float]]:
    out = {};  [out.update(d or {}) for d in dicts];  return out

# ---- Public API ----
def make_doctor_cn_simple(
    landmarks: Dict[str, List[float]],
    frame_ops: Dict[str, Any],
    cfg: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    医生端简洁版导出：复用各 compute_* 指标，仅重组输出字段。
    """
    cfg = cfg or {}
    frame, ops, wt = frame_ops['frame'], frame_ops['ops'], frame_ops['wt']
    wt_records = frame_ops.get('wt_records') or {}
    extended = bool(cfg.get('include_extended_fields', False))

    arch_form = compute_arch_form(landmarks, frame, ops)
    arch_width = compute_arch_width(landmarks, ops, cfg=cfg)
    bolton = compute_bolton(landmarks, ops, wt, wt_records=wt_records)
    canine = compute_canine_relationship(landmarks, ops)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_info = compute_spee(landmarks, ops, cfg=cfg)
    midline = compute_midline_alignment(landmarks, ops)
    molar = compute_molar_relationship(landmarks, ops, wt)
    overbite = compute_overbite(landmarks, ops)
    overjet = compute_overjet(landmarks, ops)

    def _ok(d: Any, key: Optional[str] = None) -> bool:
        if not isinstance(d, dict):
            return False
        if d.get('quality') == 'missing':
            return False
        if key is not None and d.get(key) is None:
            return False
        return True

    out: Dict[str, Any] = {}

    # 1) 牙弓形态
    out['Arch_Form'] = arch_form.get('form') if _ok(arch_form) else '缺失'

    # 2) 牙弓宽度
    if _ok(arch_width):
        j = arch_width.get('judge') or {}

        def _line(side_label: str, j_side: Dict[str, str]) -> str:
            def _pick(seg: str) -> str:
                return j_side.get(seg, '缺失')
            return f"{side_label}：前段{_pick('anterior')}、中段{_pick('middle')}、后段{_pick('posterior')}"

        if j:
            out['Arch_Width'] = "；".join([
                _line('上颌', j.get('upper', {})),
                _line('下颌', j.get('lower', {})),
            ])
        else:
            # 兼容旧版：回退到 U-L 差的三段描述
            diff = arch_width.get('diff_UL_mm') or {}
            segment_alias = {'anterior_mm': '前段', 'middle_mm': '中段', 'posterior_mm': '后段'}

            def _segment_desc(key: str, label: str) -> str:
                val = diff.get(key)
                if not isinstance(val, (int, float)):
                    return f"{label}缺失"
                if val <= -1.0:
                    return f"{label}偏窄"
                if val >= 1.0:
                    return f"{label}偏宽"
                return f"{label}正常"

            segments = [_segment_desc(k, lbl) for k, lbl in segment_alias.items()]
            base = '上牙弓较窄' if arch_width.get('upper_is_narrow') else '正常'
            out['Arch_Width'] = f"{base} {'、'.join(segments)}".strip()
    else:
        out['Arch_Width'] = '缺失'

    # 3) Bolton（前牙比换算成小数，同时输出原始比值/差值）
    anterior_ratio_txt = '缺失'
    overall_ratio_txt = '缺失'
    anterior_detail = bolton.get('anterior') if _ok(bolton, 'anterior') else None
    overall_detail = bolton.get('overall') if _ok(bolton, 'overall') else None

    if anterior_detail:
        ratio_val = anterior_detail.get('ratio')
        if isinstance(ratio_val, (int, float)):
            anterior_ratio_txt = f"{ratio_val / 100.0:.2f}"
        if extended:
            discrep_val = anterior_detail.get('discrep_mm')
            out['Bolton_Anterior_Ratio'] = round(ratio_val, 2) if isinstance(ratio_val, (int, float)) else None
            out['Bolton_Anterior_Discrepancy_mm'] = round(discrep_val, 2) if isinstance(discrep_val, (int, float)) else None
            out['Bolton_Anterior_Status'] = anterior_detail.get('status') or '缺失'
    elif extended:
        out['Bolton_Anterior_Ratio'] = None
        out['Bolton_Anterior_Discrepancy_mm'] = None
        out['Bolton_Anterior_Status'] = '缺失'

    if overall_detail:
        ratio_val = overall_detail.get('ratio')
        if isinstance(ratio_val, (int, float)):
            overall_ratio_txt = f"{ratio_val / 100.0:.2f}"
    if extended:
        if overall_detail:
            ratio_val = overall_detail.get('ratio')
            discrep_val = overall_detail.get('discrep_mm')
            out['Bolton_Overall_Ratio'] = round(ratio_val, 2) if isinstance(ratio_val, (int, float)) else None
            out['Bolton_Overall_Discrepancy_mm'] = round(discrep_val, 2) if isinstance(discrep_val, (int, float)) else None
            out['Bolton_Overall_Status'] = overall_detail.get('status') or '缺失'
        else:
            out['Bolton_Overall_Ratio'] = None
            out['Bolton_Overall_Discrepancy_mm'] = None
            out['Bolton_Overall_Status'] = '缺失'

    out['Bolton_Ratio'] = f"前牙比:{anterior_ratio_txt} 全牙比:{overall_ratio_txt}"

    # 4) 尖牙关系
    if _ok(canine):
        right = (canine.get('right') or {}).get('class')
        left = (canine.get('left') or {}).get('class')
        out['Canine_Relationship_Right'] = right if right else '缺失'
        out['Canine_Relationship_Left'] = left if left else '缺失'
    else:
        out['Canine_Relationship_Right'] = '缺失'
        out['Canine_Relationship_Left'] = '缺失'

    # 5) 锁牙合
    if _ok(crossbite):
        right_cb = crossbite.get('right') or {}
        left_cb = crossbite.get('left') or {}

        def _is_clear(section: Dict[str, Any]) -> bool:
            return section.get('premolar') == '无' and section.get('m1') == '无'

        if _is_clear(right_cb) and _is_clear(left_cb):
            out['Crossbite'] = '无'
        else:
            def _side_text(label: str, section: Dict[str, Any]) -> Optional[str]:
                if not section:
                    return None
                premolar = section.get('premolar')
                molar = section.get('m1')
                parts: List[str] = []
                if premolar in ('正锁', '反锁'):
                    parts.append('第一前磨牙')
                if molar in ('正锁', '反锁'):
                    parts.append('第一磨牙')
                if not parts:
                    return None
                typ = molar if molar in ('正锁', '反锁') else premolar
                if typ is None:
                    return None
                joined = '、'.join(parts)
                return f"{label}侧{typ}（累及{joined}）"

            segments = [
                _side_text('右', right_cb),
                _side_text('左', left_cb),
            ]
            segments = [seg for seg in segments if seg]
            out['Crossbite'] = '；'.join(segments) if segments else '无'
    else:
        out['Crossbite'] = '缺失'

    # 6) 拥挤度（以 ALD 数值表示）
    if _ok(crowding):
        upper_val = crowding.get('upper') or {}
        lower_val = crowding.get('lower') or {}
        upper_ald = upper_val.get('ald_mm') if isinstance(upper_val, dict) else None
        lower_ald = lower_val.get('ald_mm') if isinstance(lower_val, dict) else None
        out['Crowding_Up'] = float(upper_ald) if isinstance(upper_ald, (int, float)) else None
        out['Crowding_Down'] = float(lower_ald) if isinstance(lower_ald, (int, float)) else None
        if extended:
            out['Crowding_Up_Detail'] = upper_val.get('text') if isinstance(upper_val, dict) else None
            out['Crowding_Down_Detail'] = lower_val.get('text') if isinstance(lower_val, dict) else None
    else:
        out['Crowding_Up'] = None
        out['Crowding_Down'] = None
        if extended:
            out['Crowding_Up_Detail'] = None
            out['Crowding_Down_Detail'] = None

    # 7) Spee
    spee_value = spee_info.get('value_mm') if isinstance(spee_info, dict) else None
    if isinstance(spee_value, (int, float)):
        out['Curve_of_Spee'] = f"{spee_value:.1f}mm"
    else:
        out['Curve_of_Spee'] = '缺失'
    if extended:
        out['Curve_of_Spee_Trimmed'] = bool(spee_info.get('trimmed')) if isinstance(spee_info, dict) else None
        if isinstance(spee_info, dict):
            primary_side = spee_info.get('primary_side')
            if primary_side == 'right':
                out['Curve_of_Spee_Primary_Side'] = '右'
            elif primary_side == 'left':
                out['Curve_of_Spee_Primary_Side'] = '左'
            else:
                out['Curve_of_Spee_Primary_Side'] = primary_side
        else:
            out['Curve_of_Spee_Primary_Side'] = None

    # 8) 中线
    if _ok(midline):
        raw = midline.get('summary_text', '')
        text = '缺失'
        if isinstance(raw, str):
            if '*:' in raw:
                text = raw.split('*:', 1)[1].strip()
            elif ':' in raw:
                text = raw.split(':', 1)[1].strip()
            else:
                text = raw.strip() or '缺失'
        out['Midline_Alignment'] = text
    else:
        out['Midline_Alignment'] = '缺失'

    # 9) 第一磨牙关系
    if _ok(molar):
        right_info = molar.get('right') or {}
        left_info = molar.get('left') or {}
        right_molar = right_info.get('label')
        left_molar = left_info.get('label')
        out['Molar_Relationship_Right'] = right_molar if right_molar else '缺失'
        out['Molar_Relationship_Left'] = left_molar if left_molar else '缺失'
        if extended:
            out['Molar_Relationship_Right_Offset_mm'] = right_info.get('offset_mm') if isinstance(right_info.get('offset_mm'), (int, float)) else None
            out['Molar_Relationship_Right_q_mm'] = right_info.get('q_mm') if isinstance(right_info.get('q_mm'), (int, float)) else None
            out['Molar_Relationship_Right_h_mm'] = right_info.get('h_mm') if isinstance(right_info.get('h_mm'), (int, float)) else None
            out['Molar_Relationship_Left_Offset_mm'] = left_info.get('offset_mm') if isinstance(left_info.get('offset_mm'), (int, float)) else None
            out['Molar_Relationship_Left_q_mm'] = left_info.get('q_mm') if isinstance(left_info.get('q_mm'), (int, float)) else None
            out['Molar_Relationship_Left_h_mm'] = left_info.get('h_mm') if isinstance(left_info.get('h_mm'), (int, float)) else None
    else:
        out['Molar_Relationship_Right'] = '缺失'
        out['Molar_Relationship_Left'] = '缺失'
        if extended:
            out['Molar_Relationship_Right_Offset_mm'] = None
            out['Molar_Relationship_Right_q_mm'] = None
            out['Molar_Relationship_Right_h_mm'] = None
            out['Molar_Relationship_Left_Offset_mm'] = None
            out['Molar_Relationship_Left_q_mm'] = None
            out['Molar_Relationship_Left_h_mm'] = None

    # 10) Overbite
    if _ok(overbite):
        category = overbite.get('category')
        value = overbite.get('value_mm')
        if isinstance(value, (int, float)) and isinstance(category, str) and category:
            magnitude = abs(float(value))
            out['Overbite'] = f"{magnitude:.1f}mm {category}"
        else:
            out['Overbite'] = '缺失'
    else:
        out['Overbite'] = '缺失'

    # 11) Overjet
    if _ok(overjet):
        value = overjet.get('value_mm')
        category = overjet.get('category')
        if isinstance(value, (int, float)) and isinstance(category, str) and category:
            out['Overjet'] = f"{abs(float(value)):.1f}mm {category}"
        else:
            out['Overjet'] = '缺失'
    else:
        out['Overjet'] = '缺失'

    warnings: List[str] = list(frame_ops.get('warnings') or [])
    stage_label = {'mr_dr': 'mr↔dr', 'm_dr': 'm↔dr', None: '初测'}
    unique_teeth = sorted({key[0] for key in wt_records.keys()}) if isinstance(wt_records, dict) else []

    for tooth in unique_teeth:
        rec = None
        if isinstance(wt_records, dict):
            rec = wt_records.get((tooth, True)) or wt_records.get((tooth, False))
        if not rec:
            continue
        status = rec.get('status')
        if status not in ('fallback', 'removed', 'warning'):
            continue
        init = rec.get('initial_width_mm')
        thresh = rec.get('threshold_mm')
        if not isinstance(init, (int, float)) or not isinstance(thresh, (int, float)):
            continue
        if status == 'fallback':
            final = rec.get('final_width_mm')
            stage_key = rec.get('fallback_stage')
            stage_txt = stage_label.get(stage_key, stage_key or '备用点')
            if isinstance(final, (int, float)):
                warnings.append(f"{tooth} 冠宽初测 {init:.2f}mm < 阈值 {thresh:.2f}mm，回退 {stage_txt} → {final:.2f}mm")
            else:
                warnings.append(f"{tooth} 冠宽初测 {init:.2f}mm < 阈值 {thresh:.2f}mm，已回退 {stage_txt}")
        elif status == 'warning':
            final = rec.get('final_width_mm')
            stage_key = rec.get('fallback_stage')
            stage_txt = stage_label.get(stage_key, stage_key or '备用点')
            if isinstance(final, (int, float)):
                warnings.append(f"{tooth} 冠宽初测 {init:.2f}mm < 阈值 {thresh:.2f}mm，回退 {stage_txt} 后仍偏小（采用 {final:.2f}mm）")
            else:
                warnings.append(f"{tooth} 冠宽初测 {init:.2f}mm < 阈值 {thresh:.2f}mm，缺乏可靠备用点，保留原始测量")
        elif status == 'removed':
            invalid = rec.get('invalid_width_mm')
            if not isinstance(invalid, (int, float)):
                invalid = init
            warnings.append(f"{tooth} 冠宽初测 {invalid:.2f}mm < 阈值 {thresh:.2f}mm，mr↔dr/m↔dr 回退均未通过，已剔除")

    if isinstance(bolton, dict):
        removed_teeth = bolton.get('removed_teeth')
        if removed_teeth:
            warnings.append("Bolton 比计算剔除牙位：" + '、'.join(removed_teeth))

    if isinstance(spee_info, dict) and spee_info.get('trimmed'):
        trimmed_sides = spee_info.get('trimmed_sides') or []
        side_map = {'right': '右', 'left': '左'}
        if trimmed_sides:
            sides_label = '、'.join(f"{side_map.get(s, s)}侧" for s in trimmed_sides)
        else:
            sides_label = '双侧'
        warnings.append(f"Spee 曲线启用 95% 分位去极值（{sides_label}）")

    out['warnings'] = warnings

    return out

def _collect_landmarks(sources: Iterable[Any]) -> Dict[str, List[float]]:
    merged: Dict[str, List[float]] = {}
    for src in sources:
        if src is None:
            continue
        lm = _load_landmarks_json(src)
        if not lm:
            continue
        merged.update(lm)
    return merged


def generate_metrics(
    landmarks_payload: Union[Dict[str, Any], List[Any], Tuple[Any, ...], str, os.PathLike],
    out_path: str = "",
    cfg: Optional[Dict] = None
) -> Dict:
    """
    输入：点位字典（如 dots.json）、Slicer Markups 字典，或上述对象组成的列表/元组。
    若传入列表，将依次合并为统一的 {label: [x, y, z]}。
    """
    if isinstance(landmarks_payload, (list, tuple)):
        landmarks = _collect_landmarks(landmarks_payload)
    else:
        landmarks = _load_landmarks_json(landmarks_payload)
    return _generate_metrics_from_landmarks(landmarks, out_path=out_path, cfg=cfg)


def _generate_metrics_from_landmarks(
    landmarks: Dict[str, List[float]],
    out_path: str = "",
    cfg: Optional[Dict] = None
) -> Dict:
    # 2) 前置坐标系（严格 I/L/R）
    frame_ops = build_module0(landmarks)
    if frame_ops.get('frame') is None or frame_ops.get('ops') is None:
        kv = {"错误": "坐标系缺失，无法生成报告"}
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(kv, f, ensure_ascii=False, indent=2)
        return kv

    cfg = cfg or {}
    # 仅保留中文医生端键值对输出，统一口径。
    kv = make_doctor_cn_simple(landmarks, frame_ops, cfg=cfg)
    kv.pop('warnings', None)

    # 4) 可选落盘
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kv, f, ensure_ascii=False, indent=2)
    return kv


def generate_metrics_from_mapping(
    landmarks_payload: Dict[str, Any],
    out_path: str = "",
    cfg: Optional[Dict] = None
) -> Dict:
    """
    输入：已经合并的 landmarks 字典（可包含 result/result_plain）。
    """
    return generate_metrics(landmarks_payload, out_path=out_path, cfg=cfg)


def generate_metrics_from_json_text(
    json_text: str,
    out_path: str = "",
    cfg: Optional[Dict] = None
) -> Dict:
    """
    输入：JSON 字符串（与 API 返回格式一致）。
    """
    payload = json.loads(json_text)
    return generate_metrics(payload, out_path=out_path, cfg=cfg)

# 可选 CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ortho analysis → brief key-value JSON")
    ap.add_argument(
        "sources",
        nargs="+",
        help="一个或多个点位来源（支持 dots.json、Slicer Markups JSON 等路径）。",
    )
    ap.add_argument("--out", required=True, help="输出 JSON 路径")
    args = ap.parse_args()
    kv = generate_metrics(args.sources, out_path=args.out)
    print(f"saved to: {args.out} ({len(kv)} items)")
