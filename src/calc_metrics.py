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

def _lower_midline_anchor_for_relationships(
    landmarks: Dict[str, List[float]], ops: Dict, cfg: Optional[Dict[str, Any]] = None
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    估计 *用于定向* 的“下颌前段中线锚点” A_L（不影响 compute_midline_alignment）：
      使用三对对称点（避免 mc/dc）：
        (41*, 31*), (42*, 32*), (43m, 33m)
      在 P 面求中点并做加权平均；权重=1/(RMSE^2 + eps0^2)，RMSE 可从 cfg['pt_error_mm'] 注入（键如 '31ma','33m'）。
    """
    cfg = cfg or {}
    pt_error = cfg.get('pt_error_mm', {}) or {}
    eps0 = float(cfg.get('midline_anchor_eps0_mm', 0.2))

    def _safe_v(p):
        if isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3:
            try: return np.asarray(p, float)
            except Exception: return None
        return None

    def _label_err(label: str, default_kind: float) -> float:
        if label in pt_error: return float(pt_error[label])
        # 粗略后备：按后缀类型给个缺省噪声级
        for suf, val in (('mc',1.0),('dc',0.9),('ma',0.7),('da',0.7),('mb',0.7),('db',0.7),('ml',0.7),('dl',0.7),('m',0.6),('b',0.6)):
            if label.endswith(suf): return val
        return default_kind

    def _pair_center(left_label: str, right_label: str, kind: str) -> Tuple[Optional[np.ndarray], float, List[str]]:
        used = []
        if kind == 'star':
            # 用你已有的 _star_mid；内部已处理缺失回退到 m
            _, Ls = _star_mid(landmarks, left_label[:-1])   # '41*' -> '41'
            _, Rs = _star_mid(landmarks, right_label[:-1])
            L, R = Ls, Rs
            if L is not None: used.append(left_label)
            if R is not None: used.append(right_label)
            eL = min(_label_err(left_label[:-1]+'ma', 0.7), _label_err(left_label[:-1]+'da', 0.7))
            eR = min(_label_err(right_label[:-1]+'ma',0.7), _label_err(right_label[:-1]+'da',0.7))
        else:
            L = _safe_v(landmarks.get(left_label))
            R = _safe_v(landmarks.get(right_label))
            if L is not None: used.append(left_label)
            if R is not None: used.append(right_label)
            eL = _label_err(left_label, 0.6 if kind in ('m','b') else 0.7)
            eR = _label_err(right_label, 0.6 if kind in ('m','b') else 0.7)

        if (L is None) or (R is None):
            return None, 0.0, used
        center = 0.5*(ops['projP'](L) + ops['projP'](R))
        eps = 0.5*(eL + eR)
        w = 1.0 / (eps*eps + eps0*eps0)
        return center, float(w), used

    pairs = [('41*','31*','star'), ('42*','32*','star'), ('43m','33m','m')]
    centers, weights, used_pairs = [], [], []
    for L, R, kind in pairs:
        c, w, used = _pair_center(L, R, kind)
        used_pairs.append({'pair': (L,R), 'kind': kind, 'weight': w, 'used': used})
        if c is not None and w > 0: centers.append(c*w); weights.append(w)

    if not weights or sum(weights) <= 0:
        return None, {'status': 'missing', 'used_pairs': used_pairs}
    A_L = np.sum(centers, axis=0) / float(sum(weights))
    return A_L, {'status': 'ok', 'used_pairs': used_pairs, 'eps0': eps0}

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

def _apply_polarity_guard(
    landmarks: Dict[str, List[float]],
    cfg: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    牙位极性快检：针对 11/21/31/41 的 mc↔dc 极性异常执行就地修补。
    规则：
      - 预判阈值 δx（默认 2.5mm），以及最小间距阈值（默认 2.0mm）。
      - 当触发条件满足且间距足够时，交换 mc 与 dc 的坐标。
    返回触发记录列表，便于后续记录与排查。
    """
    cfg = cfg or {}
    swap_dx = float(cfg.get('polarity_swap_dx_mm', 2.5))
    min_gap = float(cfg.get('polarity_min_gap_mm', 2.0))
    actions: List[Dict[str, Any]] = []

    def _swap_if_needed(tooth: str, cond) -> None:
        key_mc = f"{tooth}mc"
        key_dc = f"{tooth}dc"
        pt_mc = landmarks.get(key_mc)
        pt_dc = landmarks.get(key_dc)
        if not (_is_xyz(pt_mc) and _is_xyz(pt_dc)):
            return
        x_mc = float(pt_mc[0])
        x_dc = float(pt_dc[0])
        gap = abs(x_mc - x_dc)
        if gap < min_gap:
            return
        if not cond(x_mc, x_dc):
            return
        landmarks[key_mc], landmarks[key_dc] = pt_dc, pt_mc
        actions.append({
            'tooth': tooth,
            'action': 'swap_mc_dc',
            'x_mc_before_mm': x_mc,
            'x_dc_before_mm': x_dc,
            'delta_mm': gap,
            'swap_dx_mm': swap_dx,
        })

    _swap_if_needed('11', lambda x_mc, x_dc: x_mc <= x_dc - swap_dx)
    _swap_if_needed('21', lambda x_mc, x_dc: x_mc >= x_dc + swap_dx)
    _swap_if_needed('31', lambda x_mc, x_dc: x_mc >= x_dc + swap_dx)
    _swap_if_needed('41', lambda x_mc, x_dc: x_mc <= x_dc - swap_dx)
    return actions

_JAW_PREFIX_RE = re.compile(r"^(upper|lower)[_\-]?(\d.*)$", flags=re.IGNORECASE)

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
def compute_arch_form(landmarks, frame, ops, dec=2, cfg=None):
    cfg = cfg or {}
    def _v(nm): 
        p = landmarks.get(nm); 
        return np.asarray(p, float) if isinstance(p,(list,tuple,np.ndarray)) and len(p)==3 else None
    def _mid2(a,b):
        pa,pb=_v(a),_v(b); 
        return 0.5*(pa+pb) if (pa is not None and pb is not None) else None
    def _dist_to_line_P(P, A, B):
        # all in 3D → 投影到 P
        Ap, Bp, Pp = ops['projP'](A), ops['projP'](B), ops['projP'](P)
        u = Bp - Ap; v = Pp - Ap
        if np.linalg.norm(u) < 1e-9: return None
        t = np.dot(u, v)/np.dot(u,u); Q = Ap + t*u
        return float(np.linalg.norm(Pp - Q))

    # --- R：宽/深比 ---
    UCI = _mid2('11ma','21ma'); UCI2 = _mid2('11da','21da')
    if UCI is None or UCI2 is None:
        # 回退：星点（脚本已有 *_star_mid）
        _, s11 = _star_mid(landmarks, '11'); _, s21 = _star_mid(landmarks, '21')
        if s11 is None or s21 is None: 
            return {'form':'缺失','summary_text':'Arch_Form_v2*: 缺失','quality':'missing'}
        UCI = 0.5*(s11+s21)
    C_R, C_L = _v('13m'), _v('23m')
    _tmp = _v('16mc')
    M_R = _tmp if _tmp is not None else _v('16mb')
    _tmp = _v('26mc')
    M_L = _tmp if _tmp is not None else _v('26mb')
    if any(x is None for x in (C_R, C_L, M_R, M_L)):
        return {'form':'缺失','summary_text':'Arch_Form_v2*: 缺失','quality':'missing'}
    Wc = ops['H'](C_R, C_L);  Wm = ops['H'](M_R, M_L)
    Dc = _dist_to_line_P(UCI, C_R, C_L);  Dm = _dist_to_line_P(UCI, M_R, M_L)
    if any(x is None for x in (Wc, Wm, Dc, Dm)) or min(Wc, Wm, Dc, Dm) < 1e-6:
        return {'form':'缺失','summary_text':'Arch_Form_v2*: 缺失','quality':'missing'}
    R = (Wc/Wm)/(Dc/Dm)

    # --- K：前/后段曲率比（基于牙间中点链） ---
    # 上弓 6–6 牙序
    U_TEETH_66 = ['16','15','14','13','12','11','21','22','23','24','25','26']
    # 构造节点：端点 16mc/26mc；中间 0.5*(a.dc + b.mc)；跨中线 0.5*(11mc + 21mc)
    nodes = []
    _tmp = _v('16mc')
    A = _tmp if _tmp is not None else _v('16mb')
    _tmp = _v('26mc')
    B = _tmp if _tmp is not None else _v('26mb')
    if A is None or B is None: 
        return {'form':'缺失','summary_text':'Arch_Form_v2*: 缺失','quality':'missing'}
    nodes.append(A)
    for a,b in zip(U_TEETH_66[:-1], U_TEETH_66[1:]):
        if a=='11' and b=='21':
            _tmpm = _mid2('11mc','21mc')
            mid = _tmpm if _tmpm is not None else _mid2('11m','21m')
        else:
            pa = _v(f'{a}dc')
            if pa is None:
                pa = _v(f'{a}dr')
            if pa is None:
                pa = _v(f'{a}m')
            pb = _v(f'{b}mc')
            if pb is None:
                pb = _v(f'{b}mr')
            if pb is None:
                pb = _v(f'{b}m')
            mid = 0.5*(pa+pb) if (pa is not None and pb is not None) else None
        if mid is None: return {'form':'缺失','summary_text':'Arch_Form_v2*: 缺失','quality':'missing'}
        nodes.append(mid)
    nodes.append(B)

    # 投影到 P → 2D
    XY = np.array([[ops['x'](ops['projP'](p)), ops['y'](ops['projP'](p))] for p in nodes], float)
    # 等弧长插值（Catmull-Rom centripetal）
    def _catmull_rom(xy, n=300, alpha=0.5):
        P = xy.copy()
        if len(P) < 4:  # 端点复制保证至少4点
            P = np.vstack([P[0], P, P[-1], P[-1]])
        t = [0.0]
        for i in range(1, len(P)):
            t.append(t[-1] + np.linalg.norm(P[i]-P[i-1])**alpha)
        t = np.array(t)
        ts = np.linspace(t[1], t[-2], n)
        out=[]
        for u in ts:
            # 找到区间 k: t[k] <= u <= t[k+1], 取 p_{k-1..k+2}
            k = np.searchsorted(t, u) - 1
            k = max(1, min(k, len(P)-3))
            t0,t1,t2,t3 = t[k-1],t[k],t[k+1],t[k+2]
            p0,p1,p2,p3 = P[k-1],P[k],P[k+1],P[k+2]
            def lerp(a,b,ta,tb,tu): 
                w = (tu-ta)/(tb-ta); return (1-w)*a + w*b
            A1 = lerp(p0,p1,t0,t1,u); A2 = lerp(p1,p2,t1,t2,u); A3 = lerp(p2,p3,t2,t3,u)
            B1 = lerp(A1,A2,t0,t2,u); B2 = lerp(A2,A3,t1,t3,u)
            C  = lerp(B1,B2,t1,t2,u)
            out.append(C)
        return np.asarray(out)
    S = _catmull_rom(XY, n=400)

    # 曲率（离散二阶差分近似）
    V1 = np.gradient(S, axis=0); V2 = np.gradient(V1, axis=0)
    speed = np.linalg.norm(V1, axis=1) + 1e-9
    kappa = np.abs(V1[:,0]*V2[:,1] - V1[:,1]*V2[:,0]) / (speed**3)
    # 分段中位数
    n = len(kappa); 
    front = kappa[int(n*0.40):int(n*0.60)];  post = np.hstack([kappa[:int(n*0.15)], kappa[int(n*0.85):]])
    if len(front)==0 or len(post)==0: 
        return {'form':'缺失','summary_text':'Arch_Form_v2*: 缺失','quality':'missing'}
    K = float(np.median(front) / (np.median(post)+1e-9))

    # 组合判型（无监督/弱监督更稳的阈值带）
    # 建议：在你们已有病例上对 R 与 K 分别做 1D K-means(k=3)得三段阈值；这里给一个保守初始带
    r_lo, r_hi = 0.95, 1.05
    k_lo, k_hi = 0.90, 1.15
    if (R <= r_lo and K >= k_hi): form = '尖圆形'
    elif (R >= r_hi and K <= k_lo): form = '方圆形'
    else: form = '卵圆形'

    return {
        'form': form,
        'indices': {'R': round(R,3), 'K': round(K,3)},
        'summary_text': f"Arch_Form_v2*: {form} (R={R:.3f}, K={K:.3f})",
        'quality': 'ok'
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
    landmarks: Dict[str, List[float]], ops: Dict, dec: int = 2, cfg: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    【新版逻辑】
    计算尖牙关系：比较上颌尖牙牙尖 (U) 的 Y 坐标（前后向）与
    下颌尖牙远中点 (L1) 和第一前磨牙近中点 (L2) 连线中点 (Target) 的 Y 坐标。
    
    坐标系：Y 轴（前后），前为正 (ops['y'])。

    偏移量 offset = y(U) - y(Target)
    
    - offset > 0: 上牙尖 U 在 目标邻接点 Target 的前方 → 远中关系 (Class II)
    - offset < 0: 上牙尖 U 在 目标邻接点 Target 的后方 → 近中关系 (Class III)
    - offset ≈ 0: 上牙尖 U 落在 目标邻接点 Target 处 → 中性关系 (Class I)
    """
    cfg = cfg or {}
    
    # 容差1：中性关系的容差（例如 ±0.5mm）
    # 沿用旧版脚本的 cfg 设置
    neutral_tol = float(cfg.get("canine_neutral_tol_mm", 0.5))
    
    # 容差2：区分 "尖对尖" 和 "完全" 的阈值
    # 默认为 2.5mm，可自定义
    cusp_tol = float(cfg.get("canine_cusp_tol_mm", 2.5))

    # _v 是 calc_metrics.py 中的全局辅助函数，这里假设它可用
    # def _v(a) -> Optional[np.ndarray]: ...

    def _one(side: str, U_label: str, L_canine_label: str, L_premolar_label: str):
        """计算单侧关系"""
        dbg = {'side': side, 'labels': [U_label, L_canine_label, L_premolar_label]}

        # 1. 获取所有需要的点
        U = _v(landmarks.get(U_label))
        L1 = _v(landmarks.get(L_canine_label))
        L2 = _v(landmarks.get(L_premolar_label))

        # 2. 检查缺失
        if U is None or L1 is None or L2 is None:
            missing = []
            if U is None: missing.append(U_label)
            if L1 is None: missing.append(L_canine_label)
            if L2 is None: missing.append(L_premolar_label)
            dbg['status'] = f"missing_landmarks: {','.join(missing)}"
            return None, dbg

        # 3. 仅比较 Y 轴（前后）坐标
        try:
            y_U = float(ops['y'](U))
            y_L1 = float(ops['y'](L1))
            y_L2 = float(ops['y'](L2))
        except Exception as e:
            dbg['status'] = f"ops_y_failed: {e}"
            return None, dbg

        # 4. 计算下颌目标邻接点 (e.g., 43dc 和 44mc) 的 Y 轴中点
        y_Target = 0.5 * (y_L1 + y_L2)

        # 5. 计算偏移量 (上牙尖 - 目标邻接点)
        offset_mm = y_U - y_Target
        
        # 6. 分类
        #    offset > 0: U 在 Target 前方 -> 远中 (Class II)
        #    offset < 0: U 在 Target 后方 -> 近中 (Class III)
        klass = '中性'
        if abs(offset_mm) <= neutral_tol:
            klass = '中性'
        elif offset_mm > neutral_tol:
            # 远中 (Class II)
            if offset_mm < cusp_tol:
                klass = '中性偏远中' # (End-to-end Class II)
            else:
                klass = '远中' # (Full cusp Class II)
        elif offset_mm < -neutral_tol:
            # 近中 (Class III)
            if abs(offset_mm) < cusp_tol:
                klass = '中性偏近中' # (End-to-end Class III)
            else:
                klass = '近中' # (Full cusp Class III)

        detail = {
            'class': klass,
            'offset_mm': float(np.round(offset_mm, dec)),
            'raw_offset_mm': offset_mm,
            'upper_label': U_label,
            'lower_target_labels': [L_canine_label, L_premolar_label],
            'y_U_mm': float(np.round(y_U, dec)),
            'y_Target_mm': float(np.round(y_Target, dec))
        }
        dbg.update({'status': 'ok', 'offset_mm': offset_mm, 'method': 'Y_axis_vs_embrasure'})
        return detail, dbg

    # --- 主流程 ---
    # 计算右侧: 13m vs (43dc + 44mc)
    right, dbgR = _one('right', '13m', '43dc', '44mc')
    # 计算左侧: 23m vs (33dc + 34mc)
    left,  dbgL = _one('left',  '23m', '33dc', '34mc')
    
    parts = [f"右侧{right['class']}" if right else "右侧缺失",
             f"左侧{left['class']}"  if left  else "左侧缺失"]
    quality = 'ok' if (right and left) else ('fallback' if (right or left) else 'missing')
    
    return {
        'right': right, 
        'left': left,
        'summary_text': "Canine_Relationship_尖牙关系*: " + " ".join(parts),
        'quality': quality,
        'debug': {'right': dbgR, 'left': dbgL},
        'thresholds': {'neutral_tol_mm': neutral_tol, 'cusp_tol_mm': cusp_tol},
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

def compute_crowding(landmarks, ops, wt_callable, dec=1, cfg=None):
    cfg = cfg or {}
    def _v(nm): 
        p = landmarks.get(nm); 
        return np.asarray(p, float) if isinstance(p,(list,tuple,np.ndarray)) and len(p)==3 else None
    def _nodes_6to6(teeth, left_mc, right_mc):
        A = _v(right_mc); B = _v(left_mc)
        if A is None or B is None: return None
        nodes=[A]
        for a,b in zip(teeth[:-1], teeth[1:]):
            if a.endswith('11') and b.endswith('21'):
                _tmp = _v('11mc')
                leftp = _tmp if _tmp is not None else _v('11m')
                _tmp2 = _v('21mc')
                rightp = _tmp2 if _tmp2 is not None else _v('21m')
                if leftp is None or rightp is None:
                    return None
                mid = 0.5 * (leftp + rightp)
            else:
                pa = _v(f"{a}dc")
                if pa is None:
                    pa = _v(f"{a}dr")
                if pa is None:
                    pa = _v(f"{a}m")
                pb = _v(f"{b}mc")
                if pb is None:
                    pb = _v(f"{b}mr")
                if pb is None:
                    pb = _v(f"{b}m")
                if pa is None or pb is None:
                    return None
                mid = 0.5 * (pa + pb)
            nodes.append(mid)
        nodes.append(B)
        return nodes

    def _arc_len(nodes3d):
        XY = np.array([[ops['x'](ops['projP'](p)), ops['y'](ops['projP'](p))] for p in nodes3d], float)
        # 折线
        L_poly = float(np.sum(np.linalg.norm(np.diff(XY, axis=0), axis=1)))
        # Catmull‑Rom
        def _cr(xy, n=400, alpha=0.5):
            P = xy.copy()
            if len(P) < 4: P = np.vstack([P[0],P,P[-1],P[-1]])
            t=[0.0]
            for i in range(1,len(P)): t.append(t[-1]+np.linalg.norm(P[i]-P[i-1])**alpha)
            t=np.array(t); ts=np.linspace(t[1],t[-2],n)
            out=[]
            for u in ts:
                k=max(1, min(np.searchsorted(t,u)-1, len(P)-3))
                t0,t1,t2,t3=t[k-1],t[k],t[k+1],t[k+2]; p0,p1,p2,p3=P[k-1],P[k],P[k+1],P[k+2]
                def L(a,b,ta,tb,tu): w=(tu-ta)/(tb-ta); return (1-w)*a+w*b
                A1,A2,A3=L(p0,p1,t0,t1,u), L(p1,p2,t1,t2,u), L(p2,p3,t2,t3,u)
                B1,B2=L(A1,A2,t0,t2,u), L(A2,A3,t1,t3,u); C=L(B1,B2,t1,t2,u); out.append(C)
            S=np.asarray(out)
            return float(np.sum(np.linalg.norm(np.diff(S,axis=0),axis=1)))
        L_spline = _cr(XY)
        # 守护：样条不得短于折线，且不应 >1.10×折线
        if (L_spline < L_poly) or (L_spline > 1.10*L_poly): 
            return L_poly, 'polyline', {'fit_len_mm': L_spline, 'polyline_len_mm': L_poly}
        return L_spline, 'spline', {'fit_len_mm': L_spline, 'polyline_len_mm': L_poly}

    def _sum_width(teeth):
        s=0.0; miss=[]
        for t in teeth:
            w = wt_callable(t, allow_fallback=False)  # 严格 mc↔dc
            if w is None: miss.append(t)
            else: s += float(w)
        return (None, miss) if miss else (s, [])

    U_teeth = ['16','15','14','13','12','11','21','22','23','24','25','26']
    L_teeth = ['36','35','34','33','32','31','41','42','43','44','45','46']
    U_nodes = _nodes_6to6(U_teeth, left_mc='26mc', right_mc='16mc')
    L_nodes = _nodes_6to6(L_teeth, left_mc='36mc', right_mc='46mc')
    if U_nodes is None and L_nodes is None:
        return {'upper':None,'lower':None,'summary_text':'Crowding_v2*: 缺失','quality':'missing'}

    out_detail = {}
    def _one(label, nodes, teeth):
        if nodes is None:
            return None, f"{label}链缺失"
        L, mode, info = _arc_len(nodes)
        S, miss = _sum_width(teeth)
        if S is None: 
            return None, f"{label}冠宽缺失({','.join(miss)})"
        ald = S - L
        txt = f"6-6{'拥挤' if ald>0 else '间隙' if ald<0 else '拥挤/间隙0.0'}{abs(ald):.{dec}f}mm(Σ{S:.1f}mm, L{L:.1f}mm, {mode})"
        return {'ald_mm': round(ald,dec), 'sum_width_mm': round(S,dec), 'arc_len_mm': round(L,dec),
                'arc_mode': mode, 'arc_detail': info, 'text': txt}, None

    up, up_msg = _one('上弓', U_nodes, U_teeth)
    lo, lo_msg = _one('下弓', L_nodes, L_teeth)
    msgs = '；'.join([m for m in (up_msg, lo_msg) if m])
    summary = 'Crowding_v2*: ' + '；'.join([up['text'] if up else '上弓缺失', lo['text'] if lo else '下弓缺失', msgs]).strip('；')
    return {'upper': up, 'lower': lo, 'summary_text': summary, 'quality': 'ok' if (up and lo) else 'fallback'}

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
# =========================
# 9) 第一磨牙关系 — 【新版：全局 Y 轴法】
# =========================
def compute_molar_relationship(
    landmarks: Dict[str, List[float]], ops: Dict, wt: Optional[Dict[str, Any]] = None, dec: int = 1, cfg: Optional[Dict[str, Any]] = None
) -> Dict:
    """
    【新版逻辑】
    计算第一磨牙关系：比较上颌第一磨牙近中颊尖 (U_mb) 的 Y 坐标（前后向）与
    下颌第一磨牙颊沟点 (L_groove) 的 Y 坐标。
    
    坐标系：Y 轴（前后），前为正 (ops['y'])。

    偏移量 offset = y(U_mb) - y(L_groove)
    
    - offset > 0: 上牙尖 U 在 下颌沟 L 的前方 → 远中关系 (Class II)
    - offset < 0: 上牙尖 U 在 下颌沟 L 的后方 → 近中关系 (Class III)
    - offset ≈ 0: 上牙尖 U 落在 下颌沟 L 处 → 中性关系 (Class I)
    """
    cfg = cfg or {}
    neutral_tol = float(cfg.get("molar_neutral_tol_mm", 0.5))
    cusp_mm = float(cfg.get("molar_cusp_mm", 2.5))
    complete_mm = float(cfg.get("molar_complete_mm", 5.0))
    match_tol = float(cfg.get("molar_match_tol_mm", 0.5))
    bg_quality_tol = float(cfg.get("molar_bg_quality_tol_mm", 1.25))

    # _v 是 calc_metrics.py 中的全局辅助函数
    # def _v(a) -> Optional[np.ndarray]: ...

    def _classify(offset: float) -> str:
        """
        根据 offset (y_U - y_L) 进行分类
        offset > 0: 上颌靠前 -> 远中 (Class II)
        offset < 0: 上颌靠后 -> 近中 (Class III)
        """
        a = abs(offset)
        if a <= neutral_tol:
            return '中性'
        
        if offset > 0: 
            # 远中关系 (Class II)
            if a < max(cusp_mm - match_tol, neutral_tol): return '中性偏远中'
            if abs(a - cusp_mm) <= match_tol: return '远中尖对尖'
            if abs(a - complete_mm) <= match_tol or a > complete_mm: return '完全远中'
            return '远中'
        else: 
            # 近中关系 (Class III)
            if a < max(cusp_mm - match_tol, neutral_tol): return '中性偏近中'
            if abs(a - cusp_mm) <= match_tol: return '近中尖对尖'
            if abs(a - complete_mm) <= match_tol or a > complete_mm: return '完全近中'
            return '近中'

    # 下颌沟点（沿用旧口径：bg→(mb+db)/2→(ml+dl)/2，并打质量标记）
    def _lower_groove_strict(tooth: str):
        info = {'status':'missing','threshold_mm': bg_quality_tol,'bg_midpoint_delta_mm':None,'sources':{}}
        bg, mb, db = _v(landmarks.get(f"{tooth}bg")), _v(landmarks.get(f"{tooth}mb")), _v(landmarks.get(f"{tooth}db"))
        ml, dl = _v(landmarks.get(f"{tooth}ml")), _v(landmarks.get(f"{tooth}dl"))
        midpoint = 0.5*(mb + db) if (mb is not None and db is not None) else None
        
        used_label = 'missing'
        
        if bg is not None:
            info['sources']['bg'] = True
            used_label = f"{tooth}bg"
            if midpoint is not None:
                dist = float(np.linalg.norm(bg - midpoint))
                info['bg_midpoint_delta_mm'] = float(np.round(dist, 3))
                if dist > bg_quality_tol:
                    info.update({'status':'midpoint_quality','used':'bg'}); info['sources']['mb_db_mean']=True; return bg, used_label, info
            info.update({'status':'bg_only','used':'bg'}); return bg, used_label, info
        
        if midpoint is not None:
            used_label = f"mean({tooth}mb,{tooth}db)"
            info.update({'status':'midpoint_fallback','used':'midpoint','sources':{'mb_db_mean':True}}); return midpoint, used_label, info
        
        if (ml is not None) and (dl is not None):
            pal_mid = 0.5*(ml + dl)
            used_label = f"mean({tooth}ml,{tooth}dl)"
            info.update({'status':'palatal_fallback','used':'palatal_midpoint','sources':{'ml_dl_mean':True}}); return pal_mid, used_label, info
        
        return None, used_label, info

    def _one(side: str, U_mb_label: str, L_tooth: str):
        dbg = {'side': side, 'upper_label': U_mb_label, 'lower_tooth': L_tooth}
        
        # 1. 获取上颌点
        U_mb = _v(landmarks.get(U_mb_label))
        if U_mb is None: 
            dbg['status']='missing_upper_mb'
            return None, dbg
        
        # 2. 获取下颌点
        L_groove, L_label_used, ginfo = _lower_groove_strict(L_tooth)
        dbg['groove_info'] = ginfo
        if L_groove is None: 
            dbg['status']='missing_lower_groove'
            return None, dbg
            
        # 3. 计算 Y 轴偏移量
        try:
            y_U = float(ops['y'](U_mb))
            y_L = float(ops['y'](L_groove))
            offset_mm = y_U - y_L
        except Exception as e:
            dbg['status'] = f"ops_y_failed: {e}"
            return None, dbg

        # 4. 分类
        klass = _classify(offset_mm)
        
        detail = {
            'class': klass, 
            'label': klass,
            'offset_mm': float(np.round(offset_mm, dec)), 
            'raw_offset_mm': offset_mm,
            'upper_label': U_mb_label, 
            'lower_label': L_label_used, # (e.g., '46bg' or 'mean(46mb,46db)')
            'bg_quality': ginfo.get('status'),
            'y_U_mm': float(np.round(y_U, dec)),
            'y_L_mm': float(np.round(y_L, dec)),
        }
        dbg.update({
            'status':'ok',
            'offset_mm': offset_mm,
            'method': 'Y_axis_direct_comparison',
            'thresholds': {'neutral_tol_mm': neutral_tol,'cusp_mm': cusp_mm,'complete_mm': complete_mm,'match_tol_mm': match_tol}
        })
        return detail, dbg

    # --- 主流程 ---
    right, dbgR = _one('right','16mb','46')
    left,  dbgL = _one('left', '26mb','36')
    
    parts = [f"右侧{right['class']}" if right else "右侧缺失",
             f"左侧{left['class']}"  if left  else "左侧缺失"]
    quality = 'ok' if (right and left) else ('fallback' if (right or left) else 'missing')
    
    return {
        'right': right, 
        'left': left,
        'summary_text': "Molar_Relationship_第一磨牙关系*: " + " ".join(parts),
        'quality': quality,
        'debug': {'right': dbgR, 'left': dbgL},
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

def compute_overbite(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    cfg: Optional[Dict[str, Any]] = None,
    dec: int = 1
) -> Dict:
    """
    口径：
      * 取点：直接使用中切 m 点（11m/21m/41m/31m）
      * 右侧 OB_R = z(11m) - z(41m)；左侧 OB_L = z(21m) - z(31m)
      * 口径切换：
          - overbite_use_midline: True 时按中线计算（上*均值 - 下*均值）
          - overbite_use_ch_ratio: True 时用比例分度 OB/CH（CH=下中切 *↔bgb 冠高）
      * “对刃”容差（默认为 0.5 mm）：abs(OB) ≤ edge_tol → “对刃”
      * 分度（“等号前不取后取”）：
          - OB < -edge_tol  → 开𬌗：|OB|≤3 Ⅰ度，≤5 Ⅱ度，>5 Ⅲ度
          - OB >  edge_tol:
              若 use_ch_ratio 且 CH 可得：
                  OB/CH ≤ 1/3 正常；≤0.5 Ⅰ度；≤2/3 Ⅱ度；>2/3 Ⅲ度
              否则（纯毫米）：
                  OB < 3 正常；≤5 Ⅰ度；≤8 Ⅱ度；>8 Ⅲ度
    """
    cfg = cfg or {}
    use_midline = bool(cfg.get('overbite_use_midline', False))
    use_ch_ratio = bool(cfg.get('overbite_use_ch_ratio', True))
    edge_tol = float(cfg.get('overbite_edge_tol_mm', 0.5))

    U11 = _v(landmarks.get('11m'))
    U21 = _v(landmarks.get('21m'))
    L41 = _v(landmarks.get('41m'))
    L31 = _v(landmarks.get('31m'))

    # 侧别值与下中切冠高（用于比例分度）
    side_vals: Dict[str, float] = {}
    ch_by_side: Dict[str, float] = {}

    def _crown_height(star: Optional[np.ndarray], base_label: str) -> Optional[float]:
        base = _v(landmarks.get(base_label))
        if star is None or base is None:
            return None
        diff = float(abs(ops['z'](star) - ops['z'](base)))
        return diff if diff > EPS else None

    if U11 is not None and L41 is not None:
        side_vals['right'] = float(ops['z'](U11) - ops['z'](L41))
        ch = _crown_height(L41, '41bgb')
        if ch is not None: ch_by_side['right'] = ch
    if U21 is not None and L31 is not None:
        side_vals['left']  = float(ops['z'](U21) - ops['z'](L31))
        ch = _crown_height(L31, '31bgb')
        if ch is not None: ch_by_side['left'] = ch

    if not side_vals:
        return {'summary_text': 'Overbite_前牙覆𬌗*: 缺失', 'quality': 'missing'}

    # 决定口径：中线 or 侧别 |OB| 最大（平局→右）
    mode_used = 'midline' if (use_midline and U11 is not None and U21 is not None and L41 is not None and L31 is not None) else 'max_side'
    if mode_used == 'midline':
        side_of_max = 'midline'
        upper_mid = 0.5*(U11 + U21)
        lower_mid = 0.5*(L41 + L31)
        OB = float(ops['z'](upper_mid) - ops['z'](lower_mid))
        CH = (sum(ch_by_side.values())/len(ch_by_side)) if ch_by_side else None
    else:
        side_of_max = max(sorted(side_vals.keys(), key=lambda s: 0 if s=='right' else 1),
                          key=lambda s: abs(side_vals[s]))
        OB = side_vals[side_of_max]
        CH = ch_by_side.get(side_of_max)

    # 分度（等号前不取、后取）
    if abs(OB) <= edge_tol:
        category = '对刃'
    elif OB < 0:  # 开𬌗
        mag = abs(OB)
        category = 'Ⅰ度开𬌗' if mag <= 3 else ('Ⅱ度开𬌗' if mag <= 5 else 'Ⅲ度开𬌗')
    else:         # 深覆𬌗
        if use_ch_ratio and CH is not None:
            r = OB / CH
            category = ('正常覆𬌗' if r <= 1.0/3.0 else
                        'Ⅰ度深覆𬌗' if r <= 0.5 else
                        'Ⅱ度深覆𬌗' if r <= 2.0/3.0 else
                        'Ⅲ度深覆𬌗')
        else:
            normal_hi = float(cfg.get('overbite_normal_hi_mm', 3.0))
            mild_hi = float(cfg.get('overbite_mild_hi_mm', 4.0))
            moderate_hi = float(cfg.get('overbite_moderate_hi_mm', 6.0))
            category = ('正常覆𬌗' if OB <= normal_hi else
                        'Ⅰ度深覆𬌗' if OB <= mild_hi else
                        'Ⅱ度深覆𬌗' if OB <= moderate_hi else
                        'Ⅲ度深覆𬌗')

    detail = {
        'OB_right_mm': round(side_vals.get('right'), dec) if 'right' in side_vals else None,
        'OB_left_mm':  round(side_vals.get('left'),  dec) if 'left' in side_vals else None,
        'CH_mm': round(CH, dec) if CH is not None else None,
        'mode': mode_used,
    }
    return {
        'value_mm': round(OB, dec),
        'side_of_max': side_of_max,
        'category': category,
        'summary_text': f"Overbite_前牙覆𬌗*: {category}",
        'quality': 'ok',
        'detail': detail
    }

# =========================
# VTP 读取 & 下颌唇面提取（31/32/41/42）
# =========================

def _guess_y_axis_from_ops(ops: Dict) -> Optional[np.ndarray]:
    """通过数值法从 ops['y'] 反推世界坐标系 Ŷ 单位向量。"""
    try:
        delta = 1e-3
        o = np.zeros(3, dtype=float)
        e = np.eye(3, dtype=float)
        g = np.array([ops['y'](e[i] * delta) - ops['y'](o) for i in range(3)], dtype=float) / float(delta)
        n = np.linalg.norm(g)
        return (g / n) if n > EPS else None
    except Exception:
        return None

def _vtk_read_polydata(path: str):
    """优先用 VTK 读取 VTP；若缺 VTK 则返回 (None, 'no_vtk')."""
    try:
        import vtk
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(path)
        reader.Update()
        return reader.GetOutput(), 'ok'
    except Exception:
        return None, 'no_vtk'

def _vtk_triangulate(poly):
    import vtk
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.PassVertsOff(); tri.PassLinesOff()
    tri.Update()
    return tri.GetOutput()

def _vtk_normals(poly):
    import vtk
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(poly)
    nf.SetFeatureAngle(150.0)
    nf.SplittingOff()
    nf.ConsistencyOn()
    nf.AutoOrientNormalsOn()
    nf.ComputePointNormalsOff()
    nf.ComputeCellNormalsOn()
    nf.Update()
    return nf.GetOutput()

def _vtk_arrays(poly, label_name_pref: List[str]):
    """拿 cell labels / points / polys 三件套（numpy）。"""
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
    cdata = poly.GetCellData()
    arr = None
    chosen = None
    for nm in label_name_pref:
        arr = cdata.GetArray(nm)
        if arr is not None:
            chosen = nm
            break
    labels = vtk_to_numpy(arr).astype(np.int32) if arr is not None else None
    pts = poly.GetPoints()
    V = vtk_to_numpy(pts.GetData()) if pts is not None else None
    polys = poly.GetPolys()
    conn = vtk_to_numpy(polys.GetData()) if polys is not None else None
    if conn is not None and conn.size % 4 == 0:
        F = conn.reshape(-1, 4)[:, 1:4]
    else:
        F = None
    return labels, chosen, V, F

def _labels_to_mask(labels: np.ndarray,
                    picked_fdi: Set[str],
                    label_kind: Optional[str],
                    cfg: Optional[Dict]) -> np.ndarray:
    """
    将 cell labels 转换为“是否属于目标牙位”的 mask。
    - 优先假设 label_kind == 'Label' 即 FDI 数字编码（11..28, 31..48）
    - 如果拿到的是 'Label_ios'，用 cfg['vtp_label_ios_to_fdi'] 做映射
    """
    if labels is None:
        return np.zeros(0, dtype=bool)
    cfg = cfg or {}
    kind = (label_kind or '').lower()
    if kind in ('label', 'tooth', 'toothid'):
        as_str = np.array([str(int(x)) for x in labels], dtype=object)
        return np.isin(as_str, list(picked_fdi))
    if kind in ('label_ios', 'labelios'):
        mapper = cfg.get('vtp_label_ios_to_fdi')
        if isinstance(mapper, dict) and mapper:
            mapped = np.array([str(mapper.get(int(x), '')) for x in labels], dtype=object)
            return np.isin(mapped, list(picked_fdi))
        return np.zeros_like(labels, dtype=bool)
    as_str = np.array([str(int(x)) for x in labels], dtype=object)
    mask = np.isin(as_str, list(picked_fdi))
    if mask.any():
        return mask
    mapper = cfg.get('vtp_label_ios_to_fdi')
    if isinstance(mapper, dict) and mapper:
        mapped = np.array([str(mapper.get(int(x), '')) for x in labels], dtype=object)
        return np.isin(mapped, list(picked_fdi))
    return np.zeros_like(labels, dtype=bool)

def _triangles_from_vtp(path: str,
                        target_fdi: Set[str],
                        ops: Dict,
                        cfg: Optional[Dict] = None,
                        cos_thresh: float = 0.2,
                        sanity_floor_ratio: float = 0.15) -> Optional[np.ndarray]:
    """
    读取 VTP → 三角化 → 统一法向 → 取目标牙位的三角片 → 过滤“唇面”（n·Ŷ > cos_thresh）。
    返回 (M,3,3) 的三角片坐标；失败时返回 None。
    """
    cfg = cfg or {}
    poly, status = _vtk_read_polydata(path)
    if poly is None:
        return None

    polyT = _vtk_triangulate(poly)
    polyN = _vtk_normals(polyT)

    label_pref = [str(x) for x in ('Label', 'label', 'Tooth', 'ToothID', 'Label_ios', 'label_ios')]
    labels, label_used, V, F = _vtk_arrays(polyN, label_pref)
    if labels is None or V is None or F is None or F.shape[1] != 3:
        return None

    mask_teeth = _labels_to_mask(labels, target_fdi, label_used, cfg)
    if mask_teeth.sum() == 0:
        mask_teeth = _labels_to_mask(labels, target_fdi, 'Label_ios', cfg)
        if mask_teeth.sum() == 0:
            return None

    F_sel = F[mask_teeth]
    T = V[F_sel]

    Y_hat = _guess_y_axis_from_ops(ops)
    if Y_hat is None:
        return None
    e1 = T[:, 1, :] - T[:, 0, :]
    e2 = T[:, 2, :] - T[:, 0, :]
    N = np.cross(e1, e2)
    nrm = np.linalg.norm(N, axis=1)
    good = nrm > EPS
    N[good] = (N[good].T / nrm[good]).T
    cosv = N @ Y_hat
    lab = cosv > float(cos_thresh)
    required = max(int(sanity_floor_ratio * len(T)), 20)
    if lab.sum() < required:
        lab = cosv > 0.0
        if lab.sum() < required:
            k = max(len(T) // 2, 1)
            idx = np.argsort(-cosv)
            sel = np.zeros_like(lab, dtype=bool)
            sel[idx[:k]] = True
            lab = sel

    return T[lab] if lab.any() else None

# =========================
# Overjet — “点到面金标准”实现（自动从 VTP 取唇面）
# =========================

def _front_trim(
    T: np.ndarray,
    ops: Dict,
    P_upper: np.ndarray,
    front_band_mm: float = 1.8,
    x_band_mm: float = 3.5,
) -> np.ndarray:
    """
    收紧候选三角片：前带优先（Y 坐标高的片元），再按与上切缘 x 距离限窗。
    若裁剪后无片元则回退原集合。
    """
    if T is None or len(T) == 0:
        return T
    C = T.mean(axis=1)
    yC = np.array([ops['y'](c) for c in C], dtype=float)
    y90 = np.quantile(yC, 0.90)
    mask = yC >= (y90 - float(front_band_mm))

    x0 = float(ops['x'](P_upper))
    xC = np.array([ops['x'](c) for c in C], dtype=float)
    mask &= (np.abs(xC - x0) <= float(x_band_mm))

    return T[mask] if mask.any() else T


def _ray_triangle_intersect(O: np.ndarray, D: np.ndarray, tri: np.ndarray) -> Optional[Tuple[float, np.ndarray]]:
    """Möller–Trumbore；返回 (t, Q)，要求 t>=0。D 需为单位向量。"""
    v0, v1, v2 = tri[0], tri[1], tri[2]
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(D, e2)
    det = float(np.dot(e1, pvec))
    if abs(det) < 1e-12:
        return None
    inv_det = 1.0 / det
    tvec = O - v0
    u = float(np.dot(tvec, pvec) * inv_det)
    if (u < -1e-9) or (u > 1.0 + 1e-9):
        return None
    qvec = np.cross(tvec, e1)
    v = float(np.dot(D, qvec) * inv_det)
    if (v < -1e-9) or (u + v > 1.0 + 1e-9):
        return None
    t = float(np.dot(e2, qvec) * inv_det)
    if t < 0.0:
        return None
    Q = O + t * D
    return t, Q

def _first_hit_dy_prefer(
    P_U: np.ndarray,
    ops: Dict,
    Y_hat: np.ndarray,
    triangles: np.ndarray,
    prefer: str,
    t_max: float = 50.0,
) -> Optional[Dict[str, Any]]:
    """
    在具备方向先验的情况下发射射线：先尝试 prefer 指定方向，失败再换向。
    prefer: 'front' → 先 +Ŷ；'back' → 先 -Ŷ。
    """
    directions = [('front', +Y_hat), ('back', -Y_hat)]
    if prefer == 'front':
        directions = directions[::-1]
    for label, D in directions:
        t_best, Q_best = None, None
        for tri in triangles:
            hit = _ray_triangle_intersect(P_U, D, tri)
            if hit is None:
                continue
            t, Q = hit
            if not np.isfinite(t) or t < 0.0 or t > float(t_max):
                continue
            if (t_best is None) or (t < t_best):
                t_best, Q_best = t, Q
        if t_best is not None:
            dy = float(ops['y'](P_U) - ops['y'](Q_best))
            return {'dy': dy, 'Q': Q_best, 'dir': label, 't': float(t_best)}
    return None

def compute_overjet(
    landmarks: Dict[str, List[float]],
    ops: Dict,
    cfg: Optional[Dict[str, Any]] = None,
    dec: int = 1
) -> Dict:
    """
    金标准 Overjet：上切缘点 → 下切牙唇面（从 VTP 自动提取）
      - 右：11m → (41,42) 唇面
      - 左：21m → (31,32) 唇面
      - 中线（可选）：(11m+21m)/2 → (31–42) 唇面
    需要：cfg['vtp_lower_path'] 指向下颌 VTP；若缺失，则回退到旧口径“点到点 y 差”。
    """
    cfg = cfg or {}
    zero_tol = float(cfg.get('overjet_edge_tol_mm', 0.5))
    baseline = float(cfg.get('overjet_baseline_mm', 0.0))
    use_mid = bool(cfg.get('overjet_use_midline', False))
    normal_hi = float(cfg.get('overjet_normal_hi_mm', 3.0))
    mild_hi = float(cfg.get('overjet_mild_hi_mm', 5.0))
    moderate_hi = float(cfg.get('overjet_moderate_hi_mm', 8.0))
    cos_thresh = float(cfg.get('labial_normal_cos_thresh', 0.2))
    t_guard = float(cfg.get('ray_max_dist_mm', 50.0))
    front_band = float(cfg.get('labial_front_band_mm', 1.8))
    x_band = float(cfg.get('labial_x_band_mm', 3.5))

    P11 = _v(landmarks.get('11m'))
    P21 = _v(landmarks.get('21m'))
    P41 = _v(landmarks.get('41m'))
    P31 = _v(landmarks.get('31m'))
    _, lower41_star = _star_mid(landmarks, '41')
    _, lower31_star = _star_mid(landmarks, '31')
    lower_star: Dict[str, Optional[np.ndarray]] = {'41': lower41_star, '31': lower31_star}
    lower_m: Dict[str, Optional[np.ndarray]] = {'41': P41, '31': P31}

    vtp_lower = cfg.get('vtp_lower_path') or cfg.get('vtp_lower') or cfg.get('vtp_path_lower')
    T_R = T_L = None
    if isinstance(vtp_lower, str):
        T_R = _triangles_from_vtp(vtp_lower, {'41', '42'}, ops, cfg=cfg, cos_thresh=cos_thresh)
        T_L = _triangles_from_vtp(vtp_lower, {'31', '32'}, ops, cfg=cfg, cos_thresh=cos_thresh)
    if T_R is not None and P11 is not None:
        T_R = _front_trim(
            T_R,
            ops,
            P11,
            front_band_mm=front_band,
            x_band_mm=x_band,
        )
    if T_L is not None and P21 is not None:
        T_L = _front_trim(
            T_L,
            ops,
            P21,
            front_band_mm=front_band,
            x_band_mm=x_band,
        )

    def _fallback_point_to_point() -> Dict:
        def _lower_point(tooth: str) -> Optional[np.ndarray]:
            if tooth not in ('41', '31'):
                return None
            pt = lower_m.get(tooth)
            if pt is not None:
                return pt
            cached = lower_star.get(tooth)
            if cached is not None:
                return cached
            _, p = _star_mid(landmarks, tooth)
            lower_star[tooth] = p
            return p

        P41_ref = _lower_point('41')
        P31_ref = _lower_point('31')
        right = (float(ops['y'](P11) - ops['y'](P41_ref)) - baseline) if (P11 is not None and P41_ref is not None) else None
        left = (float(ops['y'](P21) - ops['y'](P31_ref)) - baseline) if (P21 is not None and P31_ref is not None) else None
        if right is None and left is None:
            return {'summary_text': 'Overjet_前牙覆盖*: 缺失', 'quality': 'missing'}
        choose_left = (left is not None) and (right is None or abs(left) > abs(right))
        value = left if choose_left else right
        side = 'left' if choose_left else 'right'
        a = abs(value)
        if a <= zero_tol:
            cat = '对刃'
        elif value < -zero_tol:
            cat = '反覆盖'
        else:
            cat = ('正常覆盖' if a <= normal_hi else
                   'Ⅰ度深覆盖' if a <= mild_hi else
                   'Ⅱ度深覆盖' if a <= moderate_hi else
                   'Ⅲ度深覆盖')
        detail = {
            'mode': 'point_to_point',
            'baseline_mm': baseline,
            'right': {'status': 'ok', 'OJ_mm': round(right, dec)} if right is not None else {'status': 'missing'},
            'left': {'status': 'ok', 'OJ_mm': round(left, dec)} if left is not None else {'status': 'missing'}
        }
        summary = f"Overjet_前牙覆盖*: {abs(round(value, dec))}mm_{cat}"
        return {
            'value_mm': round(value, dec),
            'side_of_max': side,
            'category': cat,
            'summary_text': summary,
            'quality': 'fallback',
            'detail': detail,
            'mode': 'point_to_point',
            'thresholds': {
                'edge_tol_mm': zero_tol,
                'normal_hi_mm': normal_hi,
                'mild_hi_mm': mild_hi,
                'moderate_hi_mm': moderate_hi,
            },
        }

    if (T_R is None) and (T_L is None):
        return _fallback_point_to_point()

    Y_hat = _guess_y_axis_from_ops(ops)
    if Y_hat is None or (P11 is None and P21 is None):
        return _fallback_point_to_point()

    detail: Dict[str, Any] = {'right': None, 'left': None, 'midline': None}

    def _side(
        PU: Optional[np.ndarray],
        T: Optional[np.ndarray],
        lower_star_guess: Optional[np.ndarray],
        lower_m_guess: Optional[np.ndarray],
    ) -> Tuple[Optional[float], Dict[str, Any]]:
        if PU is None or T is None or len(T) == 0:
            return None, {'status': 'missing_upper_or_surface'}
        prefer = 'back'
        ref = lower_star_guess if lower_star_guess is not None else lower_m_guess
        if ref is not None and PU is not None:
            prefer = 'back' if ops['y'](PU) >= ops['y'](ref) else 'front'
        hit = _first_hit_dy_prefer(PU, ops, Y_hat, T, prefer=prefer, t_max=t_guard)
        if hit is None:
            return None, {'status': 'ray_no_hit', 't_guard_mm': t_guard, 'baseline_mm': baseline, 'n_tri': int(T.shape[0])}
        val = float(hit['dy'] - baseline)
        info = {
            'status': 'ok',
            'OJ_mm': round(val, dec),
            'raw_dy_mm': round(hit['dy'], dec),
            'dir': hit['dir'],
            't_mm': round(hit['t'], 3),
            'baseline_mm': baseline,
            'n_tri': int(T.shape[0]),
        }
        return val, info

    OJ_R, infoR = _side(P11, T_R, lower_star.get('41'), lower_m.get('41')) if T_R is not None else (None, {'status': 'missing_surface'})
    OJ_L, infoL = _side(P21, T_L, lower_star.get('31'), lower_m.get('31')) if T_L is not None else (None, {'status': 'missing_surface'})
    detail['right'] = infoR
    detail['left'] = infoL

    mode_used = 'max_side'
    primary_side = None
    OJ_val = None
    if use_mid and P11 is not None and P21 is not None:
        Pmid = 0.5 * (P11 + P21)
        T_all = None
        if T_R is not None and T_L is not None:
            T_all = np.concatenate([T_R, T_L], axis=0)
        elif T_R is not None:
            T_all = T_R
        elif T_L is not None:
            T_all = T_L
        if T_all is not None:
            prefer_mid = 'back'
            lower_mid_ref = None
            lower_points = [p for p in (lower_m.get('41'), lower_m.get('31'), lower_star.get('41'), lower_star.get('31')) if p is not None]
            if lower_points:
                lower_mid_ref = np.mean(lower_points, axis=0)
            if lower_mid_ref is not None:
                prefer_mid = 'back' if ops['y'](Pmid) >= ops['y'](lower_mid_ref) else 'front'
            hitM = _first_hit_dy_prefer(Pmid, ops, Y_hat, T_all, prefer=prefer_mid, t_max=t_guard)
            if hitM is not None:
                OJ_mid = float(hitM['dy'] - baseline)
                detail['midline'] = {
                    'status': 'ok',
                    'OJ_mid_mm': round(OJ_mid, dec),
                    'raw_dy_mm': round(hitM['dy'], dec),
                    'dir': hitM['dir'],
                    't_mm': round(hitM['t'], 3),
                    'baseline_mm': baseline,
                    'n_tri': int(T_all.shape[0]),
                }
                mode_used = 'midline'
                primary_side = 'midline'
                OJ_val = OJ_mid

    if OJ_val is None:
        pairs: List[Tuple[str, float]] = []
        if isinstance(OJ_R, (int, float)):
            pairs.append(('right', OJ_R))
        if isinstance(OJ_L, (int, float)):
            pairs.append(('left', OJ_L))
        if not pairs:
            return _fallback_point_to_point()
        pairs.sort(key=lambda kv: (abs(kv[1]), 0 if kv[0] == 'right' else 1), reverse=True)
        primary_side, OJ_val = pairs[0]

    value_mm = round(float(OJ_val), dec)
    a = abs(float(OJ_val))
    if a <= zero_tol:
        category = '对刃'
    elif OJ_val < -zero_tol:
        category = '反覆盖'
    else:
        category = ('正常覆盖' if a <= normal_hi else
                    'Ⅰ度深覆盖' if a <= mild_hi else
                    'Ⅱ度深覆盖' if a <= moderate_hi else
                    'Ⅲ度深覆盖')

    quality = 'ok' if (isinstance(OJ_R, (int, float)) and isinstance(OJ_L, (int, float))) else 'fallback'
    summary = f"Overjet_前牙覆盖*: {abs(value_mm):.{dec}f}mm_{category}"
    return {
        'value_mm': value_mm,
        'side_of_max': primary_side,
        'category': category,
        'summary_text': summary,
        'quality': quality,
        'detail': detail,
        'mode': mode_used,
        'thresholds': {
            'edge_tol_mm': zero_tol,
            'normal_hi_mm': normal_hi,
            'mild_hi_mm': mild_hi,
            'moderate_hi_mm': moderate_hi,
        },
    }

def _fmt_suffix(ok: bool) -> str: return '' if ok else ' ⚠️'

def make_brief_report(
    landmarks: Dict[str, List[float]],
    frame_ops: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None
) -> List[str]:
    frame, ops, wt = frame_ops['frame'], frame_ops['ops'], frame_ops['wt']
    wt_records = frame_ops.get('wt_records') or {}
    cfg = cfg or {}
    arch_form = compute_arch_form(landmarks, frame, ops, cfg=cfg)
    arch_width = compute_arch_width(landmarks, ops, cfg=cfg)
    bolton = compute_bolton(landmarks, ops, wt, wt_records=wt_records)
    canine = compute_canine_relationship(landmarks, ops, cfg=cfg)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_info = compute_spee(landmarks, ops)
    midline = compute_midline_alignment(landmarks, ops)
    molar = compute_molar_relationship(landmarks, ops, wt, cfg=cfg)
    ob = compute_overbite(landmarks, ops, cfg=cfg)
    oj = compute_overjet(landmarks, ops, cfg=cfg)

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

# =========================
# Emergency polarity hotfix — 仅在“强证据”下互换 mc↔dc（前牙）
# =========================
def _apply_polarity_hotfix(
    landmarks: Dict[str, List[float]],
    ops: Dict[str, Any],
    cfg: Optional[Dict[str, Any]] = None
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """
    无需 GT 的极性热修补：
      对每颗前牙 t，比较邻接一致性分数：
        as_is   = H(t.mc, M.dc) + H(t.dc, D.mc)
        swapped = H(t.dc, M.dc) + H(t.mc, D.mc)
      同时要求 X 轴次序违例（象限期望相反且越界>tol）。
      仅当 gain>=gain_min 且 swapped<=swap_max 时才互换 mc↔dc。
    只作用 1–3 号牙；邻牙点缺失则跳过。
    """
    cfg = cfg or {}
    enable = bool(cfg.get("polarity_hotfix_enable", True))
    H = ops.get("H") if isinstance(ops, dict) else None
    X = ops.get("x") if isinstance(ops, dict) else None
    if (not enable) or (H is None) or (X is None) or (not callable(H)) or (not callable(X)):
        return landmarks, {"applied": False, "changes": {}, "reason": "disabled_or_ops_missing"}

    gain_min = float(cfg.get("polarity_gain_min_mm", 4.0))
    swap_max = float(cfg.get("polarity_swapped_score_max_mm", 3.0))
    x_tol = float(cfg.get("polarity_x_tol_mm", 0.6))

    def _digits(t: str) -> Optional[Tuple[int, int]]:
        ds = "".join(ch for ch in t if ch.isdigit())
        if len(ds) >= 2:
            return int(ds[-2]), int(ds[-1])
        return None

    def _quadrant(t: str) -> Optional[int]:
        d = _digits(t)
        return d[0] if d else None

    def _mesial_distal_neighbors(t: str) -> Tuple[Optional[str], Optional[str]]:
        d = _digits(t)
        if not d:
            return None, None
        q, n = d
        if n == 1:
            across = {1: "21", 2: "11", 3: "41", 4: "31"}.get(q)
            mesial = across
            distal = f"{q}{n + 1}" if n < 7 else None
        else:
            mesial = f"{q}{n - 1}"
            distal = f"{q}{n + 1}" if n < 7 else None
        return mesial, distal

    def _get(lbl: str) -> Optional[np.ndarray]:
        return _v(landmarks.get(lbl))

    target_teeth = [f"{q}{n}" for q in (1, 2, 3, 4) for n in (1, 2, 3)]

    new_map = dict(landmarks)
    changes: Dict[str, Any] = {}

    for t in target_teeth:
        mc = _get(f"{t}mc")
        dc = _get(f"{t}dc")
        if mc is None or dc is None:
            continue

        M, D = _mesial_distal_neighbors(t)
        if not M or not D:
            continue
        Mdc = _get(f"{M}dc")
        Dmc = _get(f"{D}mc")
        if Mdc is None or Dmc is None:
            continue

        try:
            as_is = float(H(mc, Mdc) + H(dc, Dmc))
            swapped = float(H(dc, Mdc) + H(mc, Dmc))
        except Exception:
            continue

        gain = as_is - swapped
        q = _quadrant(t)
        if q is None:
            continue
        expect_mc_less = q in (1, 4)
        dx = float(X(mc) - X(dc))
        x_bad = (dx > x_tol) if expect_mc_less else (dx < -x_tol)

        if (gain >= gain_min) and (swapped <= swap_max) and x_bad:
            new_map[f"{t}mc"], new_map[f"{t}dc"] = new_map[f"{t}dc"], new_map[f"{t}mc"]
            changes[t] = {
                "action": "swap_mc_dc",
                "gain_mm": round(gain, 2),
                "as_is_mm": round(as_is, 2),
                "swapped_mm": round(swapped, 2),
                "x_mc_minus_dc_mm": round(dx, 2),
                "x_expect": "mc<dc" if expect_mc_less else "mc>dc",
            }

    return new_map, {
        "applied": bool(changes),
        "changes": changes,
        "thresholds": {
            "gain_min_mm": gain_min,
            "swapped_score_max_mm": swap_max,
            "x_tol_mm": x_tol,
        },
    }

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

    arch_form = compute_arch_form(landmarks, frame, ops, cfg=cfg)
    arch_width = compute_arch_width(landmarks, ops, cfg=cfg)
    bolton = compute_bolton(landmarks, ops, wt, wt_records=wt_records)
    canine = compute_canine_relationship(landmarks, ops, cfg=cfg)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_info = compute_spee(landmarks, ops, cfg=cfg)
    midline = compute_midline_alignment(landmarks, ops)
    molar = compute_molar_relationship(landmarks, ops, wt, cfg=cfg)
    overbite = compute_overbite(landmarks, ops, cfg=cfg)
    overjet = compute_overjet(landmarks, ops, cfg=cfg)

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
    landmarks = dict(landmarks or {})
    cfg = dict(cfg) if cfg else {}

    # 0) 极性快检：提前矫正明显的 mc↔dc 对调
    polarity_log = _apply_polarity_guard(landmarks, cfg)
    if polarity_log:
        cfg.setdefault('polarity_guard_log', []).extend(polarity_log)

    # 1) 先用（可能已修正的）点建坐标系，供热修补判定用
    frame_ops0 = build_module0(landmarks)
    if frame_ops0.get('frame') is None or frame_ops0.get('ops') is None:
        kv = {"错误": "坐标系缺失，无法生成报告"}
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(kv, f, ensure_ascii=False, indent=2)
        return kv

    # 2) 紧急极性热修补（默认关闭，可由 cfg['polarity_hotfix_enable'] 控制）
    hotfix_enabled = bool(cfg.get('polarity_hotfix_enable', False))
    if hotfix_enabled:
        landmarks_fixed, hotfix_info = _apply_polarity_hotfix(landmarks, frame_ops0['ops'], cfg)
    else:
        landmarks_fixed = landmarks
        hotfix_info = {"applied": False, "changes": {}, "reason": "disabled"}

    # 3) 用修补后的点重建坐标/算指标
    frame_ops = build_module0(landmarks_fixed)

    kv = make_doctor_cn_simple(landmarks_fixed, frame_ops, cfg=cfg)
    kv.pop('warnings', None)
    if polarity_log:
        kv['Polarity_Guard_Log'] = polarity_log
    if cfg.get('include_extended_fields'):
        kv['Polarity_Hotfix'] = hotfix_info

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
    ap.add_argument(
        "--overbite-midline",
        action="store_true",
        help="使用中线（上下中切平均）计算前牙覆𬌗，取代左右最大值。",
    )
    ap.add_argument(
        "--no-overbite-ch-ratio",
        action="store_true",
        help="禁用冠高比值口径，始终按绝对毫米阈值分度前牙覆𬌗。",
    )
    ap.add_argument(
        "--overbite-edge-tol-mm",
        type=float,
        default=0.5,
        help="对刃判定的容差（默认 0.5mm）。",
    )
    args = ap.parse_args()
    cfg = {
        "overbite_use_midline": args.overbite_midline,
        "overbite_use_ch_ratio": not args.no_overbite_ch_ratio,
        "overbite_edge_tol_mm": args.overbite_edge_tol_mm,
    }
    kv = generate_metrics(args.sources, out_path=args.out, cfg=cfg)
    print(f"saved to: {args.out} ({len(kv)} items)")
