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
    def _get(nm): return _v(landmarks.get(nm))
    # 侧切“*”
    _, U12 = _star_mid(landmarks, '12')
    _, U22 = _star_mid(landmarks, '22')
    # 尖牙、前磨
    P = {
        '12*': U12, '22*': U22,
        '13m': _get('13m'), '23m': _get('23m'),
        '14b': _get('14b'), '24b': _get('24b')
    }
    if any(P[k] is None for k in P):
        return {'form': '缺失', 'indices': {}, 'summary_text': 'Arch_Form_牙弓形态*: 缺失', 'quality': 'missing'}

    def _angle(A, B, C):
        # 在 P 面计算 ∠ABC
        Ap = ops['projP'](A); Bp = ops['projP'](B); Cp = ops['projP'](C)
        u = Ap - Bp; v = Cp - Bp
        nu, nv = _len(u), _len(v)
        if nu<EPS or nv<EPS: return None
        c = np.clip(_dot(u, v)/(nu*nv), -1.0, 1.0)
        return float(np.degrees(np.arccos(c)))

    aR = _angle(P['12*'], P['13m'], P['14b'])
    aL = _angle(P['22*'], P['23m'], P['24b'])
    if aR is None or aL is None:
        return {'form': '缺失', 'indices': {}, 'summary_text': 'Arch_Form_牙弓形态*: 缺失', 'quality': 'missing'}

    alpha_avg = 0.5*(aR + aL)
    # 分型阈值延用 160/168° 的经验口径（可在 cfg 中外部固化）
    if alpha_avg <= 160.0:
        form = '尖圆形'
    elif alpha_avg >= 168.0:
        form = '方圆形'
    else:
        form = '卵圆形'

    return {
        'form': form,
        'indices': {'alpha_deg': round(alpha_avg, 1)},
        'summary_text': f"Arch_Form_牙弓形态*: {form}",
        'quality': 'ok'
    }

# =========================
# 2) Arch Width — 前/中/后三段
# =========================
def compute_arch_width(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    need = ['13m','23m','14b','24b','16mb','26mb','33m','43m','34b','44b','36mb','46mb']
    if any(_get(n) is None for n in need):
        return {'upper':None,'lower':None,'diff_UL_mm':None,'upper_is_narrow':None,
                'summary_text':'Arch_Width_牙弓宽度*: 缺失','quality':'missing'}
    H = ops['H']
    u = {
        'anterior_mm': round(H(_get('13m'), _get('23m')), dec),
        'middle_mm'  : round(H(_get('14b'), _get('24b')), dec),
        'posterior_mm': round(H(_get('16mb'), _get('26mb')), dec)
    }
    l = {
        'anterior_mm': round(H(_get('33m'), _get('43m')), dec),
        'middle_mm'  : round(H(_get('34b'), _get('44b')), dec),
        'posterior_mm': round(H(_get('36mb'), _get('46mb')), dec)
    }
    diff = {k: round(u[k]-l[k], dec) for k in u}
    votes = sum(1 for k in diff.values() if k < 0.0)
    narrow = (votes >= 2)
    return {
        'upper': u, 'lower': l, 'diff_UL_mm': diff, 'upper_is_narrow': bool(narrow),
        'summary_text': ('Arch_Width_牙弓宽度*: 上牙弓较窄' if narrow else 'Arch_Width_牙弓宽度*: 未见上牙弓较窄'),
        'quality':'ok'
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
def compute_canine_relationship(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    tau = 0.5  # mm
    complete = 2.0

    def side(u_nm: str, l_nm: str):
        u = _get(u_nm); l = _get(l_nm)
        if u is None or l is None: return None, '缺失'
        dy = ops['y'](u) - ops['y'](l)
        # 分类
        if abs(dy) <= tau: label = '尖对尖'
        elif dy >= complete: label = '完全近中'
        elif dy <= -complete: label = '完全远中'
        else: label = ('近中尖对尖' if dy > 0 else '远中尖对尖')
        return round(dy, dec), label

    dyR, labR = side('13m', '43dc')
    dyL, labL = side('23m', '33dc')

    if labR == '缺失' and labL == '缺失':
        return {'right':None,'left':None,'summary_text':'Canine_Relationship_尖牙关系*: 缺失','quality':'missing'}

    s = f"Canine_Relationship_尖牙关系*: 右侧{labR if labR!='缺失' else '缺失'}，左侧{labL if labL!='缺失' else '缺失'}"
    return {
        'right': {'dy_mm': dyR, 'class': labR},
        'left' : {'dy_mm': dyL, 'class': labL},
        'summary_text': s, 'quality': 'ok' if ('缺失' not in (labR, labL)) else 'fallback'
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
# 6) 拥挤度 — 全弓（17↔27 / 47↔37）
# =========================
def compute_crowding(landmarks: Dict[str, List[float]], ops: Dict, wt_callable, dec:int=1) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    H = ops['H']

    # 锚点链（医生定义）
    U_chain = ['17dc','16mb','15b','14b','13m','12*','11*','21*','22*','23m','24b','25b','26mb','27dc']
    L_chain = ['47dc','46mb','45b','44b','43m','42*','41*','31*','32*','33m','34b','35b','36mb','37dc']

    # “*”展开为 (ma+da)/2
    def _resolve(tag):
        if tag.endswith('*'):
            t = tag[:2]
            _, p = _star_mid(landmarks, t)
            return p
        return _get(tag)

    def _arc(chain: List[str]) -> Optional[float]:
        pts = [ _resolve(tag) for tag in chain ]
        if any(p is None for p in pts): return None
        s = 0.0
        for a,b in zip(pts[:-1], pts[1:]):
            s += H(a, b)
        return s

    # 宽度和（14 颗，含第二磨牙，不含第三磨牙）
    U_teeth = ['17','16','15','14','13','12','11','21','22','23','24','25','26','27']
    L_teeth = ['47','46','45','44','43','42','41','31','32','33','34','35','36','37']
    def _sum_width(teeth):
        s, miss = 0.0, []
        for t in teeth:
            w = wt_callable(t, allow_fallback=True)
            if w is None: miss.append(t)
            else: s += w
        return (None if miss else s), miss

    U_len = _arc(U_chain);  L_len = _arc(L_chain)
    U_sum, U_miss = _sum_width(U_teeth)
    L_sum, L_miss = _sum_width(L_teeth)

    def _one(sumw, arclen, arch_name):
        if (sumw is None) or (arclen is None): return None, f'{arch_name}缺失'
        ald = round(sumw - arclen, dec)
        txt = (f"{arch_name}间隙{abs(ald):.{dec}f}mm" if ald < 0 else f"{arch_name}拥挤{abs(ald):.{dec}f}mm")
        return {'ald_mm': ald, 'sum_w_mm': round(sumw,dec), 'arc_len_mm': round(arclen,dec), 'text': txt}, None

    up, eU = _one(U_sum, U_len, '上牙列')
    lw, eL = _one(L_sum, L_len, '下牙列')
    if eU and eL:
        return {'upper':None,'lower':None,'summary_text':'Crowding_拥挤度*: 缺失','quality':'missing'}
    parts = []
    if up: parts.append(up['text'])
    if lw: parts.append(lw['text'])
    return {'upper': up, 'lower': lw, 'summary_text': 'Crowding_拥挤度*:' + ''.join(parts), 'quality': 'ok'}

# =========================
# 7) Curve of Spee — P 内“弦—点”稳健垂距
# =========================
def compute_spee(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict[str, Any]:
    def _get(nm): return _v(landmarks.get(nm))
    projP = ops['projP']

    # 基线端点
    A = None
    p31ma, p41ma = _get('31ma'), _get('41ma')
    if p31ma is not None and p41ma is not None:
        A = 0.5*(p31ma+p41ma)
    else:
        A = _get('31m')
        if A is None:
            A = _get('41m')
    B_R = _get('47db')
    if B_R is None:
        B_R = _get('46db')
    B_L = _get('37db')
    if B_L is None:
        B_L = _get('36db')
    if A is None or (B_R is None and B_L is None):
        return {'value_mm': None, 'quality': 'missing', 'trimmed': False, 'detail': {}, 'reason': 'anchor_missing'}

    def _side(chain: List[str], B2: Optional[np.ndarray]) -> Optional[Dict[str, Any]]:
        if B2 is None:
            return None
        B1p = projP(A)
        B2p = projP(B2)
        vec = B2p - B1p
        seg_len = _len(vec)
        if seg_len < EPS:
            return None
        u = vec / seg_len
        values: List[float] = []
        for nm in chain:
            p = _get(nm)
            if p is None:
                continue
            qp = projP(p)
            w = qp - B1p
            along = _dot(w, u)
            frac = along / seg_len if seg_len > EPS else None
            if frac is None:
                continue
            if frac < 0.10 or frac > 0.90:
                continue
            perp = w - u * along
            values.append(float(_len(perp)))
        if not values:
            return None
        trimmed = len(values) >= 4
        val = float(np.percentile(values, 95)) if trimmed else max(values)
        return {
            'raw_value_mm': val,
            'value_mm': float(np.round(val, dec)),
            'trimmed': bool(trimmed),
            'sample_count': len(values)
        }

    chain_R = ['43m','44b','45b','46mb','46db','47mb','47db']
    chain_L = ['33m','34b','35b','36mb','36db','37mb','37db']

    detail: Dict[str, Any] = {}
    r = _side(chain_R, B_R)
    if r is not None:
        detail['right'] = r
    l = _side(chain_L, B_L)
    if l is not None:
        detail['left'] = l

    if not detail:
        return {'value_mm': None, 'quality': 'missing', 'trimmed': False, 'detail': {}, 'reason': 'sampling_missing'}

    usable = [(side, info['raw_value_mm']) for side, info in detail.items()]
    side_name, raw_val = max(usable, key=lambda x: x[1])
    trimmed_sides = [side for side, info in detail.items() if info.get('trimmed')]

    return {
        'value_mm': float(np.round(raw_val, dec)),
        'raw_value_mm': float(raw_val),
        'quality': 'ok',
        'trimmed': bool(trimmed_sides),
        'trimmed_sides': trimmed_sides,
        'primary_side': side_name,
        'detail': detail,
        'used_fraction': (0.10, 0.90)
    }

# =========================
# 8) Midline — 上/下中线 x 偏移
# =========================
def compute_midline_alignment(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    def _mid_star(t1, t2):
        _, a = _star_mid(landmarks, t1); _, b = _star_mid(landmarks, t2)
        if a is None or b is None: return None
        return 0.5*(a+b)
    U = _mid_star('11','21');  L = _mid_star('31','41')
    if U is None or L is None:
        return {'summary_text': 'Midline_Alignment_牙列中线*: 缺失', 'quality':'missing'}
    dU = ops['x'](U);  dL = ops['x'](L)
    def _txt(v, arch):
        if abs(v) < 0.5:
            return f"{arch}中线居中"
        return f"{arch}中线{'右偏' if v>0 else '左偏'}{abs(round(v,dec))}mm"
    return {'summary_text': f"Midline_Alignment_牙列中线*: {_txt(dU,'上')} {_txt(dL,'下')}", 'quality':'ok'}

# =========================
# 9) 第一磨牙关系 — 16/26 对 46bg/36bg（个体化阈值）
# =========================
def compute_molar_relationship(landmarks: Dict[str, List[float]], ops: Dict, wt_callable, dec:int=1) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    def _one(u_mb, l_bg, l_mc, l_dc):
        U = _get(u_mb); Lbg = _get(l_bg); Lmc = _get(l_mc); Ldc = _get(l_dc)
        if U is None or Lbg is None: return None
        off = ops['y'](U) - ops['y'](Lbg)
        # 直接用下颌该颗牙的实际冠宽
        W = None
        if (Lmc is not None) and (Ldc is not None):
            W = ops['H'](Lmc, Ldc)
        q = (W/4.0 if W else 2.5)
        h = (W/2.0 if W else 5.0)
        tau = 0.5
        if abs(off) <= tau:
            lab = '中性'
        else:
            if off > 0:
                candidates = [
                    ('完全近中', abs(off - h)),
                    ('近中尖对尖', abs(off - q)),
                    ('中性偏近中', abs(off - 0.0)),
                ]
            else:
                candidates = [
                    ('完全远中', abs(off + h)),
                    ('远中尖对尖', abs(off + q)),
                    ('中性偏远中', abs(off - 0.0)),
                ]
            lab = min(candidates, key=lambda x: x[1])[0]
        return {'offset_mm': round(off,dec), 'label': lab, 'q_mm': round(q,1), 'h_mm': round(h,1)}

    R = _one('16mb','46bg','46mc','46dc')
    L = _one('26mb','36bg','36mc','36dc')
    if R is None and L is None:
        return {'summary_text':'Molar_Relationship_磨牙关系*: 缺失','quality':'missing'}
    return {'right': R, 'left': L,
            'summary_text': f"Molar_Relationship_磨牙关系*: 右侧{R['label'] if R else '缺失'} 左侧{L['label'] if L else '缺失'}",
            'quality': 'ok' if (R and L) else 'fallback'}

# =========================
# 10) Overbite — 垂直覆𬌗（比值分度）
# =========================
def compute_overbite(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    def _star_xy(t): _, p = _star_mid(landmarks, t); return p
    U11, U21 = _star_xy('11'), _star_xy('21')
    L41, L31 = _star_xy('41'), _star_xy('31')
    if any(p is None for p in [U11,U21,L41,L31]):
        return {'summary_text': 'Overbite_前牙覆𬌗*: 缺失', 'quality':'missing'}

    OB_R = ops['z'](U11) - ops['z'](L41)
    OB_L = ops['z'](U21) - ops['z'](L31)
    side, OB = (('right',OB_R) if abs(OB_R) >= abs(OB_L) else ('left',OB_L))

    # 下中切冠高（对应侧）
    Lm = L41 if side=='right' else L31
    Lbgb = _get('41bgb' if side=='right' else '31bgb')
    CH = (float(np.linalg.norm(Lm - Lbgb)) if (Lm is not None and Lbgb is not None) else None)

    if abs(OB) <= 0.5:
        cat = '对刃'
    elif OB < 0:
        # 开𬌗分度（mm）
        d = abs(OB)
        cat = ('Ⅰ度开𬌗' if d < 3 else ('Ⅱ度开𬌗' if d <= 5 else 'Ⅲ度开𬌗'))
    else:
        # 深覆分度（比值）
        if CH and CH>EPS:
            r = OB/CH
            if 1/3 <= r <= 1/2: cat='Ⅰ度深覆𬌗'
            elif 1/2 < r <= 2/3: cat='Ⅱ度深覆𬌗'
            elif r > 2/3: cat='Ⅲ度深覆𬌗'
            else: cat='轻度覆𬌗'
        else:
            # 没有 CH 时回退用毫米大致分界
            cat = ('Ⅰ度深覆𬌗' if 3<=OB<=5 else ('Ⅱ度深覆𬌗' if 5<OB<=8 else ('Ⅲ度深覆𬌗' if OB>8 else '轻度覆𬌗')))

    return {'value_mm': round(OB,dec), 'side_of_max': side, 'category': cat,
            'summary_text': f"Overbite_前牙覆𬌗*: {cat}", 'quality':'ok'}

# =========================
# 11) Overjet — 水平覆盖（y 前后）
# =========================
def compute_overjet(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Dict:
    def _star_xy(t): _, p = _star_mid(landmarks, t); return p
    U11, U21 = _star_xy('11'), _star_xy('21')
    L41, L31 = _star_xy('41'), _star_xy('31')
    if any(p is None for p in [U11,U21,L41,L31]):
        return {'summary_text': 'Overjet_前牙覆盖*: 缺失', 'quality':'missing'}

    OJ_R = ops['y'](U11) - ops['y'](L41)
    OJ_L = ops['y'](U21) - ops['y'](L31)
    side, OJ = (('right',OJ_R) if abs(OJ_R) >= abs(OJ_L) else ('left',OJ_L))

    zero = 0.5
    if abs(OJ) <= zero:
        cat = '对刃'
    elif OJ < -zero:
        cat = '反𬌗'
    else:
        # 深覆盖分度（mm）
        d = OJ
        if 3 <= d <= 5: cat = 'Ⅰ度深覆盖'
        elif 5 < d <= 8: cat = 'Ⅱ度深覆盖'
        elif d > 8: cat = 'Ⅲ度深覆盖'
        else: cat = '轻度覆盖'

    val_txt = f"{abs(round(OJ,dec))}mm_{cat}"
    return {'value_mm': round(OJ,dec), 'side_of_max': side, 'category': cat,
            'summary_text': f"Overjet_前牙覆盖*: {val_txt}", 'quality':'ok'}

# =========================
# 报告行（brief） & 公开接口
# =========================
def _fmt_suffix(ok: bool) -> str: return '' if ok else ' ⚠️'

def make_brief_report(landmarks: Dict[str, List[float]], frame_ops: Dict[str, Any]) -> List[str]:
    frame, ops, wt = frame_ops['frame'], frame_ops['ops'], frame_ops['wt']
    wt_records = frame_ops.get('wt_records') or {}

    arch_form = compute_arch_form(landmarks, frame, ops)
    arch_width = compute_arch_width(landmarks, ops)
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
    spee_ok = (spee_info.get('quality') != 'missing') and (not spee_info.get('trimmed'))
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
    arch_width = compute_arch_width(landmarks, ops)
    bolton = compute_bolton(landmarks, ops, wt, wt_records=wt_records)
    canine = compute_canine_relationship(landmarks, ops)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_info = compute_spee(landmarks, ops)
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
        out['Arch_Width'] = '上牙弓较窄' if arch_width.get('upper_is_narrow') else '未见上牙弓较窄'
    else:
        out['Arch_Width'] = '缺失'

    # 3) Bolton（前牙比换算成小数，同时输出原始比值/差值）
    bolton_ratio = None
    anterior_detail = bolton.get('anterior') if _ok(bolton, 'anterior') else None
    overall_detail = bolton.get('overall') if _ok(bolton, 'overall') else None

    if anterior_detail:
        ratio_val = anterior_detail.get('ratio')
        if isinstance(ratio_val, (int, float)):
            bolton_ratio = ratio_val / 100.0
        if extended:
            discrep_val = anterior_detail.get('discrep_mm')
            out['Bolton_Anterior_Ratio'] = round(ratio_val, 2) if isinstance(ratio_val, (int, float)) else None
            out['Bolton_Anterior_Discrepancy_mm'] = round(discrep_val, 2) if isinstance(discrep_val, (int, float)) else None
            out['Bolton_Anterior_Status'] = anterior_detail.get('status') or '缺失'
    elif extended:
        out['Bolton_Anterior_Ratio'] = None
        out['Bolton_Anterior_Discrepancy_mm'] = None
        out['Bolton_Anterior_Status'] = '缺失'

    if overall_detail and extended:
        ratio_val = overall_detail.get('ratio')
        discrep_val = overall_detail.get('discrep_mm')
        out['Bolton_Overall_Ratio'] = round(ratio_val, 2) if isinstance(ratio_val, (int, float)) else None
        out['Bolton_Overall_Discrepancy_mm'] = round(discrep_val, 2) if isinstance(discrep_val, (int, float)) else None
        out['Bolton_Overall_Status'] = overall_detail.get('status') or '缺失'
    elif extended:
        out['Bolton_Overall_Ratio'] = None
        out['Bolton_Overall_Discrepancy_mm'] = None
        out['Bolton_Overall_Status'] = '缺失'

    out['Bolton_Ratio'] = f"{bolton_ratio:.2f}" if bolton_ratio is not None else '缺失'

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

    # 6) 拥挤度（整弓 + 前牙 3-3）
    def _crowding_section(which: str) -> Optional[Dict[str, Any]]:
        if not isinstance(crowding, dict):
            return None
        return crowding.get(which) or None

    def _ald(which: str) -> Optional[float]:
        section = _crowding_section(which)
        val = section.get('ald_mm') if isinstance(section, dict) else None
        if isinstance(val, (int, float)):
            return float(round(val, 1))
        return None

    def _resolve_tag(tag: str) -> Optional[np.ndarray]:
        if tag.endswith('*') and len(tag) >= 3:
            _, point = _star_mid(landmarks, tag[:2])
            return point
        return _v(landmarks.get(tag))

    anterior_chain = {
        'upper': ['13m', '12*', '11*', '21*', '22*', '23m'],
        'lower': ['33m', '32*', '31*', '41*', '42*', '43m'],
    }
    anterior_teeth = {
        'upper': ['13', '12', '11', '21', '22', '23'],
        'lower': ['33', '32', '31', '41', '42', '43'],
    }

    def _anterior_ald(which: str) -> Optional[float]:
        chain = anterior_chain.get(which)
        teeth = anterior_teeth.get(which)
        if not chain or not teeth:
            return None
        pts = [_resolve_tag(tag) for tag in chain]
        if any(p is None for p in pts):
            return None
        arc = 0.0
        H = ops.get('H')
        if not callable(H):
            return None
        for a, b in zip(pts[:-1], pts[1:]):
            arc += H(a, b)
        if not callable(wt):
            return None
        width_sum = 0.0
        for tooth in teeth:
            w = wt(tooth, allow_fallback=True)
            if w is None:
                return None
            width_sum += w
        return float(round(width_sum - arc, 1))

    def _format_full_arch(val: Optional[float]) -> str:
        if not isinstance(val, (int, float)):
            return '缺失'
        v = round(float(val), 1)
        if v > 0:
            signed = f"+{v:.1f}"
            label = '拥挤'
        elif v < 0:
            signed = f"{v:.1f}"
            label = '间隙'
        else:
            signed = f"{v:.1f}"
            label = '无拥挤'
        return f"{signed}mm_{label}"

    def _format_anterior(val: Optional[float]) -> str:
        if not isinstance(val, (int, float)):
            return '缺失'
        v = round(float(val), 1)
        if v > 0:
            signed = f"+{v:.1f}"
            abs_v = abs(v)
            if abs_v < 4:
                grade = 'I度拥挤'
            elif abs_v < 8:
                grade = 'II度拥挤'
            else:
                grade = 'III度拥挤'
            label = grade
        elif v < 0:
            signed = f"{v:.1f}"
            label = '间隙'
        else:
            signed = f"{v:.1f}"
            label = '无拥挤'
        return f"{signed}mm_{label}"

    def _crowding_summary(which: str) -> str:
        whole_txt = _format_full_arch(_ald(which))
        anterior_txt = _format_anterior(_anterior_ald(which))
        if whole_txt == '缺失' and anterior_txt == '缺失':
            return '缺失'
        return f'整弓:"{whole_txt}" 前牙:"{anterior_txt}"'

    if _ok(crowding):
        out['Crowding_Up'] = _crowding_summary('upper')
        out['Crowding_Down'] = _crowding_summary('lower')
    else:
        out['Crowding_Up'] = '缺失'
        out['Crowding_Down'] = '缺失'

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
            note = '（正常范围）' if category in ('对刃', '轻度覆𬌗') else ''
            out['Overbite'] = f"{magnitude:.1f}mm_{category}{note}"
        else:
            out['Overbite'] = '缺失'
    else:
        out['Overbite'] = '缺失'

    # 11) Overjet
    if _ok(overjet):
        value = overjet.get('value_mm')
        category = overjet.get('category')
        if isinstance(value, (int, float)) and isinstance(category, str) and category:
            out['Overjet'] = f"{abs(float(value)):.1f}mm_{category}"
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
