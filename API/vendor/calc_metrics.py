# -*- coding: utf-8 -*-
"""
calc_metrics.py — 北大口腔·口扫模型 11 指标一体化实现（唯一版本）
口径与实现严格对齐《前置》与“计算指标以下为准”的医生定义。

接口：
    generate_metrics(upper_stl, lower_stl, upper_json, lower_json, out_path="", cfg=None) -> Dict

说明：
- 所有指标均在统一咬合平面 P 内计算（右+、上+、前+）。
- 依赖：仅标准库 + numpy。STL 路径参数保留（可忽略读取），JSON 必填。
- JSON 结构：Slicer Markups（提取 label->position）。

作者：ZJ06 项目
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import json, os

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
    if U_CI is None:
        U_CI = p
    elif p is not None:
        U_CI = 0.5*(U_CI + p)

    l_used, L_CI = _star_mid(landmarks, '31')
    l2, p = _star_mid(landmarks, '41'); l_used += l2
    if L_CI is None:
        L_CI = p
    elif p is not None:
        L_CI = 0.5*(L_CI + p)

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

    def wt(tooth: str, allow_fallback=True) -> Optional[float]:
        mc = _v(landmarks.get(f"{tooth}mc"))
        dc = _v(landmarks.get(f"{tooth}dc"))
        if allow_fallback:
            if mc is None:
                mc = _v(landmarks.get(f"{tooth}mr"))
            if mc is None:
                mc = _v(landmarks.get(f"{tooth}m"))
            if dc is None:
                dc = _v(landmarks.get(f"{tooth}dr"))
        if (mc is None) or (dc is None): return None
        return H(mc, dc)

    frame = {'origin': I, 'ex': X, 'ey': Y, 'ez': Z}
    ops = {'projP': projP, 'x': x, 'y': y, 'z': z, 'H': H}
    return {'frame': frame, 'ops': ops, 'wt': wt, 'quality': 'ok', 'warnings': warnings, 'used': used}

# =========================
# 1) Arch Form — 仅犬牙转折角 α_avg
# =========================
def compute_arch_form(landmarks: Dict[str, List[float]], frame: Dict, ops: Dict, dec:int=1) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    # 侧切“*”
    u12_used, U12s = _star_mid(landmarks, '12'); U12 = U12s[1] if isinstance(U12s, tuple) else U12s
    u22_used, U22s = _star_mid(landmarks, '22'); U22 = U22s[1] if isinstance(U22s, tuple) else U22s
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
    # 单阈值分类（延用 160/168° 分界的直觉口径）
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
def compute_bolton(landmarks: Dict[str, List[float]], ops: Dict, wt_callable, dec:int=2) -> Dict:
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
        for t in lst:
            w = wt_callable(t, allow_fallback=False)
            s += (w or 0.0)
        return s

    UA, LA = _sum_width(U_ant), _sum_width(L_ant)
    UO, LO = _sum_width(U_overall), _sum_width(L_overall)
    ratio_ant = (LA/UA)*100.0 if UA>EPS else None
    ratio_ovr = (LO/UO)*100.0 if UO>EPS else None

    # 目标与容差（医生口径）
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

    return {
        'anterior': {'ratio': None if ratio_ant is None else round(ratio_ant, dec),
                     'status': jA, 'discrep_mm': round(dA,2)},
        'overall' : {'ratio': None if ratio_ovr is None else round(ratio_ovr, dec),
                     'status': jO, 'discrep_mm': round(dO,2)},
        'quality':'ok',
        'summary_text': f"Bolton_Ratio_Bolton比*: {summary}"
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
# 5) 锁𬌗 — 前磨 & 第一磨牙，侧别判定
# =========================
def compute_crossbite(landmarks: Dict[str, List[float]], ops: Dict, tau:float=0.5) -> Dict:
    def _get(nm): return _v(landmarks.get(nm))
    def _pair(U_bu, U_pal, L_bu, L_pal):
        if any(_get(nm) is None for nm in [U_bu,U_pal,L_bu,L_pal]): return '缺失'
        xU_b = ops['x'](_get(U_bu)); xU_l = ops['x'](_get(U_pal))
        xL_b = ops['x'](_get(L_bu)); xL_l = ops['x'](_get(L_pal))
        S1 = abs(xU_l) - abs(xL_b)   # 正锁检测
        S2 = abs(xU_b) - abs(xL_l)   # 反锁检测
        if S1 > tau: return '正锁'
        if S2 < -tau: return '反锁'
        return '无'

    # 右侧
    pm_R = _pair('14b','14l','44b','44l')
    m1_R = _pair('16mb','16ml','46mb','46ml') if all(_is_xyz(landmarks.get(n)) for n in ['16ml','46ml']) \
            else _pair('16mb','16dl','46mb','46dl')  # 兜底：用 dl 近似
    # 左侧
    pm_L = _pair('24b','24l','34b','34l')
    m1_L = _pair('26mb','26ml','36mb','36ml') if all(_is_xyz(landmarks.get(n)) for n in ['26ml','36ml']) \
            else _pair('26mb','26dl','36mb','36dl')

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
# 7) Curve of Spee — P 内“弦—点”最大垂距
# =========================
def compute_spee(landmarks: Dict[str, List[float]], ops: Dict, dec:int=1) -> Optional[float]:
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
        return None

    def _side(chain, B2):
        if B2 is None: return None
        B1p = projP(A); B2p = projP(B2)
        u = B2p - B1p; nu = _len(u)
        if nu < EPS: return None
        u /= nu
        def _dist(q):
            qp = projP(q); w = qp - B1p
            return float(_len(w - u * _dot(w, u)))
        vals = []
        for nm in chain:
            p = _get(nm)
            if p is not None: vals.append(_dist(p))
        return (max(vals) if vals else None)

    chain_R = ['43m','44b','45b','46mb','46db','47mb','47db']
    chain_L = ['33m','34b','35b','36mb','36db','37mb','37db']

    r = _side(chain_R, B_R);  l = _side(chain_L, B_L)
    if r is None and l is None: return None
    val = max([v for v in [r,l] if v is not None])
    return float(np.round(val, dec))

# =========================
# 8) Midline — 上/下中线 x 偏移
# =========================
def compute_midline_alignment(landmarks: Dict[str, List[float]], ops: Dict, dec:int=0) -> Dict:
    def _mid_star(t1, t2):
        _, a = _star_mid(landmarks, t1); _, b = _star_mid(landmarks, t2)
        if a is None or b is None: return None
        return 0.5*(a+b)
    U = _mid_star('11','21');  L = _mid_star('31','41')
    if U is None or L is None:
        return {'summary_text': 'Midline_Alignment_牙列中线*: 缺失', 'quality':'missing'}
    dU = ops['x'](U);  dL = ops['x'](L)
    def _txt(v, arch):
        if abs(v) < 1e-6: return f"{arch}中线居中"
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
        W = (wt_callable(l_mc[:2], allow_fallback=False) if False else None)  # 占位
        # 直接用下颌该颗牙的实际冠宽
        W = None
        if (Lmc is not None) and (Ldc is not None):
            W = ops['H'](Lmc, Ldc)
        q = (W/4.0 if W else 2.5)
        h = (W/2.0 if W else 5.0)
        tau = 0.5
        lab = None
        if abs(off) <= tau: lab='中性'
        elif 0 < off < q: lab='中性偏近中'
        elif -q < off < 0: lab='中性偏远中'
        elif abs(off - (+q)) <= tau: lab='近中尖对尖'
        elif abs(off - (-q)) <= tau: lab='远中尖对尖'
        elif (off >= h - tau): lab='完全近中'
        elif (off <= -h + tau): lab='完全远中'
        else: lab='介于分度之间'
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
def _fmt_pass(ok: bool) -> str: return '✅' if ok else '⚠️'

def make_brief_report(landmarks: Dict[str, List[float]], frame_ops: Dict[str, Any]) -> List[str]:
    frame, ops, wt = frame_ops['frame'], frame_ops['ops'], frame_ops['wt']

    arch_form = compute_arch_form(landmarks, frame, ops)
    arch_width = compute_arch_width(landmarks, ops)
    bolton = compute_bolton(landmarks, ops, wt)
    canine = compute_canine_relationship(landmarks, ops)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_val = compute_spee(landmarks, ops)
    midline = compute_midline_alignment(landmarks, ops)
    molar = compute_molar_relationship(landmarks, ops, wt)
    ob = compute_overbite(landmarks, ops)
    oj = compute_overjet(landmarks, ops)

    return [
        f"{arch_form['summary_text']} {_fmt_pass(arch_form['quality']!='missing')}",
        f"{arch_width['summary_text']} {_fmt_pass(arch_width['quality']!='missing')}",
        f"{bolton['summary_text']} {_fmt_pass(bolton['quality']!='missing')}",
        f"{canine['summary_text']} {_fmt_pass(canine['quality']!='missing')}",
        f"{crossbite['summary_text']} {_fmt_pass(crossbite['quality']!='missing')}",
        f"{crowding['summary_text']} {_fmt_pass(crowding['quality']!='missing')}",
        f"Curve_of_Spee_Spee曲线*: {('%.1fmm' % spee_val) if spee_val is not None else '缺失'} {_fmt_pass(spee_val is not None)}",
        f"{midline['summary_text']} {_fmt_pass(midline['quality']!='missing')}",
        f"{molar['summary_text']} {_fmt_pass(molar['quality']!='missing')}",
        f"{ob['summary_text']} {_fmt_pass(ob['quality']!='missing')}",
        f"{oj['summary_text']} {_fmt_pass(oj['quality']!='missing')}",
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
def _load_landmarks_json(path: str) -> Dict[str, List[float]]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    out = {}
    if not data or 'markups' not in data: return out
    for mk in data['markups']:
        for cp in mk.get('controlPoints', []):
            label, pos = cp.get('label'), cp.get('position')
            if label and isinstance(pos, list) and len(pos)==3:
                try: out[str(label)] = [float(pos[0]), float(pos[1]), float(pos[2])]
                except: pass
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

    arch_form = compute_arch_form(landmarks, frame, ops)
    arch_width = compute_arch_width(landmarks, ops)
    bolton = compute_bolton(landmarks, ops, wt)
    canine = compute_canine_relationship(landmarks, ops)
    crossbite = compute_crossbite(landmarks, ops)
    crowding = compute_crowding(landmarks, ops, wt)
    spee_val = compute_spee(landmarks, ops)
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

    # 3) Bolton（前牙比换算成小数）
    bolton_ratio = None
    if _ok(bolton, 'anterior'):
        ratio = (bolton.get('anterior') or {}).get('ratio')
        if isinstance(ratio, (int, float)):
            bolton_ratio = ratio / 100.0
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

    # 6) 拥挤度（数值：正=拥挤，负=间隙）
    def _ald(which: str) -> Optional[float]:
        if not _ok(crowding):
            return None
        section = crowding.get(which) or {}
        val = section.get('ald_mm')
        if isinstance(val, (int, float)):
            return float(round(val, 1))
        return None

    out['Crowding_Up'] = _ald('upper')
    out['Crowding_Down'] = _ald('lower')

    # 7) Spee
    out['Curve_of_Spee'] = f"{spee_val:.1f}mm" if isinstance(spee_val, (int, float)) else '缺失'

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
        right_molar = (molar.get('right') or {}).get('label')
        left_molar = (molar.get('left') or {}).get('label')
        out['Molar_Relationship_Right'] = right_molar if right_molar else '缺失'
        out['Molar_Relationship_Left'] = left_molar if left_molar else '缺失'
    else:
        out['Molar_Relationship_Right'] = '缺失'
        out['Molar_Relationship_Left'] = '缺失'

    # 10) Overbite
    if _ok(overbite):
        category = overbite.get('category') or '缺失'
        if cfg.get('ob_simple', True):
            out['Overbite'] = '正常' if category in ('对刃', '轻度覆𬌗') else category
        else:
            out['Overbite'] = category
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

    return out

def generate_metrics(
    upper_stl_path: str,
    lower_stl_path: str,
    upper_json_path: str,
    lower_json_path: str,
    out_path: str = "",
    cfg: Optional[Dict] = None
) -> Dict:
    """
    输入：上/下 STL + 上/下 JSON；输出：key-value JSON（可选写盘）。
    """
    # 1) 读取 landmarks
    lm_upper = _load_landmarks_json(upper_json_path)
    lm_lower = _load_landmarks_json(lower_json_path)
    landmarks = _merge_landmarks(lm_upper, lm_lower)

    # 2) 前置坐标系（严格 I/L/R）
    frame_ops = build_module0(landmarks)
    if frame_ops.get('frame') is None or frame_ops.get('ops') is None:
        kv = {"错误": "坐标系缺失，无法生成报告"}
        if out_path:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(kv, f, ensure_ascii=False, indent=2)
        return kv

    cfg = cfg or {}
    profile = cfg.get('profile', 'brief')

    if profile == 'doctor_cn_simple':
        kv = make_doctor_cn_simple(landmarks, frame_ops, cfg=cfg)
    else:
        brief = make_brief_report(landmarks, frame_ops)
        kv = _brief_lines_to_kv(brief)

    # 4) 可选落盘
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kv, f, ensure_ascii=False, indent=2)
    return kv

# 可选 CLI
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Ortho analysis → brief key-value JSON")
    ap.add_argument("--upper_stl", required=True)
    ap.add_argument("--lower_stl", required=True)
    ap.add_argument("--upper_json", required=True)
    ap.add_argument("--lower_json", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    kv = generate_metrics(args.upper_stl, args.lower_stl, args.upper_json, args.lower_json, out_path=args.out)
    print(f"saved to: {args.out} ({len(kv)} items)")
