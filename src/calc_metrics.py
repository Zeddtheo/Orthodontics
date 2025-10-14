
# -*- coding: utf-8 -*-
"""
calc_metrics_four.py — 四项指标的极简实现（单函数、无兜底、全部在 P 内）
依赖：传入的 frame 必须来自你的 Module 0（origin/ex/ey/ez；右+、上+、前+）
  - 投影到 P：projP(p) = p - ez * dot(ez, p - origin)
  - 平面坐标：x=dot(ex, projP(p)-origin), y=dot(ey, projP(p)-origin) （y 为前后）
  - 水平距离：H(a,b) = ||projP(b) - projP(a)||
点位：严格使用 244 点字典（FDI）。
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

# ------------------------- helpers -------------------------
def _is_xyz(p) -> bool:
    return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()

def _get(landmarks: Dict[str, List[float]], nm: str) -> Optional[np.ndarray]:
    p = landmarks.get(nm)
    return np.asarray(p, float) if _is_xyz(p) else None

def _projP(frame: Dict, p: np.ndarray) -> np.ndarray:
    o = np.asarray(frame['origin'], float)
    ez = np.asarray(frame['ez'], float)
    v = np.asarray(p, float) - o
    return o + v - ez * float(np.dot(v, ez))

def _xy(frame: Dict, p: np.ndarray) -> Tuple[float, float]:
    o  = np.asarray(frame['origin'], float)
    ex = np.asarray(frame['ex'], float)
    ey = np.asarray(frame['ey'], float)
    q = _projP(frame, p) - o
    return float(np.dot(q, ex)), float(np.dot(q, ey))

def _H(frame: Dict, a: np.ndarray, b: np.ndarray) -> float:
    pa = _projP(frame, a); pb = _projP(frame, b)
    return float(np.linalg.norm(pb - pa))

# ==========================================================
# 1) Arch_Form_牙弓形态* —— 仅用“犬牙转折角 α_avg”
# ==========================================================
def compute_arch_form(landmarks: Dict[str, List[float]], frame: Dict, dec: int = 1) -> Dict:
    """
    只用犬牙转折角 α：
      αR = ∠(12* — 13m — 14b), αL = ∠(22* — 23m — 24b), α_avg=(αR+αL)/2。
      判型：α_avg ≤ 160° → 尖圆形；α_avg ≥ 168° → 方圆形；否则卵圆形。
    必要点：12ma/12da、22ma/22da、13m/23m、14b/24b。
    全部在 P 内测角（先投影，再在 ex/ey 平面做二维角度）。
    返回：{'form','indices','used','summary_text','quality'}。
    """
    req = ['12ma','12da','22ma','22da','13m','23m','14b','24b']
    used = [nm for nm in req if _get(landmarks, nm) is not None]
    if len(used) != len(req) or (not frame or any(k not in frame for k in ('origin','ex','ey','ez'))):
        return {'form':'缺失','indices':{},'used':used,'summary_text':'Arch_Form_牙弓形态*: 缺失','quality':'missing'}

    # 星号点
    p12s = 0.5*(_get(landmarks,'12ma') + _get(landmarks,'12da'))
    p22s = 0.5*(_get(landmarks,'22ma') + _get(landmarks,'22da'))

    def _angle_deg(A, B, C) -> Optional[float]:
        ax, ay = _xy(frame, A); bx, by = _xy(frame, B); cx, cy = _xy(frame, C)
        u = np.array([ax - bx, ay - by], float)
        v = np.array([cx - bx, cy - by], float)
        nu = float(np.linalg.norm(u)); nv = float(np.linalg.norm(v))
        if nu < 1e-9 or nv < 1e-9: return None
        cosv = float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))
        return float(np.degrees(np.arccos(cosv)))

    aR = _angle_deg(p12s, _get(landmarks,'13m'), _get(landmarks,'14b'))
    aL = _angle_deg(p22s, _get(landmarks,'23m'), _get(landmarks,'24b'))
    if aR is None or aL is None:
        return {'form':'缺失','indices':{},'used':req,'summary_text':'Arch_Form_牙弓形态*: 缺失','quality':'missing'}

    alpha_avg = 0.5*(aR + aL)
    if   alpha_avg <= 160.0: form = '尖圆形'
    elif alpha_avg >= 168.0: form = '方圆形'
    else:                    form = '卵圆形'

    idx = {'alpha_deg': round(float(alpha_avg), dec),
           'alpha_right_deg': round(float(aR), dec),
           'alpha_left_deg':  round(float(aL), dec)}
    return {'form': form, 'indices': idx, 'used': req,
            'summary_text': f"Arch_Form_牙弓形态*: {form}", 'quality': 'ok'}

# ==========================================================
# 2) Arch_Width_牙弓宽度* —— 三段宽度，上下比较
# ==========================================================
def compute_arch_width(landmarks: Dict[str, List[float]], frame: Dict, dec: int = 1) -> Dict:
    """
    仅在 P 平面内计算三段宽度：
      上：H(13m,23m), H(14b,24b), H(16mb,26mb)
      下：H(33m,43m), H(34b,44b), H(36mb,46mb)
    判定“上牙弓较窄”：三段里至少两段满足 上<下。
    返回：{'upper','lower','diff_UL_mm','upper_is_narrow','summary_text','quality'}。
    """
    req = ['13m','23m','14b','24b','16mb','26mb','33m','43m','34b','44b','36mb','46mb']
    used = [nm for nm in req if _get(landmarks, nm) is not None]
    if len(used) != len(req) or (not frame or any(k not in frame for k in ('origin','ex','ey','ez'))):
        return {'upper':None,'lower':None,'diff_UL_mm':None,'upper_is_narrow':None,
                'summary_text':'Arch_Width_牙弓宽度*: 缺失','quality':'missing'}

    def _w(a,b): return _H(frame, _get(landmarks,a), _get(landmarks,b))

    u_ant = _w('13m','23m'); u_mid = _w('14b','24b'); u_pos = _w('16mb','26mb')
    l_ant = _w('33m','43m'); l_mid = _w('34b','44b'); l_pos = _w('36mb','46mb')

    upper = {'anterior_mm': round(u_ant,dec), 'middle_mm': round(u_mid,dec), 'posterior_mm': round(u_pos,dec)}
    lower = {'anterior_mm': round(l_ant,dec), 'middle_mm': round(l_mid,dec), 'posterior_mm': round(l_pos,dec)}

    diff = {'anterior': round(u_ant-l_ant,dec), 'middle': round(u_mid-l_mid,dec), 'posterior': round(u_pos-l_pos,dec)}
    votes = sum(1 for v in diff.values() if v < 0.0)
    upper_is_narrow = (votes >= 2)
    summary = 'Arch_Width_牙弓宽度*: 上牙弓较窄' if upper_is_narrow else 'Arch_Width_牙弓宽度*: 未见上牙弓较窄'

    return {'upper': upper, 'lower': lower, 'diff_UL_mm': diff,
            'upper_is_narrow': bool(upper_is_narrow), 'summary_text': summary, 'quality': 'ok'}

# ==========================================================
# 3) Bolton_Ratio_Bolton比 —— 严格 mc/dc（P 内），固阈值
# ==========================================================
def compute_bolton(landmarks: Dict[str, List[float]], frame: Dict, cfg=None, dec: int = 2) -> Dict:
    """
    单牙宽度：wt = ||projP(dc) - projP(mc)||
    牙位：
      前牙比（6牙） 上：13,12,11,21,22,23   下：33,32,31,41,42,43
      全牙比（12牙）上：16..26             下：36..46
    正常阈值：前牙比 78.8% ± 1.72%；全牙比 91.5% ± 1.51%。
    缺任一牙 mc/dc → 直接 error（不做回退）。
    返回：{'kind','anterior','overall','quality','used','summary_text'}。
    """
    # 固定口径（忽略 cfg 中任何覆写）
    TARGET_A, TOL_A = 78.8, 1.72
    TARGET_O, TOL_O = 91.5, 1.51

    if not frame or any(k not in frame for k in ('origin','ex','ey','ez')):
        return {'kind':'Bolton_Ratio','anterior':None,'overall':None,'quality':'missing','used':{},'summary_text':'Bolton缺失'}

    UO = ['16','15','14','13','12','11','21','22','23','24','25','26']
    LO = ['36','35','34','33','32','31','41','42','43','44','45','46']
    UA = ['13','12','11','21','22','23']
    LA = ['33','32','31','41','42','43']

    def _wt(tooth: str) -> Optional[float]:
        pm = _get(landmarks, f'{tooth}mc')
        pd = _get(landmarks, f'{tooth}dc')
        if pm is None or pd is None: return None
        return _H(frame, pm, pd)

    # 严格完整性检查
    missing = {t: [s for s in ('mc','dc') if _get(landmarks,f'{t}{s}') is None]
               for t in sorted(set(UO+LO))}
    missing = {t:lack for t,lack in missing.items() if lack}
    if missing:
        miss_text = ', '.join([f"{t}缺{'+'.join(v)}" for t,v in sorted(missing.items())])
        return {'kind':'Bolton_Ratio','anterior':None,'overall':None,'quality':'error',
                'used':{'missing_points':missing}, 'summary_text':f'Bolton输入缺失：{miss_text}'}

    def _sum(teeth: List[str]) -> float:
        s = 0.0
        for t in teeth:
            w = _wt(t)
            s += w
        return s

    sum_UA, sum_LA = _sum(UA), _sum(LA)
    sum_UO, sum_LO = _sum(UO), _sum(LO)

    def _pack(sumU, sumL, target, tol, nU, nL):
        ratio = (sumL/sumU)*100.0 if sumU>1e-9 else None
        discrep = (sumL - sumU*target/100.0) if ratio is not None else None
        if ratio is None: status = '缺失'
        elif abs(ratio-target) <= tol: status = '正常'
        elif ratio > target + tol: status = '下颌牙量过大'
        else: status = '上颌牙量过大'
        return {
            'ratio': None if ratio is None else round(ratio, dec),
            'target': target, 'tol': tol,
            'sum_upper_mm': round(sumU, dec), 'sum_lower_mm': round(sumL, dec),
            'discrep_mm': None if discrep is None else round(discrep, dec),
            'status': status, 'n_upper': nU, 'n_lower': nL
        }

    anterior = _pack(sum_UA, sum_LA, TARGET_A, TOL_A, len(UA), len(LA))
    overall  = _pack(sum_UO, sum_LO, TARGET_O, TOL_O, len(UO), len(LO))

    both_ok = (anterior['status']=='正常' and overall['status']=='正常')
    return {'kind':'Bolton_Ratio','anterior':anterior,'overall':overall,'quality':'ok',
            'used':{'upper_overall':UO,'lower_overall':LO,'upper_anterior':UA,'lower_anterior':LA},
            'summary_text': ('正常' if both_ok else f"前牙比{anterior['status']}；全牙比{overall['status']}")}

# ==========================================================
# 4) Canine_Relationship_尖牙关系*
#    只用 y() 前后分量；阈值：尖对尖 ±0.5 mm，完全 ±2.0 mm
# ==========================================================
def compute_canine_relationship(landmarks, frame, dec: int = 1,
                                tau_mm: float = 0.5, complete_mm: float = 2.0):
    def _is_xyz(p): 
        return isinstance(p,(list,tuple,np.ndarray)) and len(p)==3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm);  return np.asarray(p,float) if _is_xyz(p) else None

    # 坐标与投影（P 面）
    if not frame or any(k not in frame for k in ('origin','ex','ey','ez')):
        return {'right':None,'left':None,'ref_point':'lower_canine.dc',
                'summary_text':'Canine_Relationship_尖牙关系*: 缺失','quality':'missing'}
    o  = np.asarray(frame['origin'],float)
    ey = np.asarray(frame['ey'],float)
    ez = np.asarray(frame['ez'],float)
    def _projP(p): 
        v = np.asarray(p,float) - o
        return o + v - ez * float(np.dot(v, ez))
    def _y(p):
        q = _projP(p) - o
        return float(np.dot(q, ey))

    # 取点（严格按 244 点字典）
    U_R, U_L = _get('13m'), _get('23m')
    L_R, L_L = _get('43dc'), _get('33dc')
    if any(v is None for v in (U_R,U_L,L_R,L_L)):
        return {'right':None,'left':None,'ref_point':'lower_canine.dc',
                'summary_text':'Canine_Relationship_尖牙关系*: 缺失','quality':'missing'}

    # 差值（前为近中，后为远中）
    dy_R = _y(U_R) - _y(L_R)
    dy_L = _y(U_L) - _y(L_L)

    def _label(dy: float) -> str:
        if abs(dy) <= tau_mm:     return '尖对尖'       # =“中性”
        if dy >=  complete_mm:    return '完全近中'
        if dy <= -complete_mm:    return '完全远中'
        return '近中尖对尖' if dy > 0 else '远中尖对尖'

    right = {'class': _label(dy_R), 'dy_mm': float(np.round(dy_R, dec))}
    left  = {'class': _label(dy_L), 'dy_mm': float(np.round(dy_L, dec))}
    summary = f"Canine_Relationship_尖牙关系*: 右侧{right['class']}，左侧{left['class']}"
    return {'right': right, 'left': left, 'ref_point':'lower_canine.dc',
            'tau_mm': float(tau_mm), 'complete_mm': float(complete_mm),
            'summary_text': summary, 'quality':'ok'}

# ==========================================================
# 5) Crossbite_锁牙合* —— 仅用 P 面 x() 横向分量
# ==========================================================
def compute_crossbite(landmarks, frame, dec: int = 1, tau_mm: float = 0.5):
    """
    Crossbite_锁牙合（按侧、按牙段：第一前磨牙段 + 第一磨牙段）
    医学判定：
      - 正锁：上颌磨牙舌尖舌斜面 对 下颌磨牙颊尖颊斜面（S1 > τ）
      - 反锁：上颌磨牙颊尖颊斜面 对 下颌磨牙舌尖舌斜面（S2 < -τ）
    实现（全部在 P 面，取 x() 横向坐标；右+=颊向右）：
      S1 = |x(U_pal)| - |x(L_buc)|     # 正锁检测
      S2 = |x(U_buc)| - |x(L_pal)|     # 反锁检测
      if S1 > τ → 正锁；elif S2 < -τ → 反锁；else → 无
    返回：
      {
        'right': {'premolar':..., 'molar':...},   # 各段：'正锁'/'反锁'/'无'/'缺失'
        'left' : {'premolar':..., 'molar':...},
        'summary_text': 'Crossbite_锁牙合: 无 / 右侧正锁𬌗（累及第一磨牙）…',
        'tau_mm': 0.5,
        'quality': 'ok|fallback|missing',
        'details': {...S1,S2 与取点...}
      }
    """
    def _is_xyz(p):
        return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm);  return (np.asarray(p, float) if _is_xyz(p) else None)

    # 坐标与投影到 P
    if not frame or any(k not in frame for k in ('origin','ex','ez')):
        return {'right': None, 'left': None, 'summary_text': 'Crossbite_锁牙合: 缺失',
                'tau_mm': float(tau_mm), 'quality': 'missing', 'details': {}}
    o  = np.asarray(frame['origin'], float)
    ex = np.asarray(frame['ex'], float)
    ez = np.asarray(frame['ez'], float)

    def _projP(p):
        v = np.asarray(p, float) - o
        return o + v - ez * float(np.dot(v, ez))
    def _x(p):
        q = _projP(p) - o
        return float(np.dot(q, ex))

    def _mean2(a, b):
        return 0.5*(a+b) if (a is not None and b is not None) else None

    # ---- 单段判定：'premolar' 或 'molar'，侧别 'right' / 'left' ----
    def _judge(segment: str, side: str):
        if segment == 'premolar':
            if side == 'right':
                U_bu = _get('14b'); U_pal = _get('14l')
                L_bu = _get('44b'); L_pal = _get('44l')
            else:
                U_bu = _get('24b'); U_pal = _get('24l')
                L_bu = _get('34b'); L_pal = _get('34l')
        else:  # 'molar' (第一磨牙段，用双尖均值代表颊/舌侧)
            if side == 'right':
                U_bu = _mean2(_get('16mb'), _get('16db'))
                U_pal= _mean2(_get('16ml'), _get('16dl'))
                L_bu = _mean2(_get('46mb'), _get('46db'))
                L_pal= _mean2(_get('46ml'), _get('46dl'))
            else:
                U_bu = _mean2(_get('26mb'), _get('26db'))
                U_pal= _mean2(_get('26ml'), _get('26dl'))
                L_bu = _mean2(_get('36mb'), _get('36db'))
                L_pal= _mean2(_get('36ml'), _get('36dl'))

        # 必需点齐全
        if any(p is None for p in (U_bu, U_pal, L_bu, L_pal)):
            return {'status': '缺失', 'S1': None, 'S2': None,
                    'x': {'U_bu': None, 'U_pal': None, 'L_bu': None, 'L_pal': None}}

        xU_b, xU_l = _x(U_bu), _x(U_pal)
        xL_b, xL_l = _x(L_bu), _x(L_pal)

        S1 = abs(xU_l) - abs(xL_b)   # 正锁检测
        S2 = abs(xU_b) - abs(xL_l)   # 反锁检测

        if S1 > tau_mm:
            st = '正锁'
        elif S2 < -tau_mm:
            st = '反锁'
        else:
            st = '无'

        return {'status': st,
                'S1': float(np.round(S1, 2)), 'S2': float(np.round(S2, 2)),
                'x': {'U_bu': float(np.round(xU_b, 2)), 'U_pal': float(np.round(xU_l, 2)),
                      'L_bu': float(np.round(xL_b, 2)), 'L_pal': float(np.round(xL_l, 2))}}

    R_pre = _judge('premolar', 'right')
    R_mol = _judge('molar',    'right')
    L_pre = _judge('premolar', 'left')
    L_mol = _judge('molar',    'left')

    def _side_summary(side_name, pre, mol):
        # 侧别总判：若两段均“无”→无；若仅出现一种类型→该类型；若两种并存→“锁𬌗”
        types = {t for t in (pre['status'], mol['status']) if t in ('正锁', '反锁')}
        if not types:
            return '无', None
        if len(types) == 1:
            label = f"{side_name}侧{list(types)[0]}𬌗"
        else:
            label = f"{side_name}侧锁𬌗"
        involved = []
        if pre['status'] in ('正锁','反锁'): involved.append('第一前磨牙')
        if mol['status'] in ('正锁','反锁'): involved.append('第一磨牙')
        ext = f"（累及{'、'.join(involved)}）" if involved else ''
        return label + ext, types

    sr_txt, _ = _side_summary('右', R_pre, R_mol)
    sl_txt, _ = _side_summary('左', L_pre, L_mol)

    # 汇总
    both_missing = (R_pre['status']=='缺失' and R_mol['status']=='缺失' and
                    L_pre['status']=='缺失' and L_mol['status']=='缺失')
    both_none = (sr_txt == '无' and sl_txt == '无')

    if both_missing:
        summary = 'Crossbite_锁牙合: 缺失'
        quality = 'missing'
    elif both_none:
        summary = 'Crossbite_锁牙合: 无'
        quality = 'ok'
    else:
        parts = []
        if sr_txt != '无' and not (R_pre['status']=='缺失' and R_mol['status']=='缺失'):
            parts.append(sr_txt)
        if sl_txt != '无' and not (L_pre['status']=='缺失' and L_mol['status']=='缺失'):
            parts.append(sl_txt)
        summary = 'Crossbite_锁牙合: ' + '，'.join(parts) if parts else 'Crossbite_锁牙合: 无'
        quality = 'ok' if '缺失' not in (R_pre['status'], R_mol['status'], L_pre['status'], L_mol['status']) else 'fallback'

    return {
        'right': {'premolar': R_pre['status'], 'molar': R_mol['status']},
        'left' : {'premolar': L_pre['status'], 'molar': L_mol['status']},
        'summary_text': summary,
        'tau_mm': float(tau_mm),
        'quality': quality,
        'details': {
            'right': {'premolar': R_pre, 'molar': R_mol},
            'left' : {'premolar': L_pre, 'molar': L_mol}
        }
    }

# ==========================================================
# 6) Crowding_拥挤度*
#    拥挤度 = Σ(17–27 或 47–37 的 wt) − 弓长
#    弓长：P 面沿“颊/切缘锚点链”折线长度（端点=第二磨牙 dc）
# ==========================================================
def compute_crowding(landmarks, frame, dec: int = 1):
    def _is_xyz(p): 
        return isinstance(p,(list,tuple,np.ndarray)) and len(p)==3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm);  return np.asarray(p,float) if _is_xyz(p) else None

    if not frame or any(k not in frame for k in ('origin','ex','ey','ez')):
        return {'upper':None,'lower':None,'summary_text':'Crowding_拥挤度*: 缺失','quality':'missing'}

    o  = np.asarray(frame['origin'],float)
    ez = np.asarray(frame['ez'],float)
    ex = np.asarray(frame['ex'],float); ey = np.asarray(frame['ey'],float)
    def _projP(p):
        v = np.asarray(p,float) - o
        return o + v - ez * float(np.dot(v, ez))
    def _xy(p):
        q = _projP(p) - o
        return float(np.dot(q, ex)), float(np.dot(q, ey))
    def _H(a,b):
        pa, pb = _projP(a), _projP(b)
        return float(np.linalg.norm(pb - pa))
    def _star(tooth: str):
        a, d = _get(f'{tooth}ma'), _get(f'{tooth}da')
        return (0.5*(a+d) if (a is not None and d is not None) else None)

    def _sum_widths(teeth):
        # 严格 mc+dc（缺任一颗则返回 None）
        s = 0.0
        for t in teeth:
            pm, pd = _get(f'{t}mc'), _get(f'{t}dc')
            if pm is None or pd is None: return None
            s += _H(pm, pd)
        return s

    def _arc_length(anchor_names):
        # 取齐锚点并转为 2D；有缺即返回 None
        P = []
        for nm in anchor_names:
            if nm.endswith('*'):
                base = nm[:-1]
                p = _star(base)
            else:
                p = _get(nm)
            if p is None: return None
            P.append(np.array(_xy(p), float))
        # 折线长度
        L = 0.0
        for i in range(1, len(P)):
            L += float(np.linalg.norm(P[i] - P[i-1]))
        return L

    # —— 上颌 ——（17..27）
    U_teeth  = ['17','16','15','14','13','12','11','21','22','23','24','25','26','27']
    U_anchor = ['17dc','16mb','15b','14b','13m','12*','11*','21*','22*','23m','24b','25b','26mb','27dc']
    sumU = _sum_widths(U_teeth)
    arcU = _arc_length(U_anchor)

    # —— 下颌 ——（47..37）
    L_teeth  = ['47','46','45','44','43','42','41','31','32','33','34','35','36','37']
    L_anchor = ['47dc','46mb','45b','44b','43m','42*','41*','31*','32*','33m','34b','35b','36mb','37dc']
    sumL = _sum_widths(L_teeth)
    arcL = _arc_length(L_anchor)

    def _pack(sumW, arc, arch_name):
        if sumW is None or arc is None:
            return None, None, None, f"{arch_name}缺失"
        ald = sumW - arc
        tag = '拥挤' if ald > 0 else '间隙'
        return (round(sumW, dec), round(arc, dec), round(ald, dec),
                f"{arch_name}{tag}{abs(round(ald, dec))}mm")

    uW,uA,uALD,uTxt = _pack(sumU, arcU, '上牙列')
    lW,lA,lALD,lTxt = _pack(sumL, arcL, '下牙列')

    if uW is None and lW is None:
        return {'upper':None,'lower':None,'summary_text':'Crowding_拥挤度*: 缺失','quality':'missing'}

    upper = None if uW is None else {'sum_widths_mm':uW,'arch_length_mm':uA,'ald_mm':uALD}
    lower = None if lW is None else {'sum_widths_mm':lW,'arch_length_mm':lA,'ald_mm':lALD}
    parts = [p for p in (uTxt, lTxt) if p and ('缺失' not in p)]
    summary = 'Crowding_拥挤度*:' + (''.join(parts) if parts else '缺失')
    quality = 'ok' if (upper or lower) else 'missing'
    return {'upper': upper, 'lower': lower, 'summary_text': summary, 'quality': quality}

# ==========================================================
# 7) Curve_of_Spee_Spee曲线*（在 P 内：基线 ⟂ 距离的最大值）
# ==========================================================
def compute_spee(landmarks, frame, dec: int = 1):
    """
    思路：在 P 内做“基线”与“颊尖链”，量二者的法向距离（线内垂距）的最大值。
      1) 基线（左右各一条）
         - 前端点：B1 = mean(31ma, 41ma)
         - 右端点：B2_R = 47db；若无则 46db
         - 左端点：B2_L = 37db；若无则 36db
      2) 颊尖链（下颌颊侧/尖端点）
         - 右：43m, 44b, 45b, 46mb, 46db, [47mb, 47db(若存在)]
         - 左：33m, 34b, 35b, 36mb, 36db, [37mb, 37db(若存在)]
      3) 距离（全部投影到 P 后计算）
         对侧别 s∈{L,R}：u_s = normalize(B2p_s - B1p)，
         dist(q) = || (qp - B1p) - dot(qp - B1p, u_s) * u_s ||
      4) Spee 值
         Spee_R = max(dist(q) for 右侧链)；Spee_L 同理；Spee = max(Spee_R, Spee_L)

    返回：float(mm)；若关键点缺失则返回 None（保持最简口径，无额外兜底）
    """
    def _is_xyz(p): 
        return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm)
        return np.asarray(p, float) if _is_xyz(p) else None

    # 坐标系与投影到 P
    if not frame or any(k not in frame for k in ('origin', 'ex', 'ey', 'ez')):
        return None
    o  = np.asarray(frame['origin'], float)
    ez = np.asarray(frame['ez'],     float)
    def _projP(p):
        v = np.asarray(p, float) - o
        return o + v - ez * float(np.dot(v, ez))

    # ---- 基线端点 ----
    p31ma, p41ma = _get('31ma'), _get('41ma')
    if p31ma is None or p41ma is None:
        return None
    B1 = 0.5 * (p31ma + p41ma)

    B2_R = _get('47db') if _get('47db') is not None else _get('46db')
    B2_L = _get('37db') if _get('37db') is not None else _get('36db')
    if B2_R is None and B2_L is None:
        return None

    # ---- 颊尖链 ----
    chain_R_names = ['43m', '44b', '45b', '46mb', '46db', '47mb', '47db']
    chain_L_names = ['33m', '34b', '35b', '36mb', '36db', '37mb', '37db']
    chain_R = [_get(nm) for nm in chain_R_names if _get(nm) is not None]
    chain_L = [_get(nm) for nm in chain_L_names if _get(nm) is not None]

    def _side_max(B2, chain):
        if B2 is None or not chain:
            return None
        B1p = _projP(B1); B2p = _projP(B2)
        u   = B2p - B1p
        L   = float(np.linalg.norm(u))
        if L < 1e-9:
            return None
        u  /= L
        dmax = 0.0
        for q in chain:
            qp = _projP(q)
            w  = qp - B1p
            # 线内法向分量
            v  = w - u * float(np.dot(w, u))
            d  = float(np.linalg.norm(v))
            if d > dmax:
                dmax = d
        return dmax

    SR = _side_max(B2_R, chain_R)
    SL = _side_max(B2_L, chain_L)
    vals = [v for v in (SR, SL) if isinstance(v, (int, float))]
    if not vals:
        return None
    return float(np.round(max(vals), dec))

# ==========================================================
# 8) Midline_Alignment_牙列中线*（在 P 内：取 x() 偏移，x>0=右，x<0=左）
# ==========================================================
def compute_midline_alignment(landmarks, frame, dec: int = 1):
    """
    U_mid = mean( (11ma+11da)/2 , (21ma+21da)/2 )
    L_mid = mean( (31ma+31da)/2 , (41ma+41da)/2 )
    δU = x(U_mid), δL = x(L_mid)    # x>0 为右偏，x<0 为左偏（全在 P 内）
    返回：
      {
        'kind': 'Midline_Alignment',
        'upper': {'dir': '右偏|左偏|居中', 'signed_x_mm': δU_round},
        'lower': {'dir': '右偏|左偏|居中', 'signed_x_mm': δL_round},
        'summary_text': 'Midline_Alignment_牙列中线*: 上中线右偏1mm 下中线右偏1mm',
        'quality': 'ok|missing'
      }
    """
    def _is_xyz(p): 
        return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm)
        return np.asarray(p, float) if _is_xyz(p) else None

    if not frame or any(k not in frame for k in ('origin', 'ex', 'ey', 'ez')):
        return {
            'kind': 'Midline_Alignment',
            'upper': None, 'lower': None,
            'summary_text': 'Midline_Alignment_牙列中线*: 缺失',
            'quality': 'missing'
        }

    o  = np.asarray(frame['origin'], float)
    ex = np.asarray(frame['ex'],     float)
    ez = np.asarray(frame['ez'],     float)

    def _projP(p):
        v = np.asarray(p, float) - o
        return o + v - ez * float(np.dot(v, ez))
    def _x(p):
        q = _projP(p) - o
        return float(np.dot(q, ex))
    def _star(tooth: str):
        ma, da = _get(f'{tooth}ma'), _get(f'{tooth}da')
        return 0.5 * (ma + da) if (ma is not None and da is not None) else None

    U11s, U21s = _star('11'), _star('21')
    L31s, L41s = _star('31'), _star('41')
    if None in (U11s, U21s, L31s, L41s):
        return {
            'kind': 'Midline_Alignment',
            'upper': None, 'lower': None,
            'summary_text': 'Midline_Alignment_牙列中线*: 缺失',
            'quality': 'missing'
        }

    U_mid = 0.5 * (U11s + U21s)
    L_mid = 0.5 * (L31s + L41s)
    dU, dL = _x(U_mid), _x(L_mid)

    def _pack(v):
        v_round = float(np.round(v, dec))
        dir_txt = '居中' if abs(v_round) < 1e-9 else ('右偏' if v_round > 0 else '左偏')
        return {'dir': dir_txt, 'signed_x_mm': v_round}

    up  = _pack(dU)
    low = _pack(dL)

    # 文案按整数 mm 输出（与示例一致）
    def _fmt_mm_abs(v):
        return f"{int(round(abs(v)))}mm"

    summary = (
        f"Midline_Alignment_牙列中线*: "
        f"上中线{up['dir']}{_fmt_mm_abs(up['signed_x_mm'])} "
        f"下中线{low['dir']}{_fmt_mm_abs(low['signed_x_mm'])}"
    )

    return {
        'kind': 'Midline_Alignment',
        'upper': up,
        'lower': low,
        'summary_text': summary,
        'quality': 'ok'
    }

# ==========================================================
# 9) Molar_Relationship_磨牙关系*（七档判读；在 P 内）
# ==========================================================
def compute_molar_relationship(landmarks, frame, dec:int=1, tau_mm:float=0.5,
                               use_width_thresholds:bool=True, default_q:float=2.5):
    """
    Molar_Relationship_磨牙关系*（七档判读；在 P 内）
    - 右侧 offset_R = y(16mb) - y(46bg)
    - 左侧 offset_L = y(26mb) - y(36bg)
    - 个体化阈值：q = W/4, h = W/2（W 为下颌第一磨牙 mc–dc 在 P 内的距离）；
      若缺失则回退 q=2.5, h=5.0（单位：mm）
    - τ=0.5mm 作为“基本重合”容差

    返回:
    {
      'right': {'class','offset_mm','W_mm','q_mm','h_mm','used':{...}},
      'left' : {'class','offset_mm','W_mm','q_mm','h_mm','used':{...}},
      'summary_text': 'Molar_Relationship_磨牙关系*: 右侧… 左侧…',
      'quality': 'ok|missing'
    }
    """
    # ---- helpers ----
    def _is_xyz(p):
        return isinstance(p,(list,tuple,np.ndarray)) and len(p)==3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm);  return np.asarray(p,float) if _is_xyz(p) else None
    if not frame or any(k not in frame for k in ('origin','ex','ey','ez')):
        return {'right':None, 'left':None, 'summary_text':'Molar_Relationship_磨牙关系*: 缺失', 'quality':'missing'}

    o  = np.asarray(frame['origin'],float)
    ey = np.asarray(frame['ey'],float)
    ez = np.asarray(frame['ez'],float)
    def _projP(p):  # 投影到 P
        v = np.asarray(p,float) - o
        return o + v - ez * float(np.dot(v, ez))
    def _y(p):
        q = _projP(p) - o
        return float(np.dot(q, ey))
    def _H(a,b):
        pa, pb = _projP(a), _projP(b)
        return float(np.linalg.norm(pb - pa))

    # ---- 个体化阈值（W->q,h） ----
    def _width_qh(mc_nm, dc_nm):
        pm, pd = _get(mc_nm), _get(dc_nm)
        if pm is not None and pd is not None:
            W = _H(pm, pd)
            q = 0.25 * W; h = 0.5 * W
            return float(W), float(q), float(h)
        # 回退常数制
        return None, float(default_q), float(2*default_q)

    # ---- 单侧计算：offset + 分类 ----
    def _classify(offset, q, h, tau=tau_mm):
        # 正向为“近中”，负向为“远中”
        if abs(offset) <= tau:                 return '中性关系'
        if offset > 0:  # 近中方向
            if abs(offset - q) <= tau:         return '近中尖对尖'
            if abs(offset - h) <= tau or offset >= h:  return '完全近中'
            if 0 < offset < q:                 return '中性偏近中'
            # 介于 q 与 h 之间 → 按更接近者贴标签（取更近的标准档）
            return '近中尖对尖' if abs(offset-q) < abs(offset-h) else '完全近中'
        else:          # 远中方向
            if abs(offset + q) <= tau:         return '远中尖对尖'
            if abs(offset + h) <= tau or offset <= -h: return '完全远中'
            if -q < offset < 0:                return '中性偏远中'
            return '远中尖对尖' if abs(offset+q) < abs(offset+h) else '完全远中'

    # ---- 右侧 ----
    U_R, G_R = _get('16mb'), _get('46bg')
    Wr, qr, hr = _width_qh('46mc','46dc') if use_width_thresholds else (None, float(default_q), float(2*default_q))
    right = None
    if (U_R is not None) and (G_R is not None):
        offR = _y(U_R) - _y(G_R)
        clsR = _classify(offR, qr, hr)
        right = {
            'class': clsR,
            'offset_mm': float(np.round(offR, dec)),
            'W_mm': (None if Wr is None else float(np.round(Wr, dec))),
            'q_mm': float(np.round(qr, dec)),
            'h_mm': float(np.round(hr, dec)),
            'used': {'upper':'16mb','lower_neutral':'46bg','width_by':['46mc','46dc'] if Wr is not None else None}
        }

    # ---- 左侧 ----
    U_L, G_L = _get('26mb'), _get('36bg')
    Wl, ql, hl = _width_qh('36mc','36dc') if use_width_thresholds else (None, float(default_q), float(2*default_q))
    left = None
    if (U_L is not None) and (G_L is not None):
        offL = _y(U_L) - _y(G_L)
        clsL = _classify(offL, ql, hl)
        left = {
            'class': clsL,
            'offset_mm': float(np.round(offL, dec)),
            'W_mm': (None if Wl is None else float(np.round(Wl, dec))),
            'q_mm': float(np.round(ql, dec)),
            'h_mm': float(np.round(hl, dec)),
            'used': {'upper':'26mb','lower_neutral':'36bg','width_by':['36mc','36dc'] if Wl is not None else None}
        }

    # ---- 汇总 ----
    if right is None and left is None:
        return {'right':None,'left':None,'summary_text':'Molar_Relationship_磨牙关系*: 缺失','quality':'missing'}

    r_txt = right['class'] if right else '缺失'
    l_txt = left['class']  if left  else '缺失'
    return {
        'right': right, 'left': left,
        'summary_text': f"Molar_Relationship_磨牙关系*: 右侧{r_txt} 左侧{l_txt}",
        'quality': 'ok' if (right and left) else 'fallback'
    }

# ==========================================================
# 10）Overjet_前牙覆盖（OJ）
# ==========================================================
def compute_overjet(landmarks, frame, dec: int = 1, zero_tau_mm: float = 0.5):
    """
    Overjet_前牙覆盖（OJ）
      右侧 OJ_R = y(U11*) - y(L41*)
      左侧 OJ_L = y(U21*) - y(L31*)
    分度：
      |OJ| ≤ 0.5 → 对刃；OJ < -0.5 → 反𬌗
      3–5 → Ⅰ度深覆盖；5–8 → Ⅱ度深覆盖；>8 → Ⅲ度深覆盖
    返回键：与现有脚本一致
    """
    def _is_xyz(p): 
        return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm);  return np.asarray(p, float) if _is_xyz(p) else None

    if not frame or any(k not in frame for k in ('origin','ex','ey','ez')):
        return {'value_mm': None, 'side_of_max': None, 'right_mm': None, 'left_mm': None,
                'category': '缺失', 'summary_text': 'Overjet_前牙覆盖*: 缺失', 'quality': 'missing', 'used': {}}

    o  = np.asarray(frame['origin'], float)
    ey = np.asarray(frame['ey'],     float)
    ez = np.asarray(frame['ez'],     float)
    def _projP(p):
        v = np.asarray(p,float) - o
        return o + v - ez * float(np.dot(v, ez))
    def _y(p):
        q = _projP(p) - o
        return float(np.dot(q, ey))

    # 切缘“星号”(ma+da)/2
    def _star(tooth: str):
        a, d = _get(f'{tooth}ma'), _get(f'{tooth}da')
        return (0.5*(a+d) if (a is not None and d is not None) else None)

    U11s, U21s = _star('11'), _star('21')
    L41s, L31s = _star('41'), _star('31')
    if None in (U11s, U21s, L41s, L31s):
        return {'value_mm': None, 'side_of_max': None, 'right_mm': None, 'left_mm': None,
                'category': '缺失', 'summary_text': 'Overjet_前牙覆盖*: 缺失', 'quality': 'missing',
                'used': {'11*': bool(U11s), '21*': bool(U21s), '41*': bool(L41s), '31*': bool(L31s)}}

    OJ_R = _y(U11s) - _y(L41s)
    OJ_L = _y(U21s) - _y(L31s)

    # 取 |值| 更大侧
    side_of_max, value = (('right', OJ_R) if abs(OJ_R) >= abs(OJ_L) else ('left', OJ_L))

    # 分类
    def _cls(v):
        av = abs(v)
        if av <= zero_tau_mm: return '对刃'
        if v < -zero_tau_mm:  return '反𬌗'
        if 3.0 <= v < 5.0:    return 'Ⅰ度深覆盖'
        if 5.0 <= v <= 8.0:   return 'Ⅱ度深覆盖'
        if v > 8.0:           return 'Ⅲ度深覆盖'
        return '轻度覆盖'  # 0.5–3.0

    cat = _cls(value)

    return {
        'value_mm': float(np.round(value, dec)),
        'side_of_max': side_of_max,
        'right_mm': float(np.round(OJ_R, dec)),
        'left_mm' : float(np.round(OJ_L, dec)),
        'category': cat,
        'summary_text': f"Overjet_前牙覆盖*: {abs(round(value, dec)):.{dec}f}mm_{cat}",
        'quality': 'ok',
        'used': {'zero_tau_mm': float(zero_tau_mm)}
    }

# ==========================================================
# 11）Overbite_前牙覆𬌗（OB）
# ==========================================================
def compute_overbite(landmarks, frame, dec: int = 1, zero_tau_mm: float = 0.5):
    """
    Overbite_前牙覆𬌗（OB）
      右侧 OB_R = z(U11*) - z(L41*)
      左侧 OB_L = z(U21*) - z(L31*)
    分度：
      |OB| ≤ 0.5 → 对刃
      OB < 0 → 开𬌗：<3/3–5/>5 → Ⅰ/Ⅱ/Ⅲ度开𬌗
      OB > 0 → 深覆𬌗：按比值 OB/CH（CH=对应侧下中切牙临床冠高，星号→bgb）
                 ∈[1/3,1/2] / [1/2,2/3] / >2/3 → Ⅰ/Ⅱ/Ⅲ度深覆𬌗
    """
    def _is_xyz(p): 
        return isinstance(p, (list, tuple, np.ndarray)) and len(p) == 3 and np.isfinite(p).all()
    def _get(nm):
        p = landmarks.get(nm);  return np.asarray(p, float) if _is_xyz(p) else None

    if not frame or any(k not in frame for k in ('origin','ex','ey','ez')):
        return {'value_mm': None, 'side_of_max': None, 'right_mm': None, 'left_mm': None,
                'ratios': {'right': None, 'left': None},
                'category': '缺失', 'summary_text': 'Overbite_前牙覆𬌗*: 缺失', 'quality': 'missing', 'used': {}}

    o  = np.asarray(frame['origin'], float)
    ez = np.asarray(frame['ez'],     float)
    def _z(p):
        return float(np.dot(np.asarray(p,float) - o, ez))
    def _star(tooth: str):
        a, d = _get(f'{tooth}ma'), _get(f'{tooth}da')
        return (0.5*(a+d) if (a is not None and d is not None) else None)

    U11s, U21s = _star('11'), _star('21')
    L41s, L31s = _star('41'), _star('31')
    B41, B31   = _get('41bgb'), _get('31bgb')  # 下中切龈缘中点（临床冠高用）
    if None in (U11s, U21s, L41s, L31s):
        return {'value_mm': None, 'side_of_max': None, 'right_mm': None, 'left_mm': None,
                'ratios': {'right': None, 'left': None},
                'category': '缺失', 'summary_text': 'Overbite_前牙覆𬌗*: 缺失', 'quality': 'missing',
                'used': {'11*': bool(U11s), '21*': bool(U21s), '41*': bool(L41s), '31*': bool(L31s),
                         '41bgb': bool(B41), '31bgb': bool(B31)}}

    OB_R = _z(U11s) - _z(L41s)
    OB_L = _z(U21s) - _z(L31s)

    # 冠高（欧氏距离，星号→bgb）
    def _ch(Lstar, Bgb):
        if Lstar is None or Bgb is None: return None
        return float(np.linalg.norm(np.asarray(Lstar,float) - np.asarray(Bgb,float)))
    CH_R = _ch(L41s, B41)
    CH_L = _ch(L31s, B31)
    ratio_R = (OB_R / CH_R) if (CH_R and CH_R > 1e-6) else None
    ratio_L = (OB_L / CH_L) if (CH_L and CH_L > 1e-6) else None

    # 取 |值| 更大侧
    side_of_max, value = (('right', OB_R) if abs(OB_R) >= abs(OB_L) else ('left', OB_L))
    ratio_pick = ratio_R if side_of_max == 'right' else ratio_L

    # 分类
    def _cls(v, r):
        av = abs(v)
        if av <= zero_tau_mm: return '对刃'
        if v < -zero_tau_mm:
            if av < 3.0:  return 'Ⅰ度开𬌗'
            if av <= 5.0: return 'Ⅱ度开𬌗'
            return 'Ⅲ度开𬌗'
        # v > 0 深覆：优先用比值
        if r is not None:
            if r > 2/3:  return 'Ⅲ度深覆𬌗'
            if r >= 1/2: return 'Ⅱ度深覆𬌗'
            if r >= 1/3: return 'Ⅰ度深覆𬌗'
            return '正常'
        # 比值缺失时用毫米粗分
        return '深覆𬌗' if v >= 5.0 else '正常'

    cat = _cls(value, ratio_pick)

    return {
        'value_mm': float(np.round(value, dec)),
        'side_of_max': side_of_max,
        'right_mm': float(np.round(OB_R, dec)),
        'left_mm' : float(np.round(OB_L, dec)),
        'ratios': {
            'right': (None if ratio_R is None else float(np.round(ratio_R, 2))),
            'left' : (None if ratio_L is None else float(np.round(ratio_L, 2))),
        },
        'category': cat,
        'summary_text': f"Overbite_前牙覆𬌗*: {abs(round(value, dec)):.{dec}f}mm_{cat}",
        'quality': 'ok',
        'used': {'41bgb': bool(B41), '31bgb': bool(B31), 'zero_tau_mm': float(zero_tau_mm)}
    }