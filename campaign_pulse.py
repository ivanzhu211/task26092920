"""CampaignPulse 分渠道精确条件推断（纯标准库实现）。

统计模型
--------
对每个渠道（stratum），固定两臂访客数 nA、nB 与总转化人数 m = xA + xB。
共同优势比为 theta 时，B 臂转化人数 K 的条件分布权重为::

    C(nB, k) * C(nA, m-k) * theta**k

k 取使两个组合数均非零的整数，权重归一化后得到该渠道分布；各渠道独立，
T = sum K 为总 B 臂转化人数，观测值 t = sum xB。

实现上只对每个渠道做一次逐 k 的对数权重，再用对数空间卷积合并渠道
（不枚举笛卡尔积），之后通过指数倾斜 O(W) 计算任意 theta 下的期望与尾概率。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

Stratum = tuple[int, int, int, int]  # (nA, nB, xA, xB)

_LN2 = math.log(2.0)
_LN_ALPHA = math.log(0.025)
_LOG_FLOAT_MAX = math.log(math.nextafter(math.inf, 0.0))


def _safe_exp(x: float) -> float:
    """exp(x)；仅当真实结果超过 float 上界才返回 inf（不抛异常）。"""
    if x >= _LOG_FLOAT_MAX:
        return math.inf
    return math.exp(x)


@dataclass(frozen=True)
class Evaluation:
    """evaluate 的结果。

    expected: 总 B 臂转化人数 T 的期望 E_theta[T]
    log_two_sided: 观测值两侧含等号尾概率（2*min(P(T<=t), P(T>=t))，
                   截到 1）的自然对数
    """

    expected: float
    log_two_sided: float


@dataclass(frozen=True)
class Inference:
    """infer 的结果。

    odds_ratio: 共同优势比条件 MLE；支持集仅一个值时为 None（不可识别）
    ci_low/ci_high: 95% 置信区间端点，可能为 0.0 或 math.inf
    p_value: theta=1 时的双侧 p 值（极小可下溢为 0）
    log_p_value: 双侧 p 值的自然对数（保持有限且正确）
    """

    odds_ratio: float | None
    ci_low: float
    ci_high: float
    p_value: float
    log_p_value: float


def _logadd(a: float, b: float) -> float:
    """log(exp(a)+exp(b))，数值稳定且不会溢出。"""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    d = b - a
    if d < -745.0:
        return a
    return a + math.log1p(math.exp(d))


def _lcomb(n: int, k: int) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
    )


def _stratum_log_weights(nA: int, nB: int, m: int, lo: int, hi: int) -> List[float]:
    """渠道 K 在 lo..hi 上的基础对数权重（theta=1），按最大值平移。"""
    raw = [
        _lcomb(nB, k) + _lcomb(nA, m - k)
        for k in range(lo, hi + 1)
    ]
    peak = max(raw)
    return [w - peak for w in raw]


def _log_convolve(acc: List[float], w: List[float]) -> List[float]:
    """对数空间卷积：合并一个新渠道的权重向量。"""
    la = len(acc)
    lw = len(w)
    out = [-math.inf] * (la + lw - 1)
    logadd = _logadd
    for i in range(la):
        ai = acc[i]
        base = i
        for j in range(lw):
            idx = base + j
            out[idx] = logadd(out[idx], ai + w[j])
    return out


class _ConditionalModel:
    """聚合后 T 的基础（theta=1）对数权重与指数倾斜计算。"""

    def __init__(self, strata: Iterable[Stratum]):
        strata = list(strata)
        t = 0
        L = 0
        acc: List[float] = [0.0]
        for nA, nB, xA, xB in strata:
            m = xA + xB
            lo = max(0, m - nA)
            hi = min(nB, m)
            t += xB
            if hi == lo:
                # 无信息渠道：K 恒为 lo，权重为尖点，只平移 T，不影响推断。
                L += lo
                continue
            w = _stratum_log_weights(nA, nB, m, lo, hi)
            acc = _log_convolve(acc, w)
            L += lo
        self.log_b = acc          # 下标 j = T - L
        self.L = L
        self.U = self.L + len(acc) - 1
        self.t = t
        self.tidx = t - self.L

    def _suffstats(self, log_theta: float):
        """单次扫描计算稳定的对数归一化常数、下尾、上尾与 T 的对数矩。

        每个和都以自身局部峰值为基准累加，避免远离全局峰端时
        因相对量级过大而整段下溢。返回 (log_z, log_lower, log_upper, log_ET)。
        """
        b = self.log_b
        L = self.L
        end = self.tidx
        lam = log_theta

        z_all = -math.inf
        z_lo = -math.inf
        z_hi = -math.inf
        z_et = -math.inf
        n = len(b)
        for j in range(n):
            v = b[j] + (L + j) * lam
            if v > z_all:
                z_all = v
            if j <= end and v > z_lo:
                z_lo = v
            if j >= end and v > z_hi:
                z_hi = v
            if L + j > 0:
                ve = v + math.log(L + j)
                if ve > z_et:
                    z_et = ve

        s_all = -math.inf
        s_lo = -math.inf
        s_hi = -math.inf
        s_et = -math.inf
        for j in range(n):
            v = b[j] + (L + j) * lam
            d = v - z_all
            if d > -745.0:
                s_all = _logadd(s_all, d)
            if j <= end:
                d = v - z_lo
                if d > -745.0:
                    s_lo = _logadd(s_lo, d)
            if j >= end:
                d = v - z_hi
                if d > -745.0:
                    s_hi = _logadd(s_hi, d)
            if L + j > 0:
                d = v + math.log(L + j) - z_et
                if d > -745.0:
                    s_et = _logadd(s_et, d)

        log_z = z_all + s_all
        log_lower = z_lo + s_lo - log_z
        log_upper = z_hi + s_hi - log_z
        log_et = z_et + s_et - log_z
        return log_z, log_lower, log_upper, log_et

    def expected(self, log_theta: float) -> float:
        _, _, _, log_et = self._suffstats(log_theta)
        return math.exp(log_et)

    def tail_logs(self, log_theta: float) -> tuple[float, float]:
        """返回 (log P(T<=t), log P(T>=t))。"""
        _, log_lower, log_upper, _ = self._suffstats(log_theta)
        return log_lower, log_upper


def _bracket(f, increasing: bool, x0: float = 0.0, step: float = 1.0, limit: float = 2000.0):
    """从 x0 向两侧倍增，返回 f 异号的区间；increasing 指定 f 的单调方向。"""
    lo = x0 - step
    hi = x0 + step
    f_lo = f(lo)
    f_hi = f(hi)
    if increasing:
        while f_lo > 0.0 and lo > -limit:
            step *= 2.0
            lo = x0 - step
            f_lo = f(lo)
        while f_hi < 0.0 and hi < limit:
            step *= 2.0
            hi = x0 + step
            f_hi = f(hi)
    else:
        while f_lo < 0.0 and lo > -limit:
            step *= 2.0
            lo = x0 - step
            f_lo = f(lo)
        while f_hi > 0.0 and hi < limit:
            step *= 2.0
            hi = x0 + step
            f_hi = f(hi)
    return lo, hi, f_lo, f_hi


def _bisect(f, lo: float, hi: float, f_lo: float, f_hi: float, increasing: bool = True, iters: int = 80) -> float:
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        root_to_right = (fm <= 0.0) if increasing else (fm >= 0.0)
        if root_to_right:
            lo, f_lo = mid, fm
        else:
            hi, f_hi = mid, fm
    return 0.5 * (lo + hi)


def evaluate(strata: Iterable[Stratum], theta: float) -> Evaluation:
    """返回 T 的期望与观测值两侧含等号尾概率的自然对数。

    尾概率定义与 infer 的 p 值一致：
    2 * min(P_theta(T<=t), P_theta(T>=t))，并截断到 1。
    theta 必须为有限正数。
    """
    if not math.isfinite(theta) or theta <= 0.0:
        raise ValueError("theta 必须为有限正数")
    model = _ConditionalModel(strata)
    log_theta = math.log(theta)
    expected = model.expected(log_theta)
    log_lo, log_hi = model.tail_logs(log_theta)
    log_two_sided = min(_LN2 + min(log_lo, log_hi), 0.0)
    return Evaluation(expected=expected, log_two_sided=log_two_sided)


def infer(strata: Iterable[Stratum]) -> Inference:
    """返回共同优势比估计、95% 置信区间、双侧 p 值及其自然对数。"""
    model = _ConditionalModel(strata)
    L, U, t = model.L, model.U, model.t

    log_p_lo, log_p_hi = model.tail_logs(0.0)
    log_p_value = min(_LN2 + min(log_p_lo, log_p_hi), 0.0)
    p_value = math.exp(log_p_value)

    if L == U:
        return Inference(
            odds_ratio=None,
            ci_low=0.0,
            ci_high=math.inf,
            p_value=1.0,
            log_p_value=0.0,
        )

    if t == L:
        estimate = 0.0
    elif t == U:
        estimate = math.inf
    else:
        def f_mean(log_theta: float) -> float:
            return model.expected(log_theta) - t

        lo, hi, flo, fhi = _bracket(f_mean, True)
        log_hat = _bisect(f_mean, lo, hi, flo, fhi, True)
        estimate = _safe_exp(log_hat)

    if t == L:
        ci_low = 0.0
    else:
        def f_low(log_theta: float) -> float:
            # 下端点解 P(T>=t)=0.025（上尾随 lambda 递增）。
            _, upper_tail = model.tail_logs(log_theta)
            return upper_tail - _LN_ALPHA

        lo, hi, flo, fhi = _bracket(f_low, True)
        ci_low = _safe_exp(_bisect(f_low, lo, hi, flo, fhi, True))

    if t == U:
        ci_high = math.inf
    else:
        def f_high(log_theta: float) -> float:
            # 上端点解 P(T<=t)=0.025（下尾随 lambda 递减）。
            lower_tail, _ = model.tail_logs(log_theta)
            return lower_tail - _LN_ALPHA

        lo, hi, flo, fhi = _bracket(f_high, False)
        ci_high = _safe_exp(_bisect(f_high, lo, hi, flo, fhi, False))

    return Inference(
        odds_ratio=estimate,
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        log_p_value=log_p_value,
    )
