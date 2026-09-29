"""CampaignPulse: stratified exact inference for A/B experiments.

Exact conditional inference on a common odds ratio across channels
(strata), using the noncentral hypergeometric distribution per stratum
and the exact discrete convolution across strata.
"""

from collections import namedtuple
from math import exp, inf, lgamma, log, log1p

__all__ = ["evaluate", "infer", "EvaluateResult", "InferResult"]

EvaluateResult = namedtuple(
    "EvaluateResult",
    ["expectation", "log_tail_ge", "log_tail_le"],
)

InferResult = namedtuple(
    "InferResult",
    ["estimate", "ci_lower", "ci_upper", "p_value", "log_p"],
)


def _parse_strata(strata):
    """Normalize input strata into a list of (nA, nB, xA, xB) tuples."""
    parsed = []
    for s in strata:
        if isinstance(s, dict):
            nA, nB, xA, xB = s["nA"], s["nB"], s["xA"], s["xB"]
        else:
            nA, nB, xA, xB = s
        parsed.append((int(nA), int(nB), int(xA), int(xB)))
    if not parsed:
        raise ValueError("strata must contain at least one channel")
    return parsed


def _log_support_weights(nA, nB, m):
    """Log unnormalized weights log C(nB,k) + log C(nA,m-k) over valid k."""
    lo = max(0, m - nA)
    hi = min(nB, m)
    if lo > hi:
        raise ValueError("inconsistent stratum: empty support")
    lg = lgamma
    w = [
        lg(nB + 1) - lg(k + 1) - lg(nB - k + 1)
        + lg(nA + 1) - lg(m - k + 1) - lg(nA - m + k + 1)
        for k in range(lo, hi + 1)
    ]
    shift = max(w)
    return lo, [v - shift for v in w]


def _log_convolve(a, b):
    """Log-space convolution of two log-weight arrays."""
    la, lb = len(a), len(b)
    out = [float("-inf")] * (la + lb - 1)
    neg_inf = float("-inf")
    for t in range(la + lb - 1):
        k_lo = max(0, t - lb + 1)
        k_hi = min(t, la - 1)
        acc = neg_inf
        for k in range(k_lo, k_hi + 1):
            av = a[k]
            if av == neg_inf:
                continue
            v = av + b[t - k]
            if v == neg_inf:
                continue
            if acc == neg_inf:
                acc = v
            elif v > acc:
                acc = v + log1p(exp(acc - v))
            else:
                acc = acc + log1p(exp(v - acc))
        out[t] = acc
    return out


def _combined_log_coeffs(parsed):
    """Log coefficients of the convolved theta=1 distribution of T."""
    lo_total = 0
    coeffs = [0.0]
    for nA, nB, xA, xB in parsed:
        m = xA + xB
        lo, w = _log_support_weights(nA, nB, m)
        lo_total += lo
        coeffs = _log_convolve(coeffs, w)
    return lo_total, coeffs


def _logsumexp(vals):
    """log(sum(exp(v) for v in vals)) computed stably."""
    neg_inf = float("-inf")
    acc = neg_inf
    for v in vals:
        if v == neg_inf:
            continue
        if acc == neg_inf:
            acc = v
        elif v > acc:
            acc = v + log1p(exp(acc - v))
        else:
            acc = acc + log1p(exp(v - acc))
    return acc


def _dist_stats(lo_total, coeffs, log_theta, t_obs):
    """(expectation, log P(T>=t_obs), log P(T<=t_obs)) under theta."""
    neg_inf = float("-inf")
    logw = [
        c + (lo_total + i) * log_theta if c != neg_inf else neg_inf
        for i, c in enumerate(coeffs)
    ]
    logz = _logsumexp(logw)
    idx = t_obs - lo_total
    log_ge = _logsumexp(logw[idx:]) - logz
    log_le = _logsumexp(logw[: idx + 1]) - logz
    expectation = 0.0
    for i, lw in enumerate(logw):
        if lw == neg_inf:
            continue
        expectation += (lo_total + i) * exp(lw - logz)
    return expectation, log_ge, log_le


_LOG_025 = -3.6888794541139363  # log(0.025)
_BISECT_LIMIT = 700.0  # exp() overflows beyond ~709; stay safely inside


def _solve(lo_total, coeffs, t_obs, target, kind):
    """Solve a monotone equation in s = log(theta) by bisection.

    kind "mean": E_s[T] = target (increasing in s).
    kind "ge":   log P_s(T >= t_obs) = target (increasing in s).
    kind "le":   log P_s(T <= t_obs) = target (decreasing in s).
    Returns None if no solution exists within float range.
    """
    def f(s):
        expct, log_ge, log_le = _dist_stats(lo_total, coeffs, s, t_obs)
        if kind == "mean":
            return expct - target
        if kind == "ge":
            return log_ge - target
        return target - log_le  # increasing in s

    lo_s, hi_s = -1.0, 1.0
    f_lo, f_hi = f(lo_s), f(hi_s)
    while f_lo > 0.0:
        hi_s, f_hi = lo_s, f_lo
        lo_s = lo_s * 2.0 if lo_s < 0 else lo_s - 1.0
        if lo_s < -_BISECT_LIMIT:
            return None
        f_lo = f(lo_s)
    while f_hi < 0.0:
        lo_s, f_lo = hi_s, f_hi
        hi_s = hi_s * 2.0 if hi_s > 0 else hi_s + 1.0
        if hi_s > _BISECT_LIMIT:
            return None
        f_hi = f(hi_s)
    for _ in range(80):
        mid = 0.5 * (lo_s + hi_s)
        if f(mid) > 0.0:
            hi_s = mid
        else:
            lo_s = mid
    return 0.5 * (lo_s + hi_s)


def evaluate(strata, theta):
    """Expectation of T and inclusive two-sided log tail probabilities."""
    if not (theta > 0.0) or theta != theta or theta == float("inf"):
        raise ValueError("theta must be a finite positive number")
    parsed = _parse_strata(strata)
    lo_total, coeffs = _combined_log_coeffs(parsed)
    t_obs = sum(xB for _, _, _, xB in parsed)
    expectation, log_ge, log_le = _dist_stats(
        lo_total, coeffs, log(theta), t_obs
    )
    return EvaluateResult(expectation, log_ge, log_le)


def infer(strata):
    """Common odds-ratio estimate, 95% CI, two-sided p-value and its log."""
    parsed = _parse_strata(strata)
    lo_total, coeffs = _combined_log_coeffs(parsed)
    t_obs = sum(xB for _, _, _, xB in parsed)
    hi_total = lo_total + len(coeffs) - 1

    if lo_total == hi_total:
        return InferResult(None, 0.0, inf, 1.0, 0.0)

    # p-value at theta = 1
    _, log_ge1, log_le1 = _dist_stats(lo_total, coeffs, 0.0, t_obs)
    log_p = min(0.0, log(2.0) + min(log_ge1, log_le1))
    p_value = exp(log_p)

    # point estimate: E_theta[T] = t_obs
    if t_obs == lo_total:
        estimate = 0.0
    elif t_obs == hi_total:
        estimate = inf
    else:
        s = _solve(lo_total, coeffs, t_obs, float(t_obs), "mean")
        estimate = inf if s is None else exp(s)

    # 95% CI: lower solves P(T >= t) = 0.025, upper solves P(T <= t) = 0.025
    if t_obs == lo_total:
        ci_lower = 0.0
    else:
        s = _solve(lo_total, coeffs, t_obs, _LOG_025, "ge")
        ci_lower = inf if s is None else exp(s)
    if t_obs == hi_total:
        ci_upper = inf
    else:
        s = _solve(lo_total, coeffs, t_obs, _LOG_025, "le")
        ci_upper = inf if s is None else exp(s)

    return InferResult(estimate, ci_lower, ci_upper, p_value, log_p)
