"""campaign_pulse 的自动化测试。

小样本用笛卡尔积穷举作为独立参照（仅测试代码枚举），
核对尾概率、MLE 与置信区间反演；另含零转化、全无信息、
支持集端点、Simpson 悖论构造，以及 5 渠道 / 50 万访客 /
宽度和 2000 的大规模性能测试。
"""

from __future__ import annotations

import math
import time
import unittest
from itertools import product

import campaign_pulse as cp


def brute_dist(strata, log_theta):
    """笛卡尔积穷举 T 的精确（对数权重）分布，仅用于小样本参照。"""
    per = []
    for nA, nB, xA, xB in strata:
        m = xA + xB
        lo = max(0, m - nA)
        hi = min(nB, m)
        ks = list(range(lo, hi + 1))
        lw = [
            math.log(math.comb(nB, k))
            + math.log(math.comb(nA, m - k))
            + k * log_theta
            for k in ks
        ]
        per.append(list(zip(ks, lw)))

    mass = {}
    z = -math.inf
    for combo in product(*per):
        t = 0
        lw = 0.0
        for k, w in combo:
            t += k
            lw += w
        mass[t] = cp._logadd(mass.get(t, -math.inf), lw)
        if lw > z:
            z = lw
    log_total = -math.inf
    for w in mass.values():
        log_total = cp._logadd(log_total, w)
    return {t: math.exp(w - log_total) for t, w in mass.items()}


def brute_tails(dist, t):
    lo = sum(p for x, p in dist.items() if x <= t)
    hi = sum(p for x, p in dist.items() if x >= t)
    return lo, hi


def brute_expected(dist):
    return sum(t * p for t, p in dist.items())


CASES = [
    [(10, 10, 4, 2), (5, 5, 3, 4)],
    [(8, 7, 3, 1), (6, 9, 2, 2), (5, 5, 1, 3)],
    [(20, 20, 15, 1)],
    [(20, 20, 1, 15)],
    [(10, 10, 0, 3), (8, 8, 5, 1)],
    [(3, 4, 0, 0), (5, 6, 2, 1)],          # 含零转化渠道
    [(4, 4, 2, 2), (3, 3, 1, 1)],          # 含可能的单点渠道
]


class TestAgainstBruteForce(unittest.TestCase):
    def test_tails_expected(self):
        for strata in CASES:
            t = sum(s[3] for s in strata)
            for lt in (-2.0, 0.0, 1.5):
                with self.subTest(strata=strata, lt=lt):
                    dist = brute_dist(strata, lt)
                    exp_b = brute_expected(dist)
                    lo_b, hi_b = brute_tails(dist, t)
                    ev = cp.evaluate(strata, math.exp(lt))
                    self.assertAlmostEqual(ev.expected, exp_b, delta=1e-9)
                    want_log = min(math.log(2) + math.log(min(lo_b, hi_b)), 0.0)
                    self.assertAlmostEqual(ev.log_two_sided, want_log, delta=1e-9)

    def test_mle_and_ci(self):
        for strata in CASES:
            with self.subTest(strata=strata):
                res = cp.infer(strata)
                model = cp._ConditionalModel(strata)
                L, U, t = model.L, model.U, model.t
                informative = any(
                    min(s[1], s[2] + s[3]) - max(0, s[2] + s[3] - s[0]) > 0
                    for s in strata
                )
                if not informative:
                    self.assertIsNone(res.odds_ratio)
                    self.assertEqual(res.ci_low, 0.0)
                    self.assertEqual(res.ci_high, math.inf)
                    self.assertEqual(res.p_value, 1.0)
                    continue
                # MLE：E[T]=t（端点除外）
                if t == L:
                    self.assertEqual(res.odds_ratio, 0.0)
                elif t == U:
                    self.assertEqual(res.odds_ratio, math.inf)
                else:
                    e = cp.evaluate(strata, res.odds_ratio).expected
                    self.assertAlmostEqual(e, t, delta=1e-6)
                    self.assertGreater(res.odds_ratio, 0.0)
                # CI 端点尾概率 = 0.025
                if t != L:
                    _, upper = model.tail_logs(math.log(res.ci_low))
                    self.assertAlmostEqual(math.exp(upper), 0.025, delta=1e-9)
                else:
                    self.assertEqual(res.ci_low, 0.0)
                if t != U:
                    lower, _ = model.tail_logs(math.log(res.ci_high))
                    self.assertAlmostEqual(math.exp(lower), 0.025, delta=1e-9)
                else:
                    self.assertEqual(res.ci_high, math.inf)
                # 与穷举尾概率反演核对
                if math.isfinite(res.ci_low):
                    d = brute_dist(strata, math.log(res.ci_low))
                    _, hi_b = brute_tails(d, t)
                    self.assertAlmostEqual(hi_b, 0.025, delta=1e-7)
                if math.isfinite(res.ci_high):
                    d = brute_dist(strata, math.log(res.ci_high))
                    lo_b, _ = brute_tails(d, t)
                    self.assertAlmostEqual(lo_b, 0.025, delta=1e-7)


if __name__ == "__main__":
    unittest.main()


class TestEndpointsAndNull(unittest.TestCase):
    def test_at_lower_endpoint(self):
        strata = [(20, 20, 20, 0)]
        res = cp.infer(strata)
        self.assertEqual(res.odds_ratio, 0.0)
        self.assertEqual(res.ci_low, 0.0)
        self.assertTrue(math.isfinite(res.ci_high))
        self.assertGreater(res.ci_high, 0.0)

    def test_at_upper_endpoint(self):
        strata = [(20, 20, 0,20)]
        res = cp.infer(strata)
        self.assertEqual(res.odds_ratio, math.inf)
        self.assertEqual(res.ci_high, math.inf)
        self.assertTrue(math.isfinite(res.ci_low))
        self.assertGreater(res.ci_low, 0.0)

    def test_zero_conversions_single_support(self):
        # 全无信息：所有渠道 m=0，T 恒为 0
        res = cp.infer([(10, 10, 0, 0), (5, 8, 0, 0)])
        self.assertIsNone(res.odds_ratio)
        self.assertEqual(res.ci_low, 0.0)
        self.assertEqual(res.ci_high, math.inf)
        self.assertEqual(res.p_value, 1.0)
        self.assertEqual(res.log_p_value, 0.0)

    def test_point_mass_stratum(self):
        # 合法 k 宽度为 0 的渠道不改变其他渠道推断。
        # nA=5,nB=5,m=10 => k 只能取 5（xA=5,xB=5）。
        point = (5, 5, 5, 5)
        base = [(12, 12, 7, 2)]
        a = cp.infer(base)
        b = cp.infer(base + [point])
        self.assertAlmostEqual(a.odds_ratio, b.odds_ratio, delta=1e-12)
        self.assertAlmostEqual(a.ci_low, b.ci_low, delta=1e-12)
        self.assertAlmostEqual(a.ci_high, b.ci_high, delta=1e-12)
        self.assertAlmostEqual(a.log_p_value, b.log_p_value, delta=1e-12)

    def test_uninformative_does_not_shift_tails(self):
        point = (6, 4, 6, 4)  # m=10=nB+nA? 10 vs 10 -> k 只能为 4
        base = [(10, 10, 6, 1)]
        ea = cp.evaluate(base, 1.3)
        eb = cp.evaluate(base + [point], 1.3)
        self.assertAlmostEqual(ea.log_two_sided, eb.log_two_sided, delta=1e-12)

    def test_log_p_finite_when_p_underflows(self):
        # 中等但极端的端点，p 可能下溢但 log 必须有限
        strata = [(200, 200, 200, 0)]
        res = cp.infer(strata)
        self.assertTrue(math.isfinite(res.log_p_value))
        self.assertLess(res.log_p_value, 0.0)


class TestSimpsonParadox(unittest.TestCase):
    def test_pooled_reverses_but_or_above_one(self):
        # 渠道1：A 流量大且转化率高，B 流量小、转化率略低但仍高；
        # 渠道2：B 流量大但基础转化率低。逐渠道 B 的条件优势比 > 1，
        # 但合并成一张总表后 B 转化率低于 A。
        strata = [
            # nA, nB, xA, xB
            (5000, 500, 4500, 475),  # A:90.0%  B:95.0%  (B 更好, OR~2.11)
            (500, 5000, 150, 1700),  # A:30.0%  B:34.0%  (B 更好, OR~1.21)
        ]
        nA = sum(s[0] for s in strata)
        nB = sum(s[1] for s in strata)
        xA = sum(s[2] for s in strata)
        xB = sum(s[3] for s in strata)
        rate_a = xA / nA
        rate_b = xB / nB
        # 合并后 B 转化率低于 A
        self.assertLess(rate_b, rate_a)
        res = cp.infer(strata)
        self.assertIsNotNone(res.odds_ratio)
        self.assertGreater(res.odds_ratio, 1.0)
        self.assertGreater(res.ci_low, 1.0)
        # 用逐渠道条件优势比核对方向（条件 MLE 应接近共同真值）
        for nA_i, nB_i, xA_i, xB_i in strata:
            or_i = (xB_i * (nA_i - xA_i)) / ((nB_i - xB_i) * xA_i)
            self.assertGreater(or_i, 1.0)


class TestLargeScale(unittest.TestCase):
    def test_500k_width2000_near_endpoint(self):
        # 1 个信息渠道：nA=nB=249998（宽度=min(nB,m)=2000），
        # 观测取靠近下端点但不在端点（k=60）；4 个无信息零转化渠道。
        nA = nB = 249996
        m = 2000
        xA, xB = m - 60, 60
        strata = [
            (nA, nB, xA, xB),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
            (1, 1, 0, 0),
        ]
        total_visitors = sum(s[0] + s[1] for s in strata)
        self.assertEqual(total_visitors, 500000)
        self.assertEqual(len(strata), 5)
        width_sum = sum(
            min(s[1], s[2] + s[3]) - max(0, s[2] + s[3] - s[0])
            for s in strata
        )
        self.assertEqual(width_sum, 2000)

        t0 = time.perf_counter()
        res = cp.infer(strata)
        elapsed = time.perf_counter() - t0

        model = cp._ConditionalModel(strata)
        self.assertNotEqual(model.t, model.L)
        self.assertNotEqual(model.t, model.U)

        self.assertLess(res.log_p_value, -1000.0)
        self.assertTrue(math.isfinite(res.log_p_value))
        self.assertEqual(res.p_value, 0.0)  # 极小 p 自然下溢

        self.assertTrue(math.isfinite(res.odds_ratio))
        self.assertTrue(math.isfinite(res.ci_low))
        self.assertTrue(math.isfinite(res.ci_high))
        self.assertLess(res.ci_low, res.odds_ratio)
        self.assertLess(res.odds_ratio, res.ci_high)

        # 尾概率反演仍精确
        _, upper = model.tail_logs(math.log(res.ci_low))
        lower, _ = model.tail_logs(math.log(res.ci_high))
        self.assertAlmostEqual(math.exp(upper), 0.025, delta=1e-9)
        self.assertAlmostEqual(math.exp(lower), 0.025, delta=1e-9)

        self.assertLess(elapsed, 10.0)
        print(f"\n[large] infer 耗时 {elapsed:.3f}s, log p={res.log_p_value:.1f}, "
              f"CI=({res.ci_low:.6g},{res.ci_high:.6g})")


class TestEvaluateValidation(unittest.TestCase):
    def test_theta_must_be_positive_finite(self):
        for bad in (0.0, -1.0, math.inf, math.nan):
            with self.assertRaises(ValueError):
                cp.evaluate([(10, 10, 4, 2)], bad)

    def test_evaluate_expected_monotone(self):
        strata = [(12, 14, 5, 3), (8, 9, 2, 2)]
        prev = -1.0
        for theta in (1e-6, 0.1, 1.0, 10.0, 1e6):
            e = cp.evaluate(strata, theta).expected
            self.assertGreaterEqual(e, prev - 1e-12)
            prev = e


class TestEvaluateInferConsistency(unittest.TestCase):
    def test_null_tail_matches_pvalue(self):
        for strata in CASES + [[(200, 200, 199, 1)], [(20, 20, 20, 0)]]:
            inf = cp.infer(strata)
            ev = cp.evaluate(strata, 1.0)
            self.assertAlmostEqual(ev.log_two_sided, inf.log_p_value, delta=1e-12)
            # expected 必须落在支持集内
            self.assertGreaterEqual(ev.expected, cp._ConditionalModel(strata).L - 1e-9)
            self.assertLessEqual(ev.expected, cp._ConditionalModel(strata).U + 1e-9)
