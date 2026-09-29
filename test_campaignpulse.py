"""Tests for campaignpulse (stdlib unittest)."""

import itertools
import math
import time
import unittest

from campaignpulse import evaluate, infer


def brute_dist(strata, theta):
    """Independent reference: exact distribution of T via Cartesian product."""
    supports = []
    for nA, nB, xA, xB in strata:
        m = xA + xB
        supports.append(range(max(0, m - nA), min(nB, m) + 1))
    dist = {}
    for ks in itertools.product(*supports):
        t = sum(ks)
        w = float(theta) ** t
        for (nA, nB, xA, xB), k in zip(strata, ks):
            m = xA + xB
            w *= math.comb(nB, k) * math.comb(nA, m - k)
        dist[t] = dist.get(t, 0.0) + w
    z = sum(dist.values())
    return {t: w / z for t, w in dist.items()}


def brute_stats(strata, theta, t_obs):
    dist = brute_dist(strata, theta)
    expct = sum(t * p for t, p in dist.items())
    ge = sum(p for t, p in dist.items() if t >= t_obs)
    le = sum(p for t, p in dist.items() if t <= t_obs)
    return expct, ge, le


class EvaluateAgainstBruteForceTest(unittest.TestCase):
    STRATA_SETS = [
        [(30, 25, 5, 8), (20, 40, 3, 10), (15, 12, 4, 2)],
        [(10, 10, 0, 0), (25, 30, 6, 9)],          # zero-conversion stratum
        [(10, 10, 10, 10), (25, 30, 6, 9)],        # all-converted stratum
        [(8, 12, 2, 5)],                            # single stratum
        [(50, 3, 20, 1), (4, 60, 2, 40)],          # skewed traffic split
    ]

    def test_tail_probabilities_and_expectation(self):
        for strata in self.STRATA_SETS:
            t_obs = sum(s[3] for s in strata)
            for theta in (0.05, 0.37, 1.0, 2.7, 41.0):
                r = evaluate(strata, theta)
                expct, ge, le = brute_stats(strata, theta, t_obs)
                self.assertAlmostEqual(r.expectation, expct, places=10)
                self.assertAlmostEqual(r.log_tail_ge, math.log(ge), places=9)
                self.assertAlmostEqual(r.log_tail_le, math.log(le), places=9)

    def test_evaluate_rejects_bad_theta(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                evaluate([(10, 10, 1, 2)], bad)


class InferAgainstBruteForceTest(unittest.TestCase):
    STRATA_SETS = [
        [(30, 25, 5, 8), (20, 40, 3, 10), (15, 12, 4, 2)],
        [(10, 10, 0, 0), (25, 30, 6, 9)],
        [(8, 12, 2, 5)],
        [(50, 3, 20, 1), (4, 60, 2, 40)],
        [(12, 9, 7, 1), (11, 14, 2, 8)],
    ]

    def test_estimate_solves_unbiasedness(self):
        for strata in self.STRATA_SETS:
            t_obs = sum(s[3] for s in strata)
            r = infer(strata)
            self.assertIsNotNone(r.estimate)
            expct, _, _ = brute_stats(strata, r.estimate, t_obs)
            self.assertAlmostEqual(expct, t_obs, places=8)

    def test_ci_inversion(self):
        for strata in self.STRATA_SETS:
            t_obs = sum(s[3] for s in strata)
            r = infer(strata)
            _, ge, _ = brute_stats(strata, r.ci_lower, t_obs)
            _, _, le = brute_stats(strata, r.ci_upper, t_obs)
            self.assertAlmostEqual(ge, 0.025, places=9)
            self.assertAlmostEqual(le, 0.025, places=9)

    def test_p_value(self):
        for strata in self.STRATA_SETS:
            t_obs = sum(s[3] for s in strata)
            r = infer(strata)
            _, ge, le = brute_stats(strata, 1.0, t_obs)
            expected = min(1.0, 2.0 * min(ge, le))
            self.assertAlmostEqual(r.p_value, expected, places=12)
            self.assertAlmostEqual(r.log_p, math.log(expected), places=9)

    def test_uninformative_strata_do_not_change_inference(self):
        base = [(30, 25, 5, 8), (20, 40, 3, 10)]
        extra = base + [(10, 10, 0, 0), (7, 9, 7, 9)]
        r1, r2 = infer(base), infer(extra)
        self.assertAlmostEqual(r1.estimate, r2.estimate, places=12)
        self.assertAlmostEqual(r1.ci_lower, r2.ci_lower, places=12)
        self.assertAlmostEqual(r1.ci_upper, r2.ci_upper, places=12)
        self.assertAlmostEqual(r1.log_p, r2.log_p, places=12)


class DegenerateAndEndpointTest(unittest.TestCase):
    def test_all_uninformative_is_unidentifiable(self):
        r = infer([(10, 10, 0, 0), (5, 5, 5, 5), (8, 4, 8, 4)])
        self.assertIsNone(r.estimate)
        self.assertEqual(r.ci_lower, 0.0)
        self.assertEqual(r.ci_upper, float("inf"))
        self.assertEqual(r.p_value, 1.0)
        self.assertEqual(r.log_p, 0.0)

    def test_observation_at_lower_support_end(self):
        strata = [(20, 15, 10, 0), (12, 18, 6, 0)]
        r = infer(strata)
        self.assertEqual(r.estimate, 0.0)
        self.assertEqual(r.ci_lower, 0.0)
        self.assertTrue(math.isfinite(r.ci_upper))
        t_obs = 0
        _, _, le = brute_stats(strata, r.ci_upper, t_obs)
        self.assertAlmostEqual(le, 0.025, places=9)

    def test_observation_at_upper_support_end(self):
        strata = [(20, 15, 0, 15), (12, 18, 0, 18)]
        # per-stratum k max: min(15, 15) = 15 and min(18, 18) = 18
        r = infer(strata)
        self.assertEqual(r.estimate, float("inf"))
        self.assertEqual(r.ci_upper, float("inf"))
        self.assertTrue(math.isfinite(r.ci_lower))
        t_obs = 15 + 18
        _, ge, _ = brute_stats(strata, r.ci_lower, t_obs)
        self.assertAlmostEqual(ge, 0.025, places=9)


class SimpsonsParadoxTest(unittest.TestCase):
    def test_common_odds_ratio_above_one(self):
        # Within each channel B converts better; pooled B converts worse.
        strata = [(100, 10, 20, 3), (10, 100, 1, 15)]
        pool_a = (20 + 1) / 110
        pool_b = (3 + 15) / 110
        self.assertLess(pool_b, pool_a)  # pooled B looks worse
        for nA, nB, xA, xB in strata:
            self.assertGreater(xB / nB, xA / nA)  # but B wins per channel
        r = infer(strata)
        self.assertGreater(r.estimate, 1.0)
        self.assertGreater(r.ci_lower, 0.0)


class LargeScaleTest(unittest.TestCase):
    def test_max_scale_near_endpoint(self):
        # 5 channels, 500k visitors total, support widths summing to 2000.
        strata = [(50000, 50000, 395, 5) for _ in range(5)]
        self.assertEqual(sum(s[0] + s[1] for s in strata), 500000)
        start = time.perf_counter()
        r = infer(strata)
        elapsed = time.perf_counter() - start
        print(f"\nmax-scale infer elapsed: {elapsed:.3f}s")
        self.assertLess(elapsed, 10.0)

        # support is [0, 2000]; t_obs = 25 is near but not at the endpoint
        self.assertLess(r.log_p, -1000.0)
        self.assertEqual(r.p_value, 0.0)  # underflows, log stays finite
        self.assertTrue(math.isfinite(r.log_p))
        self.assertTrue(math.isfinite(r.estimate))
        self.assertGreater(r.estimate, 0.0)
        self.assertTrue(math.isfinite(r.ci_lower))
        self.assertTrue(math.isfinite(r.ci_upper))
        self.assertGreater(r.ci_lower, 0.0)
        self.assertLess(r.estimate, 1.0)

        # log tail probabilities from evaluate stay finite and consistent
        ev = evaluate(strata, 1.0)
        self.assertTrue(math.isfinite(ev.log_tail_le))
        self.assertAlmostEqual(
            min(0.0, math.log(2.0) + min(ev.log_tail_ge, ev.log_tail_le)),
            r.log_p,
            places=9,
        )


if __name__ == "__main__":
    unittest.main()
