"""Synthetic regression tests; this module never reads participant data.

Run from this directory with:
    python -m unittest -v test_setting_reassignment_sensitivity
"""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.special import expit, logsumexp
from statsmodels.discrete.discrete_model import MNLogit

import run_setting_reassignment_sensitivity as sensitivity


def _probabilities(design, params):
    logits = np.column_stack((np.zeros(len(design)), design @ params))
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))


def _synthetic_sample(n=640, seed=29031):
    rng = np.random.default_rng(seed)
    design = np.column_stack(
        (np.ones(n), rng.normal(size=n), rng.integers(0, 2, size=n))
    )
    params = np.array([[-0.25, -0.6], [0.45, -0.3], [-0.2, 0.35]])
    probabilities = _probabilities(design, params)
    uniforms = rng.random(n)
    labels = (uniforms[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
    counts = np.eye(3)[labels]
    return design, labels, counts


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.steps = np.linspace(-2.5, 2.5, 601)
        self.weights = np.linspace(0.1, 3.0, len(self.steps)) ** 2

    def test_unweighted_mean_and_logistic_parameterization(self):
        for target in (0.01, 0.2, 0.5, 0.85, 0.99):
            for slope in (-1.4, 0.0, 1.4):
                with self.subTest(target=target, slope=slope):
                    probabilities, intercept = sensitivity.calibrate_probabilities(
                        self.steps, target, slope
                    )
                    self.assertEqual(probabilities.shape, self.steps.shape)
                    self.assertTrue(np.isfinite(intercept))
                    self.assertAlmostEqual(probabilities.mean(), target, places=10)
                    assert_allclose(
                        probabilities, expit(intercept + slope * self.steps),
                        rtol=1e-11, atol=1e-12,
                    )

    def test_weighted_mean_and_slope_direction(self):
        for slope in (-2.0, 0.0, 2.0):
            with self.subTest(slope=slope):
                probabilities, _ = sensitivity.calibrate_probabilities(
                    self.steps, 0.3, slope, weights=self.weights
                )
                self.assertAlmostEqual(
                    np.average(probabilities, weights=self.weights), 0.3, places=10
                )
                if slope > 0:
                    self.assertTrue(np.all(np.diff(probabilities) > 0))
                elif slope < 0:
                    self.assertTrue(np.all(np.diff(probabilities) < 0))
                else:
                    assert_allclose(probabilities, 0.3, atol=1e-11)

    def test_zero_weight_rows_and_weight_rescaling(self):
        weights = self.weights.copy()
        weights[::3] = 0.0
        probabilities, intercept = sensitivity.calibrate_probabilities(
            self.steps, 0.37, 0.9, weights=weights
        )
        scaled, scaled_intercept = sensitivity.calibrate_probabilities(
            self.steps, 0.37, 0.9, weights=weights * 17.0
        )
        included = weights > 0
        subset, subset_intercept = sensitivity.calibrate_probabilities(
            self.steps[included], 0.37, 0.9, weights=weights[included]
        )
        assert_allclose(probabilities, scaled, atol=1e-11)
        assert_allclose(probabilities[included], subset, atol=1e-11)
        self.assertAlmostEqual(intercept, scaled_intercept, places=10)
        self.assertAlmostEqual(intercept, subset_intercept, places=10)

    def test_endpoints_are_exact_and_independent_of_slope(self):
        for target in (0.0, 1.0):
            for slope in (-10.0, 0.0, 10.0):
                with self.subTest(target=target, slope=slope):
                    probabilities, _ = sensitivity.calibrate_probabilities(
                        self.steps, target, slope, weights=self.weights
                    )
                    assert_array_equal(probabilities, np.full(len(self.steps), target))

    def test_invalid_calibration_inputs_raise(self):
        invalid_cases = [
            (self.steps, -0.01, 1.0, None),
            (self.steps, 1.01, 1.0, None),
            (self.steps, np.nan, 1.0, None),
            (self.steps, np.inf, 1.0, None),
            (self.steps, 0.3, np.nan, None),
            (self.steps, 0.3, np.inf, None),
            (self.steps, 0.3, -np.inf, None),
            (np.array([0.0, np.nan, 1.0]), 0.3, 1.0, None),
            (np.array([0.0, np.inf, 1.0]), 0.3, 1.0, None),
            (np.array([]), 0.3, 1.0, None),
            (self.steps, 0.3, 1.0, np.ones(len(self.steps) - 1)),
            (self.steps, 0.3, 1.0, np.zeros(len(self.steps))),
            (self.steps, 0.3, 1.0, -np.ones(len(self.steps))),
            (self.steps, 0.3, 1.0, np.full(len(self.steps), np.nan)),
            (self.steps, 0.3, 1.0, np.full(len(self.steps), np.inf)),
        ]
        for index, (steps, target, slope, weights) in enumerate(invalid_cases):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    sensitivity.calibrate_probabilities(
                        steps, target, slope, weights=weights
                    )


class MultinomialLikelihoodTests(unittest.TestCase):
    def test_loglike_score_and_information_against_finite_differences(self):
        rng = np.random.default_rng(412)
        design = np.column_stack((np.ones(37), rng.normal(size=(37, 2))))
        counts = rng.uniform(0.05, 2.5, size=(37, 3))
        counts[::7] = 0.0
        params = rng.normal(scale=0.35, size=(3, 2))
        loglike, score, information = sensitivity.multinomial_loglik_score_info(
            design, counts, params
        )
        expected_loglike = np.sum(counts * np.log(_probabilities(design, params)))
        self.assertAlmostEqual(loglike, expected_loglike, places=10)
        self.assertEqual(np.asarray(score).shape, (6,))
        self.assertEqual(np.asarray(information).shape, (6, 6))
        assert_allclose(information, information.T, atol=1e-11)
        self.assertTrue(np.all(np.linalg.eigvalsh(information) > 0))

        flat_params = params.ravel(order="F")
        epsilon = 1e-5
        numerical_score = np.empty(6)
        numerical_information = np.empty((6, 6))
        for column in range(6):
            delta = np.zeros(6)
            delta[column] = epsilon
            plus = sensitivity.multinomial_loglik_score_info(
                design, counts, (flat_params + delta).reshape(3, 2, order="F")
            )
            minus = sensitivity.multinomial_loglik_score_info(
                design, counts, (flat_params - delta).reshape(3, 2, order="F")
            )
            numerical_score[column] = (plus[0] - minus[0]) / (2 * epsilon)
            numerical_information[:, column] = -(plus[1] - minus[1]) / (2 * epsilon)
        assert_allclose(score, numerical_score, rtol=2e-7, atol=2e-8)
        assert_allclose(information, numerical_information, rtol=2e-7, atol=2e-8)

    def test_likelihood_remains_finite_with_extreme_logits(self):
        design = np.array([[1.0, -1000.0], [1.0, 0.0], [1.0, 1000.0]])
        counts = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0], [0.1, 0.8, 0.1]])
        params = np.array([[0.2, -0.5], [1.2, -0.8]])
        loglike, score, information = sensitivity.multinomial_loglik_score_info(
            design, counts, params
        )
        self.assertTrue(np.isfinite(loglike))
        self.assertTrue(np.all(np.isfinite(score)))
        self.assertTrue(np.all(np.isfinite(information)))

    def test_invalid_likelihood_inputs_are_not_silently_dropped(self):
        design, _, counts = _synthetic_sample(n=90)
        params = np.zeros((design.shape[1], 2))
        nan_design = design.copy()
        nan_design[2, 1] = np.nan
        inf_design = design.copy()
        inf_design[2, 1] = np.inf
        nan_counts = counts.copy()
        nan_counts[4, 2] = np.nan
        negative_counts = counts.copy()
        negative_counts[4, 2] = -0.1
        inf_counts = counts.copy()
        inf_counts[4, 2] = np.inf
        invalid_cases = [
            (nan_design, counts),
            (inf_design, counts),
            (design, nan_counts),
            (design, negative_counts),
            (design, inf_counts),
            (design[:-1], counts),
            (design, counts[:, :2]),
            (design[:, 0], counts),
            (design, counts[:, 0]),
        ]
        for index, (bad_design, bad_counts) in enumerate(invalid_cases):
            with self.subTest(case=index):
                with self.assertRaises((ValueError, TypeError)):
                    sensitivity.multinomial_loglik_score_info(
                        bad_design, bad_counts, params
                    )
                with self.assertRaises((ValueError, TypeError)):
                    sensitivity.fit_multinomial_counts(bad_design, bad_counts)
        for bad_params in (np.zeros((3, 3)), np.full((3, 2), np.nan)):
            with self.subTest(params=bad_params):
                with self.assertRaises((ValueError, TypeError)):
                    sensitivity.multinomial_loglik_score_info(design, counts, bad_params)
        with self.assertRaises((ValueError, TypeError)):
            sensitivity.fit_multinomial_counts(design, np.zeros_like(counts))


class MultinomialFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.design, cls.labels, cls.counts = _synthetic_sample()
        cls.result = sensitivity.fit_multinomial_counts(
            cls.design, cls.counts, with_covariance=True
        )
        cls.reference = MNLogit(cls.labels, cls.design).fit(
            method="newton", maxiter=100, tol=1e-12, disp=False
        )

    def test_hard_labels_match_statsmodels_parameters_and_loglike(self):
        self.assertEqual(np.asarray(self.result["params"]).shape, (3, 2))
        assert_allclose(self.result["params"], self.reference.params, atol=2e-6, rtol=2e-6)
        self.assertAlmostEqual(self.result["loglike"], self.reference.llf, places=7)
        self.assertGreaterEqual(self.result["iterations"], 0)
        self.assertLess(self.result["max_abs_score"], 1e-6)
        _, score, _ = sensitivity.multinomial_loglik_score_info(
            self.design, self.counts, self.result["params"]
        )
        self.assertAlmostEqual(
            self.result["max_abs_score"],
            np.max(np.abs(score)) / self.counts.sum(),
            places=10,
        )

    def test_covariance_and_step_contrasts_match_statsmodels(self):
        covariance = self.result["covariance"]
        self.assertEqual(np.asarray(covariance).shape, (6, 6))
        assert_allclose(covariance, self.reference.cov_params(), atol=2e-7, rtol=2e-5)
        # Fortran ordering groups all outpatient coefficients before all acute ones.
        contrast_matrix = np.zeros((3, 6))
        contrast_matrix[0, 1] = 1.0
        contrast_matrix[1, 4] = 1.0
        contrast_matrix[2, 4] = 1.0
        contrast_matrix[2, 1] = -1.0
        expected = contrast_matrix @ self.reference.params.ravel(order="F")
        observed = sensitivity.step_contrasts(self.result["params"], step_index=1)
        self.assertEqual(np.asarray(observed).shape, (3,))
        assert_allclose(observed, expected, atol=2e-6, rtol=2e-6)
        assert_allclose(
            contrast_matrix @ covariance @ contrast_matrix.T,
            contrast_matrix @ self.reference.cov_params() @ contrast_matrix.T,
            atol=2e-7, rtol=2e-5,
        )
        assert_allclose(
            sensitivity.step_contrasts(self.result["params"], step_index=2),
            [self.result["params"][2, 0], self.result["params"][2, 1],
             self.result["params"][2, 1] - self.result["params"][2, 0]],
        )

    def test_fractional_counts_match_rationally_replicated_rows(self):
        rng = np.random.default_rng(6204)
        design, _, _ = _synthetic_sample(n=100, seed=924)
        probabilities = _probabilities(
            design, np.array([[-0.3, 0.2], [0.4, -0.2], [0.1, -0.4]])
        )
        integer_counts = np.array([rng.multinomial(6, p) for p in probabilities])
        fractional = sensitivity.fit_multinomial_counts(
            design, integer_counts / 6.0, with_covariance=True
        )
        row_indices = np.repeat(np.arange(len(design)), integer_counts.sum(axis=1))
        labels = np.concatenate(
            [np.repeat(np.arange(3), row_counts) for row_counts in integer_counts]
        )
        self.assertEqual(len(labels), 600)
        replicated = MNLogit(labels, design[row_indices]).fit(
            method="newton", maxiter=100, tol=1e-12, disp=False
        )
        assert_allclose(fractional["params"], replicated.params, atol=3e-6, rtol=3e-6)
        self.assertAlmostEqual(fractional["loglike"] * 6.0, replicated.llf, places=7)
        assert_allclose(
            fractional["covariance"], replicated.cov_params() * 6.0,
            atol=3e-7, rtol=3e-5,
        )

    def test_integer_bootstrap_counts_match_repeated_rows(self):
        rng = np.random.default_rng(43021)
        n = len(self.design)
        multiplicities = rng.multinomial(n, np.full(n, 1.0 / n))
        self.assertTrue(np.any(multiplicities == 0))
        row_indices = np.repeat(np.arange(n), multiplicities)
        weighted = sensitivity.fit_multinomial_counts(
            self.design, self.counts * multiplicities[:, None],
            start=self.result["params"], with_covariance=True,
        )
        repeated = sensitivity.fit_multinomial_counts(
            self.design[row_indices], self.counts[row_indices],
            start=self.result["params"], with_covariance=True,
        )
        assert_allclose(weighted["params"], repeated["params"], atol=2e-7, rtol=2e-7)
        self.assertAlmostEqual(weighted["loglike"], repeated["loglike"], places=8)
        assert_allclose(weighted["covariance"], repeated["covariance"], atol=2e-8, rtol=2e-7)

    def test_warm_start_preserves_estimate(self):
        warm = sensitivity.fit_multinomial_counts(
            self.design, self.counts, start=self.result["params"]
        )
        assert_allclose(warm["params"], self.result["params"], atol=2e-6, rtol=2e-6)
        self.assertAlmostEqual(warm["loglike"], self.result["loglike"], places=7)


if __name__ == "__main__":
    unittest.main()
