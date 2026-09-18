"""Focused, synthetic-data tests; no participant data or analysis outputs written."""

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import run_robustness_and_figure2 as analysis


class StepsSplineDiagnosticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(71249)
        n = 1500
        steps = rng.normal(size=n)
        age = rng.normal(size=n)
        eta = np.column_stack(
            [np.zeros(n), -0.6 - 0.3 * steps + 0.4 * age,
             -1.5 - 0.5 * steps + 0.2 * age]
        )
        probabilities = np.exp(eta - eta.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        outcome = (rng.random(n)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
        cls.data = pd.DataFrame({
            "avg_steps": 7000 + 1200 * steps,
            "age": 55 + 10 * age,
            "sex_at_birth": rng.choice(["Female", "Male"], n),
            "is_smoker": rng.integers(0, 2, n),
            "is_drinker": rng.integers(0, 3, n),
            "Outcome_Status": outcome,
        })
        cls.diagnostic = analysis.fit_spline_test(cls.data).iloc[0]

    def test_independent_parameter_counts_and_aic_identity(self):
        result = self.diagnostic
        self.assertEqual(result["n"], len(self.data))
        self.assertEqual(result["linear_design_rank"], 7)
        self.assertEqual(result["spline_design_rank"], 9)
        self.assertEqual(result["linear_parameter_count"], 14)
        self.assertEqual(result["spline_parameter_count"], 18)
        self.assertEqual(result["df"], 4)
        self.assertEqual(result["spline_df"], 3)
        self.assertTrue(result["spline_converged"])
        self.assertAlmostEqual(
            result["spline_AIC"] - result["linear_AIC"],
            -result["likelihood_ratio_chi2"] + 2 * result["df"], places=8,
        )

    def test_centered_basis_preserves_legacy_function_space(self):
        data = self.data.assign(
            steps_z=analysis.z_standardize(self.data["avg_steps"]),
            age_z=analysis.z_standardize(self.data["age"]),
        )
        legacy = smf.mnlogit(
            "Outcome_Status ~ cr(steps_z, df=4) + age_z "
            "+ C(sex_at_birth) + C(is_smoker) + C(is_drinker)", data=data,
        )
        self.assertEqual(np.linalg.matrix_rank(legacy.exog), 9)
        self.assertEqual(legacy.exog.shape[1], 10)
        keep = [i for i, name in enumerate(legacy.exog_names)
                if name != "cr(steps_z, df=4)[3]"]
        identifiable_legacy = sm.MNLogit(
            data["Outcome_Status"], legacy.exog[:, keep]
        ).fit(method="newton", maxiter=250, disp=False)
        self.assertAlmostEqual(
            identifiable_legacy.llf, self.diagnostic["spline_log_likelihood"],
            places=8,
        )

    def test_redundant_intercept_is_rejected_before_fitting(self):
        data = self.data.assign(steps_z=analysis.z_standardize(self.data["avg_steps"]))
        with self.assertRaisesRegex(ValueError, "rank deficient"):
            analysis._fit_checked_spline_diagnostic_model(
                "Outcome_Status ~ cr(steps_z, df=4)", data, "Legacy spline"
            )

    def test_nonconvergence_is_not_suppressed(self):
        with patch.object(analysis.smf, "mnlogit") as constructor:
            model = constructor.return_value
            model.exog = np.column_stack([np.ones(10), np.arange(10)])
            model.fit.return_value.mle_retvals = {"converged": False}
            with self.assertRaisesRegex(RuntimeError, "did not converge"):
                analysis._fit_checked_spline_diagnostic_model("unused", self.data, "Test")

    def test_standalone_diagnostic_does_not_refit_pca_or_write_outputs(self):
        with patch.object(analysis, "load_analysis_data", return_value=self.data), \
                patch.object(analysis, "initialize_analysis", side_effect=AssertionError), \
                patch.object(analysis, "compute_reference_loadings", side_effect=AssertionError), \
                patch.object(analysis, "archive_previous_outputs", side_effect=AssertionError), \
                patch.object(pd.DataFrame, "to_csv", side_effect=AssertionError):
            result = analysis.run_spline_diagnostic_only().iloc[0]
        self.assertAlmostEqual(result["p_value"], self.diagnostic["p_value"], places=12)

    def test_code_guard_accepts_only_original_or_exact_reviewed_revision(self):
        manifest = {"code_sha256": "original"}
        with patch.object(analysis, "sha256_file", return_value="original"):
            analysis.validate_canonical_code(manifest)
        with patch.object(analysis, "sha256_file", return_value="changed"):
            with self.assertRaisesRegex(RuntimeError, "without a recorded"):
                analysis.validate_canonical_code(manifest)
        manifest["post_run_code_revisions"] = [{
            "source_run_code_sha256": "original",
            "revised_code_sha256": "reviewed",
            "scope": "steps-spline-diagnostic-only",
            "mediation_refitted": False,
        }]
        with patch.object(analysis, "sha256_file", return_value="reviewed"):
            analysis.validate_canonical_code(manifest)
        with patch.object(analysis, "sha256_file", return_value="later-unreviewed"):
            with self.assertRaises(RuntimeError):
                analysis.validate_canonical_code(manifest)

    def test_code_guard_rejects_revision_for_a_different_source_run(self):
        manifest = {"code_sha256": "original", "post_run_code_revisions": [{
            "source_run_code_sha256": "different-run",
            "revised_code_sha256": "reviewed",
            "scope": "steps-spline-diagnostic-only",
            "mediation_refitted": False,
        }]}
        with patch.object(analysis, "sha256_file", return_value="reviewed"):
            with self.assertRaises(RuntimeError):
                analysis.validate_canonical_code(manifest)


if __name__ == "__main__":
    unittest.main()
