"""Synthetic-only tests for the unadjusted/adjusted Steps supplement."""

import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

import run_crude_adjusted_steps_comparison as analysis


class CrudeAdjustedStepsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260924)
        n = 1800
        steps = rng.normal(size=n)
        age = -.3 * steps + rng.normal(size=n)
        eta = np.column_stack([np.zeros(n), -.5 - .25 * steps + .55 * age,
                               -1.8 - .5 * steps + .35 * age])
        probability = np.exp(eta - eta.max(axis=1, keepdims=True))
        probability /= probability.sum(axis=1, keepdims=True)
        outcomes = (rng.random(n)[:, None] > probability.cumsum(axis=1)).sum(axis=1)
        cls.data = pd.DataFrame({
            "avg_steps": 7300 + 2100 * steps, "age": 58 + 12 * age,
            "sex_at_birth": rng.choice(["Female", "Male"], n),
            "is_smoker": rng.integers(0, 2, n), "is_drinker": rng.integers(0, 3, n),
            "Outcome_Status": outcomes,
        }, index=np.arange(n) * 3)
        cls.scales = {"steps_mean": 7300., "steps_SD": 2100., "age_mean": 58., "age_SD": 12.}
        cls.estimates, cls.diagnostics = analysis.analyze(cls.data, cls.scales)

    def test_original_scales_and_identical_rows(self):
        designs = analysis.build_designs(self.data, self.scales)
        self.assertEqual(designs["Unadjusted"].shape, (len(self.data), 2))
        self.assertEqual(designs["M1 adjusted"].shape, (len(self.data), 7))
        for design in designs.values():
            self.assertTrue(design.index.equals(self.data.index))
            np.testing.assert_allclose(design.steps_z, (self.data.avg_steps - 7300) / 2100)
        self.assertTrue(self.estimates.n.eq(len(self.data)).all())
        self.assertTrue(self.estimates.steps_SD.eq(2100).all())

    def test_direct_crude_fit_agrees(self):
        design = analysis.build_designs(self.data, self.scales)["Unadjusted"]
        direct = sm.MNLogit(self.data.Outcome_Status, design).fit(disp=False)
        row = self.estimates.query('model == "Unadjusted" and contrast == "Acute vs control"').iloc[0]
        self.assertAlmostEqual(row.log_OR, direct.params.loc["steps_z", 1], places=11)

    def test_acute_outpatient_full_cross_covariance(self):
        for label, design in analysis.build_designs(self.data, self.scales).items():
            fit = analysis.fit_model(self.data.Outcome_Status, design)
            k = design.shape[1]
            vector = np.zeros(k * 2)
            vector[1], vector[k + 1] = -1, 1
            test = fit.t_test(vector)
            row = self.estimates.loc[self.estimates.model.eq(label)
                                     & self.estimates.contrast.eq("Acute vs outpatient")].iloc[0]
            self.assertAlmostEqual(row.log_OR, float(np.asarray(test.effect).item()), places=11)
            self.assertAlmostEqual(row.SE, float(np.asarray(test.sd).item()), places=11)

    def test_reproduction_allows_legacy_rounded_critical_value(self):
        reference = self.estimates.loc[self.estimates.model.eq("M1 adjusted")].copy()
        reference["term"] = "steps_z"
        reference["CI_lower"] = np.exp(reference.log_OR - 1.96 * reference.SE)
        reference["CI_upper"] = np.exp(reference.log_OR + 1.96 * reference.SE)
        self.assertEqual(len(analysis.verify_adjusted(self.estimates, reference)), 3)
        reference.loc[reference.index[0], "OR"] += .01
        with self.assertRaisesRegex(AssertionError, "failed to reproduce"):
            analysis.verify_adjusted(self.estimates, reference)

    def test_missing_and_unknown_categories_raise(self):
        data = self.data.copy()
        data.loc[data.index[0], "is_smoker"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing model"):
            analysis.build_designs(data, self.scales)
        data = self.data.copy()
        data.loc[data.index[0], "is_drinker"] = 7
        with self.assertRaisesRegex(ValueError, "Unexpected category"):
            analysis.build_designs(data, self.scales)

    def test_missing_outcome_level_and_nonfinite_scale_raise(self):
        with self.assertRaisesRegex(ValueError, "All three"):
            analysis.build_designs(self.data.loc[self.data.Outcome_Status.ne(2)], self.scales)
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            analysis.build_designs(self.data, dict(self.scales, steps_SD=np.inf))

    def test_original_group_and_setting_consistency(self):
        data = pd.DataFrame({
            "Group": ["Control", "Heart Disease", "Heart Disease", "Heart Disease"],
            "Onset_Type": ["N/A (Control)", "Chronic (Office/Outpatient)",
                           "Acute (Hospital/ER)", np.nan],
        })
        analysis.validate_export_categories(data)
        invalid = data.copy()
        invalid.loc[0, "Onset_Type"] = "Acute (Hospital/ER)"
        with self.assertRaisesRegex(ValueError, "must not carry a case setting"):
            analysis.validate_export_categories(invalid)
        invalid = data.copy()
        invalid.loc[1, "Group"] = "Unknown"
        with self.assertRaisesRegex(ValueError, "original Group"):
            analysis.validate_export_categories(invalid)
        invalid = data.copy()
        invalid.loc[2, "Onset_Type"] = "N/A (Control)"
        with self.assertRaisesRegex(ValueError, "among Heart Disease"):
            analysis.validate_export_categories(invalid)

    def test_input_unchanged_and_models_converged(self):
        original = self.data.copy(deep=True)
        scales = self.scales.copy()
        analysis.analyze(self.data, self.scales)
        pd.testing.assert_frame_equal(self.data, original)
        self.assertEqual(self.scales, scales)
        self.assertTrue(all(row["converged"] for row in self.diagnostics))

    def test_renderer_reports_distinct_crude_adjusted_columns(self):
        text = analysis.render_s15(self.estimates)
        self.assertIn("S15 Table", text)
        self.assertIn("Unadjusted", text)
        self.assertIn("M1 adjusted", text)
        self.assertIn("non-collapsible", text)
        self.assertNotIn("person_id", text)


if __name__ == "__main__":
    unittest.main()
