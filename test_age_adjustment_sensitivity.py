"""Synthetic-only tests for the age-form sensitivity implementation."""

import unittest

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

import run_age_adjustment_sensitivity as analysis


class AgeAdjustmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20260910)
        n = 1600
        steps = rng.normal(size=n)
        age = rng.normal(size=n)
        eta = np.column_stack([np.zeros(n),
                               -0.7 - 0.25 * steps + 0.55 * age - 0.18 * age ** 2,
                               -1.6 - 0.45 * steps + 0.25 * age])
        probabilities = np.exp(eta - eta.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        outcome = (rng.random(n)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
        cls.data = pd.DataFrame({
            "avg_steps": 7200 + 1800 * steps, "age": 57 + 11 * age,
            "sex_at_birth": rng.choice(["Female", "Male"], n),
            "is_smoker": rng.integers(0, 2, n), "is_drinker": rng.integers(0, 3, n),
            "Outcome_Status": outcome,
        }, index=np.arange(n) * 2)
        cls.result = analysis.analyze(cls.data)

    def test_design_ranks_and_nested_parameter_counts(self):
        diagnostic = self.result["diagnostics"].set_index("model")
        self.assertEqual(diagnostic.loc["M1 multinomial", "LR_df"], 4)
        self.assertEqual(diagnostic.loc["Case-only binary", "LR_df"], 2)
        self.assertTrue(diagnostic.linear_rank.eq(7).all())
        self.assertTrue(diagnostic.spline_rank.eq(9).all())
        np.testing.assert_allclose(
            diagnostic.spline_AIC - diagnostic.linear_AIC,
            -diagnostic.LR_chi2 + 2 * diagnostic.LR_df, atol=1e-8,
        )

    def test_samples_and_scales_are_constant_within_comparison(self):
        table = self.result["estimates"]
        self.assertEqual(len(table), 8)
        self.assertTrue(table.steps_SD.eq(self.data.avg_steps.std(ddof=1)).all())
        self.assertTrue(table.loc[table.model.eq("M1 multinomial"), "n"].eq(len(self.data)).all())
        n_case = self.data.Outcome_Status.isin([1, 2]).sum()
        self.assertTrue(table.loc[table.model.eq("Case-only binary"), "n"].eq(n_case).all())

    def test_explicit_basis_matches_fixed_df_natural_spline(self):
        linear, spline, metadata = analysis.build_designs(self.data)
        automatic = patsy.dmatrix('cr(age_z, df=3, constraints="center") - 1',
                                  {"age_z": linear.age_z}, return_type="dataframe")
        np.testing.assert_allclose(spline.filter(like="age_spline").values,
                                   automatic.values, atol=1e-10)
        np.testing.assert_allclose(metadata["age_knots_years"],
                                   np.quantile(self.data.age, [0, 1/3, 2/3, 1]))

    def test_multinomial_contrast_includes_cross_covariance(self):
        _, spline, _ = analysis.build_designs(self.data)
        fit = analysis.fit_checked(self.data.Outcome_Status, spline, True)[0]
        k = spline.shape[1]
        position = spline.columns.get_loc("steps_z")
        contrast = np.zeros(2 * k)
        contrast[position] = -1
        contrast[k + position] = 1
        native = fit.t_test(contrast)
        row = self.result["estimates"].query(
            'model == "M1 multinomial" and age_form == "Natural cubic spline age" '
            'and contrast == "Acute vs outpatient"').iloc[0]
        self.assertAlmostEqual(row.log_OR, float(np.asarray(native.effect).item()), places=10)
        self.assertAlmostEqual(row.SE, float(np.asarray(native.sd).item()), places=10)

    def test_case_only_uses_pooled_design_without_restandardisation(self):
        _, spline, _ = analysis.build_designs(self.data)
        keep = self.data.Outcome_Status.isin([1, 2])
        fit = sm.Logit(self.data.loc[keep, "Outcome_Status"].eq(2).astype(int),
                       spline.loc[keep]).fit(disp=False, maxiter=250)
        row = self.result["estimates"].query(
            'model == "Case-only binary" and age_form == "Natural cubic spline age"').iloc[0]
        self.assertAlmostEqual(row.log_OR, fit.params["steps_z"], places=10)

    def test_missing_and_invalid_codes_are_rejected(self):
        data = self.data.copy()
        data.loc[data.index[0], "is_smoker"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing model"):
            analysis.build_designs(data)
        data = self.data.copy()
        data.loc[data.index[0], "is_drinker"] = 8
        with self.assertRaisesRegex(ValueError, "Unexpected category"):
            analysis.build_designs(data)

    def test_redundant_design_rejected(self):
        linear, _, _ = analysis.build_designs(self.data)
        linear["duplicate_intercept"] = linear.Intercept
        with self.assertRaisesRegex(ValueError, "rank-deficient"):
            analysis.fit_checked(self.data.Outcome_Status, linear, True)

    def test_input_and_primary_globals_not_modified(self):
        original = self.data.copy(deep=True)
        before = (analysis.primary.STEPS_MEAN, analysis.primary.STEPS_SD,
                  analysis.primary.AGE_MEAN, analysis.primary.AGE_SD)
        analysis.analyze(self.data)
        pd.testing.assert_frame_equal(self.data, original)
        self.assertEqual(before, (analysis.primary.STEPS_MEAN, analysis.primary.STEPS_SD,
                                  analysis.primary.AGE_MEAN, analysis.primary.AGE_SD))


if __name__ == "__main__":
    unittest.main()
