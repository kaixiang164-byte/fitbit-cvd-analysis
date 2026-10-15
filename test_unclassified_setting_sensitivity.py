"""Synthetic tests for the supplementary setting-agnostic CVD models."""

import tempfile
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import statsmodels.api as sm

import run_unclassified_setting_sensitivity as analysis


class AnyCVDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(20261014)
        n = 2400
        steps = rng.normal(size=n)
        age = rng.normal(size=n) - .2 * steps
        eta = -.4 - .3 * steps + .6 * age
        cases = rng.random(n) < 1 / (1 + np.exp(-eta))
        setting = rng.choice([1, 2, 3], n, p=[.60, .20, .20])
        outcome = np.where(cases, setting, 0)
        cls.data = pd.DataFrame({
            "person_id": np.arange(n), "Group": np.where(cases, "Heart Disease", "Control"),
            "outcome": outcome, "avg_steps": 7323.3 + 3298.82 * steps,
            "age": 59.86 + 14.64 * age, "sex_at_birth": rng.choice(["Female", "Male"], n),
            "is_smoker": rng.integers(0, 2, n), "is_drinker": rng.integers(0, 3, n),
        }, index=np.arange(n) * 3)
        cls.data["Onset_Type"] = cls.data.outcome.map({
            0: "N/A (Control)", 1: "Chronic (Office/Outpatient)",
            2: "Acute (Hospital/ER)", 3: "Other/Unknown",
        })
        missing_indices = cls.data.index[cls.data.outcome.eq(3)][::2]
        cls.data.loc[missing_indices, "Onset_Type"] = np.nan
        cls.parameters = pd.DataFrame({"mean": [7323.3, 59.86], "SD": [3298.82, 14.64]},
                                      index=["avg_steps", "age"])
        cls.results, cls.diagnostics = analysis.fit_any_cvd_models(cls.data, cls.parameters)

    def test_nested_cohorts_keep_all_unclassified_cases(self):
        rows = self.results.loc[self.results.term.eq("steps_z")].set_index("cohort")
        first, full = rows.loc["Original classified cohort"], rows.loc["Full prepared cohort"]
        n_unclassified = int(self.data.outcome.eq(3).sum())
        self.assertEqual(first.n_unclassified_cvd, 0)
        self.assertEqual(full.n_unclassified_cvd, n_unclassified)
        self.assertEqual(full.n - first.n, n_unclassified)
        self.assertEqual(full.n_cvd - first.n_cvd, n_unclassified)
        self.assertEqual(first.n_no_recorded_cvd, full.n_no_recorded_cvd)
        self.assertEqual(full.n_cvd, int(self.data.Group.eq("Heart Disease").sum()))

    def test_scales_and_categorical_design_are_unchanged(self):
        for cohort in (self.data, self.data.loc[self.data.outcome.ne(3)]):
            _, design = analysis.build_design(cohort, self.parameters)
            np.testing.assert_allclose(design.steps_z, (cohort.avg_steps - 7323.3) / 3298.82)
            np.testing.assert_allclose(design.age_z, (cohort.age - 59.86) / 14.64)
            self.assertEqual(design.shape[1], 7)
            np.testing.assert_array_equal(design["C(is_drinker)[T.1]"], cohort.is_drinker.eq(1))
            np.testing.assert_array_equal(design["C(is_drinker)[T.2]"], cohort.is_drinker.eq(2))
        self.assertTrue(self.results.steps_SD.eq(3298.82).all())
        self.assertTrue(self.results.age_SD.eq(14.64).all())

    def test_direct_binary_fit_and_wald_results_agree(self):
        for label, cohort in (("Original classified cohort", self.data.loc[self.data.outcome.ne(3)]),
                              ("Full prepared cohort", self.data)):
            complete, design = analysis.build_design(cohort, self.parameters)
            model = sm.Logit(complete.Group.eq("Heart Disease").astype(int), design).fit(disp=False)
            rows = self.results.loc[self.results.cohort.eq(label)].set_index("term")
            np.testing.assert_allclose(rows.log_OR, model.params.reindex(rows.index), atol=1e-12)
            np.testing.assert_allclose(rows.SE, model.bse.reindex(rows.index), atol=1e-12)
            np.testing.assert_allclose(rows.OR, np.exp(rows.log_OR), atol=1e-12)
            np.testing.assert_allclose(rows.CI_lower, np.exp(rows.log_OR - 1.96 * rows.SE), atol=1e-12)
            np.testing.assert_allclose(rows.CI_upper, np.exp(rows.log_OR + 1.96 * rows.SE), atol=1e-12)

    def test_formula_glm_and_logit_converge(self):
        for diagnostic in self.diagnostics:
            self.assertTrue(diagnostic["converged"])
            self.assertTrue(diagnostic["formula_GLM_converged"])
            self.assertEqual(diagnostic["n_excluded_missing_model_covariates"], 0)
            self.assertLess(diagnostic["maximum_absolute_average_score"], 1e-8)
            self.assertLess(diagnostic["formula_GLM_maximum_absolute_coefficient_difference"], 1e-8)
            self.assertGreater(diagnostic["minimum_covariance_eigenvalue"], 0)

    def test_input_and_parameters_not_modified(self):
        data, parameters = self.data.copy(deep=True), self.parameters.copy(deep=True)
        analysis.fit_any_cvd_models(data, parameters)
        pd.testing.assert_frame_equal(data, self.data)
        pd.testing.assert_frame_equal(parameters, self.parameters)

    def test_missing_invalid_and_inconsistent_data_raise(self):
        data = self.data.copy()
        data.loc[data.index[0], "is_smoker"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing"):
            analysis.fit_any_cvd_models(data, self.parameters)
        data = self.data.copy()
        data.loc[data.index[0], "is_drinker"] = 7
        with self.assertRaisesRegex(ValueError, "Unexpected category"):
            analysis.fit_any_cvd_models(data, self.parameters)
        data = self.data.copy()
        data.loc[data.index[0], "Group"] = "Control" if data.iloc[0].outcome else "Heart Disease"
        with self.assertRaisesRegex(ValueError, "disagree"):
            analysis.fit_any_cvd_models(data, self.parameters)
        data = self.data.copy()
        data.loc[data.index[1], "person_id"] = data.iloc[0].person_id
        with self.assertRaisesRegex(ValueError, "one non-missing ID"):
            analysis.fit_any_cvd_models(data, self.parameters)

    def test_invalid_scale_counts_and_outcome_levels_raise(self):
        parameters = self.parameters.copy()
        parameters.loc["avg_steps", "SD"] = 0
        with self.assertRaisesRegex(ValueError, "positive SDs"):
            analysis.fit_any_cvd_models(self.data, parameters)
        parameters.loc["avg_steps", "SD"] = np.inf
        with self.assertRaisesRegex(ValueError, "finite means"):
            analysis.fit_any_cvd_models(self.data, parameters)
        with self.assertRaisesRegex(ValueError, "Unexpected cohort counts"):
            analysis.fit_any_cvd_models(self.data, self.parameters,
                                       expected_counts={"Original classified cohort": (1, 1, 0)})
        with self.assertRaisesRegex(ValueError, "Both binary outcome levels"):
            analysis.fit_any_cvd_models(self.data.loc[self.data.outcome.eq(0)], self.parameters)

    def test_s7_renderer_includes_both_binary_rows_and_scope(self):
        first, _ = analysis.fit_and_contrast(self.data.loc[self.data.outcome.ne(3)], self.parameters,
                                            "Reproduced three-category M1")
        full, _ = analysis.fit_and_contrast(self.data, self.parameters,
                                           "Four-category unclassified-setting sensitivity")
        steps = pd.concat([first, full]).query('term == "steps_z"')
        descriptive, _ = analysis.setting_specific_descriptives(self.data)
        descriptive, _ = analysis.apply_descriptive_disclosure_controls(descriptive)
        n_unclassified = int(self.data.outcome.eq(3).sum())
        metadata = {"input_n": len(self.data), "primary_n": int(self.data.outcome.ne(3).sum()),
                    "unclassified_n": n_unclassified,
                    "unclassified_percent_of_all_cvd": 100 * n_unclassified / self.data.outcome.ne(0).sum()}
        binary_steps = self.results.query('term == "steps_z"')
        with tempfile.TemporaryDirectory() as temporary:
            analysis.write_supporting_table(Path(temporary), descriptive, steps, metadata,
                                            int(self.data.Onset_Type.eq("Other/Unknown").sum()),
                                            int(self.data.Onset_Type.isna().sum()), binary_steps)
            rendered = (Path(temporary) / "S7_Table.tex").read_text()
        self.assertIn("C. Any qualifying primary CVD recording", rendered)
        for row in binary_steps.itertuples():
            self.assertIn(f"{row.cohort} & {row.n:,} & {row.n_cvd:,} & {row.n_no_recorded_cvd:,}", rendered)
            self.assertIn(f"{row.OR:.3f} ({row.CI_lower:.3f}--{row.CI_upper:.3f})", rendered)
        self.assertIn("nested", rendered)
        self.assertIn("post hoc", rendered)
        self.assertNotIn("person_id", rendered)


class SettingDescriptiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        frames = []
        for number, label in enumerate(("Chronic (Office/Outpatient)", "Acute (Hospital/ER)",
                                        None, "Other/Unknown")):
            j = np.arange(120)
            frames.append(pd.DataFrame({
                "Group": "Heart Disease", "Onset_Type": label,
                "age": 40 + number * 4 + j / 6,
                "avg_steps": 5000 + number * 700 + j * (20 + number * 3),
                "sex_at_birth": np.where(j % 2, "Female", "Male"),
                "is_smoker": j % 2, "is_drinker": j % 3,
            }))
        cls.data = pd.concat(frames, ignore_index=True)
        cls.table, cls.reference = analysis.setting_specific_descriptives(cls.data)

    def test_four_disjoint_groups_and_pooled_reference(self):
        self.assertEqual(len(self.table), 8)
        self.assertEqual(len(self.reference), 8)
        for group in analysis.DESCRIPTIVE_GROUPS:
            self.assertTrue(self.table[f"{group}_n_available"].eq(120).all())
            self.assertTrue(self.table[f"{group}_n_missing"].eq(0).all())
        self.assertTrue(self.reference.n_available.eq(240).all())
        mean = self.table.loc[self.table.variable.eq("Daily steps, mean (SD)")].iloc[0]
        self.assertNotEqual(mean.missing_mean, mean.other_unknown_mean)
        with self.assertRaisesRegex(ValueError, "Unexpected four-group"):
            analysis.setting_specific_descriptives(self.data, {name: 121 for name in analysis.DESCRIPTIVE_GROUPS})

    def test_median_quartiles_and_no_median_smd(self):
        row = self.table.loc[self.table.type.eq("median")].iloc[0]
        for group, label in zip(analysis.DESCRIPTIVE_GROUPS,
                               ("Chronic (Office/Outpatient)", "Acute (Hospital/ER)", None, "Other/Unknown")):
            subset = self.data.loc[self.data.Onset_Type.isna() if label is None else self.data.Onset_Type.eq(label)]
            self.assertAlmostEqual(row[f"{group}_median"], subset.avg_steps.median())
            self.assertAlmostEqual(row[f"{group}_q1"], subset.avg_steps.quantile(.25, interpolation="linear"))
            self.assertAlmostEqual(row[f"{group}_q3"], subset.avg_steps.quantile(.75, interpolation="linear"))
        self.assertTrue(np.isnan(row.SMD_missing_minus_classified))
        self.assertTrue(np.isnan(row.SMD_other_unknown_minus_classified))

    def test_smd_uses_same_pooled_classified_comparator(self):
        row = self.table.loc[self.table.variable.eq("Daily steps, mean (SD)")].iloc[0]
        classified = self.data.loc[self.data.Onset_Type.isin(["Chronic (Office/Outpatient)", "Acute (Hospital/ER)"]), "avg_steps"]
        missing = self.data.loc[self.data.Onset_Type.isna(), "avg_steps"]
        other = self.data.loc[self.data.Onset_Type.eq("Other/Unknown"), "avg_steps"]
        for name, subgroup in (("missing", missing), ("other_unknown", other)):
            expected = (subgroup.mean() - classified.mean()) / np.sqrt((subgroup.var(ddof=1) + classified.var(ddof=1)) / 2)
            self.assertAlmostEqual(row[f"SMD_{name}_minus_classified"], expected)
        wrong_comparator = (missing.mean() - other.mean()) / np.sqrt((missing.var(ddof=1) + other.var(ddof=1)) / 2)
        self.assertNotAlmostEqual(row.SMD_missing_minus_classified, wrong_comparator)
        female = self.table.loc[self.table.variable.eq("Female sex")].iloc[0]
        self.assertEqual(female.SMD_missing_minus_classified, 0)

    def test_complete_covariates_and_unmodified_inputs(self):
        original = self.data.copy(deep=True)
        analysis.setting_specific_descriptives(self.data)
        pd.testing.assert_frame_equal(self.data, original)
        missing = self.data.copy()
        missing.loc[0, "age"] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing descriptive covariates"):
            analysis.setting_specific_descriptives(missing)
        invalid = self.data.copy()
        invalid.loc[0, "Onset_Type"] = "Not a setting category"
        with self.assertRaisesRegex(ValueError, "four disjoint"):
            analysis.setting_specific_descriptives(invalid)

    def test_small_counts_trigger_complementary_suppression(self):
        data = self.data.copy()
        other_index = data.index[data.Onset_Type.eq("Other/Unknown")]
        data.loc[other_index, "is_drinker"] = 1
        data.loc[other_index[:40], "is_drinker"] = 0
        data.loc[other_index[-5:], "is_drinker"] = 2
        raw, _ = analysis.setting_specific_descriptives(data)
        public, report = analysis.apply_descriptive_disclosure_controls(raw)
        self.assertEqual(len(report["flagged_cells"]), 1)
        self.assertNotIn("count", report["flagged_cells"][0])
        affected = public.variable.isin(["Drinking, 2-4/month or 2-3/week", "Drinking, >=4/week"])
        for group in ("missing", "other_unknown"):
            self.assertTrue(public.loc[affected, f"{group}_suppressed"].all())
            self.assertTrue(public.loc[affected, f"{group}_reporting_status"].eq("suppressed").all())
            for field in (f"{group}_count", f"{group}_percent", f"SMD_{group}_minus_classified"):
                self.assertTrue(public.loc[affected, field].isna().all())
        self.assertFalse(public.outpatient_suppressed.any())
        self.assertFalse(public.acute_suppressed.any())
        self.assertTrue(raw.loc[affected, "other_unknown_count"].notna().all())

    def test_small_complements_and_unreviewed_patterns(self):
        data = self.data.copy()
        other_index = data.index[data.Onset_Type.eq("Other/Unknown")]
        data.loc[other_index, "is_drinker"] = 2
        data.loc[other_index[:4], "is_drinker"] = 1
        raw, _ = analysis.setting_specific_descriptives(data)
        flags = analysis.descriptive_disclosure_flags(raw)
        self.assertEqual({flag["variable"] for flag in flags},
                         {"Drinking, 2-4/month or 2-3/week", "Drinking, >=4/week"})
        public, _ = analysis.apply_descriptive_disclosure_controls(raw)
        self.assertEqual(int(public.other_unknown_suppressed.sum()), 2)
        data.loc[other_index, "sex_at_birth"] = "Female"
        data.loc[other_index[:4], "sex_at_birth"] = "Male"
        raw, _ = analysis.setting_specific_descriptives(data)
        with self.assertRaisesRegex(ValueError, "Additional small-cell"):
            analysis.apply_descriptive_disclosure_controls(raw)


if __name__ == "__main__":
    unittest.main()
