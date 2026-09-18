"""Generated-data checks of the extracted wrapper; never reads study inputs.

All records are manufactured from a fixed RNG seed. The pathways smoke also
executes the primary stage. The 2000-draw moderation stage is syntax/hash
checked only, and no original notebook is opened.
"""

import contextlib
import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_age_adjustment_sensitivity as age_sensitivity
import prepare_age_reference as age_reference
import run_primary_and_moderated_notebook_source as wrapper
import run_robustness_and_figure2 as pooled
import run_sex_stratified_mediation as sex_stratified
import run_unclassified_setting_sensitivity as setting_sensitivity


class NotebookWrapperSyntheticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(prefix="notebook-wrapper-synthetic-")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.base = Path(cls.temporary.name)
        rng = np.random.default_rng(917240)
        n = 1200
        steps = rng.normal(size=n)
        age = rng.normal(size=n)
        burden = -0.15 * steps + 0.15 * age + rng.normal(size=n)
        flags = (burden[:, None] + rng.normal(size=(n, 8)) > 0.4).astype(int)
        eta = np.column_stack([
            np.zeros(n), -0.25 - 0.15 * steps + 0.15 * burden,
            -0.8 - 0.25 * steps + 0.25 * burden,
        ])
        probabilities = np.exp(eta - eta.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        outcome = (rng.random(n)[:, None] > probabilities.cumsum(axis=1)).sum(axis=1)
        cls.cohort = pd.DataFrame({
            "person_id": np.arange(n) + 90000000,
            "avg_steps": 7400 + 1200 * steps,
            "age": 57 + 9 * age,
            "sex_at_birth": rng.choice(["Female", "Male"], n),
            "is_smoker": rng.integers(0, 2, n),
            "is_drinker": rng.integers(0, 3, n),
            "Group": np.where(outcome == 0, "Control", "Heart Disease"),
            "Onset_Type": np.array([
                "N/A (Control)", "Chronic (Office/Outpatient)",
                "Acute (Hospital/ER)",
            ])[outcome],
        })
        for index, name in enumerate([
            "has_sleep_disorder", "has_depression", "has_anxiety",
            "has_hypertension", "has_diabetes", "has_hyperlipidemia",
            "has_high_cholesterol", "has_ckd",
        ]):
            cls.cohort[name] = flags[:, index]
        cls.synthetic_input = cls.base / "manufactured_cohort.csv"
        cls.cohort.to_csv(cls.synthetic_input, index=False)
        cls.output = cls.base / "runtime-output"
        previous_directory = Path.cwd()
        with contextlib.redirect_stdout(io.StringIO()):
            wrapper.run_analysis(cls.synthetic_input, cls.output, stage="pathways")
        if Path.cwd() != previous_directory:
            raise AssertionError("Wrapper failed to restore the working directory.")

    def test_all_embedded_sources_including_moderation_verify(self):
        self.assertEqual(wrapper.verify_embedded_sources(), (2, 7, 9, 11, 13, 18, 20))
        self.assertEqual(wrapper.STAGE_CELLS["primary"], (2, 7, 9, 11, 13))
        self.assertNotIn(20, wrapper.STAGE_CELLS["pathways"])

    def test_primary_exports_supply_sex_and_setting_interfaces(self):
        frame, by_sex = sex_stratified._load_data(
            self.output / "pca_mediation_eligible_dataset.csv"
        )
        self.assertEqual(len(frame), len(self.cohort))
        self.assertEqual(set(by_sex), {"Female", "Male"})
        self.assertTrue(frame.person_id.is_unique)
        self.assertAlmostEqual(frame.vulnerability.mean(), 0, places=12)
        self.assertAlmostEqual(frame.vulnerability.std(ddof=1), 1, places=12)
        loadings = pd.read_csv(self.output / "pca_vulnerability_loadings.csv")
        self.assertEqual(set(loadings.raw_feature), {"Sleep", "Mental", "Clinical"})
        self.assertTrue(np.isfinite(loadings.PC1_loading).all())
        # The setting script expects its cohort beside the generated reference
        # files. This is another copy of our manufactured fixture, never study data.
        self.cohort.to_csv(self.output / wrapper.SOURCE_INPUT_BASENAME, index=False)
        data, scales = setting_sensitivity.load_data(self.output)
        reproduced, _ = setting_sensitivity.fit_and_contrast(data, scales, "Synthetic M1")
        setting_sensitivity.verify_primary(reproduced, self.output)

    def test_age_linear_estimates_match_primary_reference(self):
        data = self.cohort.copy()
        data["Outcome_Status"] = np.select([
            data.Group.eq("Control"),
            data.Onset_Type.eq("Chronic (Office/Outpatient)"),
            data.Onset_Type.eq("Acute (Hospital/ER)"),
        ], [0, 1, 2])
        result = age_sensitivity.analyze(data)["estimates"]
        actual = result.loc[
            result.model.eq("M1 multinomial") & result.age_form.eq("Linear age")
        ].set_index("contrast")
        reference = pd.read_csv(self.output / "m1_primary_pairwise_results_python.csv")
        reference = reference.loc[reference.term.eq("steps_z")].set_index("contrast")
        for column in ["log_OR", "SE", "OR"]:
            np.testing.assert_allclose(
                actual.loc[reference.index, column], reference[column], atol=1e-9, rtol=1e-9
            )

    def test_age_adapter_preserves_values_and_passes_reference_checks(self):
        reference_path, _ = age_reference.prepare_reference(self.output)
        with (self.output / age_reference.SOURCE_FILENAME).open(newline="") as handle:
            original = list(csv.reader(handle))
        with reference_path.open(newline="") as handle:
            adapted = list(csv.reader(handle))
        model_column = original[0].index("model")
        expected = [row.copy() for row in original]
        for row in expected[1:]:
            row[model_column] = "M1_primary"
        self.assertEqual(adapted, expected)
        self.assertFalse(age_reference.prepare_reference(self.output)[1])
        data = self.cohort.copy()
        data["Outcome_Status"] = np.select([
            data.Group.eq("Control"),
            data.Onset_Type.eq("Chronic (Office/Outpatient)"),
            data.Onset_Type.eq("Acute (Hospital/ER)"),
        ], [0, 1, 2])
        # Produce the second required reference from the same manufactured data.
        with patch.multiple(
            pooled, STEPS_MEAN=data.avg_steps.mean(), STEPS_SD=data.avg_steps.std(ddof=1),
            AGE_MEAN=data.age.mean(), AGE_SD=data.age.std(ddof=1),
        ):
            pooled.fit_case_only_setting_sensitivity(data).to_csv(
                self.output / "case_only_acute_vs_outpatient_results.csv", index=False
            )
        with patch.object(age_sensitivity, "BASE_DIR", self.output):
            checks = age_sensitivity.verify_linear_reproduction(
                age_sensitivity.analyze(data)["estimates"]
            )
        self.assertEqual(len(checks), 4)

    def test_age_adapter_rejects_invalid_inputs_and_existing_different_output(self):
        source_rows = list(csv.DictReader(
            io.StringIO((self.output / age_reference.SOURCE_FILENAME).read_text())
        ))
        for case in ["wrong-model", "nonfinite", "duplicate", "existing"]:
            with self.subTest(case=case), tempfile.TemporaryDirectory(
                prefix="age-reference-synthetic-"
            ) as temporary:
                directory = Path(temporary)
                rows = [row.copy() for row in source_rows]
                if case == "wrong-model":
                    rows[0]["model"] = "M2"
                elif case == "nonfinite":
                    rows[0]["SE"] = "nan"
                elif case == "duplicate":
                    rows.append(rows[0].copy())
                with (directory / age_reference.SOURCE_FILENAME).open("w", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                destination = directory / age_reference.DESTINATION_FILENAME
                if case == "existing":
                    destination.write_text("Existing synthetic sentinel\n")
                with self.assertRaises(FileExistsError if case == "existing" else ValueError):
                    age_reference.prepare_reference(directory)
                if case == "existing":
                    self.assertEqual(destination.read_text(), "Existing synthetic sentinel\n")
                else:
                    self.assertFalse(destination.exists())
        with self.assertRaisesRegex(ValueError, "outside the code release"):
            age_reference.prepare_reference(Path(age_reference.__file__).parent)

    def test_pathways_export_has_exactly_six_holm_adjusted_tests(self):
        table = pd.read_csv(self.output / "moderated_path_omnibus_tests.csv")
        adjusted = table.loc[table.Holm_p_value.notna()]
        self.assertEqual(len(adjusted), 6)
        self.assertEqual(adjusted.Holm_family.nunique(), 1)
        self.assertTrue(np.isfinite(table.Wald_chi2).all())
        self.assertFalse((self.output / "moderated_mediation_conditional_draws.csv").exists())

    def test_output_boundary_rejects_reuse_and_code_directory(self):
        with self.assertRaises(FileExistsError):
            wrapper.run_analysis(self.synthetic_input, self.output)
        with self.assertRaisesRegex(ValueError, "outside the code release"):
            wrapper.run_analysis(self.synthetic_input, Path(wrapper.__file__).parent / "forbidden")


if __name__ == "__main__":
    unittest.main()
