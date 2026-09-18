"""Manufactured Table 1 fixtures only; no participant inputs or notebook outputs."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

import run_table1_descriptive as table1


def manufactured_cohort():
    data = pd.DataFrame({
        "person_id": range(100001, 100009),
        "Group": ["Control"] * 2 + ["Heart Disease"] * 6,
        "Onset_Type": ["N/A (Control)"] * 2 + ["Chronic (Office/Outpatient)"] * 2
        + ["Acute (Hospital/ER)"] * 2 + ["Other/Unknown", None],
        "age": [40, 60, 50, 70, 60, 80, 55, 65],
        "avg_steps": [6000, 8000, 5000, 7000, 4000, 6000, 3000, 4000],
        "sex_at_birth": ["Female", "Male"] * 4,
        "is_smoker": [0, 1] * 4,
        "is_drinker": [0, 1, 2, 2, 0, 2, 1, 1],
    })
    for index, flag in enumerate(table1.BINARY_FLAGS):
        data[flag] = ([0, 1] if index % 2 == 0 else [1, 1]) * 4
    return data


class Table1SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.cohort = manufactured_cohort()

    def test_outcomes_denominators_and_no_input_mutation(self):
        original = self.cohort.copy(deep=True)
        result, metadata = table1.summarize_table1(self.cohort)
        self.assertEqual(len(result), 51)
        self.assertTrue(result.denominator.eq(2).all())
        self.assertEqual(metadata["classified_n"], 6)
        self.assertEqual(metadata["excluded_unclassified_setting_n"], 2)
        female = result.loc[result.field.eq("sex_at_birth")]
        self.assertTrue(female["count"].eq(1).all())
        self.assertTrue(female.percent.eq(50).all())
        pd.testing.assert_frame_equal(original, self.cohort)

    def test_sample_sd_and_derived_scores(self):
        summary, _ = table1.summarize_table1(self.cohort)
        control = summary.loc[summary.outcome.eq("Control")].set_index("field")
        self.assertAlmostEqual(control.loc["age", "mean"], 50)
        self.assertAlmostEqual(control.loc["age", "sd"], np.sqrt(200))
        self.assertEqual(control.loc["age", "display"], "50.0 (14.1)")
        self.assertEqual(control.loc["Mental", "mean"], 1.5)
        self.assertEqual(control.loc["Clinical", "mean"], 4)

    def test_drinking_categories_partition_each_column(self):
        summary, _ = table1.summarize_table1(self.cohort)
        drinking = summary.loc[summary.field.eq("is_drinker")]
        self.assertTrue(drinking.groupby("outcome")["count"].sum().eq(2).all())
        self.assertTrue(drinking.groupby("outcome").percent.sum().eq(100).all())

    def test_missing_classified_values_never_become_zero(self):
        for field in ("has_sleep_disorder", "has_depression", "has_hypertension", "age", "is_drinker"):
            with self.subTest(field=field):
                data = self.cohort.copy()
                data.loc[0, field] = np.nan
                with self.assertRaisesRegex(ValueError, "missing required values"):
                    table1.summarize_table1(data)

    def test_unclassified_fields_are_not_used_for_summary(self):
        data = self.cohort.copy()
        data.loc[6:, "has_depression"] = np.nan
        result, metadata = table1.summarize_table1(data)
        self.assertEqual(metadata["classified_n"], 6)
        self.assertTrue(result.denominator.eq(2).all())

    def test_invalid_codes_ids_groups_and_counts_rejected(self):
        for field, value in [("has_sleep_disorder", 2), ("is_drinker", 3),
                             ("avg_steps", float("inf")), ("Group", "Unknown"),
                             ("sex_at_birth", "Unknown"), ("Onset_Type", "Other label"),
                             ("person_id", 100002), ("age", -1)]:
            with self.subTest(field=field):
                data = self.cohort.copy()
                if field == "avg_steps":
                    data[field] = data[field].astype(float)
                data.loc[0, field] = value
                with self.assertRaises((ValueError, TypeError)):
                    table1.summarize_table1(data)
        with self.assertRaisesRegex(ValueError, "Outcome counts differ"):
            table1.summarize_table1(self.cohort, expected_counts=table1.EXPECTED_COUNTS)
        with self.assertRaisesRegex(ValueError, "Missing required"):
            table1.summarize_table1(self.cohort.drop(columns="has_anxiety"))

    def test_contradictory_control_and_case_labels_rejected(self):
        for index, setting in [(0, "Acute (Hospital/ER)"), (0, "Other/Unknown"), (2, "N/A (Control)")]:
            with self.subTest(index=index, setting=setting):
                data = self.cohort.copy()
                data.loc[index, "Onset_Type"] = setting
                with self.assertRaises(ValueError):
                    table1.summarize_table1(data)

    def test_tex_contains_current_definitions_and_no_identifiers(self):
        summary, metadata = table1.summarize_table1(self.cohort)
        rendered = table1.render_table1_tex(summary, metadata)
        self.assertIn(r"Control ($N=2$)", rendered)
        self.assertIn(r"Smoking, $>3$/month", rendered)
        self.assertIn(r"1 (50.0\%)", rendered)
        self.assertIn("Low sleep-efficiency indicator", rendered)
        self.assertNotIn("person_id", rendered)
        self.assertNotIn("p value", rendered)
        self.assertEqual(rendered.count(r"\begin{table}"), 1)
        self.assertEqual(rendered.count(r"\end{table}"), 1)

    def test_runtime_io_is_aggregate_exclusive_and_outside_code(self):
        with tempfile.TemporaryDirectory(prefix="table1-manufactured-") as temporary:
            directory = Path(temporary)
            source = directory / "manufactured.csv"
            self.cohort.to_csv(source, index=False)
            destination = directory / "output"
            table1.generate_table1(source, destination, expected_counts={x: 2 for x in table1.EXPECTED_COUNTS})
            self.assertEqual({x.name for x in destination.iterdir()}, {
                "table1_descriptive_summary.csv", "table1_descriptive.tex", "table1_generation_metadata.json"
            })
            for output in destination.iterdir():
                content = output.read_text()
                self.assertNotIn("100001", content)
                self.assertNotIn(str(source), content)
            metadata = json.loads((destination / "table1_generation_metadata.json").read_text())
            self.assertEqual(metadata["classified_n"], 6)
            with self.assertRaises(FileExistsError):
                table1.generate_table1(source, destination)
            with self.assertRaisesRegex(ValueError, "outside the code release"):
                table1.generate_table1(source, table1.CODE_DIRECTORY / "forbidden")
            rejected = directory / "wrong-count-output"
            with self.assertRaises(ValueError):
                table1.generate_table1(source, rejected, expected_counts=table1.EXPECTED_COUNTS)
            self.assertFalse(rejected.exists())


if __name__ == "__main__":
    unittest.main()
