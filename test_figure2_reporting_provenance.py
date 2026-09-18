"""Synthetic integrity tests for Figure 2 reporting-only revisions; no fits."""

import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd

import run_robustness_and_figure2 as analysis


class Figure2ReportingProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.originals = {
            "Figure2_results.pdf": "original-pdf",
            "Figure2_results.png": "original-png",
            "Fig2.tif": "original-tif",
            "mediation_bootstrap_summary.csv": "original-summary",
            "mediation_bootstrap_metadata.csv": "original-metadata",
        }
        self.visuals = {
            "Figure2_results.pdf": "revised-pdf",
            "Figure2_results.png": "revised-png",
            "Fig2.tif": "revised-tif",
        }
        self.revision = {
            "scope": "figure2-exposure-label-reporting-only",
            "source_run_code_sha256": "original-source",
            "previous_reviewed_code_sha256": "diagnostic-source",
            "revised_code_sha256": "reporting-source",
            "mediation_refitted": False,
            "fitted_results_unchanged": True,
            "source_reporting_artifacts_sha256": {
                name: self.originals[name] for name in self.visuals
            },
            "reporting_artifacts_sha256": dict(self.visuals),
        }
        self.manifest = {
            "code_sha256": "original-source",
            "canonical_version": analysis.CANONICAL_VERSION,
            "data_file_sha256": "original-data",
            "residual_quantile_nodes": 51,
            "artifacts_sha256": dict(self.originals),
            "post_run_code_revisions": [{
                "scope": "steps-spline-diagnostic-only",
                "source_run_code_sha256": "original-source",
                "revised_code_sha256": "diagnostic-source",
                "mediation_refitted": False,
                "diagnostic_artifacts_sha256": {
                    "steps_spline_nonlinearity_test.csv": "corrected-diagnostic"
                },
            }, self.revision],
        }
        self.hashes = {
            **self.originals,
            **self.visuals,
            Path(analysis.__file__).name: "reporting-source",
            analysis.DATA_FILE.name: "original-data",
            "steps_spline_nonlinearity_test.csv": "corrected-diagnostic",
        }

    def fake_hash(self, path):
        return self.hashes[Path(path).name]

    def validate(self):
        with patch.object(analysis, "sha256_file", side_effect=self.fake_hash):
            return analysis.validate_canonical_code(self.manifest)

    def test_accepts_exact_reporting_revision_without_mutating_originals(self):
        before = copy.deepcopy(self.manifest)
        self.assertEqual(self.validate(), self.visuals)
        self.assertEqual(self.manifest, before)

    def test_rejects_scientific_csv_override(self):
        self.revision["reporting_artifacts_sha256"]["mediation_bootstrap_summary.csv"] = "new-summary"
        with self.assertRaisesRegex(RuntimeError, "only the three Figure 2"):
            self.validate()

    def test_rejects_unreviewed_source(self):
        self.hashes[Path(analysis.__file__).name] = "unreviewed-source"
        with self.assertRaisesRegex(RuntimeError, "without a recorded"):
            self.validate()

    def test_rejects_different_original_run(self):
        self.revision["source_run_code_sha256"] = "different-run"
        with self.assertRaisesRegex(RuntimeError, "without a recorded"):
            self.validate()

    def test_rejects_unreviewed_predecessor(self):
        self.revision["previous_reviewed_code_sha256"] = "unreviewed-predecessor"
        with self.assertRaisesRegex(RuntimeError, "no unique reviewed"):
            self.validate()

    def test_rejects_changed_graphic(self):
        self.hashes["Figure2_results.pdf"] = "unexpected-pdf"
        with self.assertRaisesRegex(RuntimeError, "Reporting artifact changed"):
            self.validate()

    def test_rejects_relabelled_original_artifact(self):
        self.revision["source_reporting_artifacts_sha256"]["Fig2.tif"] = "wrong-original"
        with self.assertRaisesRegex(RuntimeError, "original hash mismatch"):
            self.validate()

    def test_keeps_diagnostic_integrity_check(self):
        self.hashes["steps_spline_nonlinearity_test.csv"] = "changed-diagnostic"
        with self.assertRaisesRegex(RuntimeError, "Corrected diagnostic artifact changed"):
            self.validate()

    def test_rejects_claim_that_results_were_refitted(self):
        self.revision["mediation_refitted"] = True
        with self.assertRaisesRegex(RuntimeError, "without a recorded"):
            self.validate()

    def test_loader_uses_visual_overrides_but_keeps_scientific_hashes(self):
        summary = pd.DataFrame({"estimate": [0.0]})
        metadata = pd.DataFrame({"n": [2]})
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps(self.manifest)), \
                patch.object(analysis, "sha256_file", side_effect=self.fake_hash), \
                patch.object(analysis, "initialize_analysis") as initialize, \
                patch.object(analysis.pd, "read_csv", side_effect=[summary, metadata]):
            result = analysis.load_canonical_pooled_results()
        self.assertIs(result[0], summary)
        initialize.assert_called_once_with(51)
        self.hashes["mediation_bootstrap_summary.csv"] = "changed-summary"
        with patch.object(Path, "exists", return_value=True), \
                patch.object(Path, "read_text", return_value=json.dumps(self.manifest)), \
                patch.object(analysis, "sha256_file", side_effect=self.fake_hash), \
                patch.object(analysis, "initialize_analysis") as initialize:
            with self.assertRaisesRegex(RuntimeError, "Canonical artifact changed"):
                analysis.load_canonical_pooled_results()
        initialize.assert_not_called()


if __name__ == "__main__":
    unittest.main()
