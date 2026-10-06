# Fitbit steps and cardiovascular recording: downstream analysis code

Code and variable documentation for the analysis of wearable-derived daily
steps, cardiovascular diagnosis recording setting, and exploratory statistical
indirect associations in the All of Us Research Program.

**Repository:** https://github.com/kaixiang164-byte/fitbit-cvd-analysis

This source-only package documents the downstream analysis. Cite the Git commit
used for an analysis; no archival DOI has been assigned. No open-source licence
has been selected.

## Scope

The workflow starts from a prepared participant-level cohort. It does not
perform database extraction or upstream eligibility filtering. It contains
no participant data, notebook outputs, fitted-result exports or credentials.
Restricted inputs and any participant-level intermediate files must remain
inside the authorised All of Us Researcher Workbench. Independent researchers
must obtain their own access through https://www.researchallofus.org/register/.

See [VARIABLES.md](VARIABLES.md) for input fields, measurement windows and
coding; [ENVIRONMENT.md](ENVIRONMENT.md) for dependencies and numerical settings;
and [RUNNING.md](RUNNING.md) for execution order and output dependencies.
Internal names such as `vulnerability`, `has_sleep_disorder`, `smoker` and
`moderate/heavy` are retained for compatibility. Their scientific meanings are
defined in the variable dictionary rather than inferred from those names.

## Analysis coverage

| Component | Source / output role |
| --- | --- |
| Table 1 descriptive characteristics | `run_table1_descriptive.py`; documented row definitions, column-total percentages and sample SDs; see the validation scope below |
| Primary M1, Steps-by-sex, pooled PCA and M2 | `run_primary_and_moderated_notebook_source.py`, primary stage: exported current notebook source cells and prerequisite input producers |
| Unadjusted and adjusted Steps comparison (S15) | `run_crude_adjusted_steps_comparison.py`; identical classified sample and original Steps scale, full multinomial covariance, with reproduction of the saved M1 Steps estimates |
| Formal pooled pathway interactions and conditional sex/age contrasts | Same wrapper, pathways/moderation stages; not interchangeable with separately fitted sex models |
| Pooled decomposition, score-construction robustness, case-only consistency, Steps spline, Figure 2 | `run_robustness_and_figure2.py` |
| Separate female/male mediation fits (S2) | `run_sex_stratified_mediation.py` |
| Unclassified-setting sensitivity (S7) | `run_unclassified_setting_sensitivity.py` |
| Hypothetical allocation of missing settings (S14/main Fig 3) | `run_setting_reassignment_sensitivity.py`; 455 missing-setting cases allocated under fixed scenarios, with 148 Other/Unknown cases excluded |
| S14/main Fig 3 typesetting | `build_setting_reassignment_si.py`; reads aggregate outputs and reports scenario-conditional sampling CIs |
| Nonlinear age-adjustment sensitivity (S8) | `run_age_adjustment_sensitivity.py` |
| Compatibility export for the age script | `prepare_age_reference.py`; filename/model-label adapter only, no model fitting |
| Sleep-omission sensitivity (S10) | `run_sleep_omission_sensitivity.py` |
| Shared pooled estimates for the main table and S3/S4 | `render_pooled_mediation_tex.py` |

The primary/moderation wrapper contains notebook source cells without saved
execution output. The pooled bootstrap is implemented separately in
`run_robustness_and_figure2.py`. Figure 1/S1 Fig are manuscript diagrams; S9 is
phenotype documentation rather than a computed statistical table. This package
does not automatically generate every manuscript layout.

The Table 1 generator implements the documented definitions and rejects missing
required fields rather than recoding them as absence. It was written for this
release and tested on synthetic inputs; agreement with the historical study
table has not been checked during packaging.

## Checks without study data

```bash
python -B -m unittest discover -v -p 'test_*.py'
python -B validate_release.py
```

Tests use synthetic data and mocked I/O. The inventory validator checks hashes,
Python syntax and selected disclosure hazards; it is not proof of scientific
validity or exhaustive privacy review. None of these checks reruns study data.
See [VALIDATION.md](VALIDATION.md) for test coverage and
[PROVENANCE.md](PROVENANCE.md) for source history and verification boundaries.
Allocation tests cover calibration, fractional multinomial estimation and
bootstrap behaviour on synthetic inputs. The allocation script also writes
optional random-assignment scenario intervals for diagnostic use; these are
not confidence intervals for the fractional point estimate. The S14/main
Fig 3 renderer uses only the conditional sampling CIs.

## Release safety

Do not upload a working analysis directory, participant CSVs, notebook files,
runtime logs, generated manifests, backups or unreviewed aggregate outputs.
The `.gitignore` helps avoid accidents but is not a security boundary; publish
only the reviewed files listed in `release_manifest.json` and that manifest.
New data/result files belong outside the repository. Only the reviewed code and
documentation are intended for this public repository; participant-level data
are not available here. An open source licence requires the authors' approval.
