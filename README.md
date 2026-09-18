# Fitbit steps and cardiovascular recording: downstream analysis code

Code and variable documentation for the analysis of wearable-derived daily
steps, cardiovascular diagnosis recording setting, and exploratory statistical
indirect associations in the All of Us Research Program.

**Repository:** https://github.com/kaixiang164-byte/fitbit-cvd-analysis

This source-only package documents the downstream analysis. Use the Git commit
identifier to identify a version; no archival DOI has been assigned. An open
source licence has not yet been selected, so no MIT or other licence is implied.

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
Legacy source names such as `vulnerability`, `has_sleep_disorder`, `smoker` and
`moderate/heavy` are retained to preserve the existing code: their current
scientific meanings are defined in the variable dictionary. Code labels are
not evidence of clinical diagnoses, causal effects or alcohol-volume categories.

## Analysis coverage

| Component | Source / output role |
| --- | --- |
| Primary M1, Steps-by-sex, pooled PCA and M2 | `run_primary_and_moderated_notebook_source.py`, primary stage: exported current notebook source cells and prerequisite input producers |
| Formal pooled pathway interactions and conditional sex/age contrasts | Same wrapper, pathways/moderation stages; not interchangeable with separately fitted sex models |
| Pooled decomposition, score-construction robustness, case-only consistency, Steps spline, Figure 2 | `run_robustness_and_figure2.py` |
| Separate female/male mediation fits (S2) | `run_sex_stratified_mediation.py` |
| Unclassified-setting sensitivity (S7) | `run_unclassified_setting_sensitivity.py` |
| Nonlinear age-adjustment sensitivity (S8) | `run_age_adjustment_sensitivity.py` |
| Compatibility export for the age script | `prepare_age_reference.py`; filename/model-label adapter only, no model fitting |
| Sleep-omission sensitivity (S10) | `run_sleep_omission_sensitivity.py` |
| Shared pooled estimates for the main table and S3/S4 | `render_pooled_mediation_tex.py` |

The primary/moderation wrapper contains source-only notebook cells, not saved
execution output. The separately maintained pooled script is the canonical
implementation of the pooled bootstrap; the wrapper does not substitute an
older pooled bootstrap cell. Downstream source coverage does not establish
independent validation of upstream measurement dates or selection rules.
Figure 1/S1 Fig are manuscript diagrams; S9 is phenotype documentation rather
than a computed statistical table. A complete Table 1 descriptive-table
generator has not been recovered in this package; do not claim that every
manuscript asset is generated automatically.

## Checks without study data

```bash
python -B -m unittest discover -v -p 'test_*.py'
python -B validate_release.py
```

Tests use synthetic data and mocked I/O. The inventory validator checks hashes,
Python syntax and selected disclosure hazards; it is not proof of scientific
validity or exhaustive privacy review. None of these checks reruns study data.
The source-only provenance manifest distinguishes unmodified standalone scripts
from the new notebook-source packaging wrapper and release documentation.

## Release safety

Do not upload a working analysis directory, participant CSVs, notebook files,
runtime logs, generated manifests, backups or unreviewed aggregate outputs.
The `.gitignore` helps avoid accidents but is not a security boundary; publish
only the reviewed files listed in `release_manifest.json` and that manifest.
New data/result files belong outside the repository. Only the reviewed code and
documentation are intended for this public repository; participant-level data
are not available here. An open source licence requires the authors' approval.
