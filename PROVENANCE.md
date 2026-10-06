# Source provenance and verification boundary

Initial package preparation: 2026-09-17. Repository owner: `kaixiang164-byte`.
Repository: https://github.com/kaixiang164-byte/fitbit-cvd-analysis.
This is a code-only downstream analysis package, not a new study-data analysis.
Its Git commit identifies the uploaded version; no archival DOI is assigned.

## Standalone sources and version history

The initial release copied six standalone analysis/rendering scripts and three
test modules from the working sources without numerical changes.
`release_manifest.json` records the current files and SHA-256 digests; Git
history records subsequent changes.

The 2026-09-24 additions comprise the post-hoc missing-setting allocation
analysis, its synthetic tests, the S14/main Fig 3 renderer, and the S15
unadjusted/adjusted comparison and tests. They postdate release `9e03508`.
The allocation analysis imports the unclassified-setting helper and uses the
existing scientific dependencies. Its renderer reads aggregate outputs,
requires 2,000 bootstrap replicates, and uses conditional sampling CIs rather
than the optional random-assignment scenario intervals.

`run_robustness_and_figure2.py` includes the Steps spline rank correction and
Figure 2 labels identifying the fixed exposure contrast, M1 probabilities and
the no-recorded-CVD reference group.
Its source therefore differs from the earlier canonical bootstrap run.
Run manifests distinguish fitted-output provenance from reporting changes.
Each authorised run creates its own manifest; historical hashes must not be
overwritten to bypass a mismatch.

## Notebook source wrapper

`run_primary_and_moderated_notebook_source.py` embeds only the source text of
zero-based cells 2, 7, 9, 11, 13, 18 and 20 from `fitbit_analyis.ipynb`.
It records each original cell's hash, verifies it before execution, and provides
explicit primary/pathways/moderation stages. The notebook installation magic
is removed; presentation-only IPython display is routed to ordinary printing.
The prepared CSV path and working directory are explicit. No notebook JSON,
saved outputs or original participant inputs are distributed.

The source cells retain historical labels and a descriptive coefficient-change
export, which is not a reported mediation result. Coefficient attenuation does
not establish mediation. The variable dictionary defines measurement and
interpretation terms.

## Release-only additions

- `prepare_age_reference.py`: compatibility export for an older reference
  filename and model label; no coefficient estimation or numerical modification.
- `test_pooled_effects_synthetic.py`: generated-data decomposition, scale and
  determinism tests.
- `test_notebook_wrapper_synthetic.py`: source-wrapper/adapter tests using
  generated data only.
- `run_table1_descriptive.py` and `test_table1_descriptive.py`: a descriptive-table
  generator and synthetic tests implementing the reported Table 1 definitions.
  This is a release implementation, not the historical table-generation code;
  agreement with the historical study table has not been checked during packaging.
- Release README, variable dictionary, environment specification, execution
  instructions, manifest and inventory validator.

Tests cover the documented functions and I/O boundaries using synthetic data.
They do not independently reproduce the original cohort or study results.
Upstream eligibility and source-field processing are prerequisites; these
scripts operate on the prepared cohort. Not all manuscript layouts are generated
automatically.

## Confidentiality

No participant-level inputs or outputs belong in this package. The notebook
wrapper creates a participant-level PCA intermediate at runtime; keep it in the
authorised Workbench. Runtime logs, backup directories, manifests and aggregate
exports require separate review before any public sharing. The repository must
start from this reviewed allowlist, not the working directory's Git history.
