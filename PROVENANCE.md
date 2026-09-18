# Source provenance and verification boundary

Package preparation date: 2026-09-17. Intended account: `kaixiang164-byte`.
Repository: https://github.com/kaixiang164-byte/fitbit-cvd-analysis.
This is a code-only downstream analysis package, not a new study-data analysis.
Its Git commit identifies the uploaded version; no archival DOI is assigned.

## Preserved standalone sources

The six existing analysis/rendering scripts and three existing test modules
were copied byte-for-byte from the current working sources. The manifest records
their SHA-256 digests. Study source files, notebooks, input data and fitted
results were not edited during this packaging work.

`run_robustness_and_figure2.py` is the current reviewed source, including the
Steps spline rank correction and the Figure 2 fixed-exposure-contrast label.
It is not relabelled as the exact source of the earlier canonical bootstrap.
The model-fitting functions are preserved. A new authorised run creates its
own manifest; existing historical provenance must not be overwritten simply
to make a newer source pass an older run's checks.

## Notebook source wrapper

`run_primary_and_moderated_notebook_source.py` embeds only the source text of
zero-based cells 2, 7, 9, 11, 13, 18 and 20 from `fitbit_analyis.ipynb`.
It records each original cell's hash, verifies it before execution, and provides
explicit primary/pathways/moderation stages. The notebook installation magic
is removed; presentation-only IPython display is routed to ordinary printing.
The prepared CSV path and working directory are explicit. No notebook JSON,
saved outputs or original participant inputs are distributed.

The source cells retain historical labels and a descriptive coefficient-change
export. Those outputs are not additional findings endorsed by the current
manuscript; coefficient attenuation is not proof of mediation. The variable
dictionary and manuscript define the current measurement/interpretation terms.

## Release-only additions

- `prepare_age_reference.py`: compatibility export for an older reference
  filename and model label; no coefficient estimation or numerical modification.
- `test_pooled_effects_synthetic.py`: generated-data decomposition, scale and
  determinism tests.
- `test_notebook_wrapper_synthetic.py`: source-wrapper/adapter tests using
  generated data only.
- `run_table1_descriptive.py` and `test_table1_descriptive.py`: newly written
  descriptive-table generator and synthetic checks following the reported
  Table 1 definitions. These are not recovered historical extraction or
  table-generation code; study-data agreement has not been checked during
  packaging.
- Release README, variable dictionary, environment specification, execution
  instructions, manifest and inventory validator.

Test results document the tested functions and I/O boundaries, not independent
recovery of the original cohort. In particular, the upstream eligibility and
source-field definitions are prerequisites, not reconstructed by these scripts.
The source-only package does not automatically generate every manuscript
layout. The new Table 1 generator requires the documented prepared cohort and
does not establish that historical source records met upstream eligibility.

## Confidentiality

No participant-level inputs or outputs belong in this package. The notebook
wrapper creates a participant-level PCA intermediate at runtime; keep it in the
authorised Workbench. Runtime logs, backup directories, manifests and aggregate
exports require separate review before any public sharing. The repository must
start from this reviewed allowlist, not the working directory's Git history.
