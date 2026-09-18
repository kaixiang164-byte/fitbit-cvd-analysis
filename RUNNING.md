# Execution order and output boundaries

Run participant-level analyses only in the authorised All of Us Researcher
Workbench. Start in a **separate working copy of these scripts inside that
environment**; do not run them in the public-release checkout or upload its
runtime outputs. Some historical scripts read/write next to their source and
may replace existing outputs. Use a new working directory for every analysis.
These commands document the workflow; no study-data rerun was performed during
code packaging. The original prepared cohort and row order are needed for
exact resample comparisons. Models guard the manuscript cohort size/counts;
small arbitrary synthetic CSVs cannot replace the study input in the canonical
pooled CLI. The extracted primary/pathway notebook cells are also exercised on
manufactured cohorts by tests and do not enforce the manuscript's exact counts.

## 1. Primary models and prerequisite exports

Table 1 can be generated independently from the same prepared cohort, without
fitting models. Run this new generator from the code directory in the authorised
Workbench and use a separate, non-existent output directory:

```bash
python -B run_table1_descriptive.py --prepared-cohort /authorized/input/final_analytic_cohort_with_habits.csv --output-dir /authorized/new_table1_run
```

It writes a numeric aggregate CSV, a LaTeX table, and aggregate generation
metadata. The CLI checks classified counts against 5,780/2,127/385; unclassified
CVD settings are excluded. Every displayed variable must be complete in the
classified sample: missing values fail rather than becoming zero or silently
changing denominators. Binary percentages use the full outcome-column total;
continuous summaries use sample SD (`ddof=1`). This newly written script follows
the current documented Table 1 definitions and has synthetic tests only; an
authorised run must check agreement with historical values before replacing
the manuscript table. It does not reconstruct upstream selection or source
measurements. Keep outputs in the Workbench until disclosure review.

For example, from the code directory inside an authorised Workbench environment
(replace these illustrative paths with your approved locations):

```bash
python -B run_primary_and_moderated_notebook_source.py --prepared-cohort /authorized/input/final_analytic_cohort_with_habits.csv --output-dir /authorized/new_analysis_run --stage primary
python -B prepare_age_reference.py --directory /authorized/new_analysis_run
```

`run_primary_and_moderated_notebook_source.py` provides a guarded command-line
wrapper around selected existing notebook source cells. Use its `--help` for
the prepared input path, explicit new output directory and stage selection.
The primary stage supplies M1 scaling, M1/M2 exports, pooled PCA loadings and the
prepared PCA dataset needed by later branches. The PCA dataset is a
participant-level **private intermediate**, never a public release artifact.

Place the prepared cohort in the new working analysis directory under the
filename `final_analytic_cohort_with_habits.csv`, and copy the reviewed source
scripts into that working directory. Retain the producers' filenames unchanged.
Invoke the wrapper and age-reference adapter from the original code directory
as shown above: both intentionally reject runtime writes inside their own code
directory. Run the following standalone commands from the separate working
analysis directory containing their copied scripts and private input files.

```bash
cd /authorized/new_analysis_run
```

## 2. Pooled decomposition, score sensitivity, case-only model and Figure 2

```bash
python -B run_robustness_and_figure2.py --bootstrap 2000 --workers 8 --residual-nodes 51 --probability-simulations 500 --seed 20260903
python -B render_pooled_mediation_tex.py
```

The pooled script produces the canonical summary/draw/metadata files,
`case_only_acute_vs_outpatient_results.csv`, Figure 2, and a new
`canonical_mediation_manifest.json`. It backs up recognised earlier outputs,
but that is not a substitute for starting in a new directory. The renderer
checks agreement of primary result views and requires 2,000 successful draws.
**The renderer has no `--help` parser**: use its source/docstring for help.

## 3. Formal pooled sex/age moderation

```bash
python -B run_primary_and_moderated_notebook_source.py --prepared-cohort /authorized/input/final_analytic_cohort_with_habits.csv --output-dir /authorized/new_moderation_run --stage moderation
```

This separate new directory intentionally regenerates the primary prerequisites
before moderation; do not reuse an existing output directory. Use `--stage
pathways` for the interaction regressions without the 2,000-replicate bootstrap.

Use the notebook-source wrapper's moderation stage to generate the pathway
interaction tests and conditional/difference summaries underlying S1/S5/S6.
It preserves the source implementation and its original configuration rather
than substituting the separately fitted sex-stratified analysis below. The
source is retained for provenance; legacy descriptive attenuation output does
not establish mediation and is not a main finding of the current manuscript.

## 4. Separate sex-stratified models (supplementary)

```bash
python -B run_sex_stratified_mediation.py --input pca_mediation_eligible_dataset.csv --reference-loadings pca_vulnerability_loadings.csv --scaling m1_standardization_parameters.csv --output-dir sex_stratified_results --bootstrap 2000 --workers 8 --seed 20260904 --residual-nodes 51 --sensitivity-residual-nodes 101 --grid-size 1201
```

This requires the primary-stage PCA dataset, loading and scaling exports.
Significance within one sex and not another is not a heterogeneity test.

## 5. Remaining sensitivities

```bash
python -B run_unclassified_setting_sensitivity.py --directory .
python -B run_age_adjustment_sensitivity.py --output-dir age_sensitivity
python -B run_sleep_omission_sensitivity.py --bootstrap 2000 --workers 8 --output-dir sleep_omission
```

| Script | Prerequisites |
| --- | --- |
| Unclassified setting | Prepared 8,895-person cohort, `m1_standardization_parameters.csv`, `m1_primary_pairwise_results_python.csv` |
| Age adjustment | Prepared cohort, `manuscript_multinomial_pairwise_results.csv` (M1 rows), case-only results from step 2 |
| Sleep omission | Completed pooled run with 51 nodes, 2,000 successful draws, all manifest-listed files, saved child seeds and primary draws |

`prepare_age_reference.py` is a release-packaging compatibility adapter, not a
new statistical analysis. It exports the generated M1 rows under the older
reference filename expected by the age script, changing only the model label
from `M1 primary` to `M1_primary` and preserving all numerical columns. This
M1-only reference should not be described as the historical combined M1/M2 file.

The sleep output directory must not already exist. Do not bypass its provenance
guards: the alternative score uses matching participant resamples. The age
script verifies reproduction of both linear-age reference fits before comparing
age specifications. The four-category model tests a changed outcome model, not
all possible acute/outpatient assignments of missing settings.

## Generated output review

Keep participant-level PCA intermediates inside the Workbench. Generated logs,
manifests and backup folders can contain local paths or sensitive context.
Even aggregate outputs require disclosure review before release. Figure and
table production commands do not confer permission to export every file.

Record the code version, run configuration, convergence/failure counts and
environment with each real run. Tests on synthetic data do not certify equality
to the historical manuscript estimates; that comparison needs authorised input.
