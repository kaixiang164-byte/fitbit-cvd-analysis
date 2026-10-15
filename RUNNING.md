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

## Vocabulary-only Clinical dictionary export

Replace every `PROJECT.DATASET` placeholder in `export_clinical_concepts.sql`
with the exact dataset identifier for the pinned CDR v8 release; do not
substitute a newer vocabulary release. Submit the complete Standard SQL script
in an authorised Workbench BigQuery session; no lipid query parameters are
needed. Hyperlipidaemia uses exactly concept 432867, and high cholesterol uses
exactly concept 437827. Neither lipid definition includes descendants or
automatically assigns the other indicator. A participant may have both
indicators when both exact concepts were recorded during the window.

The script reads only `concept` and `concept_ancestor` and uses temporary tables.
The 14 explicit non-lipid seeds are expanded to all descendants. All components
are deduplicated within component, with cross-component membership retained.
The output distinguishes non-lipid seed/descendant membership from exact-match
lipid membership using `definition_source`, and marks direct inclusions with
`is_explicit_inclusion`. It does not apply a
diagnosis-status restriction or filter by standard status, domain, or validity;
these metadata are displayed in the final result.

The script fails if any seed, including `443238`, is absent from the selected
dictionary, or if an included descendant or exact-match lipid concept lacks a
`concept` row. Resolve such
failures against the selected CDR v8 vocabulary; do not remove the seed or
silently change versions to obtain a result. Export the final result table as
a CSV dictionary, retaining its component and concept-ID columns. The output
contains vocabulary metadata only, not participant identifiers or counts.
Record the CDR dataset identifier and code commit with that dictionary.

This query does not select dated clinical records, construct participant-level
flags, or run models. Its static unit tests check the specified non-lipid seeds,
the two exact-match lipid definitions and
query contract, not execution against BigQuery or historical study results:

```bash
python -B -m unittest -v test_clinical_concept_export
```

## 1. Primary models and prerequisite exports

Table 1 can be generated independently from the same prepared cohort, without
fitting models. Run the generator from the code directory in the authorised
Workbench and use a separate, non-existent output directory:

```bash
python -B run_table1_descriptive.py --prepared-cohort /authorized/input/final_analytic_cohort_with_habits.csv --output-dir /authorized/new_table1_run
```

It writes a numeric aggregate CSV, a LaTeX table, and aggregate generation
metadata. The CLI checks classified counts against 5,780/2,127/385; unclassified
CVD settings are excluded. Every displayed variable must be complete in the
classified sample: missing values fail rather than becoming zero or silently
changing denominators. Binary percentages use the full outcome-column total;
continuous summaries use sample SD (`ddof=1`). This release's Table 1 generator
follows the documented definitions and has synthetic tests only; an
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

### 1a. Unadjusted and adjusted Steps comparison (S15)

This reporting supplement fits a Steps-only multinomial model and reproduces
the adjusted M1 Steps estimates in the identical classified sample. It retains
the original pooled Steps scale and uses the full cross-logit covariance for
acute-care versus outpatient. It does not refit mediation or alter the cohort.
Use explicit private inputs and output locations outside the code checkout:

```bash
python -B run_crude_adjusted_steps_comparison.py --data /authorized/input/final_analytic_cohort_with_habits.csv --scales /authorized/new_analysis_run/m1_standardization_parameters.csv --reference /authorized/new_analysis_run/m1_primary_pairwise_results_python.csv --output-dir /authorized/new_analysis_run/crude_adjusted_steps_outputs --table /authorized/new_analysis_run/S15_Table.tex
```

Outputs are aggregate estimates, final-analysis-field missingness counts,
reproduction diagnostics and S15 LaTeX. Re-running the command can replace these
outputs; use a new working/output location and review outputs before sharing.
Differences between unadjusted and adjusted ORs are not percentages of
confounding removed. Synthetic tests use no participant data.

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
| Unclassified setting and binary any-CVD recording | Prepared 8,895-person cohort, `m1_standardization_parameters.csv`, `m1_primary_pairwise_results_python.csv` |
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

S7 Panel A describes outpatient (N=2,127), acute-care (N=385), missing-setting
(N=455) and Other/Unknown-setting (N=148) cases separately. It reports means
and sample SDs, a Steps median with first and third quartiles, and categorical
counts and percentages. The two SMD columns compare each unclassified subgroup
with the pooled 2,512 classified cases; they are not missing-versus-outpatient
or Other/Unknown-versus-acute comparisons. No descriptive hypothesis tests are
performed, and the median row has no SMD. The outputs are
`unclassified_setting_case_characteristics_by_setting.csv`,
`unclassified_setting_classified_reference.csv`, and
`unclassified_setting_descriptive_metadata.json`. Small-cell and complementary
suppression applies to the affected counts, percentages and SMDs before the
four-group descriptive table is written. These summaries describe observed
groups; they do not establish why setting information is incomplete.

The S7 script also fits two supplementary binary logistic models of any
qualifying primary CVD recording versus no recorded CVD. The first uses the
classified cohort (N=8,292; 2,512 cases and 5,780 participants without recorded
CVD); the second uses the full prepared cohort (N=8,895; 3,115 cases and the
same 5,780 participants without recorded CVD). All 603 unclassified cases,
including 455 with missing setting and 148 with Other/Unknown setting, enter
the second binary model as cases without an acute/outpatient assignment.
Both models retain the original M1 Steps and age standardisation, linear age,
sex at birth, recorded smoking status and categorical drinking frequency.

The binary outputs are `any_cvd_binary_results.csv` (all coefficients),
`any_cvd_binary_steps_comparison.csv` (the two Steps estimates), and
`any_cvd_binary_metadata.json` (sample counts, scaling, fit diagnostics and
run information). S7 Panel C and the aggregate report present the two ORs and
Wald 95% CIs. These nested cohorts retain the same upstream eligibility rules;
the binary models are not independent replications and do not test the
acute-care versus outpatient contrast. No model coefficients are averaged
across the multinomial outcomes to construct these estimates.

### 5a. Missing-setting allocation scenarios

This analysis is distinct from the S7 four-category model. It adds the
455 missing-setting cases to the original
8,292 classified-cohort participants under hypothetical acute/outpatient
probabilities, excludes the 148 combined Other/Unknown cases, and does not
refit PCA or mediation. The original prepared 8,895-person cohort, M1 scaling
and `m1_primary_pairwise_results_python.csv` are required. The existing
`run_unclassified_setting_sensitivity.py` module supplies the compatible
input/design/reference helpers and must remain beside the allocation script.

Run from the source directory, using private inputs and a new output directory
in the authorised Workbench (not this public-release checkout):

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -B run_setting_reassignment_sensitivity.py --data-dir /authorized/new_analysis_run --output-dir /authorized/new_analysis_run/setting_reassignment_sensitivity --bootstrap 2000 --workers 4 --seed 20260924
python -B build_setting_reassignment_si.py --directory /authorized/new_analysis_run
```

The renderer expects the sensitivity CSV and metadata within the named
`setting_reassignment_sensitivity` child directory. It writes S14 Table and
main Figure 3 LaTeX source plus the figure PDF/PNG/TIFF into the parent working
directory; a LaTeX installation is needed separately to compile the table and
captioned-figure documents. It does not rerun models or read participant data.
These commands can replace existing output files: always start with a new
working/output location. The renderer requires all 2,000 bootstrap replicates
and the documented cohort counts; a small smoke run cannot generate the final
manuscript assets.

The main point estimator maximises a fractional multinomial log-likelihood.
`sampling_lower/upper` are percentile CIs conditional on each fixed allocation
scenario. `scenario_lower/upper` additionally include randomly completed labels
and are not calibrated CIs for that point estimator or the unknown true
association. The analysis script's diagnostic heatmap displays these latter
scenario intervals; the manuscript `build_setting_reassignment_si.py` renderer
uses only `sampling_lower/upper`. Neither output estimates the true missing
settings or uncertainty over which allocation assumptions are correct.

## Generated output review

Keep participant-level PCA intermediates inside the Workbench. Generated logs,
manifests and backup folders can contain local paths or sensitive context.
Even aggregate outputs require disclosure review before release. Figure and
table production commands do not confer permission to export every file.

Record the code version, run configuration, convergence/failure counts and
environment with each real run. Tests on synthetic data do not certify equality
to the historical manuscript estimates; that comparison needs authorised input.
