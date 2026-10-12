# Analysis input and variable dictionary

This package starts from a prepared, one-row-per-participant analysis dataset.
It does not perform database extraction or reconstruct upstream eligibility.
Upstream measurement definitions below describe the study's input requirements;
the scripts consume the supplied fields rather than checking source-record dates
or completeness. Internal field names are retained for
compatibility and are not clinical diagnoses unless explicitly stated.

## Study timeline and eligibility

- Baseline `t0`: the earliest first day of a calendar month for which both
  Fitbit requirements are satisfied: at least 12 months of observation before
  that date, and at least 9 valid months within the immediately preceding
  12 complete calendar months. The selected dates range from 2018-02-01 to
  2020-10-01.
- Steps exposure: `[t0 - 12 months, t0)`, the 12 complete calendar months
  immediately preceding the selected month-start boundary.
- A valid Steps month contains at least 21 valid days. A valid day has at least
  100 steps and at least 600 distinct minutes of heart-rate records.
- Health-burden assessment endpoint `tM = t0 + 1 month`; window `[t0, tM)`.
- Outcome landmark `t1 = t0 + 12 months`; both M1 and exploratory mediation ascertain
  outcomes over `[t1, t0 + 36 months]`, a 24-month risk period.
- The reported starting cohort of 10,365 already incorporates prior-CVD and
  diagnosis-status exclusions described below, together with
  wearable, EHR, sleep-coverage, survey-completeness, recorded-smoking-status and drinking-frequency eligibility. Excluding
  1,470 participants with CVD during `[t0,t1)` leaves 8,895 exported participants.
- The same classified cohort of 8,292 is used for primary association and exploratory mediation analyses:
  5,780 Control, 2,127 Outpatient, 385 Acute. A four-category sensitivity retains
  the additional 603 CVD cases with unclassified setting (148 other/unknown,
  455 missing setting). These counts are validation targets, not filters that
  reconstruct the historical selection.
- Upstream eligibility includes at least 7 valid main-sleep days in `[t0,tM)`
  and complete depression/anxiety responses; missing responses or inadequate
  sleep coverage are not encoded as zero. EHR endpoint coverage was required.

## Prepared cohort: `final_analytic_cohort_with_habits.csv`

| Field | Meaning / allowed values |
| --- | --- |
| `person_id` | Unique, nonmissing participant identifier for within-Workbench linkage. Never distribute. |
| `Group` | `Control` or `Heart Disease`; the latter is a legacy label for qualifying CVD. |
| `Onset_Type` | `N/A (Control)`, `Chronic (Office/Outpatient)`, `Acute (Hospital/ER)`, `Other/Unknown`, or missing setting for CVD cases. `Chronic` does not imply chronic rather than acute disease biology. |
| `avg_steps` | Mean valid-day steps/day in the pre-t0 exposure window. |
| `age` | Age in years at t0. |
| `sex_at_birth` | `Female` or `Male`, as represented in the analyzed export. |
| `is_smoker` | Recorded smoking status: 0 for never-smoker or non-smoker concepts; 1 for moderate, heavy, daily, occasional or unspecified smoker concepts. Code 0 is not restricted to lifetime never smokers; see the concept mapping below. |
| `is_drinker` | Past-year drinking frequency from question concept 1586201: 0 for never or monthly or less; 1 for 2-4 times/month or 2-3 times/week; 2 for at least 4 times/week. These are grouped response categories, not alcohol-volume thresholds; see below. |
| `has_sleep_disorder` | Study-defined low-sleep-efficiency indicator, not a clinical diagnosis or Fitbit Sleep Score; see below. |
| `has_depression` | Survey report of ever having been told by a health professional of depression: 0/1 among complete responses. |
| `has_anxiety` | Corresponding survey report of anxiety reaction/panic disorder: 0/1 among complete responses. |
| `has_hypertension` | Hypertension recorded during the mediator window, 0/1. |
| `has_diabetes` | Diabetes recorded during that window, 0/1. |
| `has_hyperlipidemia` | Hyperlipidaemia recorded during that window, 0/1. |
| `has_high_cholesterol` | High cholesterol recorded during that window, 0/1. |
| `has_ckd` | Chronic kidney disease recorded during that window, 0/1. |

Clinical flags describe records in the window, not necessarily incident disease.
An absent record is not proof of disease absence. Recorded smoking status and
drinking-frequency categories are distinct adjustment variables. Drinking
responses are grouped directly, without converting weekly categories into
monthly counts. These categories do not measure consumption quantity; smoking status also does not
measure cumulative tobacco exposure. Upstream coding and eligibility are input
requirements rather than operations performed by these downstream scripts.

### Recorded smoking status

Smoking status is obtained from the `observation` table by selecting
`observation_concept_id = 4275495` (`Tobacco smoking behavior finding`). The
status concepts below are stored in `value_as_concept_id`, not in
`observation_concept_id`. The most recent eligible record before `t0` supplies
the status indicator.

| Recorded concept | `value_as_concept_id` (OMOP ID) | SNOMED CT code | `is_smoker` |
| --- | --- | --- | --- |
| Never smoked tobacco | 4144272 | 266919005 | 0 |
| Non-smoker | 4222303 | 8392000 | 0 |
| Moderate smoker (20 or less per day) | 4209585 | 56578002 | 1 |
| Heavy smoker (over 20 per day) | 4209006 | 56771006 | 1 |
| Smokes tobacco daily | 42709996 | 449868002 | 1 |
| Occasional tobacco smoker | 37395605 | 428041000124106 | 1 |
| Smoker | 4298794 | 77176002 | 1 |

Unknown, no-information and missing pre-`t0` smoking-status records are ignored
when an eligible record mapping to 0 or 1 is available; the most recent eligible
pre-`t0` record supplies the indicator. Participants with no eligible record are
excluded before the reported starting cohort of 10,365. The separate upstream
exclusion of participants with explicit former-smoker/ex-smoker classifications
also applies.
The unspecified `Smoker` concept is an affirmative smoking record, distinct from
unknown smoking status. The reference category combines documented never-smoker
and non-smoker concepts and is not a lifetime-never-smoking category. The daily
quantity descriptions in the moderate/heavy concepts do not define a separate
cutpoint in this binary adjustment variable.

### Drinking frequency

The most recent eligible pre-`t0` response to question concept `1586201`
describes alcohol-use frequency in the past year. The response categories are
grouped as follows, with code 0 as the model reference:

| Original response category | `is_drinker` |
| --- | --- |
| Never | 0 |
| Monthly or less | 0 |
| 2-4 times a month | 1 |
| 2-3 times a week | 1 |
| 4 or more times a week | 2 |

Missing or indeterminate responses fail upstream eligibility and are excluded
before the reported starting cohort of 10,365. Weekly responses are not
converted into occasions per month. Alcohol quantity and binge-drinking
responses are not used to assign these categories. Code 0 includes occasional
drinking and is not an abstainer-only category; codes 1 and 2 do not establish
moderate or heavy alcohol consumption. The downstream scripts consume the
existing 0/1/2 categories rather than performing questionnaire extraction.

### Sleep and survey measurement

For each main-sleep episode, sleep efficiency is **total minutes asleep /
minutes in bed for the same episode**, not classic-specific asleep-stage
duration. Both fields must be available and minutes in bed positive. With at
least 7 valid main-sleep days, the binary indicator is 1 if strictly more than
75% of valid days have a ratio <=0.60; otherwise it is 0. A value of 0 means
that the study threshold was not met, not that normal sleep was established.
Missing/invalid days are excluded from the denominator. Fewer than 7 valid
days fails eligibility.
Sleep-efficiency calculations were checked against the underlying main-sleep
records during cohort preparation. This downstream package consumes the resulting
binary flag, not night-level values or record types; episode-level processing
and sleep-coverage assessment precede its prepared-cohort inputs. Omitting
Sleep tests score composition within the same selected cohort, not the
accuracy of the underlying sleep measurements or sleep-coverage selection.

Depression/anxiety are Personal Medical History **ever-diagnosed** responses
reported as completed in `[t0,tM)`, not current symptom scales or proof of new
diagnosis during the window. The publicly available October 2018 questionnaire
is a reference for question wording, not verified metadata for every
participant's administered edition. Exact item/answer identifiers are not
consumed by this package.

### Outcome semantics

Upstream CVD screening used `condition_status_concept_id IN (32902, 32908)`:
32902 denotes Primary diagnosis and 32908 denotes Secondary diagnosis. Both
statuses were used for prior-CVD and pre-landmark screening and to determine
control eligibility. The outcome is the first qualifying primary CVD record:
among eligible cases, only primary CVD records (32902) qualify, and the earliest
such record determines the outcome date and setting. The same status rules
apply across encounter settings.
Eligible participants are linked to diagnosis records by `person_id`.
Outcome ascertainment includes primary CVD records dated within
`[t1, t0 + 36 months]`, including both endpoints. `condition_occurrence` is
left-joined to `visit_occurrence` using both `visit_occurrence_id` and
`person_id`, so absence of a linked encounter does not discard a diagnosis.
The earliest qualifying primary **date** is selected before excluding
unclassified settings; records with missing setting remain eligible for this date
selection. Among records on that same earliest date, acute care takes priority
over other setting categories. A later acute-care record cannot replace an
earlier qualifying primary date.

| Linked `visit_concept_id` | Recording-setting category |
| --- | --- |
| 9201, 9203, 262 | Acute care |
| 9202, 581477 | Outpatient |
| NULL or 0, including no linked encounter | Missing |
| All other values | Other/Unknown |

This downstream package consumes the
prepared outcome labels; it does not execute or independently validate the
upstream extraction.
Before the reported starting cohort of 10,365, upstream screening excluded
participants with prior CVD, missing or unknown CVD diagnosis status, or
secondary-only CVD. A missing or unknown status prevents application of the
primary-outcome rule; secondary-only CVD meets neither the primary-outcome
definition nor the requirement for no recorded CVD. For participants with both
types of record, the entire participant was excluded if the earliest secondary
record preceded the earliest primary record, because CVD had already been
recorded before the qualifying primary outcome. A later primary record did not
restore eligibility. Those upstream exclusion counts and their ordering or overlap
with other eligibility exclusions are not recoverable from the prepared table.
Controls had neither a primary nor a secondary CVD record through the endpoint,
subject to the study's prior-CVD and pre-landmark eligibility requirements.
The 603 exported unclassified-setting cases are distinct from diagnosis-status
exclusions. The package does not reclassify secondary diagnoses as controls.

## Clinical component concept sets

The hypertension, diabetes and kidney-disease components include their explicit
seeds below and all descendants in the pinned CDR v8 `concept_ancestor` table.
Seeds are inclusions in their own right; listing several seeds does not assert
that they are direct children of one another. Clinical flags use qualifying records in
`[t0, tM)` without a diagnosis-status restriction. This definition is separate
from the primary-diagnosis restriction used for the CVD outcome.

| Clinical component | Explicit `condition_concept_id` inclusion seeds |
| --- | --- |
| Hypertension | 316866 |
| Diabetes | 201820, 201254, 201826, 4024659, 195771, 442793, 443238, 435216, 761051, 4058243 |
| Chronic kidney disease | 46271022, 192279, 201313 |

Hyperlipidaemia and high cholesterol are separately coded using exact matches
to `condition_concept_id` 432867 (Hyperlipidemia) and 437827 (Pure
hypercholesterolemia), respectively. Neither lipid component includes
descendants. A qualifying window-specific record of either exact concept
sets only its corresponding indicator; neither is inferred from the other.
Both indicators can be 1 when both concepts were recorded during the window.
These are study matching rules, not a claim about the medical or vocabulary
hierarchy relationships between the two diagnoses.

`export_clinical_concepts.sql` expands the 14 non-lipid seeds using vocabulary
tables only and includes the two exact-match lipid concepts without descendant
expansion or automatic cross-assignment. The output
has one row per component/concept pair, retains cross-component membership,
and reports `definition_source` and `is_explicit_inclusion` to distinguish
non-lipid hierarchy expansion from the exact-match lipid memberships.
Standard-concept status, domain and invalid-reason metadata are reported rather
than used as implicit exclusion filters. Missing seeds or missing `concept`
rows for included descendants or exact-match lipid concepts cause an assertion
failure instead of being dropped.
The exported dictionary contains no participant records or counts. It defines
vocabulary membership, not a complete upstream extraction pipeline; the model
scripts consume the five prepared binary indicators.

## Derived fields and statistical definitions

| Derived variable | Definition |
| --- | --- |
| `Outcome_Status` | 0 Control, 1 Outpatient, 2 Acute care in the classified cohort. The four-category sensitivity retains all 603 unclassified cases; the allocation sensitivity adds the 455 missing-setting cases under hypothetical assignments and excludes the 148 other/unknown cases. |
| `steps_z`, `age_z` | Pooled classified-cohort mean/SD standardization; sample SD (`ddof=1`). Fixed original scales are used in bootstrap contrasts. |
| `Sleep` | `has_sleep_disorder`. |
| `Mental` | `has_depression + has_anxiety`, range 0-2. |
| `Clinical` | Sum of the five clinical flags, range 0-5. |
| `Clinical_deduplicated` | HTN + diabetes + CKD + max(hyperlipidaemia, high cholesterol), range 0-4. |
| `vulnerability` | Legacy code name for the standardized pooled PCA-derived health-burden score; separately standardize the three domains, retain/orient PC1, then standardize the score. |

The fixed probability-scale contrast is `steps_z: 0 -> 1`, approximately
7,323 -> 10,622 steps/day. It is not the probability change for an arbitrary
1-SD increment from any starting value. Logit-scale Steps ORs are per 1 pooled
SD under the corresponding specified model.

Bootstrap refits domain standardization/PCA (with sign alignment), mediator and
outcome models. Primary pooled effects use 2,000 participant draws, 51 midpoint
empirical-residual integration nodes, and a 101-node precision check. For
pairwise effects, probabilities are normalized within each individual before
population averaging. Raw category effects share one multinomial denominator.
ACME/ADE names are retained as computational estimand labels; interpretations
are exploratory statistical indirect/direct associations, not established
biological mechanisms.

Formal moderated analyses and descriptive sex-stratified models are separate
analyses. Holm families are six secondary pathway-interaction tests, six sex/age ACME
differences (two moderator comparisons by three outcome contrasts), and three
exploratory sex-by-age difference-in-differences. Reported 95% intervals are
pointwise; subgroup significance alone is not a test of heterogeneity.
