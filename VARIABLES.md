# Analysis input and variable dictionary

This package starts from a prepared, one-row-per-participant analysis dataset.
It does not perform database extraction or reconstruct upstream eligibility.
Upstream measurement definitions below document investigator-confirmed study
rules; the scripts consume the supplied fields and cannot independently check
their dates or source-record completeness. Legacy field names are retained for
compatibility and are not clinical diagnoses unless explicitly stated.

## Study timeline and eligibility

- Baseline `t0`: investigator-confirmed range 2018-02-01 to 2020-10-01.
- Steps exposure: `[t0 - 12 months, t0)`; more than 12 months of prior Fitbit
  observation and at least 9 valid calendar months in the exposure window.
- A valid Steps month contains at least 21 valid days. A valid day has at least
  100 steps and at least 600 distinct minutes of heart-rate records.
- Mediator-assessment endpoint `tM = t0 + 1 month`; window `[t0, tM)`.
- Outcome landmark `t1 = t0 + 12 months`; both M1 and mediation ascertain
  outcomes over `[t1, t0 + 36 months]`, a 24-month risk period.
- The reported starting cohort of 10,365 already incorporates prior-CVD,
  wearable, EHR, sleep-coverage and survey-completeness eligibility. Excluding
  1,470 participants with CVD during `[t0,t1)` leaves 8,895 exported participants.
- The same classified cohort of 8,292 is used for primary and mediation analyses:
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
| `is_smoker` | Monthly smoking frequency: 0 for <=3 occasions/month, 1 for >3. Code 0 does not necessarily mean no smoking. |
| `is_drinker` | Monthly drinking frequency: 0 for <=1 occasion/month; 1 for >1 to <=10; 2 for >10. These are frequency categories, not alcohol-volume thresholds. |
| `has_sleep_disorder` | Study-defined low-sleep-efficiency indicator, not a clinical diagnosis or Fitbit Sleep Score; see below. |
| `has_depression` | Survey report of ever having been told by a health professional of depression: 0/1 among complete responses. |
| `has_anxiety` | Corresponding survey report of anxiety reaction/panic disorder: 0/1 among complete responses. |
| `has_hypertension` | Hypertension recorded during the mediator window, 0/1. |
| `has_diabetes` | Diabetes recorded during that window, 0/1. |
| `has_hyperlipidemia` | Hyperlipidaemia recorded during that window, 0/1. |
| `has_high_cholesterol` | High cholesterol recorded during that window, 0/1. |
| `has_ckd` | Chronic kidney disease recorded during that window, 0/1. |

Clinical flags describe records in the window, not necessarily incident disease.
An absent record is not proof of disease absence. Smoking/drinking definitions
and timing were confirmed by the investigator; original question-to-code
mappings are not independently validated by these downstream scripts.

### Sleep and survey measurement

For each main-sleep episode, sleep efficiency is **total minutes asleep /
minutes in bed for the same episode**, not classic-specific asleep-stage
duration. Both fields must be available and minutes in bed positive. With at
least 7 valid main-sleep days, the binary indicator is 1 if strictly more than
75% of valid days have a ratio <=0.60; otherwise it is 0. Missing/invalid days
are excluded from the denominator. Fewer than 7 valid days fails eligibility.
Additional upstream quality/duplicate-episode rules are not checked here.

Depression/anxiety are Personal Medical History **ever-diagnosed** responses
reported as completed in `[t0,tM)`, not current symptom scales or proof of new
diagnosis during the window. The publicly available October 2018 questionnaire
is a reference for question wording, not verified metadata for every
participant's administered edition. Exact item/answer identifiers are not
consumed by this package.

### Outcome semantics

Qualifying records were restricted upstream to principal CVD diagnoses using
`condition_type_concept_id`, then the earliest qualifying record determined
date/setting. Principal-status numeric mappings are not inferred by this code.
Upstream participant exclusions include prior CVD, unknown diagnosis position,
secondary-only CVD, and secondary CVD preceding a later principal CVD record.
Those upstream exclusion counts are not recoverable from the prepared table.
Controls have no recorded CVD of any diagnosis position through the endpoint.
The 603 exported unclassified-setting cases are distinct from diagnosis-position
exclusions. The package does not reclassify secondary diagnoses as controls.

## Derived fields and statistical definitions

| Derived variable | Definition |
| --- | --- |
| `Outcome_Status` | 0 Control, 1 Outpatient, 2 Acute care; unclassified cases enter only the separate four-category sensitivity. |
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
analyses. Holm families are six primary pathway tests, six sex/age ACME
differences (two moderator comparisons by three outcome contrasts), and three
exploratory sex-by-age difference-in-differences. Reported 95% intervals are
pointwise; subgroup significance alone is not a test of heterogeneity.
