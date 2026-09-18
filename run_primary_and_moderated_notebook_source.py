#!/usr/bin/env python3
"""Standalone packaging wrapper for selected downstream notebook source cells.

Extracted 2026-09-17 from code cells 2, 7, 9, 11, 13, 18, and 20 (zero-based)
of fitbit_analyis.ipynb. Only source text is embedded: no notebook, saved
outputs, participant data, credentials, or input/output results are included.
SOURCE_CELL_SHA256 records the original source bytes for each selected cell.

This is a new extraction/packaging wrapper, not an exact historical standalone
script. Embedded mathematical source is retained verbatim except removal of
cell 2's notebook-only %pip installation line. The wrapper routes its prepared
CSV read to --prepared-cohort, runs relative writes in a new --output-dir,
and replaces rich IPython display with plain printing of the same aggregate
objects. It does not install packages, access the original notebook at runtime,
reconstruct upstream eligibility/phenotyping, or change the bootstrap settings.

Stages:
  primary (default): cells 2, 7, 9, 11, 13; M1, Steps-by-sex, pooled PCA,
      mediator regression, and M2 exports.
  pathways: primary plus cell 18; formal pooled sex/age pathway tests (S5).
  moderation: pathways plus cell 20; original 2000-replicate pooled sex/age
      moderated-mediation bootstrap, conditional effects and contrasts (S1/S6).

Cell 9 produces a PARTICIPANT-LEVEL prepared PCA dataset in the output
directory, required by separate downstream scripts. Keep all runtime outputs
within the authorized data environment and out of the code release.
A fresh output directory outside this code directory is required.

Historical output labels such as "Smoker vs nonsmoker", "Moderate", "Heavy",
and "Vulnerability" are preserved to retain source provenance. The manuscript
defines smoking/drinking codes as occasions-per-month groups; those labels do
not establish abstinence, consumption quantity, or a validated latent trait.
Cell 13's historical coefficient-attenuation export does not establish
mediation; the current manuscript uses probability-scale decompositions.
The source cells do not implement Table 1 or the upstream cohort extraction.

Runtime packages: numpy, pandas, scipy, statsmodels, and patsy (for formula
models). Importing this module or using --help/--verify-sources runs no analysis
and requires only the Python standard library. Analyses use process-wide cwd
and pandas input routing during execution; use this CLI in a dedicated process.
"""

from __future__ import annotations

import argparse
import builtins
import hashlib
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch
import uuid

SOURCE_NOTEBOOK = "fitbit_analyis.ipynb"
SOURCE_CELL_SHA256 = {
    2: "6ee4110120dbfb626135cdd9640c4bca10e5d4fb5394e8fa96b05443192aa06f",
    7: "1067c565e4460e1fb68ab430d46c33c9fa70189b9957b4005bc16427f9f5517d",
    9: "089f172e854c065c307a06086b22bfd02211399f8d5266f09e798ca0466b6208",
    11: "17b748c0791c6c3f5d718e1164d7140a667efe50ee132ea6d87fc2a2813fcf9b",
    13: "0835d7d0503e8983bdf9a9365366ae7fc12a9d1695310dd511c057ecd4bafee4",
    18: "dc353adf69360fb19601eb361bb4f8945204107ab0ebeb85917543ae620b20c8",
    20: "e4a91db17b3ee6f93662ba4186cec8013aa2fb9db68a36315d9fad6eceb10568"
}
REMOVED_INSTALL_LINE = "%pip install statsmodels\n"
SOURCE_INPUT_BASENAME = "final_analytic_cohort_with_habits.csv"
STAGE_CELLS = {
    "primary": (2, 7, 9, 11, 13),
    "pathways": (2, 7, 9, 11, 13, 18),
    "moderation": (2, 7, 9, 11, 13, 18, 20),
}

# The raw multiline strings below contain source code, not executed statements.
# Their original cell hashes are checked before every explicit analysis run.
SOURCE_CELLS = {
    2: r'''
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm

# ------------------------------------------------------------------
# M1 primary
# Outcome ~ Steps + Age + Sex + Smoking + Drinking
# ------------------------------------------------------------------
m1_data = pd.read_csv("final_analytic_cohort_with_habits.csv")

# Retain the three manuscript outcome groups and exclude unknown onset.
is_control = m1_data["Group"].eq("Control")
is_outpatient = m1_data["Onset_Type"].eq("Chronic (Office/Outpatient)")
is_acute = m1_data["Onset_Type"].eq("Acute (Hospital/ER)")
m1_data = m1_data.loc[is_control | is_outpatient | is_acute].copy()

# Numeric coding fixes Control as the reference outcome:
# 0 = Control, 1 = Outpatient, 2 = Acute.
m1_data["Outcome_Status"] = np.select(
    [
        m1_data["Group"].eq("Control"),
        m1_data["Onset_Type"].eq("Chronic (Office/Outpatient)"),
        m1_data["Onset_Type"].eq("Acute (Hospital/ER)"),
    ],
    [0, 1, 2],
    default=np.nan,
)
# Explicit category order fixes the covariate reference categories.
m1_data["sex_at_birth"] = pd.Categorical(
    m1_data["sex_at_birth"], categories=["Female", "Male"]
)
m1_data["is_smoker"] = pd.Categorical(
    m1_data["is_smoker"], categories=[0, 1]
)
m1_data["is_drinker"] = pd.Categorical(
    m1_data["is_drinker"], categories=[0, 1, 2]
)

# Define the complete analytic sample before estimating means and SDs.
m1_complete_columns = [
    "Outcome_Status", "avg_steps", "age",
    "sex_at_birth", "is_smoker", "is_drinker",
]
m1_analysis = m1_data[m1_complete_columns].dropna().copy()
m1_analysis["Outcome_Status"] = m1_analysis["Outcome_Status"].astype(int)

# Z-standardize continuous variables using the complete analytic sample.
m1_steps_mean = m1_analysis["avg_steps"].mean()
m1_steps_sd = m1_analysis["avg_steps"].std(ddof=1)
m1_age_mean = m1_analysis["age"].mean()
m1_age_sd = m1_analysis["age"].std(ddof=1)
if m1_steps_sd == 0 or m1_age_sd == 0:
    raise ValueError("Cannot standardize a continuous variable with SD=0.")
m1_analysis["steps_z"] = (
    m1_analysis["avg_steps"] - m1_steps_mean
) / m1_steps_sd
m1_analysis["age_z"] = (
    m1_analysis["age"] - m1_age_mean
) / m1_age_sd

m1_standardization_parameters = pd.DataFrame({
    "variable": ["avg_steps", "age"],
    "mean": [m1_steps_mean, m1_age_mean],
    "SD": [m1_steps_sd, m1_age_sd],
    "ddof": [1, 1],
    "n": [len(m1_analysis), len(m1_analysis)],
})
m1_standardization_parameters.to_csv(
    "m1_standardization_parameters.csv", index=False
)

m1_formula = (
    "Outcome_Status ~ steps_z + age_z + C(sex_at_birth) "
    "+ C(is_smoker) + C(is_drinker)"
)
m1_fit = smf.mnlogit(m1_formula, data=m1_analysis).fit(
    method="newton", maxiter=200, disp=False
)

if not bool(m1_fit.mle_retvals.get("converged", False)):
    raise RuntimeError("M1 multinomial model did not converge.")

# statsmodels stores one coefficient column per non-reference outcome.
# With Outcome_Status coded 0/1/2, column 0 is Outpatient vs Control
# and column 1 is Acute vs Control.
outpatient_column = 0
acute_column = 1
parameters = m1_fit.params
covariance = m1_fit.cov_params()
n_terms = len(parameters.index)

term_labels = {
    "steps_z": "Steps (per 1 SD)",
    "age_z": "Age (per 1 SD)",
    "C(sex_at_birth)[T.Male]": "Male vs Female",
    "C(is_smoker)[T.1]": "Smoker vs nonsmoker",
    "C(is_drinker)[T.1]": "Moderate vs nondrinker",
    "C(is_drinker)[T.2]": "Heavy vs nondrinker",
}

def coefficient_position(term, outcome_column):
    """Position in statsmodels' equation-major covariance matrix."""
    return outcome_column * n_terms + parameters.index.get_loc(term)

def result_row(term, label, contrast, log_or, variance):
    se = np.sqrt(variance)
    z_value = log_or / se
    return {
        "model": "M1 primary",
        "term": term,
        "variable": label,
        "contrast": contrast,
        "log_OR": log_or,
        "SE": se,
        "OR": np.exp(log_or),
        "CI_lower": np.exp(log_or - 1.96 * se),
        "CI_upper": np.exp(log_or + 1.96 * se),
        "p_value": 2 * norm.sf(abs(z_value)),
        "n": len(m1_analysis),
    }

m1_rows = []
for term, label in term_labels.items():
    if term not in parameters.index:
        continue

    outpatient_position = coefficient_position(term, outpatient_column)
    acute_position = coefficient_position(term, acute_column)
    beta_outpatient = parameters.iloc[parameters.index.get_loc(term), outpatient_column]
    beta_acute = parameters.iloc[parameters.index.get_loc(term), acute_column]

    m1_rows.append(result_row(
        term, label, "Outpatient vs control", beta_outpatient,
        covariance.iloc[outpatient_position, outpatient_position],
    ))
    m1_rows.append(result_row(
        term, label, "Acute vs control", beta_acute,
        covariance.iloc[acute_position, acute_position],
    ))

    # Acute vs outpatient = beta_acute - beta_outpatient.
    contrast_variance = (
        covariance.iloc[acute_position, acute_position]
        + covariance.iloc[outpatient_position, outpatient_position]
        - 2 * covariance.iloc[acute_position, outpatient_position]
    )
    m1_rows.append(result_row(
        term, label, "Acute vs outpatient",
        beta_acute - beta_outpatient, contrast_variance,
    ))

m1_primary_results = pd.DataFrame(m1_rows)
m1_primary_results.to_csv("m1_primary_pairwise_results_python.csv", index=False)

print(
    f"M1 analytic N={len(m1_analysis):,}; "
    f"Control={(m1_analysis['Outcome_Status'] == 0).sum():,}, "
    f"Outpatient={(m1_analysis['Outcome_Status'] == 1).sum():,}, "
    f"Acute={(m1_analysis['Outcome_Status'] == 2).sum():,}"
)
m1_primary_results.round({
    "OR": 3, "CI_lower": 3, "CI_upper": 3, "p_value": 4
})
''',
    7: r'''from IPython.display import display
from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

if "m1_analysis" not in globals():
    raise RuntimeError("Run the M1 primary cell before this cell.")

# Effect-modification model:
# Outcome ~ Steps_z + Age_z + Sex + Steps_z*Sex + Smoking + Drinking
em_formula = (
    "Outcome_Status ~ steps_z * C(sex_at_birth) + age_z "
    "+ C(is_smoker) + C(is_drinker)"
)
em_fit = smf.mnlogit(em_formula, data=m1_analysis).fit(
    method="newton", maxiter=200, disp=False
)

if not bool(em_fit.mle_retvals.get("converged", False)):
    raise RuntimeError("Effect-modification model did not converge.")

em_parameters = em_fit.params
em_covariance = em_fit.cov_params().to_numpy()
em_beta = em_parameters.to_numpy().T.reshape(-1)  # equation-major order
em_n_terms = len(em_parameters.index)
em_outpatient_column = 0
em_acute_column = 1
step_term = "steps_z"
interaction_term = "steps_z:C(sex_at_birth)[T.Male]"

if interaction_term not in em_parameters.index:
    raise KeyError(f"Interaction coefficient not found: {interaction_term}")

def em_position(term, outcome_column):
    return outcome_column * em_n_terms + em_parameters.index.get_loc(term)

step_positions = [
    em_position(step_term, em_outpatient_column),
    em_position(step_term, em_acute_column),
]
interaction_positions = [
    em_position(interaction_term, em_outpatient_column),
    em_position(interaction_term, em_acute_column),
]

# Weights apply to [Outpatient vs control, Acute vs control].
outcome_contrasts = {
    "Outpatient vs control": np.array([1.0, 0.0]),
    "Acute vs control": np.array([0.0, 1.0]),
    "Acute vs outpatient": np.array([-1.0, 1.0]),
}

def linear_combination_summary(weight_vector):
    log_or = float(weight_vector @ em_beta)
    variance = float(weight_vector @ em_covariance @ weight_vector)
    se = np.sqrt(max(variance, 0.0))
    z_value = log_or / se
    return {
        "log_OR": log_or,
        "SE": se,
        "OR": np.exp(log_or),
        "CI_lower": np.exp(log_or - 1.96 * se),
        "CI_upper": np.exp(log_or + 1.96 * se),
        "p_value": 2 * norm.sf(abs(z_value)),
    }

interaction_rows = []
sex_specific_rows = []
for contrast_name, equation_weights in outcome_contrasts.items():
    # Core test: H0 beta_(Steps x Sex) = 0 for this outcome contrast.
    interaction_vector = np.zeros_like(em_beta)
    interaction_vector[interaction_positions] = equation_weights
    interaction_result = linear_combination_summary(interaction_vector)
    interaction_result.update({
        "contrast": contrast_name,
        "estimand": "Steps x Male interaction (ratio of ORs)",
        "null_hypothesis": "beta_StepsxSex = 0",
    })
    interaction_rows.append(interaction_result)

    # Female Steps slope is beta_Steps.
    female_vector = np.zeros_like(em_beta)
    female_vector[step_positions] = equation_weights
    female_result = linear_combination_summary(female_vector)
    female_result.update({
        "contrast": contrast_name,
        "sex": "Female",
        "estimand": "Steps OR per 1 SD",
    })
    sex_specific_rows.append(female_result)

    # Male Steps slope is beta_Steps + beta_(Steps x Male).
    male_vector = female_vector.copy()
    male_vector[interaction_positions] = equation_weights
    male_result = linear_combination_summary(male_vector)
    male_result.update({
        "contrast": contrast_name,
        "sex": "Male",
        "estimand": "Steps OR per 1 SD",
    })
    sex_specific_rows.append(male_result)

effect_modification_tests = pd.DataFrame(interaction_rows)[[
    "contrast", "estimand", "OR", "CI_lower", "CI_upper",
    "p_value", "log_OR", "SE", "null_hypothesis",
]]
sex_specific_steps = pd.DataFrame(sex_specific_rows)[[
    "contrast", "sex", "estimand", "OR", "CI_lower",
    "CI_upper", "p_value", "log_OR", "SE",
]]

# Global multinomial test:
# H0: beta_interaction,outpatient = beta_interaction,acute = 0.
joint_beta = em_beta[interaction_positions]
joint_covariance = em_covariance[np.ix_(interaction_positions, interaction_positions)]
joint_wald_statistic = float(
    joint_beta @ np.linalg.solve(joint_covariance, joint_beta)
)
joint_effect_modification_test = pd.DataFrame([{
    "test": "Global Steps x Sex interaction",
    "null_hypothesis": (
        "beta_interaction,outpatient = beta_interaction,acute = 0"
    ),
    "Wald_chi2": joint_wald_statistic,
    "df": 2,
    "p_value": chi2.sf(joint_wald_statistic, df=2),
    "n": len(m1_analysis),
}])

effect_modification_tests.to_csv(
    "effect_modification_interaction_tests.csv", index=False
)
sex_specific_steps.to_csv(
    "effect_modification_sex_specific_steps.csv", index=False
)
joint_effect_modification_test.to_csv(
    "effect_modification_joint_wald_test.csv", index=False
)

print("Core contrast-specific interaction tests (OR = ratio of Steps ORs):")
display(effect_modification_tests.round({
    "OR": 3, "CI_lower": 3, "CI_upper": 3, "p_value": 4
}))
print("Global 2-df interaction test:")
display(joint_effect_modification_test.round({
    "Wald_chi2": 3, "p_value": 4
}))
print("Sex-specific Steps associations:")
display(sex_specific_steps.round({
    "OR": 3, "CI_lower": 3, "CI_upper": 3, "p_value": 4
}))
''',
    9: r'''from IPython.display import display

if "m1_data" not in globals() or "m1_analysis" not in globals():
    raise RuntimeError("Run the M1 primary cell before the PCA cell.")

diagnosis_columns = [
    "has_sleep_disorder", "has_depression", "has_anxiety",
    "has_hypertension", "has_diabetes",
    "has_hyperlipidemia", "has_high_cholesterol", "has_ckd",
]
mediation_required_columns = [
    "person_id", "Outcome_Status", "avg_steps", "age",
    "sex_at_birth", "is_smoker", "is_drinker",
] + diagnosis_columns

# Start from M1-eligible participants and then require complete PCA inputs.
mediation_data = (
    m1_data.loc[m1_analysis.index, mediation_required_columns]
    .dropna()
    .copy()
)
mediation_data["Outcome_Status"] = (
    mediation_data["Outcome_Status"].astype(int)
)

# All component diagnoses must be binary indicators.
for column in diagnosis_columns:
    observed_values = set(mediation_data[column].unique())
    if not observed_values.issubset({0, 1}):
        raise ValueError(
            f"{column} must be binary 0/1; observed {observed_values}"
        )

mediation_data["Sleep"] = mediation_data["has_sleep_disorder"].astype(int)
mediation_data["Mental"] = mediation_data[[
    "has_depression", "has_anxiety"
]].sum(axis=1).astype(int)
mediation_data["Clinical"] = mediation_data[[
    "has_hypertension", "has_diabetes",
    "has_hyperlipidemia", "has_high_cholesterol", "has_ckd",
]].sum(axis=1).astype(int)

assert mediation_data["Mental"].between(0, 2).all()
assert mediation_data["Clinical"].between(0, 5).all()

# Reuse the M1 standardization for the continuous model covariates.
mediation_data["steps_z"] = (
    mediation_data["avg_steps"] - m1_steps_mean
) / m1_steps_sd
mediation_data["age_z"] = (
    mediation_data["age"] - m1_age_mean
) / m1_age_sd

pca_raw_features = ["Sleep", "Mental", "Clinical"]
pca_z_features = ["Z_Sleep", "Z_Mental", "Z_Clinical"]
pca_means = mediation_data[pca_raw_features].mean()
pca_sds = mediation_data[pca_raw_features].std(ddof=1)
if (pca_sds == 0).any():
    zero_sd = pca_sds.index[pca_sds == 0].tolist()
    raise ValueError(f"Cannot standardize SD=0 PCA feature(s): {zero_sd}")

for raw_feature, z_feature in zip(pca_raw_features, pca_z_features):
    mediation_data[z_feature] = (
        mediation_data[raw_feature] - pca_means[raw_feature]
    ) / pca_sds[raw_feature]

pca_matrix = mediation_data[pca_z_features].copy()
assert "Outcome_Status" not in pca_matrix.columns

# PCA by SVD, fitted exactly once in the pooled mediation-eligible cohort.
_, singular_values, right_singular_vectors = np.linalg.svd(
    pca_matrix.to_numpy(), full_matrices=False
)
pc1_loadings = right_singular_vectors[0].copy()
pc1_raw = pca_matrix.to_numpy() @ pc1_loadings
standardized_total_burden = pca_matrix.sum(axis=1).to_numpy()
orientation_correlation = np.corrcoef(
    pc1_raw, standardized_total_burden
)[0, 1]
if orientation_correlation < 0:
    pc1_loadings *= -1
    pc1_raw *= -1
    orientation_correlation *= -1

pc1_mean = pc1_raw.mean()
pc1_sd = pc1_raw.std(ddof=1)
if pc1_sd == 0:
    raise ValueError("PC1 has SD=0 and cannot be standardized.")
mediation_data["vulnerability"] = (pc1_raw - pc1_mean) / pc1_sd

explained_variance_ratio = (
    singular_values**2 / np.sum(singular_values**2)
)
pca_loadings = pd.DataFrame({
    "raw_feature": pca_raw_features,
    "standardized_feature": pca_z_features,
    "mean_before_standardization": pca_means.to_numpy(),
    "SD_before_standardization": pca_sds.to_numpy(),
    "PC1_loading": pc1_loadings,
})
pca_summary = pd.DataFrame([{
    "n_mediation_eligible": len(mediation_data),
    "PC1_explained_variance_ratio": explained_variance_ratio[0],
    "orientation_correlation_with_total_burden": orientation_correlation,
    "PC1_mean_before_final_standardization": pc1_mean,
    "PC1_SD_before_final_standardization": pc1_sd,
    "final_vulnerability_mean": mediation_data["vulnerability"].mean(),
    "final_vulnerability_SD": mediation_data["vulnerability"].std(ddof=1),
}])

pca_loadings.to_csv("pca_vulnerability_loadings.csv", index=False)
pca_summary.to_csv("pca_vulnerability_summary.csv", index=False)
mediation_data[[
    "person_id", "Outcome_Status", "steps_z", "age_z",
    "sex_at_birth", "is_smoker", "is_drinker",
    "Sleep", "Mental", "Clinical",
    "Z_Sleep", "Z_Mental", "Z_Clinical", "vulnerability",
]].to_csv("pca_mediation_eligible_dataset.csv", index=False)

print("PCA loadings (positive direction = greater vulnerability):")
display(pca_loadings.round(4))
print("PCA summary:")
display(pca_summary.round(4))
''',
    11: r'''if "mediation_data" not in globals():
    raise RuntimeError("Run the pooled PCA cell before the mediator model.")

mediator_formula = (
    "vulnerability ~ steps_z + age_z + C(sex_at_birth) "
    "+ C(is_smoker) + C(is_drinker)"
)
mediator_fit = smf.ols(mediator_formula, data=mediation_data).fit()
mediator_ci = mediator_fit.conf_int(alpha=0.05)
mediator_model_results = pd.DataFrame({
    "term": mediator_fit.params.index,
    "estimate": mediator_fit.params.to_numpy(),
    "SE": mediator_fit.bse.to_numpy(),
    "CI_lower": mediator_ci[0].to_numpy(),
    "CI_upper": mediator_ci[1].to_numpy(),
    "p_value": mediator_fit.pvalues.to_numpy(),
    "n": int(mediator_fit.nobs),
    "R_squared": mediator_fit.rsquared,
})
mediator_steps_path = mediator_model_results.loc[
    mediator_model_results["term"].eq("steps_z")
].copy()

mediator_model_results.to_csv("mediator_model_results.csv", index=False)
print("Core Steps -> Vulnerability path:")
display(mediator_steps_path.round({
    "estimate": 3, "SE": 3, "CI_lower": 3,
    "CI_upper": 3, "p_value": 4, "R_squared": 4,
}))
print("Full mediator model:")
display(mediator_model_results.round({
    "estimate": 3, "SE": 3, "CI_lower": 3,
    "CI_upper": 3, "p_value": 4, "R_squared": 4,
}))
''',
    13: r'''if "mediation_data" not in globals():
    raise RuntimeError("Run the pooled PCA cell before M2.")

m2_formula = (
    "Outcome_Status ~ steps_z + vulnerability + age_z "
    "+ C(sex_at_birth) + C(is_smoker) + C(is_drinker)"
)
m2_fit = smf.mnlogit(m2_formula, data=mediation_data).fit(
    method="newton", maxiter=200, disp=False
)
if not bool(m2_fit.mle_retvals.get("converged", False)):
    raise RuntimeError("M2 multinomial model did not converge.")

m2_term_labels = {
    "steps_z": "Steps (per 1 SD)",
    "vulnerability": "Vulnerability PC1 (per 1 SD)",
    "age_z": "Age (per 1 SD)",
    "C(sex_at_birth)[T.Male]": "Male vs Female",
    "C(is_smoker)[T.1]": "Smoker vs nonsmoker",
    "C(is_drinker)[T.1]": "Moderate vs nondrinker",
    "C(is_drinker)[T.2]": "Heavy vs nondrinker",
}

def extract_multinomial_pairwise(fit, model_name, term_labels):
    params = fit.params
    cov = fit.cov_params().to_numpy()
    beta = params.to_numpy().T.reshape(-1)
    number_of_terms = len(params.index)
    equation_contrasts = {
        "Outpatient vs control": np.array([1.0, 0.0]),
        "Acute vs control": np.array([0.0, 1.0]),
        "Acute vs outpatient": np.array([-1.0, 1.0]),
    }
    rows = []
    for term, variable_label in term_labels.items():
        if term not in params.index:
            continue
        term_index = params.index.get_loc(term)
        positions = [term_index, number_of_terms + term_index]
        for contrast_name, equation_weights in equation_contrasts.items():
            weight_vector = np.zeros_like(beta)
            weight_vector[positions] = equation_weights
            log_or = float(weight_vector @ beta)
            variance = float(weight_vector @ cov @ weight_vector)
            se = np.sqrt(max(variance, 0.0))
            rows.append({
                "model": model_name,
                "term": term,
                "variable": variable_label,
                "contrast": contrast_name,
                "log_OR": log_or,
                "SE": se,
                "OR": np.exp(log_or),
                "CI_lower": np.exp(log_or - 1.96 * se),
                "CI_upper": np.exp(log_or + 1.96 * se),
                "p_value": 2 * norm.sf(abs(log_or / se)),
                "n": len(mediation_data),
            })
    return pd.DataFrame(rows)

m2_pairwise_results = extract_multinomial_pairwise(
    m2_fit, "M2 mediation outcome", m2_term_labels
)
m2_pairwise_results.to_csv(
    "m2_mediation_outcome_pairwise_results.csv", index=False
)

# Descriptive M1-to-M2 change in the standardized Steps log-OR.
m1_steps_for_comparison = m1_primary_results.loc[
    m1_primary_results["term"].eq("steps_z"),
    ["contrast", "log_OR", "OR"],
].rename(columns={"log_OR": "M1_log_OR", "OR": "M1_OR"})
m2_steps_for_comparison = m2_pairwise_results.loc[
    m2_pairwise_results["term"].eq("steps_z"),
    ["contrast", "log_OR", "OR"],
].rename(columns={"log_OR": "M2_log_OR", "OR": "M2_OR"})
m1_m2_steps_attenuation = m1_steps_for_comparison.merge(
    m2_steps_for_comparison, on="contrast", validate="one_to_one"
)
m1_m2_steps_attenuation["percent_change_in_log_OR"] = 100 * (
    1 - m1_m2_steps_attenuation["M2_log_OR"]
    / m1_m2_steps_attenuation["M1_log_OR"]
)
m1_m2_steps_attenuation.to_csv(
    "m1_m2_steps_attenuation.csv", index=False
)

print("M2 Steps and Vulnerability results:")
display(m2_pairwise_results.loc[
    m2_pairwise_results["term"].isin(["steps_z", "vulnerability"])
].round({
    "OR": 3, "CI_lower": 3, "CI_upper": 3, "p_value": 4
}))
print("Descriptive change in the Steps coefficient from M1 to M2:")
display(m1_m2_steps_attenuation.round({
    "M1_OR": 3, "M2_OR": 3, "percent_change_in_log_OR": 1
}))
''',
    18: r'''from scipy.stats import chi2, norm
from statsmodels.stats.multitest import multipletests

if "mediation_data" not in globals():
    raise RuntimeError("Run the pooled PCA cell before pathway models.")

def _wald_block(beta, covariance, positions):
    positions = list(positions)
    block_beta = np.asarray(beta)[positions]
    block_covariance = np.asarray(covariance)[np.ix_(positions, positions)]
    inverse_covariance = np.linalg.pinv(block_covariance)
    statistic = float(block_beta @ inverse_covariance @ block_beta)
    degrees_of_freedom = int(np.linalg.matrix_rank(block_covariance))
    return statistic, degrees_of_freedom, chi2.sf(
        statistic, degrees_of_freedom
    )

def _ols_block_test(fit, terms, analysis, pathway):
    positions = [fit.params.index.get_loc(term) for term in terms]
    statistic, df_test, p_value = _wald_block(
        fit.params.to_numpy(), fit.cov_params().to_numpy(), positions
    )
    return {
        "analysis": analysis, "model": "Mediator OLS",
        "pathway": pathway, "terms": " | ".join(terms),
        "Wald_chi2": statistic, "df": df_test,
        "p_value": p_value, "n": int(fit.nobs),
    }

def _mnlogit_term_positions(fit, terms):
    number_of_terms = len(fit.params.index)
    term_indices = [fit.params.index.get_loc(term) for term in terms]
    return [
        equation * number_of_terms + term_index
        for equation in range(fit.params.shape[1])
        for term_index in term_indices
    ]

def _check_mnlogit_fit(fit, label):
    if not bool(fit.mle_retvals.get("converged", False)):
        raise RuntimeError(f"{label} did not converge.")
    expected_covariance_index = [
        (str(int(column) + 1), term)
        for column in fit.params.columns
        for term in fit.params.index
    ]
    if list(fit.cov_params().index) != expected_covariance_index:
        raise RuntimeError(
            f"Unexpected equation/term covariance ordering in {label}."
        )

def _mnlogit_block_test(fit, terms, analysis, pathway):
    beta = fit.params.to_numpy().T.reshape(-1)
    positions = _mnlogit_term_positions(fit, terms)
    statistic, df_test, p_value = _wald_block(
        beta, fit.cov_params().to_numpy(), positions
    )
    return {
        "analysis": analysis, "model": "Outcome MNLogit",
        "pathway": pathway, "terms": " | ".join(terms),
        "Wald_chi2": statistic, "df": df_test,
        "p_value": p_value, "n": int(fit.nobs),
    }

def _mnlogit_pairwise_interaction(fit, term, analysis, pathway):
    params = fit.params
    covariance = fit.cov_params().to_numpy()
    beta = params.to_numpy().T.reshape(-1)
    number_of_terms = len(params.index)
    term_index = params.index.get_loc(term)
    positions = [term_index, number_of_terms + term_index]
    contrasts = {
        "Outpatient vs control": np.array([1.0, 0.0]),
        "Acute vs control": np.array([0.0, 1.0]),
        "Acute vs outpatient": np.array([-1.0, 1.0]),
    }
    rows = []
    for contrast_name, equation_weights in contrasts.items():
        weight_vector = np.zeros_like(beta)
        weight_vector[positions] = equation_weights
        estimate = float(weight_vector @ beta)
        variance = float(weight_vector @ covariance @ weight_vector)
        standard_error = np.sqrt(max(variance, 0.0))
        rows.append({
            "analysis": analysis, "pathway": pathway,
            "term": term, "contrast": contrast_name,
            "log_ratio_of_ORs": estimate, "SE": standard_error,
            "ratio_of_ORs": np.exp(estimate),
            "CI_lower": np.exp(estimate - 1.96 * standard_error),
            "CI_upper": np.exp(estimate + 1.96 * standard_error),
            "p_value": 2 * norm.sf(abs(estimate / standard_error)),
            "n": int(fit.nobs),
        })
    return rows

pathway_omnibus_rows = []
pathway_pairwise_rows = []

# ------------------------- Sex moderation -------------------------
sex_mediator_formula = (
    "vulnerability ~ steps_z * C(sex_at_birth) + age_z "
    "+ C(is_smoker) + C(is_drinker)"
)
sex_mediator_fit = smf.ols(sex_mediator_formula, mediation_data).fit()
sex_a_term = "steps_z:C(sex_at_birth)[T.Male]"
pathway_omnibus_rows.append(_ols_block_test(
    sex_mediator_fit, [sex_a_term], "Sex moderation",
    "Steps -> Vulnerability (Steps x Sex)",
))

sex_outcome_formula = (
    "Outcome_Status ~ steps_z * C(sex_at_birth) "
    "+ vulnerability * C(sex_at_birth) + age_z "
    "+ C(is_smoker) + C(is_drinker)"
)
sex_outcome_fit = smf.mnlogit(
    sex_outcome_formula, mediation_data
).fit(method="newton", maxiter=200, disp=False)
_check_mnlogit_fit(sex_outcome_fit, "Sex-moderated outcome model")
sex_steps_term = "steps_z:C(sex_at_birth)[T.Male]"
sex_vulnerability_term = "vulnerability:C(sex_at_birth)[T.Male]"
pathway_omnibus_rows.extend([
    _mnlogit_block_test(
        sex_outcome_fit, [sex_steps_term], "Sex moderation",
        "Residual Steps -> Outcome (Steps x Sex)",
    ),
    _mnlogit_block_test(
        sex_outcome_fit, [sex_vulnerability_term], "Sex moderation",
        "Vulnerability -> Outcome (Vulnerability x Sex)",
    ),
    _mnlogit_block_test(
        sex_outcome_fit, [sex_steps_term, sex_vulnerability_term],
        "Sex moderation", "Joint outcome-path sex interactions",
    ),
])
pathway_pairwise_rows.extend(_mnlogit_pairwise_interaction(
    sex_outcome_fit, sex_steps_term, "Sex moderation",
    "Residual Steps -> Outcome",
))
pathway_pairwise_rows.extend(_mnlogit_pairwise_interaction(
    sex_outcome_fit, sex_vulnerability_term, "Sex moderation",
    "Vulnerability -> Outcome",
))

# --------------------- Linear age moderation ---------------------
age_mediator_formula = (
    "vulnerability ~ steps_z * age_z + C(sex_at_birth) "
    "+ C(is_smoker) + C(is_drinker)"
)
age_mediator_fit = smf.ols(age_mediator_formula, mediation_data).fit()
age_a_term = "steps_z:age_z"
pathway_omnibus_rows.append(_ols_block_test(
    age_mediator_fit, [age_a_term], "Linear age moderation",
    "Steps -> Vulnerability (Steps x Age)",
))

age_outcome_formula = (
    "Outcome_Status ~ steps_z * age_z + vulnerability * age_z "
    "+ C(sex_at_birth) + C(is_smoker) + C(is_drinker)"
)
age_outcome_fit = smf.mnlogit(
    age_outcome_formula, mediation_data
).fit(method="newton", maxiter=200, disp=False)
_check_mnlogit_fit(age_outcome_fit, "Age-moderated outcome model")
age_steps_term = "steps_z:age_z"
age_vulnerability_term = "vulnerability:age_z"
pathway_omnibus_rows.extend([
    _mnlogit_block_test(
        age_outcome_fit, [age_steps_term], "Linear age moderation",
        "Residual Steps -> Outcome (Steps x Age)",
    ),
    _mnlogit_block_test(
        age_outcome_fit, [age_vulnerability_term],
        "Linear age moderation",
        "Vulnerability -> Outcome (Vulnerability x Age)",
    ),
    _mnlogit_block_test(
        age_outcome_fit, [age_steps_term, age_vulnerability_term],
        "Linear age moderation", "Joint outcome-path age interactions",
    ),
])
pathway_pairwise_rows.extend(_mnlogit_pairwise_interaction(
    age_outcome_fit, age_steps_term, "Linear age moderation",
    "Residual Steps -> Outcome",
))
pathway_pairwise_rows.extend(_mnlogit_pairwise_interaction(
    age_outcome_fit, age_vulnerability_term, "Linear age moderation",
    "Vulnerability -> Outcome",
))

# ----------------- Nonlinear age sensitivity --------------------
age_spline = 'cr(age_z, df=3, constraints="center")'
spline_mediator_formula = (
    f"vulnerability ~ steps_z * {age_spline} + C(sex_at_birth) "
    "+ C(is_smoker) + C(is_drinker)"
)
spline_mediator_fit = smf.ols(
    spline_mediator_formula, mediation_data
).fit()
spline_a_terms = [
    term for term in spline_mediator_fit.params.index
    if term.startswith("steps_z:cr(")
]
pathway_omnibus_rows.append(_ols_block_test(
    spline_mediator_fit, spline_a_terms, "Flexible age sensitivity",
    "Steps -> Vulnerability (Steps x spline[Age])",
))

spline_outcome_formula = (
    f"Outcome_Status ~ steps_z * {age_spline} "
    f"+ vulnerability * {age_spline} + C(sex_at_birth) "
    "+ C(is_smoker) + C(is_drinker)"
)
spline_outcome_fit = smf.mnlogit(
    spline_outcome_formula, mediation_data
).fit(method="newton", maxiter=300, disp=False)
_check_mnlogit_fit(spline_outcome_fit, "Flexible-age outcome model")
spline_steps_terms = [
    term for term in spline_outcome_fit.params.index
    if term.startswith("steps_z:cr(")
]
spline_vulnerability_terms = [
    term for term in spline_outcome_fit.params.index
    if term.startswith("vulnerability:cr(")
]
pathway_omnibus_rows.extend([
    _mnlogit_block_test(
        spline_outcome_fit, spline_steps_terms,
        "Flexible age sensitivity",
        "Residual Steps -> Outcome (Steps x spline[Age])",
    ),
    _mnlogit_block_test(
        spline_outcome_fit, spline_vulnerability_terms,
        "Flexible age sensitivity",
        "Vulnerability -> Outcome (Vulnerability x spline[Age])",
    ),
])

# -------- Exploratory joint Sex x linear-Age moderation ---------
joint_mediator_formula = (
    "vulnerability ~ steps_z * C(sex_at_birth) * age_z "
    "+ C(is_smoker) + C(is_drinker)"
)
joint_mediator_fit = smf.ols(
    joint_mediator_formula, mediation_data
).fit()
joint_a_term = "steps_z:C(sex_at_birth)[T.Male]:age_z"
pathway_omnibus_rows.append(_ols_block_test(
    joint_mediator_fit, [joint_a_term],
    "Exploratory Sex x Age",
    "Steps -> Vulnerability (Steps x Sex x Age)",
))

joint_outcome_formula = (
    "Outcome_Status ~ steps_z * C(sex_at_birth) * age_z "
    "+ vulnerability * C(sex_at_birth) * age_z "
    "+ C(is_smoker) + C(is_drinker)"
)
joint_outcome_fit = smf.mnlogit(
    joint_outcome_formula, mediation_data
).fit(method="newton", maxiter=300, disp=False)
_check_mnlogit_fit(joint_outcome_fit, "Joint Sex-by-Age outcome model")
joint_steps_term = "steps_z:C(sex_at_birth)[T.Male]:age_z"
joint_vulnerability_term = (
    "vulnerability:C(sex_at_birth)[T.Male]:age_z"
)
pathway_omnibus_rows.extend([
    _mnlogit_block_test(
        joint_outcome_fit, [joint_steps_term],
        "Exploratory Sex x Age",
        "Residual Steps -> Outcome (Steps x Sex x Age)",
    ),
    _mnlogit_block_test(
        joint_outcome_fit, [joint_vulnerability_term],
        "Exploratory Sex x Age",
        "Vulnerability -> Outcome (Vulnerability x Sex x Age)",
    ),
    _mnlogit_block_test(
        joint_outcome_fit, [joint_steps_term, joint_vulnerability_term],
        "Exploratory Sex x Age",
        "Joint three-way outcome interactions",
    ),
])
pathway_pairwise_rows.extend(_mnlogit_pairwise_interaction(
    joint_outcome_fit, joint_steps_term, "Exploratory Sex x Age",
    "Residual Steps -> Outcome three-way",
))
pathway_pairwise_rows.extend(_mnlogit_pairwise_interaction(
    joint_outcome_fit, joint_vulnerability_term,
    "Exploratory Sex x Age",
    "Vulnerability -> Outcome three-way",
))

moderated_path_omnibus_tests = pd.DataFrame(pathway_omnibus_rows)
confirmatory_pathways = {
    "Steps -> Vulnerability (Steps x Sex)",
    "Residual Steps -> Outcome (Steps x Sex)",
    "Vulnerability -> Outcome (Vulnerability x Sex)",
    "Steps -> Vulnerability (Steps x Age)",
    "Residual Steps -> Outcome (Steps x Age)",
    "Vulnerability -> Outcome (Vulnerability x Age)",
}
confirmatory_mask = moderated_path_omnibus_tests[
    "pathway"
].isin(confirmatory_pathways)
moderated_path_omnibus_tests["Holm_family"] = ""
moderated_path_omnibus_tests["Holm_p_value"] = np.nan
moderated_path_omnibus_tests.loc[
    confirmatory_mask, "Holm_family"
] = "Secondary sex/linear-age pathway interaction family"
moderated_path_omnibus_tests.loc[
    confirmatory_mask, "Holm_p_value"
] = multipletests(
    moderated_path_omnibus_tests.loc[confirmatory_mask, "p_value"],
    method="holm",
)[1]
moderated_path_pairwise_tests = pd.DataFrame(pathway_pairwise_rows)
moderated_path_omnibus_tests.to_csv(
    "moderated_path_omnibus_tests.csv", index=False
)
moderated_path_pairwise_tests.to_csv(
    "moderated_path_pairwise_tests.csv", index=False
)
print("Pathway-level omnibus tests:")
display(moderated_path_omnibus_tests.round({
    "Wald_chi2": 3, "p_value": 4
}))
print("Pairwise multinomial interaction contrasts:")
display(moderated_path_pairwise_tests.round({
    "ratio_of_ORs": 3, "CI_lower": 3,
    "CI_upper": 3, "p_value": 4
}))
''',
    20: r'''"""Bootstrap g-computation for moderated mediation in the Fitbit analysis.

The public entry point is ``run_moderated_mediation_bootstrap``.  The function
is deliberately written around NumPy design matrices so the complete pooled
PCA + three-model pipeline can be refitted many times without changing the
model specifications used in the notebook.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from statsmodels.stats.multitest import multipletests


CONTRASTS = (
    "Outpatient vs control",
    "Acute vs control",
    "Acute vs outpatient",
)
EFFECTS = ("ACME", "ADE", "Total")


@dataclass(frozen=True)
class _State:
    label: str
    sex_male: float | None
    age_z: float | None


def _z_standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    sd = values.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError("Cannot standardize a variable with invalid/zero SD.")
    return (values - values.mean()) / sd


def _pca_vulnerability(
    raw_burdens: np.ndarray,
    reference_loadings: np.ndarray,
) -> np.ndarray:
    """Refit the pooled PCA without using outcome and align the PC1 sign."""
    pca_z = np.column_stack([
        _z_standardize(raw_burdens[:, column])
        for column in range(raw_burdens.shape[1])
    ])
    _, _, right_vectors = np.linalg.svd(pca_z, full_matrices=False)
    loading = right_vectors[0].copy()
    if float(loading @ reference_loadings) < 0:
        loading *= -1
    return _z_standardize(pca_z @ loading)


def _as_array(value, n: int) -> np.ndarray:
    if np.ndim(value) == 0:
        return np.full(n, float(value))
    value = np.asarray(value, dtype=float)
    if len(value) != n:
        raise ValueError("Pseudo-population vector has the wrong length.")
    return value


def _mediator_design(
    variant: str,
    exposure,
    sex_male,
    age_z,
    smoker: np.ndarray,
    drink_moderate: np.ndarray,
    drink_heavy: np.ndarray,
) -> np.ndarray:
    n = len(smoker)
    a = _as_array(exposure, n)
    s = _as_array(sex_male, n)
    h = _as_array(age_z, n)
    one = np.ones(n)
    if variant == "sex":
        columns = [one, a, s, a * s, h]
    elif variant == "age":
        columns = [one, a, h, a * h, s]
    elif variant == "joint":
        columns = [
            one, a, s, h, a * s, a * h, s * h, a * s * h,
        ]
    else:
        raise ValueError(f"Unknown model variant: {variant}")
    columns.extend([smoker, drink_moderate, drink_heavy])
    return np.column_stack(columns)


def _outcome_design(
    variant: str,
    exposure,
    mediator,
    sex_male,
    age_z,
    smoker: np.ndarray,
    drink_moderate: np.ndarray,
    drink_heavy: np.ndarray,
) -> np.ndarray:
    n = len(smoker)
    a = _as_array(exposure, n)
    m = _as_array(mediator, n)
    s = _as_array(sex_male, n)
    h = _as_array(age_z, n)
    one = np.ones(n)
    if variant == "sex":
        columns = [one, a, m, s, a * s, m * s, h]
    elif variant == "age":
        columns = [one, a, m, h, a * h, m * h, s]
    elif variant == "joint":
        columns = [
            one, a, m, s, h,
            a * s, a * h, s * h, a * s * h,
            m * s, m * h, m * s * h,
        ]
    else:
        raise ValueError(f"Unknown model variant: {variant}")
    columns.extend([smoker, drink_moderate, drink_heavy])
    return np.column_stack(columns)


def _fit_mnlogit(y: np.ndarray, design: np.ndarray) -> np.ndarray:
    """Fit MNLogit, with a deterministic fallback optimizer."""
    fit = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, maxiter in (("newton", 100), ("bfgs", 300)):
            candidate = sm.MNLogit(y, design).fit(
                method=method, maxiter=maxiter, disp=False
            )
            if bool(candidate.mle_retvals.get("converged", False)):
                fit = candidate
                break
    if fit is None:
        raise RuntimeError("A moderated MNLogit model did not converge.")
    parameters = np.asarray(fit.params, dtype=float)
    if parameters.shape != (design.shape[1], 2):
        raise RuntimeError("Unexpected MNLogit parameter layout.")
    return parameters


def _pairwise_coefficients(parameters: np.ndarray) -> np.ndarray:
    """Columns correspond to the three pairwise log-odds contrasts."""
    return np.column_stack([
        parameters[:, 0],
        parameters[:, 1],
        parameters[:, 1] - parameters[:, 0],
    ])


def _residual_nodes(residuals: np.ndarray, node_count: int) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    residuals = residuals - residuals.mean()
    probabilities = (np.arange(node_count) + 0.5) / node_count
    return np.quantile(residuals, probabilities)


def _integrated_pair_means(
    linear_predictors: np.ndarray,
    mediator_slopes: np.ndarray,
    residual_nodes: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Integrate pairwise probabilities over empirical mediator residuals.

    The one-dimensional logistic convolution is evaluated on a fine fixed
    grid and linearly interpolated.  This retains the full pseudo-population
    average while avoiding a large person x residual-node array.
    """
    answers = np.empty(linear_predictors.shape[1])
    for contrast_index in range(linear_predictors.shape[1]):
        convolution = expit(
            grid[:, None]
            + mediator_slopes[contrast_index] * residual_nodes[None, :]
        ).mean(axis=1)
        answers[contrast_index] = np.mean(np.interp(
            linear_predictors[:, contrast_index],
            grid,
            convolution,
            left=convolution[0],
            right=convolution[-1],
        ))
    return answers


def _fit_variant(
    data: dict[str, np.ndarray],
    vulnerability: np.ndarray,
    variant: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mediator_x = _mediator_design(
        variant,
        data["steps_z"],
        data["sex_male"],
        data["age_z"],
        data["smoker"],
        data["drink_moderate"],
        data["drink_heavy"],
    )
    mediator_parameters, _, mediator_rank, _ = np.linalg.lstsq(
        mediator_x, vulnerability, rcond=None
    )
    if mediator_rank != mediator_x.shape[1]:
        raise RuntimeError("Rank-deficient moderated mediator model.")
    residuals = vulnerability - mediator_x @ mediator_parameters

    outcome_x = _outcome_design(
        variant,
        data["steps_z"],
        vulnerability,
        data["sex_male"],
        data["age_z"],
        data["smoker"],
        data["drink_moderate"],
        data["drink_heavy"],
    )
    outcome_parameters = _fit_mnlogit(data["outcome"], outcome_x)
    return mediator_parameters, outcome_parameters, residuals


def _conditional_effects(
    data: dict[str, np.ndarray],
    variant: str,
    state: _State,
    mediator_parameters: np.ndarray,
    outcome_parameters: np.ndarray,
    residuals: np.ndarray,
    residual_node_count: int,
    integration_grid: np.ndarray,
) -> np.ndarray:
    """Return contrast x (ACME, ADE, Total) for one moderator state."""
    s = data["sex_male"] if state.sex_male is None else state.sex_male
    h = data["age_z"] if state.age_z is None else state.age_z
    nuisance = (
        data["smoker"], data["drink_moderate"], data["drink_heavy"]
    )

    mediator_means = []
    for exposure in (0.0, 1.0):
        x = _mediator_design(variant, exposure, s, h, *nuisance)
        mediator_means.append(x @ mediator_parameters)

    pair_parameters = _pairwise_coefficients(outcome_parameters)
    residual_nodes = _residual_nodes(residuals, residual_node_count)
    theta = {}
    for outcome_exposure in (0, 1):
        for mediator_exposure in (0, 1):
            mediator_values = mediator_means[mediator_exposure]
            x = _outcome_design(
                variant,
                float(outcome_exposure),
                mediator_values,
                s,
                h,
                *nuisance,
            )
            x_plus_one = _outcome_design(
                variant,
                float(outcome_exposure),
                mediator_values + 1.0,
                s,
                h,
                *nuisance,
            )
            eta = x @ pair_parameters
            slopes_by_person = (x_plus_one - x) @ pair_parameters
            slopes = slopes_by_person[0]
            if not np.allclose(slopes_by_person, slopes, atol=1e-10):
                raise RuntimeError(
                    "Mediator slope varies within a fixed moderator cell."
                )
            theta[(outcome_exposure, mediator_exposure)] = (
                _integrated_pair_means(
                    eta, slopes, residual_nodes, integration_grid
                )
            )

    q00, q01 = theta[(0, 0)], theta[(0, 1)]
    q10, q11 = theta[(1, 0)], theta[(1, 1)]
    acme = ((q01 - q00) + (q11 - q10)) / 2
    ade = ((q10 - q00) + (q11 - q01)) / 2
    total = q11 - q00
    result = np.column_stack([acme, ade, total])
    if not np.allclose(result[:, 0] + result[:, 1], result[:, 2], atol=1e-10):
        raise AssertionError("ACME + ADE must equal Total.")
    return result


def _prepare_arrays(
    mediation_data: pd.DataFrame,
    positions: np.ndarray,
    steps_mean: float,
    steps_sd: float,
    age_mean: float,
    age_sd: float,
) -> dict[str, np.ndarray]:
    frame = mediation_data.iloc[positions]
    return {
        "outcome": frame["Outcome_Status"].to_numpy(int),
        "steps_z": (
            frame["avg_steps"].to_numpy(float) - steps_mean
        ) / steps_sd,
        "age_z": (
            frame["age"].to_numpy(float) - age_mean
        ) / age_sd,
        "sex_male": frame["sex_at_birth"].eq("Male").to_numpy(float),
        "smoker": frame["is_smoker"].eq(1).to_numpy(float),
        "drink_moderate": frame["is_drinker"].eq(1).to_numpy(float),
        "drink_heavy": frame["is_drinker"].eq(2).to_numpy(float),
        "raw_burdens": frame[["Sleep", "Mental", "Clinical"]].to_numpy(float),
    }


def _key(variant: str, state: str, contrast: str, effect: str) -> str:
    return "__".join([variant, state, contrast, effect])


def _evaluate_draw(
    mediation_data: pd.DataFrame,
    positions: np.ndarray,
    state_specification: dict[str, tuple[_State, ...]],
    reference_loadings: np.ndarray,
    steps_mean: float,
    steps_sd: float,
    age_mean: float,
    age_sd: float,
    residual_node_count: int,
    integration_grid: np.ndarray,
) -> dict[str, float]:
    data = _prepare_arrays(
        mediation_data, positions, steps_mean, steps_sd, age_mean, age_sd
    )
    vulnerability = _pca_vulnerability(
        data["raw_burdens"], reference_loadings
    )
    values = {}
    for variant, states in state_specification.items():
        mediator_parameters, outcome_parameters, residuals = _fit_variant(
            data, vulnerability, variant
        )
        for state in states:
            effect_matrix = _conditional_effects(
                data,
                variant,
                state,
                mediator_parameters,
                outcome_parameters,
                residuals,
                residual_node_count,
                integration_grid,
            )
            for contrast_index, contrast in enumerate(CONTRASTS):
                for effect_index, effect in enumerate(EFFECTS):
                    values[_key(variant, state.label, contrast, effect)] = (
                        effect_matrix[contrast_index, effect_index]
                    )
    return values


def _tail_p_value(draws: np.ndarray) -> float:
    draws = draws[np.isfinite(draws)]
    lower = (np.sum(draws <= 0) + 1) / (len(draws) + 1)
    upper = (np.sum(draws >= 0) + 1) / (len(draws) + 1)
    return min(1.0, 2 * min(lower, upper))


def _summarize_columns(
    point_values: dict[str, float],
    draws: pd.DataFrame,
    descriptors: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for descriptor in descriptors.to_dict("records"):
        column = descriptor["column"]
        valid = draws[column].dropna().to_numpy(float)
        if len(valid) == 0:
            raise RuntimeError(f"No successful bootstrap draws for {column}.")
        lower, upper = np.quantile(valid, [0.025, 0.975])
        row = dict(descriptor)
        row.update({
            "estimate": point_values[column],
            "CI_lower": lower,
            "CI_upper": upper,
            "p_value": _tail_p_value(valid),
            "estimate_percentage_points": 100 * point_values[column],
            "CI_lower_percentage_points": 100 * lower,
            "CI_upper_percentage_points": 100 * upper,
            "successful_bootstrap": len(valid),
        })
        rows.append(row)
    return pd.DataFrame(rows)


def _conditional_descriptors(columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        variant, state, contrast, effect = column.split("__", 3)
        rows.append({
            "column": column,
            "analysis": variant,
            "moderator_state": state,
            "contrast": contrast,
            "effect": effect,
        })
    return pd.DataFrame(rows)


def _difference_definitions() -> list[dict]:
    definitions = []
    for contrast in CONTRASTS:
        for effect in EFFECTS:
            definitions.extend([
                {
                    "column": _key("difference", "Female - Male", contrast, effect),
                    "analysis": "Sex moderation",
                    "comparison": "Female - Male",
                    "contrast": contrast,
                    "effect": effect,
                    "weights": {
                        _key("sex", "Female", contrast, effect): 1.0,
                        _key("sex", "Male", contrast, effect): -1.0,
                    },
                },
                {
                    "column": _key("difference", "Age 64 - Age 50", contrast, effect),
                    "analysis": "Linear age moderation",
                    "comparison": "Age 64 - Age 50",
                    "contrast": contrast,
                    "effect": effect,
                    "weights": {
                        _key("age", "Age 64", contrast, effect): 1.0,
                        _key("age", "Age 50", contrast, effect): -1.0,
                    },
                },
                {
                    "column": _key("difference", "Sex difference at age 50", contrast, effect),
                    "analysis": "Exploratory joint Sex x Age",
                    "comparison": "Female - Male at age 50",
                    "contrast": contrast,
                    "effect": effect,
                    "weights": {
                        _key("joint", "Female, age 50", contrast, effect): 1.0,
                        _key("joint", "Male, age 50", contrast, effect): -1.0,
                    },
                },
                {
                    "column": _key("difference", "Sex difference at age 64", contrast, effect),
                    "analysis": "Exploratory joint Sex x Age",
                    "comparison": "Female - Male at age 64",
                    "contrast": contrast,
                    "effect": effect,
                    "weights": {
                        _key("joint", "Female, age 64", contrast, effect): 1.0,
                        _key("joint", "Male, age 64", contrast, effect): -1.0,
                    },
                },
                {
                    "column": _key("difference", "Sex x age DID", contrast, effect),
                    "analysis": "Exploratory joint Sex x Age",
                    "comparison": "[Female - Male] age 64 - age 50",
                    "contrast": contrast,
                    "effect": effect,
                    "weights": {
                        _key("joint", "Female, age 64", contrast, effect): 1.0,
                        _key("joint", "Male, age 64", contrast, effect): -1.0,
                        _key("joint", "Female, age 50", contrast, effect): -1.0,
                        _key("joint", "Male, age 50", contrast, effect): 1.0,
                    },
                },
            ])
    return definitions


def _construct_differences(
    point_values: dict[str, float],
    conditional_draws: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    point_differences = {}
    difference_draws = {}
    descriptor_rows = []
    for definition in _difference_definitions():
        column = definition["column"]
        point_differences[column] = sum(
            weight * point_values[source]
            for source, weight in definition["weights"].items()
        )
        combined = np.zeros(len(conditional_draws))
        for source, weight in definition["weights"].items():
            combined += weight * conditional_draws[source].to_numpy(float)
        difference_draws[column] = combined
        descriptor_rows.append({
            key: value for key, value in definition.items()
            if key != "weights"
        })
    return (
        point_differences,
        pd.DataFrame(difference_draws),
        pd.DataFrame(descriptor_rows),
    )


def _add_multiplicity_adjustments(summary: pd.DataFrame) -> pd.DataFrame:
    summary = summary.copy()
    summary["Holm_family"] = ""
    summary["Holm_p_value"] = np.nan

    primary = (
        summary["analysis"].isin(["Sex moderation", "Linear age moderation"])
        & summary["effect"].eq("ACME")
    )
    if primary.any():
        summary.loc[primary, "Holm_family"] = (
            "Secondary moderated mediation: sex/age Delta-ACME across 3 outcome contrasts"
        )
        summary.loc[primary, "Holm_p_value"] = multipletests(
            summary.loc[primary, "p_value"], method="holm"
        )[1]

    exploratory = (
        summary["analysis"].eq("Exploratory joint Sex x Age")
        & summary["comparison"].eq("[Female - Male] age 64 - age 50")
        & summary["effect"].eq("ACME")
    )
    if exploratory.any():
        summary.loc[exploratory, "Holm_family"] = (
            "Exploratory: Sex x Age ACME DID across 3 outcome contrasts"
        )
        summary.loc[exploratory, "Holm_p_value"] = multipletests(
            summary.loc[exploratory, "p_value"], method="holm"
        )[1]
    return summary


def run_moderated_mediation_bootstrap(
    mediation_data: pd.DataFrame,
    reference_loadings: np.ndarray,
    steps_mean: float,
    steps_sd: float,
    age_mean: float,
    age_sd: float,
    *,
    bootstrap_draws: int = 2000,
    seed: int = 20260903,
    residual_node_count: int = 51,
    output_directory: str | Path = ".",
    progress_every: int = 100,
) -> dict[str, pd.DataFrame]:
    """Run pooled sex/age moderated-mediation g-computation and bootstrap."""
    if bootstrap_draws < 1:
        raise ValueError("bootstrap_draws must be positive.")
    if residual_node_count < 5:
        raise ValueError("Use at least five empirical residual nodes.")
    if set(mediation_data["Outcome_Status"].unique()) != {0, 1, 2}:
        raise ValueError("Outcome must be coded Control=0, Outpatient=1, Acute=2.")

    ages = (35, 50, 57, 64, 81)
    to_age_z = lambda age: (age - age_mean) / age_sd
    state_specification = {
        "sex": (
            _State("Female", 0.0, None),
            _State("Male", 1.0, None),
        ),
        "age": tuple(
            _State(f"Age {age}", None, to_age_z(age)) for age in ages
        ),
        "joint": tuple(
            _State(f"{sex}, age {age}", float(sex == "Male"), to_age_z(age))
            for age in (50, 57, 64)
            for sex in ("Female", "Male")
        ),
    }
    # Step 0.05 on the logit scale keeps interpolation error negligible.
    integration_grid = np.linspace(-30.0, 30.0, 1201)
    original_positions = np.arange(len(mediation_data))
    point_values = _evaluate_draw(
        mediation_data,
        original_positions,
        state_specification,
        np.asarray(reference_loadings, dtype=float),
        steps_mean,
        steps_sd,
        age_mean,
        age_sd,
        residual_node_count,
        integration_grid,
    )

    # Numerical sensitivity: K versus 101 empirical residual quantiles.
    point_values_k101 = _evaluate_draw(
        mediation_data,
        original_positions,
        state_specification,
        np.asarray(reference_loadings, dtype=float),
        steps_mean,
        steps_sd,
        age_mean,
        age_sd,
        101,
        integration_grid,
    )
    maximum_k_sensitivity = max(
        abs(point_values[key] - point_values_k101[key])
        for key in point_values
    )

    columns = list(point_values)
    bootstrap_matrix = np.full((bootstrap_draws, len(columns)), np.nan)
    rng = np.random.default_rng(seed)
    failed_draws = 0
    for draw_index in range(bootstrap_draws):
        positions = rng.integers(
            0, len(mediation_data), size=len(mediation_data)
        )
        try:
            draw_values = _evaluate_draw(
                mediation_data,
                positions,
                state_specification,
                np.asarray(reference_loadings, dtype=float),
                steps_mean,
                steps_sd,
                age_mean,
                age_sd,
                residual_node_count,
                integration_grid,
            )
            bootstrap_matrix[draw_index] = [
                draw_values[column] for column in columns
            ]
        except (
            ValueError,
            RuntimeError,
            np.linalg.LinAlgError,
            FloatingPointError,
        ):
            failed_draws += 1
        if progress_every and (
            (draw_index + 1) % progress_every == 0
            or draw_index + 1 == bootstrap_draws
        ):
            print(
                f"Moderated-mediation bootstrap: {draw_index + 1}/"
                f"{bootstrap_draws}; failed={failed_draws}"
            )

    conditional_draws = pd.DataFrame(bootstrap_matrix, columns=columns)
    conditional_descriptors = _conditional_descriptors(columns)
    conditional_summary = _summarize_columns(
        point_values, conditional_draws, conditional_descriptors
    )

    point_differences, difference_draws, difference_descriptors = (
        _construct_differences(point_values, conditional_draws)
    )
    difference_summary = _summarize_columns(
        point_differences, difference_draws, difference_descriptors
    )
    difference_summary = _add_multiplicity_adjustments(difference_summary)

    for frame in (conditional_summary, difference_summary):
        frame["requested_bootstrap"] = bootstrap_draws
        frame["failed_bootstrap"] = failed_draws
        frame["seed"] = seed
        frame["residual_integration"] = (
            f"{residual_node_count} empirical residual mid-quantiles"
        )
        frame["exposure_contrast"] = (
            "steps_z 0 to 1 using fixed original pooled mean/SD"
        )
        frame["effect_scale"] = "pairwise-normalized probability difference"

    metadata = pd.DataFrame([{
        "n": len(mediation_data),
        "bootstrap_requested": bootstrap_draws,
        "bootstrap_successful": bootstrap_draws - failed_draws,
        "bootstrap_failed": failed_draws,
        "seed": seed,
        "steps_mean_fixed": steps_mean,
        "steps_SD_fixed": steps_sd,
        "age_mean_fixed": age_mean,
        "age_SD_fixed": age_sd,
        "residual_nodes": residual_node_count,
        "max_absolute_K_vs_101_sensitivity": maximum_k_sensitivity,
        "effect_scale": "pairwise-normalized probability difference",
        "standardization_population": (
            "same pooled covariate distribution in every moderator cell"
        ),
        "PCA_bootstrap": (
            "pooled refit each draw; sign aligned to original pooled loading"
        ),
    }])

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    conditional_draws.to_csv(
        output_directory / "moderated_mediation_conditional_draws.csv",
        index=False,
    )
    difference_draws.to_csv(
        output_directory / "moderated_mediation_difference_draws.csv",
        index=False,
    )
    conditional_summary.to_csv(
        output_directory / "moderated_mediation_conditional_summary.csv",
        index=False,
    )
    difference_summary.to_csv(
        output_directory / "moderated_mediation_difference_summary.csv",
        index=False,
    )
    metadata.to_csv(
        output_directory / "moderated_mediation_bootstrap_metadata.csv",
        index=False,
    )
    return {
        "conditional_summary": conditional_summary,
        "difference_summary": difference_summary,
        "metadata": metadata,
        "conditional_draws": conditional_draws,
        "difference_draws": difference_draws,
    }

if "mediation_data" not in globals():
    raise RuntimeError("Run the pooled PCA cell before this analysis.")

MODERATED_BOOTSTRAP_DRAWS = 2000
MODERATED_BOOTSTRAP_SEED = 20260903
MEDIATOR_RESIDUAL_NODES = 51

moderated_mediation_results = run_moderated_mediation_bootstrap(
    mediation_data=mediation_data,
    reference_loadings=pc1_loadings,
    steps_mean=m1_steps_mean,
    steps_sd=m1_steps_sd,
    age_mean=m1_age_mean,
    age_sd=m1_age_sd,
    bootstrap_draws=MODERATED_BOOTSTRAP_DRAWS,
    seed=MODERATED_BOOTSTRAP_SEED,
    residual_node_count=MEDIATOR_RESIDUAL_NODES,
    output_directory=".",
)
moderated_mediation_conditional_summary = (
    moderated_mediation_results["conditional_summary"]
)
moderated_mediation_difference_summary = (
    moderated_mediation_results["difference_summary"]
)
moderated_mediation_bootstrap_metadata = (
    moderated_mediation_results["metadata"]
)

print("Direct tests of effect differences (percentage-point scale):")
display(moderated_mediation_difference_summary[[
    "analysis", "comparison", "contrast", "effect",
    "estimate_percentage_points",
    "CI_lower_percentage_points",
    "CI_upper_percentage_points",
    "p_value", "Holm_p_value", "successful_bootstrap",
]].round({
    "estimate_percentage_points": 3,
    "CI_lower_percentage_points": 3,
    "CI_upper_percentage_points": 3,
    "p_value": 4, "Holm_p_value": 4,
}))
print("Bootstrap and numerical-integration checks:")
display(moderated_mediation_bootstrap_metadata)
''',
}


def verify_embedded_sources() -> tuple[int, ...]:
    """Check original cell hashes and syntax without importing analysis packages."""
    verified = []
    for cell_index in STAGE_CELLS["moderation"]:
        source = SOURCE_CELLS[cell_index]
        original_source = (
            REMOVED_INSTALL_LINE + source if cell_index == 2 else source
        )
        digest = hashlib.sha256(original_source.encode("utf-8")).hexdigest()
        if digest != SOURCE_CELL_SHA256[cell_index]:
            raise ValueError(f"Embedded source hash mismatch in cell {cell_index}.")
        compile(
            source, f"{SOURCE_NOTEBOOK}:source-cell-{cell_index}", "exec",
            dont_inherit=True,
        )
        verified.append(cell_index)
    return tuple(verified)


def _cell_builtins() -> dict:
    """Retain cell imports while replacing presentation-only IPython display."""
    ordinary_import = builtins.__import__
    display_module = types.ModuleType("IPython.display")

    def plain_display(*objects, **unused_options):
        for value in objects:
            print(value)

    display_module.display = plain_display

    def standalone_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "IPython.display" and level == 0:
            return display_module
        return ordinary_import(name, globals, locals, fromlist, level)

    scope_builtins = dict(vars(builtins))
    scope_builtins["__import__"] = standalone_import
    return scope_builtins


def run_analysis(
    prepared_cohort: str | Path,
    output_dir: str | Path,
    stage: str = "primary",
) -> None:
    """Execute the selected exact-source stages on explicitly supplied inputs."""
    verify_embedded_sources()
    if stage not in STAGE_CELLS:
        raise ValueError(f"Unknown stage: {stage}")
    input_path = Path(prepared_cohort).expanduser().resolve(strict=True)
    if not input_path.is_file():
        raise ValueError("The prepared cohort must be an existing CSV file.")
    output_path = Path(output_dir).expanduser().resolve()
    code_directory = Path(__file__).resolve().parent
    if output_path == code_directory or code_directory in output_path.parents:
        raise ValueError("Choose an output directory outside the code release.")
    if output_path.exists():
        raise FileExistsError(
            "Choose a new output directory; existing files will not be overwritten."
        )

    # Analysis dependencies are deliberately imported only for an explicit run.
    import pandas as pd

    original_read_csv = pd.read_csv

    def read_prepared_csv(path, *args, **kwargs):
        if os.fspath(path) != SOURCE_INPUT_BASENAME:
            raise ValueError("Unexpected CSV read requested by embedded source.")
        return original_read_csv(input_path, *args, **kwargs)

    output_path.mkdir(parents=True, exist_ok=False)
    previous_directory = Path.cwd()
    module_name = "_released_notebook_source_" + uuid.uuid4().hex
    analysis_module = types.ModuleType(module_name)
    analysis_module.__dict__.update({
        "__builtins__": _cell_builtins(),
        "__file__": str(Path(__file__).resolve()),
    })
    # Registration is required by dataclass annotation handling in cell 20.
    sys.modules[module_name] = analysis_module
    try:
        with patch.object(pd, "read_csv", side_effect=read_prepared_csv):
            os.chdir(output_path)
            for cell_index in STAGE_CELLS[stage]:
                print(f"Running downstream source cell {cell_index} ({stage}).")
                code = compile(
                    SOURCE_CELLS[cell_index],
                    f"{SOURCE_NOTEBOOK}:source-cell-{cell_index}",
                    "exec",
                    dont_inherit=True,
                )
                exec(code, analysis_module.__dict__, analysis_module.__dict__)
    finally:
        os.chdir(previous_directory)
        sys.modules.pop(module_name, None)
    print(f"Completed stage: {stage}. Runtime outputs are in the requested directory.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected downstream notebook source on an explicitly supplied "
            "prepared cohort; never reads the original notebook at runtime."
        ),
        epilog=(
            "The output directory must be new and outside the code release. "
            "The PCA export contains participant records and must remain private. "
            "Use --stage moderation explicitly for the original 2000 bootstraps."
        ),
    )
    parser.add_argument("--prepared-cohort", type=Path, help="Prepared cohort CSV.")
    parser.add_argument("--output-dir", type=Path, help="New private output directory.")
    parser.add_argument(
        "--stage", choices=tuple(STAGE_CELLS), default="primary",
        help="primary (default), pathways (adds S5), or moderation (adds S1/S6 bootstrap).",
    )
    parser.add_argument(
        "--verify-sources", action="store_true",
        help="Verify embedded source hashes and syntax, then exit without reading data.",
    )
    args = parser.parse_args(argv)
    if args.verify_sources:
        checked = verify_embedded_sources()
        print("Verified source hashes and syntax for cells: " + ", ".join(map(str, checked)))
        return 0
    if args.prepared_cohort is None or args.output_dir is None:
        parser.error("--prepared-cohort and --output-dir are required for an analysis.")
    run_analysis(args.prepared_cohort, args.output_dir, args.stage)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
