#!/usr/bin/env python3
"""Assess CVD recording associations with and without setting classification.

This is an association-model sensitivity, not a new mediation analysis.
The input is the already upstream-restricted cohort: no additional landmark
exclusions are applied. Outputs contain aggregates only, including the
reproducible S7 supporting-information table approved for manuscript inclusion.
"""

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

BASE_DIR = Path(__file__).resolve().parent
TERM_LABELS = {
    "Intercept": "Intercept",
    "steps_z": "Mean daily steps, per original pooled SD",
    "age_z": "Age, per original pooled SD",
    "C(sex_at_birth)[T.Male]": "Male sex",
    "C(is_smoker)[T.1]": "Recorded smoking",
    "C(is_drinker)[T.1]": "Drinking, 2-4/month or 2-3/week",
    "C(is_drinker)[T.2]": "Drinking, >=4/week",
}
CATEGORY_LABELS = {0: "Control", 1: "Outpatient", 2: "Acute", 3: "Unclassified"}
CONTRASTS = (
    (1, 0, "Outpatient vs control"),
    (2, 0, "Acute vs control"),
    (2, 1, "Acute vs outpatient"),
    (3, 0, "Unclassified vs control"),
)
DESCRIPTIVE_GROUPS = {
    "outpatient": "Outpatient", "acute": "Acute care",
    "missing": "Missing setting", "other_unknown": "Other/Unknown",
}


def load_data(directory):
    data = pd.read_csv(directory / "final_analytic_cohort_with_habits.csv", usecols=[
        "person_id", "Group", "Onset_Type", "avg_steps", "age",
        "sex_at_birth", "is_smoker", "is_drinker",
    ])
    if data["person_id"].isna().any() or not data["person_id"].is_unique:
        raise ValueError("Input must have one non-missing ID per participant.")
    expected = {
        "Group": {"Control", "Heart Disease"},
        "sex_at_birth": {"Female", "Male"},
        "is_smoker": {0, 1},
        "is_drinker": {0, 1, 2},
        "Onset_Type": {
            "N/A (Control)", "Chronic (Office/Outpatient)",
            "Acute (Hospital/ER)", "Other/Unknown",
        },
    }
    for variable, allowed in expected.items():
        if not set(data[variable].dropna()).issubset(allowed):
            raise ValueError(f"Unexpected category in {variable}.")
    if data["Group"].isna().any():
        raise ValueError("CVD status must be observed.")
    controls = data["Group"].eq("Control")
    cases = data["Group"].eq("Heart Disease")
    outpatient = cases & data["Onset_Type"].eq("Chronic (Office/Outpatient)")
    acute = cases & data["Onset_Type"].eq("Acute (Hospital/ER)")
    unclassified = cases & (data["Onset_Type"].isna() | data["Onset_Type"].eq("Other/Unknown"))
    if not (controls | outpatient | acute | unclassified).all():
        raise ValueError("Some records cannot be assigned to a sensitivity category.")
    data["outcome"] = np.select([controls, outpatient, acute, unclassified], [0, 1, 2, 3])
    parameters = pd.read_csv(directory / "m1_standardization_parameters.csv").set_index("variable")
    primary = data.loc[data["outcome"].ne(3)]
    for name in ("avg_steps", "age"):
        if not np.isclose(primary[name].mean(), parameters.loc[name, "mean"], atol=1e-9):
            raise ValueError(f"Original M1 mean does not reproduce for {name}.")
        if not np.isclose(primary[name].std(ddof=1), parameters.loc[name, "SD"], atol=1e-9):
            raise ValueError(f"Original M1 SD does not reproduce for {name}.")
    return data, parameters


def descriptive_comparison(data):
    """Positive SMD denotes a larger mean/proportion in unclassified cases."""
    cases = data.loc[data["outcome"].ne(0)].copy()
    for name, source, level in (
        ("Female sex", "sex_at_birth", "Female"),
        ("Recorded smoking", "is_smoker", 1),
        ("Drinking, never/monthly or less", "is_drinker", 0),
        ("Drinking, 2-4/month or 2-3/week", "is_drinker", 1),
        ("Drinking, >=4/week", "is_drinker", 2),
    ):
        cases[name] = cases[source].eq(level).astype(float).where(cases[source].notna())
    retained = cases.loc[cases["outcome"].isin([1, 2])]
    excluded = cases.loc[cases["outcome"].eq(3)]
    rows = []
    for column, label, kind in (
        ("age", "Age, years", "continuous"),
        ("avg_steps", "Mean daily steps, steps/day", "continuous"),
        *[(name, name, "binary") for name in (
            "Female sex", "Recorded smoking", "Drinking, never/monthly or less",
            "Drinking, 2-4/month or 2-3/week", "Drinking, >=4/week",
        )],
    ):
        first, second = retained[column].dropna(), excluded[column].dropna()
        mean_first, mean_second = first.mean(), second.mean()
        row = {"variable": label, "type": kind,
               "retained_n_available": len(first), "unclassified_n_available": len(second),
               "retained_n_missing": len(retained) - len(first),
               "unclassified_n_missing": len(excluded) - len(second)}
        if kind == "continuous":
            denominator = np.sqrt((first.var(ddof=1) + second.var(ddof=1)) / 2)
            row.update(retained_mean=mean_first, retained_sd=first.std(ddof=1),
                       unclassified_mean=mean_second, unclassified_sd=second.std(ddof=1))
        else:
            denominator = np.sqrt((mean_first * (1 - mean_first) + mean_second * (1 - mean_second)) / 2)
            row.update(retained_count=int(first.sum()), retained_percent=100 * mean_first,
                       unclassified_count=int(second.sum()), unclassified_percent=100 * mean_second)
        row["SMD_unclassified_minus_retained"] = (mean_second - mean_first) / denominator if denominator else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def setting_specific_descriptives(data, expected_counts=None):
    """Describe four observed case-setting groups and pooled classified cases.

    SMDs compare each unclassified subgroup with the pooled outpatient/acute
    cases, not with the other unclassified subgroup. This function returns
    aggregates in memory; disclosure controls must precede any public export.
    """
    cases = data.loc[data.Group.eq("Heart Disease")].copy()
    required = ["age", "avg_steps", "sex_at_birth", "is_smoker", "is_drinker"]
    if cases[required].isna().any().any():
        raise ValueError("Missing descriptive covariates; complete-case reporting cannot be assumed.")
    if not np.isfinite(cases[["age", "avg_steps"]].to_numpy(float)).all():
        raise ValueError("Nonfinite descriptive covariates.")
    for column, levels in (("sex_at_birth", {"Female", "Male"}),
                           ("is_smoker", {0, 1}), ("is_drinker", {0, 1, 2})):
        if not set(cases[column]).issubset(levels):
            raise ValueError(f"Unexpected descriptive category in {column}.")
    masks = {
        "outpatient": cases.Onset_Type.eq("Chronic (Office/Outpatient)"),
        "acute": cases.Onset_Type.eq("Acute (Hospital/ER)"),
        "missing": cases.Onset_Type.isna(),
        "other_unknown": cases.Onset_Type.eq("Other/Unknown"),
    }
    if not np.all(np.column_stack(list(masks.values())).sum(axis=1) == 1):
        raise ValueError("CVD cases do not form the four disjoint descriptive groups.")
    groups = {name: cases.loc[mask].copy() for name, mask in masks.items()}
    if any(len(group) < 2 for group in groups.values()):
        raise ValueError("Each descriptive group requires at least two participants.")
    counts = {name: len(group) for name, group in groups.items()}
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("Unexpected four-group descriptive cohort counts.")
    groups["classified"] = cases.loc[masks["outpatient"] | masks["acute"]].copy()
    specs = [
        ("age", "Age, years", "continuous", None),
        ("avg_steps", "Daily steps, mean (SD)", "continuous", None),
        ("avg_steps", "Daily steps, median [Q1--Q3]", "median", None),
        ("sex_at_birth", "Female sex", "binary", "Female"),
        ("is_smoker", "Recorded smoking", "binary", 1),
        ("is_drinker", "Drinking, never/monthly or less", "binary", 0),
        ("is_drinker", "Drinking, 2-4/month or 2-3/week", "binary", 1),
        ("is_drinker", "Drinking, >=4/week", "binary", 2),
    ]
    rows, references = [], []
    for column, label, kind, level in specs:
        row = {"variable": label, "type": kind}
        values = {}
        for name, group in groups.items():
            values[name] = group[column].eq(level).astype(float) if kind == "binary" else group[column]
            series = values[name]
            row[f"{name}_n_available"] = len(series)
            row[f"{name}_n_missing"] = 0
            if kind == "continuous":
                row.update({f"{name}_mean": series.mean(), f"{name}_sd": series.std(ddof=1)})
            elif kind == "median":
                row.update({f"{name}_median": series.median(),
                            f"{name}_q1": series.quantile(.25, interpolation="linear"),
                            f"{name}_q3": series.quantile(.75, interpolation="linear")})
            else:
                row.update({f"{name}_count": int(series.sum()), f"{name}_percent": 100 * series.mean()})
        for name in ("missing", "other_unknown"):
            first, second = values["classified"], values[name]
            if kind == "median":
                smd = np.nan
            else:
                variance = (lambda series: series.mean() * (1 - series.mean())) if kind == "binary" else (lambda series: series.var(ddof=1))
                denominator = np.sqrt((variance(first) + variance(second)) / 2)
                smd = (second.mean() - first.mean()) / denominator if denominator else np.nan
            row[f"SMD_{name}_minus_classified"] = smd
        references.append({"variable": label, "type": kind,
                           **{key.removeprefix("classified_"): value for key, value in row.items() if key.startswith("classified_")}})
        rows.append({key: value for key, value in row.items() if not key.startswith("classified_")})
    return pd.DataFrame(rows), pd.DataFrame(references)


def descriptive_disclosure_flags(table, threshold=20):
    """Return cell labels only, never the sensitive counts themselves."""
    flags = []
    for row in table.itertuples():
        for group in DESCRIPTIVE_GROUPS:
            n = int(getattr(row, f"{group}_n_available"))
            if 0 < n <= threshold:
                flags.append({"group": group, "variable": row.variable, "kind": "denominator"})
            if row.type == "binary":
                count = int(getattr(row, f"{group}_count"))
                if any(0 < value <= threshold for value in (count, n - count)):
                    flags.append({"group": group, "variable": row.variable, "kind": "count_or_complement"})
    return flags


def apply_descriptive_disclosure_controls(table):
    """Suppress sparse drinking cells and complementary reconstruction routes.

    The pooled unclassified counts are already reported. When a drinking cell
    in either unclassified subgroup is sparse, both upper drinking categories
    are suppressed in both subgroups, together with their percentages and SMDs.
    Other sparse patterns require a separate disclosure review before export.
    """
    flags = descriptive_disclosure_flags(table)
    upper_drinking = {"Drinking, 2-4/month or 2-3/week", "Drinking, >=4/week"}
    if any(flag["group"] not in {"missing", "other_unknown"}
           or flag["variable"] not in upper_drinking or flag["kind"] != "count_or_complement"
           for flag in flags):
        raise ValueError("Additional small-cell patterns require disclosure review before export.")
    public = table.copy(deep=True)
    for group in DESCRIPTIVE_GROUPS:
        public[f"{group}_suppressed"] = False
        public[f"{group}_reporting_status"] = "reported"
    if flags:
        affected = public.variable.isin(upper_drinking)
        for group in ("missing", "other_unknown"):
            public.loc[affected, f"{group}_suppressed"] = True
            public.loc[affected, f"{group}_reporting_status"] = "suppressed"
            for field in (f"{group}_count", f"{group}_percent", f"SMD_{group}_minus_classified"):
                public.loc[affected, field] = np.nan
    return public, {"threshold": 20, "flagged_cells": flags,
                    "suppressed_categories": sorted(upper_drinking) if flags else [],
                    "suppressed_groups": ["missing", "other_unknown"] if flags else [],
                    "policy": "Counts of 1-20 and recoverable complements are not disclosed. Upper drinking categories in both unclassified subgroups, associated percentages and SMDs are suppressed together when required; pooled unclassified totals cannot recover an individual suppressed cell."}


def build_design(data, parameters):
    required = ["avg_steps", "age", "sex_at_birth", "is_smoker", "is_drinker"]
    complete = data.dropna(subset=required).copy()
    if not np.isfinite(complete[["avg_steps", "age"]].to_numpy(float)).all():
        raise ValueError("Nonfinite continuous covariate.")
    design = pd.DataFrame(index=complete.index)
    design["Intercept"] = 1.0
    design["steps_z"] = (complete["avg_steps"] - parameters.loc["avg_steps", "mean"]) / parameters.loc["avg_steps", "SD"]
    design["age_z"] = (complete["age"] - parameters.loc["age", "mean"]) / parameters.loc["age", "SD"]
    design["C(sex_at_birth)[T.Male]"] = complete["sex_at_birth"].eq("Male").astype(float)
    design["C(is_smoker)[T.1]"] = complete["is_smoker"].eq(1).astype(float)
    design["C(is_drinker)[T.1]"] = complete["is_drinker"].eq(1).astype(float)
    design["C(is_drinker)[T.2]"] = complete["is_drinker"].eq(2).astype(float)
    return complete, design


def fit_and_contrast(data, parameters, model_name):
    complete, design = build_design(data, parameters)
    categories = sorted(complete["outcome"].unique())
    if categories != list(range(len(categories))):
        raise ValueError("Outcome codes must be contiguous from control = 0.")
    model = sm.MNLogit(complete["outcome"], design).fit(method="newton", maxiter=250, disp=False)
    if not model.mle_retvals.get("converged", False):
        raise RuntimeError(f"{model_name} did not converge.")
    estimates = np.asarray(model.params, dtype=float)
    covariance = np.asarray(model.cov_params(), dtype=float)
    flat = estimates.ravel(order="F")
    # statsmodels orders the covariance matrix by nonreference category, then term.
    np.testing.assert_allclose(np.diag(covariance), np.asarray(model.bse).ravel(order="F") ** 2, rtol=1e-10)
    if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
        raise RuntimeError("Invalid model covariance matrix.")
    predictions = np.asarray(model.predict(design))
    np.testing.assert_allclose(predictions.sum(axis=1), 1.0, atol=1e-12)
    rows = []
    n_terms = design.shape[1]
    for numerator, denominator, contrast_name in CONTRASTS:
        if numerator not in categories:
            continue
        for term_index, term in enumerate(design.columns):
            contrast = np.zeros_like(flat)
            if numerator:
                contrast[(numerator - 1) * n_terms + term_index] += 1
            if denominator:
                contrast[(denominator - 1) * n_terms + term_index] -= 1
            estimate = float(contrast @ flat)
            se = float(np.sqrt(contrast @ covariance @ contrast))
            rows.append({
                "model": model_name, "term": term, "variable": TERM_LABELS[term],
                "contrast": contrast_name, "log_OR": estimate, "SE": se,
                # Match the existing M1's 1.96 Wald critical value exactly.
                "OR": np.exp(estimate), "CI_lower": np.exp(estimate - 1.96 * se),
                "CI_upper": np.exp(estimate + 1.96 * se),
                "p_value": 2 * norm.sf(abs(estimate / se)), "n": len(complete),
                **{f"n_{label.lower()}": int(complete["outcome"].eq(code).sum()) for code, label in CATEGORY_LABELS.items()},
                "steps_SD": parameters.loc["avg_steps", "SD"], "age_SD": parameters.loc["age", "SD"],
            })
    diagnostics = {"model": model_name, "n": len(complete),
                   "n_excluded_missing_model_covariates": len(data) - len(complete),
                   "converged": bool(model.mle_retvals["converged"]),
                   "iterations": int(model.mle_retvals["iterations"]),
                   "log_likelihood": float(model.llf),
                   "maximum_absolute_average_score": float(np.max(np.abs(model.model.score(flat))) / len(complete))}
    return pd.DataFrame(rows), diagnostics


def verify_primary(reproduced, directory):
    reference = pd.read_csv(directory / "m1_primary_pairwise_results_python.csv")
    merged = reference.merge(reproduced, on=["term", "contrast"], suffixes=("_existing", "_reproduced"), validate="one_to_one")
    if len(merged) != len(reference):
        raise RuntimeError("Some original primary-model terms were not reproduced.")
    errors = {}
    for field in ("log_OR", "SE", "OR", "CI_lower", "CI_upper", "p_value"):
        existing, current = merged[f"{field}_existing"], merged[f"{field}_reproduced"]
        np.testing.assert_allclose(existing, current, rtol=1e-8, atol=1e-9)
        errors[field] = float(np.max(np.abs(existing - current)))
    return errors


def fit_any_cvd_models(data, parameters, expected_counts=None):
    """Fit two nested-cohort binary models using the original M1 scales.

    CVD status comes from Group, including cases with unavailable settings.
    This supplementary outcome does not reassign a case to a recording setting.
    Missing covariates or inconsistent outcome labels raise rather than silently
    changing the supplied cohorts. ``expected_counts`` maps each cohort name to
    its required (total, cases, no-recorded-CVD) counts when supplied.
    """
    required = ["Group", "outcome", "avg_steps", "age", "sex_at_birth",
                "is_smoker", "is_drinker"]
    if data[required].isna().any().any():
        raise ValueError("Missing binary-outcome or model covariate values.")
    allowed = {"Group": {"Control", "Heart Disease"}, "outcome": {0, 1, 2, 3},
               "sex_at_birth": {"Female", "Male"}, "is_smoker": {0, 1},
               "is_drinker": {0, 1, 2}}
    for column, levels in allowed.items():
        if not set(data[column]).issubset(levels):
            raise ValueError(f"Unexpected category in {column}.")
    if not data["Group"].eq("Heart Disease").equals(data["outcome"].ne(0)):
        raise ValueError("Group and setting-category CVD status disagree.")
    if "person_id" in data and (data.person_id.isna().any() or not data.person_id.is_unique):
        raise ValueError("Input must have one non-missing ID per participant.")
    scales = parameters.loc[["avg_steps", "age"], ["mean", "SD"]]
    if not np.isfinite(scales.to_numpy(float)).all() or scales["SD"].le(0).any():
        raise ValueError("Original standardization requires finite means and positive SDs.")
    cohorts = [("Original classified cohort", data.loc[data.outcome.ne(3)]),
               ("Full prepared cohort", data)]
    estimates, diagnostics = [], []
    for label, cohort in cohorts:
        complete, design = build_design(cohort, parameters)
        if len(complete) != len(cohort):
            raise ValueError("Binary analysis must retain all supplied cohort members.")
        outcome = complete["Group"].eq("Heart Disease").astype(int)
        counts = (len(complete), int(outcome.sum()), int(outcome.eq(0).sum()))
        if expected_counts is not None and counts != tuple(expected_counts[label]):
            raise ValueError(f"Unexpected cohort counts for {label}: {counts}.")
        if outcome.nunique() != 2:
            raise ValueError("Both binary outcome levels are required.")
        if np.linalg.matrix_rank(design.to_numpy(float)) != design.shape[1]:
            raise ValueError("Binary-model design is not full rank.")
        model = sm.Logit(outcome, design).fit(method="newton", maxiter=250, disp=False)
        if not model.mle_retvals.get("converged", False):
            raise RuntimeError(f"Binary model did not converge for {label}.")
        covariance = np.asarray(model.cov_params(), dtype=float)
        if not np.isfinite(covariance).all() or np.linalg.eigvalsh(covariance).min() <= 0:
            raise RuntimeError("Invalid binary-model covariance matrix.")
        average_score = float(np.max(np.abs(model.model.score(model.params))) / counts[0])
        if average_score > 1e-8:
            raise RuntimeError("Binary-model score did not reach the convergence tolerance.")

        # Independently construct categorical regressors through the formula API
        # and fit the same likelihood with binomial GLM/IRLS.
        verification = complete.copy()
        verification["any_cvd"] = outcome
        for source, target in (("avg_steps", "steps_z"), ("age", "age_z")):
            verification[target] = ((verification[source] - parameters.loc[source, "mean"])
                                    / parameters.loc[source, "SD"])
        for name, levels in (("sex_at_birth", ["Female", "Male"]),
                             ("is_smoker", [0, 1]), ("is_drinker", [0, 1, 2])):
            verification[name] = pd.Categorical(verification[name], categories=levels)
        check = smf.glm("any_cvd ~ steps_z + age_z + C(sex_at_birth) + C(is_smoker) + C(is_drinker)",
                        data=verification, family=sm.families.Binomial()).fit(maxiter=250, tol=1e-12)
        if not check.converged or int(check.nobs) != counts[0]:
            raise RuntimeError("Independent formula GLM failed convergence or cohort-size verification.")
        check_estimates = check.params.reindex(design.columns)
        check_covariance = check.cov_params().reindex(index=design.columns, columns=design.columns)
        np.testing.assert_allclose(model.params, check_estimates, rtol=1e-8, atol=1e-10)
        np.testing.assert_allclose(covariance, check_covariance, rtol=1e-7, atol=1e-10)
        for term in design.columns:
            estimate, se = float(model.params[term]), float(model.bse[term])
            estimates.append({
                "cohort": label, "term": term, "variable": TERM_LABELS[term],
                "contrast": "Any qualifying primary CVD recording vs no recorded CVD",
                "log_OR": estimate, "SE": se, "OR": np.exp(estimate),
                "CI_lower": np.exp(estimate - 1.96 * se),
                "CI_upper": np.exp(estimate + 1.96 * se),
                "p_value": 2 * norm.sf(abs(estimate / se)),
                "n": counts[0], "n_cvd": counts[1], "n_no_recorded_cvd": counts[2],
                "n_unclassified_cvd": int(complete.outcome.eq(3).sum()),
                "steps_mean": parameters.loc["avg_steps", "mean"],
                "steps_SD": parameters.loc["avg_steps", "SD"],
                "age_mean": parameters.loc["age", "mean"],
                "age_SD": parameters.loc["age", "SD"],
            })
        diagnostics.append({
            "cohort": label, "n": counts[0], "n_cvd": counts[1],
            "n_no_recorded_cvd": counts[2], "n_excluded_missing_model_covariates": 0,
            "converged": bool(model.mle_retvals["converged"]),
            "iterations": int(model.mle_retvals["iterations"]),
            "log_likelihood": float(model.llf),
            "maximum_absolute_average_score": average_score,
            "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(covariance).min()),
            "formula_GLM_converged": bool(check.converged),
            "formula_GLM_maximum_absolute_coefficient_difference": float(np.max(np.abs(model.params - check_estimates))),
            "formula_GLM_maximum_absolute_covariance_difference": float(np.max(np.abs(covariance - check_covariance.to_numpy()))),
        })
    return pd.DataFrame(estimates), diagnostics


def write_supporting_table(directory, descriptive, steps, metadata, n_unknown, n_missing, binary_steps):
    """Render S7 directly from the aggregate outputs used by the manuscript."""
    def p_text(value):
        return "$<0.001$" if value < 0.001 else f"{value:.3f}"

    def estimate_text(row):
        return f"{row.OR:.3f} ({row.CI_lower:.3f}--{row.CI_upper:.3f})"

    n_classified = int(descriptive.iloc[0].outpatient_n_available + descriptive.iloc[0].acute_n_available)
    n_unclassified = int(metadata["unclassified_n"])
    primary = steps.loc[steps["model"].eq("Reproduced three-category M1")].set_index("contrast")
    expanded = steps.loc[steps["model"].eq("Four-category unclassified-setting sensitivity")].set_index("contrast")
    lines = [
        "% Generated by run_unclassified_setting_sensitivity.py; do not hand-edit values.",
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=0.65in]{geometry}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{booktabs,tabularx,array,microtype}",
        r"\setlength{\parindent}{0pt}",
        r"\setlength{\parskip}{0.6em}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\begin{document}",
        r"\section*{S7 Table. Sensitivity to retaining CVD records with unclassified recording settings}",
        r"\textbf{A. Characteristics of CVD cases by recording-setting completeness}\par",
        r"{\footnotesize\setlength{\tabcolsep}{2.5pt}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}X*{4}{>{\raggedleft\arraybackslash}p{2.35cm}}rr@{}}",
        r"\toprule",
        "Characteristic & " + " & ".join(
            f"\\shortstack{{{DESCRIPTIVE_GROUPS[group]}\\\\$N={int(descriptive.iloc[0][f'{group}_n_available'])}$}}"
            for group in DESCRIPTIVE_GROUPS
        ) + r" & \shortstack{SMD\\M--C} & \shortstack{SMD\\O--C} \\",
        r"\midrule",
    ]
    for row in descriptive.itertuples():
        cells = []
        for group in DESCRIPTIVE_GROUPS:
            if getattr(row, f"{group}_suppressed"):
                cells.append("Suppressed")
            elif row.type == "continuous":
                cells.append(f"{getattr(row, f'{group}_mean'):.1f} ({getattr(row, f'{group}_sd'):.1f})")
            elif row.type == "median":
                cells.append(f"{getattr(row, f'{group}_median'):.1f} [{getattr(row, f'{group}_q1'):.1f}--{getattr(row, f'{group}_q3'):.1f}]")
            else:
                cells.append(f"{int(getattr(row, f'{group}_count'))} ({getattr(row, f'{group}_percent'):.1f}\\%)")
        for group in ("missing", "other_unknown"):
            if getattr(row, f"{group}_suppressed"):
                cells.append("Suppressed")
            elif row.type == "median":
                cells.append("---")
            else:
                cells.append(f"${getattr(row, f'SMD_{group}_minus_classified'):.3f}$")
        label = {"Drinking, 2-4/month or 2-3/week": "Drinking, 2--4/month or 2--3/week",
                 "Drinking, >=4/week": r"Drinking, $\geq4$/week",
                 "Daily steps, median [Q1--Q3]": "Daily steps, median [Q1--Q3]"}.get(row.variable, row.variable)
        lines.append(f"{label} & " + " & ".join(cells) + r" \\")
    lines.extend([
        r"\bottomrule\end{tabularx}\par}",
        r"{\footnotesize Values are mean (SD), median [first--third quartile], or $n$ (\%). Step summaries describe participants' mean daily step counts, in steps/day. SMD=standardised mean difference; M=missing setting; O=other/unknown setting; C=pooled classified cases (outpatient and acute care, $N=" + str(n_classified) + r"$). Both SMD columns compare their indicated subgroup with this same pooled reference, using the square root of the unweighted mean of the two sample variances for continuous variables and Bernoulli variances for binary variables. Positive SMDs indicate larger subgroup values or proportions. The median row is descriptive and has no separate SMD. Quartiles use linear interpolation. All listed covariates were observed. Each drinking category is treated as a separate indicator; no descriptive hypothesis tests were performed. Selected drinking cells, percentages, and corresponding SMDs are suppressed to protect small counts and prevent their reconstruction from published totals.\par}",
        r"\medskip",
        r"\textbf{B. Daily step-count associations in the original and expanded models}\par",
        r"{\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xcccc@{}}",
        r"\toprule",
        f"& \\multicolumn{{2}}{{c}}{{Three-category M1 ($N={metadata['primary_n']}$)}} & \\multicolumn{{2}}{{c}}{{Four-category model ($N={metadata['input_n']}$)}} " + r"\\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"Outcome comparison & OR (95\% CI) & $p$ & OR (95\% CI) & $p$ \\",
        r"\midrule",
    ])
    for _, _, contrast in CONTRASTS:
        one = primary.loc[contrast] if contrast in primary.index else None
        two = expanded.loc[contrast]
        label = contrast.replace("Acute", "Acute care").replace("vs control", "vs no recorded CVD")
        first = f"{estimate_text(one)} & {p_text(one.p_value)}" if one is not None else "--- & ---"
        lines.append(f"{label} & {first} & {estimate_text(two)} & {p_text(two.p_value)} " + r"\\")
    row = expanded.iloc[0]
    lines.extend([
        r"\bottomrule\end{tabularx}\par}",
        r"{\footnotesize",
        f"ORs are per 1 original pooled SD of mean daily steps ({row.steps_SD:.2f} steps/day). Both multinomial models adjust for age, sex, recorded smoking status, and drinking frequency using the same original M1 means and SDs; age SD is {row.age_SD:.2f} years. Female sex, recorded never/non-smoker status (smoking code 0), and never/monthly-or-less drinking during the past year (drinking code 0) are predictor reference categories. The smoking-concept mapping, drinking-response grouping, and measurement limitations are given in S1 Text. Participants without recorded CVD are the outcome reference. Acute-care versus outpatient estimates use the jointly estimated coefficients and cross-logit covariance, not a separate binary model. Intervals are two-sided Wald 95\\% CIs.",
        f"The three-category model includes {int(row.n_control)} participants without recorded CVD, {int(row.n_outpatient)} outpatient recordings, and {int(row.n_acute)} acute-care recordings. The expanded model adds {n_unclassified} CVD records ({metadata['unclassified_percent_of_all_cvd']:.1f}\\% of all CVD cases): {n_unknown} other/unknown and {n_missing} missing settings. All model covariates were observed; no additional participants were excluded. The dataset was already restricted upstream to the $t_1$-landmark cohort; no further landmark exclusions were applied.",
        f"Unclassified is an administrative analysis category, not a new clinical phenotype or an imputation of the true recording setting. Similar known-setting associations support stability under this expanded outcome specification, but do not assess sensitivity to the unknown acute-care or outpatient assignments of these {n_unclassified} cases, establish random setting missingness, or rule out selection bias. PCA and exploratory mediation models were not re-estimated in this sensitivity analysis. CI=confidence interval; CVD=cardiovascular disease; OR=odds ratio; SD=standard deviation.",
        r"\par}",
        r"\clearpage",
        r"\textbf{C. Any qualifying primary CVD recording, regardless of setting}\par",
        r"{\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrcc@{}}",
        r"\toprule",
        r"Cohort & $N$ & \shortstack{CVD\\recording} & \shortstack{No recorded\\CVD} & OR (95\% CI) & $p$ \\",
        r"\midrule",
    ])
    for row in binary_steps.itertuples():
        lines.append(f"{row.cohort} & {row.n:,} & {row.n_cvd:,} & {row.n_no_recorded_cvd:,} & {estimate_text(row)} & {p_text(row.p_value)} " + r"\\")
    lines.extend([
        r"\bottomrule\end{tabularx}\par}",
        r"{\footnotesize",
        "Binary logistic regression compares any qualifying primary CVD recording with no recorded CVD over the same 24-month outcome window, from $t_1=t_0+12$ months through $t_0+36$ months. Each model adjusts for linear age, sex assigned at birth, recorded smoking status, and categorical drinking frequency, using the same reference categories and original M1 standardisation as Panel B. ORs are per 3298.82 steps/day; CIs are two-sided Wald intervals, and $p$ values are unadjusted two-sided Wald tests.",
        f"The full prepared cohort retains all {n_unclassified} unclassified CVD cases ({n_missing} missing and {n_unknown} other/unknown settings) as cases without assigning them to outpatient or acute care. All covariates were observed, and no additional participants were excluded. Both cohorts retain the same upstream eligibility and landmark restrictions; this post hoc supplementary endpoint analysis does not broaden those eligibility criteria.",
        "The cohorts are nested and share all participants without recorded CVD; their estimates are not independent replications, and no between-cohort test of the OR difference was performed. This analysis assesses association with CVD recording irrespective of setting, not sensitivity of the acute-care versus outpatient contrast to unknown settings.",
        r"\par}",
        r"\end{document}",
    ])
    (directory / "S7_Table.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=BASE_DIR)
    args = parser.parse_args()
    directory = args.directory.resolve()
    data, parameters = load_data(directory)
    original, diagnostic_original = fit_and_contrast(data.loc[data["outcome"].ne(3)], parameters, "Reproduced three-category M1")
    reproduction_errors = verify_primary(original, directory)
    expanded, diagnostic_expanded = fit_and_contrast(data, parameters, "Four-category unclassified-setting sensitivity")
    binary_results, binary_diagnostics = fit_any_cvd_models(data, parameters, expected_counts={
        "Original classified cohort": (8292, 2512, 5780),
        "Full prepared cohort": (8895, 3115, 5780),
    })
    binary_steps = binary_results.loc[binary_results.term.eq("steps_z")].copy()
    descriptive = descriptive_comparison(data)
    group_descriptive, classified_reference = setting_specific_descriptives(data, expected_counts={
        "outpatient": 2127, "acute": 385, "missing": 455, "other_unknown": 148,
    })
    public_descriptive, disclosure = apply_descriptive_disclosure_controls(group_descriptive)
    results = pd.concat([original, expanded], ignore_index=True)
    steps = results.loc[results["term"].eq("steps_z")].copy()
    counts = data.groupby(["Group", "Onset_Type"], dropna=False).size().reset_index(name="n")
    counts["Onset_Type"] = counts["Onset_Type"].fillna("Missing setting")
    results.to_csv(directory / "unclassified_setting_multinomial_results.csv", index=False)
    steps.to_csv(directory / "unclassified_setting_steps_comparison.csv", index=False)
    descriptive.to_csv(directory / "unclassified_setting_case_characteristics.csv", index=False)
    public_descriptive.to_csv(directory / "unclassified_setting_case_characteristics_by_setting.csv", index=False)
    classified_reference.to_csv(directory / "unclassified_setting_classified_reference.csv", index=False)
    counts.to_csv(directory / "unclassified_setting_counts.csv", index=False)
    binary_results.to_csv(directory / "any_cvd_binary_results.csv", index=False)
    binary_steps.to_csv(directory / "any_cvd_binary_steps_comparison.csv", index=False)
    metadata = {
        "input_n": len(data), "primary_n": int(data["outcome"].ne(3).sum()),
        "unclassified_n": int(data["outcome"].eq(3).sum()),
        "unclassified_percent_of_all_cvd": 100 * data["outcome"].eq(3).sum() / data["outcome"].ne(0).sum(),
        "standardization": parameters.reset_index().to_dict(orient="records"),
        "model_diagnostics": [diagnostic_original, diagnostic_expanded],
        "original_M1_reproduction_max_absolute_errors": reproduction_errors,
        "descriptive_SMD": "Unclassified minus retained; continuous pooled unweighted group variances; binary pooled Bernoulli variances; available-case denominators.",
        "four_group_descriptive_SMD": "Missing setting minus pooled classified cases, and other/unknown setting minus pooled classified cases; sample variance ddof=1 for continuous variables, Bernoulli variance for each binary indicator, unweighted mean of the two variances in each SMD denominator; no median-based SMD.",
        "confidence_intervals": "Two-sided Wald 95% intervals with critical value 1.96, matching original M1, using the full jointly estimated multinomial covariance including cross-logit covariance for acute vs outpatient.",
        "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in (
            "final_analytic_cohort_with_habits.csv", "m1_standardization_parameters.csv", "m1_primary_pairwise_results_python.csv")},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "scope": "Already upstream-restricted participants; no further temporal screening, no mediation re-estimation, no assignment of true setting to unclassified cases.",
    }
    (directory / "unclassified_setting_sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    descriptive_metadata = {
        "groups": {name: int(public_descriptive.iloc[0][f"{name}_n_available"]) for name in DESCRIPTIVE_GROUPS},
        "reference_group": {"name": "Pooled classified cases (outpatient and acute care)", "n": 2512},
        "cohort": "Existing 3115 qualifying primary CVD cases; no reclassification or additional exclusions",
        "missing_setting": "Onset_Type is null in the prepared analytic extract",
        "other_unknown_setting": "Onset_Type equals Other/Unknown in the prepared analytic extract",
        "continuous_summary": "Participant-level age and average daily steps: mean and sample SD (ddof=1); steps also median, Q1 and Q3 using linear interpolation",
        "categorical_summary": "Count and percentage with all group members as denominator; no missing model covariates",
        "SMD": metadata["four_group_descriptive_SMD"],
        "hypothesis_tests": "None; descriptive comparisons only",
        "disclosure": disclosure,
        "input_sha256": metadata["input_sha256"],
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": metadata["versions"],
    }
    (directory / "unclassified_setting_descriptive_metadata.json").write_text(json.dumps(descriptive_metadata, indent=2) + "\n")
    binary_metadata = {
        "endpoint": "Any qualifying primary CVD recording, regardless of setting, versus no recorded CVD",
        "outcome_source": "Group == Heart Disease; all unclassified cases remain cases",
        "outcome_window": "t1 = t0 + 12 months through t0 + 36 months; unchanged upstream landmark cohort",
        "model": "Binary logistic regression with linear steps_z, linear age_z, sex assigned at birth, recorded smoking status, and categorical drinking frequency",
        "reference_categories": {"sex_at_birth": "Female", "is_smoker": 0, "is_drinker": 0, "outcome": "No recorded CVD"},
        "standardization": metadata["standardization"],
        "confidence_intervals": "Two-sided Wald 95% intervals using critical value 1.96, matching M1",
        "p_values": "Unadjusted two-sided Wald tests; supplementary post hoc endpoint analysis",
        "model_diagnostics": binary_diagnostics,
        "input_sha256": metadata["input_sha256"],
        "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "versions": metadata["versions"],
        "scope": "Nested cohorts; no between-cohort OR-difference test, setting imputation, additional upstream exclusions, or mediation re-estimation",
    }
    (directory / "any_cvd_binary_metadata.json").write_text(json.dumps(binary_metadata, indent=2) + "\n")
    n_cases = int(data["outcome"].ne(0).sum())
    n_classified = int(data["outcome"].isin([1, 2]).sum())
    n_unclassified = int(data["outcome"].eq(3).sum())
    n_unknown = int((data["outcome"].eq(3) & data["Onset_Type"].eq("Other/Unknown")).sum())
    n_missing = int((data["outcome"].eq(3) & data["Onset_Type"].isna()).sum())
    write_supporting_table(directory, public_descriptive, steps, metadata, n_unknown, n_missing, binary_steps)
    report = ["# Unclassified-setting sensitivity", "",
              "Aggregate descriptive comparisons and association-model results are reported in S7_Table.tex.", "",
              f"The exported cohort includes {len(data):,} participants. Of {n_cases:,} CVD cases, {n_classified:,} have classified settings and {n_unclassified:,} ({100 * n_unclassified / n_cases:.2f}%) are unclassified: {n_unknown:,} other/unknown and {n_missing:,} missing setting. No additional upstream landmark exclusions were applied.", "",
              "## Steps associations", "",
              "All estimates use the original M1 scale: 1 SD = 3298.822232 steps/day. Adjustment: age, sex, recorded smoking status and drinking-frequency categories; control is the outcome reference. Female sex, documented never/non-smoker status, and never/monthly-or-less drinking are predictor references. Smoking concepts and upstream eligibility, grouped past-year drinking responses, and measurement definitions are documented in S1 Text.", "",
              "| Model | Contrast | OR (95% CI) | p |", "|---|---|---|---|"]
    for row in steps.itertuples():
        report.append(f"| {row.model} | {row.contrast} | {row.OR:.4f} ({row.CI_lower:.4f}, {row.CI_upper:.4f}) | {row.p_value:.4g} |")
    report += ["", "## Any qualifying primary CVD recording, regardless of setting", "",
               "Two post hoc supplementary binary logistic models use the original M1 standardization and adjustment set. Both use the existing 24-month outcome window and upstream eligibility rules. No settings are imputed; the full prepared cohort retains all 603 unclassified CVD cases as cases.", "",
               "| Cohort | N | CVD recording | No recorded CVD | OR (95% CI) | p |",
               "|---|---:|---:|---:|---|---:|"]
    for row in binary_steps.itertuples():
        report.append(f"| {row.cohort} | {row.n} | {row.n_cvd} | {row.n_no_recorded_cvd} | {row.OR:.4f} ({row.CI_lower:.4f}, {row.CI_upper:.4f}) | {row.p_value:.4g} |")
    report += ["", "Higher steps were inversely associated with qualifying CVD recording in both cohorts. The full-cohort estimate includes the 603 unclassified cases without guessing their encounter settings. The cohorts are nested, not independent replications; no test compares the two ORs. These estimates do not resolve uncertainty in the acute-care versus outpatient contrast or remove the original cohort-selection and confounding limitations.", "",
               "Both binary fits were independently checked using a formula-based binomial GLM with the same scales and categorical covariates. No participants were dropped for missing covariates. Coefficients, covariance matrices, convergence diagnostics and input hashes are recorded in any_cvd_binary_metadata.json and the aggregate result CSVs."]
    report += ["", "## Descriptive comparison of CVD cases", "",
               "Positive standardized mean differences (SMDs) indicate larger values/proportions in unclassified cases. No null-hypothesis tests were performed. Each alcohol category is compared as an indicator, not as a single omnibus SMD.", "",
               f"| Characteristic | Classified (n={n_classified}) | Unclassified (n={n_unclassified}) | SMD |", "|---|---|---|---|"]
    for row in descriptive.itertuples():
        if row.type == "continuous":
            first, second = f"{row.retained_mean:.1f} ({row.retained_sd:.1f})", f"{row.unclassified_mean:.1f} ({row.unclassified_sd:.1f})"
        else:
            first, second = f"{int(row.retained_count)} ({row.retained_percent:.1f}%)", f"{int(row.unclassified_count)} ({row.unclassified_percent:.1f}%)"
        report.append(f"| {row.variable} | {first} | {second} | {row.SMD_unclassified_minus_retained:.3f} |")
    report += ["", "## Four-group descriptive comparison (S7 Panel A)", "",
               "The missing-setting and other/unknown groups are distinct. Both SMD columns use all 2512 classified cases (outpatient plus acute care) as the common reference. Step summaries describe participant-level mean daily steps; quartiles use linear interpolation. All listed covariates are observed. These are descriptive comparisons without hypothesis tests.", "",
               "| Characteristic | Outpatient | Acute care | Missing setting | Other/Unknown | SMD missing minus classified | SMD other minus classified |",
               "|---|---|---|---|---|---|---|"]
    for row in public_descriptive.itertuples():
        cells = [row.variable]
        for group in DESCRIPTIVE_GROUPS:
            if getattr(row, f"{group}_suppressed"):
                cells.append("Suppressed")
            elif row.type == "continuous":
                cells.append(f"{getattr(row, f'{group}_mean'):.1f} ({getattr(row, f'{group}_sd'):.1f})")
            elif row.type == "median":
                cells.append(f"{getattr(row, f'{group}_median'):.1f} [{getattr(row, f'{group}_q1'):.1f}, {getattr(row, f'{group}_q3'):.1f}]")
            else:
                cells.append(f"{int(getattr(row, f'{group}_count'))} ({getattr(row, f'{group}_percent'):.1f}%)")
        for group in ("missing", "other_unknown"):
            if getattr(row, f"{group}_suppressed"):
                cells.append("Suppressed")
            else:
                cells.append("Not applicable" if row.type == "median" else f"{getattr(row, f'SMD_{group}_minus_classified'):.3f}")
        report.append("| " + " | ".join(cells) + " |")
    report += ["", "Selected drinking cells, percentages and corresponding SMDs are suppressed together to protect small counts and prevent reconstruction from previously reported pooled totals. The aggregate export retains blank numeric cells and explicit suppressed status for these entries."]
    primary_setting = steps.loc[steps["model"].eq("Reproduced three-category M1") & steps["contrast"].eq("Acute vs outpatient")].iloc[0]
    expanded_setting = steps.loc[steps["model"].eq("Four-category unclassified-setting sensitivity") & steps["contrast"].eq("Acute vs outpatient")].iloc[0]
    step_characteristic = descriptive.loc[descriptive["variable"].eq("Mean daily steps, steps/day")].iloc[0]
    setting_steps = public_descriptive.loc[public_descriptive.variable.eq("Daily steps, mean (SD)")].iloc[0]
    report += ["", "## Interpretation", "",
               f"The acute-versus-outpatient Steps OR changes from {primary_setting.OR:.3f} to {expanded_setting.OR:.3f}; the expanded-model interval remains below one ({expanded_setting.CI_lower:.3f}, {expanded_setting.CI_upper:.3f}). The known-setting contrasts are numerically similar after retaining unclassified records. Mean daily steps in cases with missing settings are {setting_steps.missing_mean:.1f}, compared with {step_characteristic.retained_mean:.1f} in pooled classified cases (SMD {setting_steps.SMD_missing_minus_classified:.3f}); the other/unknown group averages {setting_steps.other_unknown_mean:.1f} (SMD {setting_steps.SMD_other_unknown_minus_classified:.3f} against the same reference). Thus, the pooled unclassified average of {step_characteristic.unclassified_mean:.1f} combines descriptively different groups and should not be attributed to either group individually. These observed summaries do not identify the cause of missingness or establish random setting missingness.", "",
               "## Verification and limits", "",
               "The three-category M1 was reproduced against every existing coefficient, SE, OR, confidence limit and p value before fitting the expanded model. Both models converged. All required M1 covariates were observed in this input; no additional participants were excluded. Full diagnostics, software versions and input hashes are in the accompanying metadata JSON.", "",
               "The fourth category combines other/unknown encounters with missing setting; it is an administrative analysis category, not a distinct disease phenotype. This sensitivity retains those participants in a joint multinomial model, but does not recover their true acute/outpatient settings, establish that setting missingness is random, remove possible informative missingness or assess subtype composition. Similar known-setting contrasts therefore indicate stability under this expanded outcome specification, not proof that exclusion bias is absent. The model retains the multinomial-logit specification and its assumptions. The unclassified-versus-control association should not be interpreted as a new clinical presentation endpoint.", "",
               "This analysis does not rerun PCA or mediation because its question is the robustness of the primary Steps–setting association to retaining unclassified CVD records. S7_Table.tex is regenerated with the aggregate outputs by this script.", ""]
    (directory / "unclassified_setting_sensitivity_report.md").write_text("\n".join(report))
    print("Input and model validation passed; aggregate outputs written.")
    print(steps[["model", "contrast", "OR", "CI_lower", "CI_upper", "p_value", "n"]].to_string(index=False))
    print(binary_steps[["cohort", "n", "n_cvd", "n_no_recorded_cvd", "OR", "CI_lower", "CI_upper", "p_value"]].to_string(index=False))
    print(descriptive.to_string(index=False))
    print(json.dumps(metadata["model_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
