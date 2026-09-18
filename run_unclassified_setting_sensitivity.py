#!/usr/bin/env python3
"""Assess the 603 CVD records without a classified presentation setting.

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

BASE_DIR = Path(__file__).resolve().parent
TERM_LABELS = {
    "Intercept": "Intercept",
    "steps_z": "Mean daily steps, per original pooled SD",
    "age_z": "Age, per original pooled SD",
    "C(sex_at_birth)[T.Male]": "Male sex",
    "C(is_smoker)[T.1]": "Smoking code 1",
    "C(is_drinker)[T.1]": "Drinking code 1",
    "C(is_drinker)[T.2]": "Drinking code 2",
}
CATEGORY_LABELS = {0: "Control", 1: "Outpatient", 2: "Acute", 3: "Unclassified"}
CONTRASTS = (
    (1, 0, "Outpatient vs control"),
    (2, 0, "Acute vs control"),
    (2, 1, "Acute vs outpatient"),
    (3, 0, "Unclassified vs control"),
)


def load_data(directory):
    data = pd.read_csv(directory / "final_analytic_cohort_with_habits.csv")
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
        ("Smoking code 1", "is_smoker", 1),
        ("Drinking code 0", "is_drinker", 0),
        ("Drinking code 1", "is_drinker", 1),
        ("Drinking code 2", "is_drinker", 2),
    ):
        cases[name] = cases[source].eq(level).astype(float).where(cases[source].notna())
    retained = cases.loc[cases["outcome"].isin([1, 2])]
    excluded = cases.loc[cases["outcome"].eq(3)]
    rows = []
    for column, label, kind in (
        ("age", "Age, years", "continuous"),
        ("avg_steps", "Mean daily steps, steps/day", "continuous"),
        *[(name, name, "binary") for name in (
            "Female sex", "Smoking code 1", "Drinking code 0",
            "Drinking code 1", "Drinking code 2",
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


def write_supporting_table(directory, descriptive, steps, metadata, n_unknown, n_missing):
    """Render S7 directly from the aggregate outputs used by the manuscript."""
    def p_text(value):
        return "$<0.001$" if value < 0.001 else f"{value:.3f}"

    def estimate_text(row):
        return f"{row.OR:.3f} ({row.CI_lower:.3f}--{row.CI_upper:.3f})"

    n_classified = int(descriptive.iloc[0].retained_n_available + descriptive.iloc[0].retained_n_missing)
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
        r"{\small\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\linewidth}{@{}>{\raggedright\arraybackslash}Xrrrr@{}}",
        r"\toprule",
        f"Characteristic & \\shortstack{{Classified\\\\$N={n_classified}$}} & \\shortstack{{Unclassified\\\\$N={n_unclassified}$}} & SMD & \\shortstack{{Missing\\\\C/U}} " + r"\\",
        r"\midrule",
    ]
    for row in descriptive.itertuples():
        if row.type == "continuous":
            first = f"{row.retained_mean:.1f} ({row.retained_sd:.1f})"
            second = f"{row.unclassified_mean:.1f} ({row.unclassified_sd:.1f})"
        else:
            first = f"{int(row.retained_count)} ({row.retained_percent:.1f}\\%)"
            second = f"{int(row.unclassified_count)} ({row.unclassified_percent:.1f}\\%)"
        lines.append(f"{row.variable} & {first} & {second} & ${row.SMD_unclassified_minus_retained:.3f}$ & {row.retained_n_missing}/{row.unclassified_n_missing} " + r"\\")
    lines.extend([
        r"\bottomrule\end{tabularx}\par}",
        r"{\footnotesize Values are mean (SD) or $n$ (\%). SMD is the standardised mean difference, unclassified minus classified, using the square root of the mean of the two group variances; binary variables use Bernoulli variances. Each alcohol category is a separate indicator. Missing C/U gives the missing count in classified/unclassified cases for each variable. Percentages use available observations. No descriptive null-hypothesis tests were performed.\par}",
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
        label = contrast.replace("Acute", "Acute care")
        first = f"{estimate_text(one)} & {p_text(one.p_value)}" if one is not None else "--- & ---"
        lines.append(f"{label} & {first} & {estimate_text(two)} & {p_text(two.p_value)} " + r"\\")
    row = expanded.iloc[0]
    lines.extend([
        r"\bottomrule\end{tabularx}\par}",
        r"{\footnotesize",
        f"ORs are per 1 original pooled SD of mean daily steps ({row.steps_SD:.2f} steps/day). Both multinomial models adjust for age, sex, and the exported smoking and drinking codes using the same original M1 means and SDs; age SD is {row.age_SD:.2f} years. Female sex, smoking code 0, and drinking code 0 are predictor reference categories. The investigator-confirmed monthly-frequency cutpoints and source-mapping limitations are given in S1 Text. Controls are the outcome reference. Acute-care versus outpatient estimates use the jointly estimated coefficients and cross-logit covariance, not a separate binary model. Intervals are two-sided Wald 95\\% CIs.",
        f"The three-category model includes {int(row.n_control)} controls, {int(row.n_outpatient)} outpatient recordings, and {int(row.n_acute)} acute-care recordings. The expanded model adds {n_unclassified} CVD records ({metadata['unclassified_percent_of_all_cvd']:.1f}\\% of all CVD cases): {n_unknown} other/unknown and {n_missing} missing settings. All model covariates were observed; no additional participants were excluded. The dataset was already restricted upstream to the $t_1$-landmark cohort; no further landmark exclusions were applied.",
        f"Unclassified is an administrative analysis category, not a new clinical phenotype or an imputation of the true recording setting. Similar known-setting associations support stability under this expanded outcome specification, but do not assess sensitivity to the unknown acute-care or outpatient assignments of these {n_unclassified} cases, establish random setting missingness, or rule out selection bias. PCA and mediation were not re-estimated in this sensitivity analysis. CI=confidence interval; CVD=cardiovascular disease; OR=odds ratio; SD=standard deviation.",
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
    descriptive = descriptive_comparison(data)
    results = pd.concat([original, expanded], ignore_index=True)
    steps = results.loc[results["term"].eq("steps_z")].copy()
    counts = data.groupby(["Group", "Onset_Type"], dropna=False).size().reset_index(name="n")
    counts["Onset_Type"] = counts["Onset_Type"].fillna("Missing setting")
    results.to_csv(directory / "unclassified_setting_multinomial_results.csv", index=False)
    steps.to_csv(directory / "unclassified_setting_steps_comparison.csv", index=False)
    descriptive.to_csv(directory / "unclassified_setting_case_characteristics.csv", index=False)
    counts.to_csv(directory / "unclassified_setting_counts.csv", index=False)
    metadata = {
        "input_n": len(data), "primary_n": int(data["outcome"].ne(3).sum()),
        "unclassified_n": int(data["outcome"].eq(3).sum()),
        "unclassified_percent_of_all_cvd": 100 * data["outcome"].eq(3).sum() / data["outcome"].ne(0).sum(),
        "standardization": parameters.reset_index().to_dict(orient="records"),
        "model_diagnostics": [diagnostic_original, diagnostic_expanded],
        "original_M1_reproduction_max_absolute_errors": reproduction_errors,
        "descriptive_SMD": "Unclassified minus retained; continuous pooled unweighted group variances; binary pooled Bernoulli variances; available-case denominators.",
        "confidence_intervals": "Two-sided Wald 95% intervals with critical value 1.96, matching original M1, using the full jointly estimated multinomial covariance including cross-logit covariance for acute vs outpatient.",
        "input_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in (
            "final_analytic_cohort_with_habits.csv", "m1_standardization_parameters.csv", "m1_primary_pairwise_results_python.csv")},
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "scope": "Already upstream-restricted participants; no further temporal screening, no mediation re-estimation, no assignment of true setting to unclassified cases.",
    }
    (directory / "unclassified_setting_sensitivity_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    n_cases = int(data["outcome"].ne(0).sum())
    n_classified = int(data["outcome"].isin([1, 2]).sum())
    n_unclassified = int(data["outcome"].eq(3).sum())
    n_unknown = int((data["outcome"].eq(3) & data["Onset_Type"].eq("Other/Unknown")).sum())
    n_missing = int((data["outcome"].eq(3) & data["Onset_Type"].isna()).sum())
    write_supporting_table(directory, descriptive, steps, metadata, n_unknown, n_missing)
    report = ["# Unclassified-setting sensitivity", "",
              "Status: approved for manuscript inclusion; the reproducible supporting-information table is S7_Table.tex.", "",
              f"The exported cohort includes {len(data):,} participants. Of {n_cases:,} CVD cases, {n_classified:,} have classified settings and {n_unclassified:,} ({100 * n_unclassified / n_cases:.2f}%) are unclassified: {n_unknown:,} other/unknown and {n_missing:,} missing setting. No additional upstream landmark exclusions were applied.", "",
              "## Steps associations", "",
              "All estimates use the original M1 scale: 1 SD = 3298.822232 steps/day. Adjustment: age, sex, and the exported smoking and drinking codes; control is the outcome reference. Female sex, smoking code 0, and drinking code 0 are predictor references. Investigator-confirmed monthly-frequency cutpoints and source-mapping limitations are documented in S1 Text; the original source questions remain unavailable for independent verification.", "",
              "| Model | Contrast | OR (95% CI) | p |", "|---|---|---|---|"]
    for row in steps.itertuples():
        report.append(f"| {row.model} | {row.contrast} | {row.OR:.4f} ({row.CI_lower:.4f}, {row.CI_upper:.4f}) | {row.p_value:.4g} |")
    report += ["", "## Descriptive comparison of CVD cases", "",
               "Positive standardized mean differences (SMDs) indicate larger values/proportions in unclassified cases. No null-hypothesis tests were performed. Each alcohol category is compared as an indicator, not as a single omnibus SMD.", "",
               f"| Characteristic | Classified (n={n_classified}) | Unclassified (n={n_unclassified}) | SMD |", "|---|---|---|---|"]
    for row in descriptive.itertuples():
        if row.type == "continuous":
            first, second = f"{row.retained_mean:.1f} ({row.retained_sd:.1f})", f"{row.unclassified_mean:.1f} ({row.unclassified_sd:.1f})"
        else:
            first, second = f"{int(row.retained_count)} ({row.retained_percent:.1f}%)", f"{int(row.unclassified_count)} ({row.unclassified_percent:.1f}%)"
        report.append(f"| {row.variable} | {first} | {second} | {row.SMD_unclassified_minus_retained:.3f} |")
    primary_setting = steps.loc[steps["model"].eq("Reproduced three-category M1") & steps["contrast"].eq("Acute vs outpatient")].iloc[0]
    expanded_setting = steps.loc[steps["model"].eq("Four-category unclassified-setting sensitivity") & steps["contrast"].eq("Acute vs outpatient")].iloc[0]
    step_characteristic = descriptive.loc[descriptive["variable"].eq("Mean daily steps, steps/day")].iloc[0]
    report += ["", "## Interpretation", "",
               f"The acute-versus-outpatient Steps OR changes from {primary_setting.OR:.3f} to {expanded_setting.OR:.3f}; the expanded-model interval remains below one ({expanded_setting.CI_lower:.3f}, {expanded_setting.CI_upper:.3f}). The known-setting contrasts are numerically similar after retaining unclassified records. However, mean daily steps are lower in unclassified than classified cases ({step_characteristic.unclassified_mean:.1f} vs {step_characteristic.retained_mean:.1f}; SMD {step_characteristic.SMD_unclassified_minus_retained:.3f}), so comparable recorded characteristics or random exclusion must not be assumed.", "",
               "## Verification and limits", "",
               "The three-category M1 was reproduced against every existing coefficient, SE, OR, confidence limit and p value before fitting the expanded model. Both models converged. All required M1 covariates were observed in this input; no additional participants were excluded. Full diagnostics, software versions and input hashes are in the accompanying metadata JSON.", "",
               "The fourth category combines other/unknown encounters with missing setting; it is an administrative analysis category, not a distinct disease phenotype. This sensitivity retains those participants in a joint multinomial model, but does not recover their true acute/outpatient settings, establish that setting missingness is random, remove possible informative missingness or assess subtype composition. Similar known-setting contrasts therefore indicate stability under this expanded outcome specification, not proof that exclusion bias is absent. The model retains the multinomial-logit specification and its assumptions. The unclassified-versus-control association should not be interpreted as a new clinical presentation endpoint.", "",
               "This analysis does not rerun PCA or mediation because its question is the robustness of the primary Steps–setting association to retaining unclassified CVD records. S7_Table.tex is regenerated with the aggregate outputs by this script.", ""]
    (directory / "unclassified_setting_sensitivity_report.md").write_text("\n".join(report))
    print("Input and model validation passed; aggregate outputs written.")
    print(steps[["model", "contrast", "OR", "CI_lower", "CI_upper", "p_value", "n"]].to_string(index=False))
    print(descriptive.to_string(index=False))
    print(json.dumps(metadata["model_diagnostics"], indent=2))


if __name__ == "__main__":
    main()
