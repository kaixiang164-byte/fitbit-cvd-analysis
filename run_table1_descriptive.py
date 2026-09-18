#!/usr/bin/env python3
"""New Table 1 generator from documented, already-prepared cohort fields.

This release addition is not a recovered historical script. It uses the current
table's sample SDs and column-total denominators; it does not reconstruct
eligibility, validate measurement dates or fit models. Run in an authorised
Workbench environment only. Aggregate outputs still need disclosure review.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


CODE_DIRECTORY = Path(__file__).resolve().parent
EXPECTED_COUNTS = {"Control": 5780, "Outpatient": 2127, "Acute": 385}
SETTINGS = {"Chronic (Office/Outpatient)": "Outpatient", "Acute (Hospital/ER)": "Acute"}
CLINICAL_FLAGS = (
    "has_hypertension", "has_diabetes", "has_hyperlipidemia",
    "has_high_cholesterol", "has_ckd",
)
BINARY_FLAGS = ("has_sleep_disorder", "has_depression", "has_anxiety") + CLINICAL_FLAGS
REQUIRED_COLUMNS = (
    "person_id", "Group", "Onset_Type", "age", "avg_steps",
    "sex_at_birth", "is_smoker", "is_drinker",
) + BINARY_FLAGS

# window, input/derived field, label, counted value (None means mean/sample SD)
ROWS = (
    ("pre_t0", "age", "Age at t0, years", None),
    ("pre_t0", "sex_at_birth", "Female sex", "Female"),
    ("pre_t0", "is_smoker", "Smoking, >3/month", 1),
    ("pre_t0", "is_drinker", "Drinking, <=1/month", 0),
    ("pre_t0", "is_drinker", "Drinking, >1 to 10/month", 1),
    ("pre_t0", "is_drinker", "Drinking, >10/month", 2),
    ("pre_t0", "avg_steps", "Mean daily step count during [t0-12m,t0)", None),
    ("mediator_month", "has_sleep_disorder", "Low sleep-efficiency indicator", 1),
    ("mediator_month", "has_depression", "Self-reported depression diagnosis", 1),
    ("mediator_month", "has_anxiety", "Self-reported anxiety diagnosis", 1),
    ("mediator_month", "Mental", "Mental burden score, 0-2", None),
    ("mediator_month", "has_hypertension", "Recorded hypertension", 1),
    ("mediator_month", "has_diabetes", "Recorded diabetes", 1),
    ("mediator_month", "has_hyperlipidemia", "Recorded hyperlipidaemia", 1),
    ("mediator_month", "has_high_cholesterol", "Recorded high cholesterol", 1),
    ("mediator_month", "has_ckd", "Recorded chronic kidney disease", 1),
    ("mediator_month", "Clinical", "Clinical burden score, 0-5", None),
)


def prepare_classified_cohort(cohort, expected_counts=None):
    """Validate without silently imputing or dropping classified participants."""
    missing = set(REQUIRED_COLUMNS) - set(cohort.columns)
    if missing:
        raise ValueError("Missing required Table 1 fields: " + ", ".join(sorted(missing)))
    data = cohort.loc[:, list(REQUIRED_COLUMNS)].copy()
    if data.person_id.isna().any() or not data.person_id.is_unique:
        raise ValueError("Each supplied row needs a nonmissing unique person_id.")
    if not data.Group.isin(["Control", "Heart Disease"]).all():
        raise ValueError("Group must be Control or the legacy Heart Disease label.")
    controls = data.Group.eq("Control")
    allowed = set(SETTINGS) | {"Other/Unknown", "N/A (Control)"}
    if not (data.Onset_Type.isna() | data.Onset_Type.isin(allowed)).all():
        raise ValueError("Unexpected recording-setting label.")
    if (controls & data.Onset_Type.isin(set(SETTINGS) | {"Other/Unknown"})).any():
        raise ValueError("Control rows cannot have a CVD recording setting.")
    if ((~controls) & data.Onset_Type.eq("N/A (Control)")).any():
        raise ValueError("CVD rows cannot carry the control setting label.")
    data["outcome"] = data.Onset_Type.map(SETTINGS)
    data.loc[controls, "outcome"] = "Control"
    excluded = int(data.outcome.isna().sum())
    data = data.loc[data.outcome.notna()].copy()
    complete_fields = ["age", "avg_steps", "sex_at_birth", "is_smoker", "is_drinker"] + list(BINARY_FLAGS)
    if data[complete_fields].isna().any().any():
        raise ValueError("Classified participants have missing required values; no imputation is allowed.")
    if not data.sex_at_birth.isin(["Female", "Male"]).all():
        raise ValueError("Unexpected sex-at-birth category in the prepared analytic cohort.")
    for field in ("age", "avg_steps", "is_smoker", "is_drinker") + BINARY_FLAGS:
        data[field] = pd.to_numeric(data[field], errors="raise")
        if not np.isfinite(data[field].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite values in required field: " + field)
    for field in ("is_smoker",) + BINARY_FLAGS:
        if not data[field].isin([0, 1]).all():
            raise ValueError("Expected complete binary 0/1 values: " + field)
    if not data.is_drinker.isin([0, 1, 2]).all():
        raise ValueError("Drinking must use the prepared 0/1/2 categories.")
    if (data.age < 0).any() or (data.avg_steps < 0).any():
        raise ValueError("Age and mean step count cannot be negative.")
    counts = data.outcome.value_counts().to_dict()
    if set(counts) != set(EXPECTED_COUNTS) or min(counts.values()) < 2:
        raise ValueError("Each of the three outcomes needs at least two observations for sample SDs.")
    if expected_counts is not None and counts != expected_counts:
        raise ValueError("Outcome counts differ from expected Table 1 cohort; investigate before reporting.")
    data["Mental"] = data.has_depression + data.has_anxiety
    data["Clinical"] = data.loc[:, list(CLINICAL_FLAGS)].sum(axis=1)
    return data, excluded


def summarize_table1(cohort, expected_counts=None):
    """Return numeric aggregate summaries and counts, never participant IDs."""
    data, excluded = prepare_classified_cohort(cohort, expected_counts)
    summaries = []
    for order, (window, field, label, value) in enumerate(ROWS, start=1):
        for outcome in EXPECTED_COUNTS:
            series = data.loc[data.outcome.eq(outcome), field]
            item = {
                "row_order": order, "measurement_window": window, "field": field,
                "characteristic": label, "outcome": outcome, "denominator": int(series.size),
                "summary_type": "mean_sd" if value is None else "n_percent",
                "count": None, "percent": None, "mean": None, "sd": None,
            }
            if value is None:
                item["mean"], item["sd"] = float(series.mean()), float(series.std(ddof=1))
                item["display"] = f"{item['mean']:.1f} ({item['sd']:.1f})"
            else:
                item["count"] = int(series.eq(value).sum())
                item["percent"] = 100 * item["count"] / series.size
                item["display"] = f"{item['count']} ({item['percent']:.1f}%)"
            summaries.append(item)
    metadata = {
        "input_n": int(len(cohort)), "classified_n": int(len(data)),
        "excluded_unclassified_setting_n": excluded,
        "outcome_counts": {group: int((data.outcome == group).sum()) for group in EXPECTED_COUNTS},
        "standard_deviation_ddof": 1, "percent_denominator": "full outcome column total",
        "provenance": "New generator from documented Table 1 definitions; not historical script recovery.",
        "verification_boundary": "No extraction, date verification or independent manuscript-value reproduction.",
    }
    return pd.DataFrame(summaries), metadata


def render_table1_tex(summary, metadata):
    """Render current documented rows; values need review in the Workbench."""
    labels = {
        "Age at t0, years": r"Age at $t_0$, years",
        "Smoking, >3/month": r"Smoking, $>3$/month",
        "Drinking, <=1/month": r"Drinking, $\leq1$/month",
        "Drinking, >1 to 10/month": r"Drinking, $>1$ to $10$/month",
        "Drinking, >10/month": r"Drinking, $>10$/month",
        "Mean daily step count during [t0-12m,t0)": r"Mean daily step count during $[t_0-12m,t_0)$",
        "Mental burden score, 0-2": "Mental burden score, 0--2",
        "Clinical burden score, 0-5": "Clinical burden score, 0--5",
    }
    lines = [
        r"\begin{table}[htbp]", r"\centering",
        r"\caption{\textbf{Participant characteristics by cardiovascular recording and landmark measurement window}}",
        r"\label{tab:baseline}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}Xrrr@{}}", r"\toprule",
        "Characteristic & " + " & ".join(
            f"{group} ($N={metadata['outcome_counts'][group]}$)" for group in EXPECTED_COUNTS
        ) + r" \\", r"\midrule",
    ]
    previous = None
    for _, rows in summary.groupby("row_order", sort=True):
        row = rows.iloc[0]
        if row.measurement_window != previous:
            if previous is not None:
                lines.append(r"\addlinespace")
            title = (r"Before $t_0$: adjustment variables and Fitbit exposure"
                     if row.measurement_window == "pre_t0" else
                     r"$[t_0,t_M)$: health burden components (first post-baseline month)")
            lines.append(r"\multicolumn{4}{@{}l}{\textit{" + title + r"}} \\")
            previous = row.measurement_window
        cells = rows.set_index("outcome").loc[list(EXPECTED_COUNTS), "display"]
        lines.append(labels.get(row.characteristic, row.characteristic) + " & " +
                     " & ".join(value.replace("%", r"\%") for value in cells) + r" \\")
    lines += [
        r"\bottomrule", r"\end{tabularx}", r"\begin{minipage}{\textwidth}", r"\footnotesize",
        r"Data are mean (sample SD) or n (\%). Percentages use the full column total; missing required values are not recoded as absence. "
        r"Mental sums depression/anxiety survey indicators; Clinical sums five recorded diagnosis indicators. "
        r"Smoking and drinking frequencies are occasions per month. Source-variable definitions are given in S1 Text. "
        r"CVD=cardiovascular disease.", r"\end{minipage}", r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def generate_table1(prepared_cohort, output_dir, expected_counts=None):
    """Write only aggregates to a new directory outside the source tree."""
    source = Path(prepared_cohort).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    if destination == CODE_DIRECTORY or CODE_DIRECTORY in destination.parents:
        raise ValueError("Runtime output must remain outside the code release directory.")
    if destination.exists():
        raise FileExistsError("Output directory already exists; refusing to overwrite.")
    if source == CODE_DIRECTORY or CODE_DIRECTORY in source.parents:
        raise ValueError("Participant input must be outside the code release directory.")
    summary, metadata = summarize_table1(pd.read_csv(source), expected_counts)
    latex = render_table1_tex(summary, metadata)
    destination.mkdir(parents=True, exist_ok=False)
    summary.to_csv(destination / "table1_descriptive_summary.csv", index=False)
    (destination / "table1_descriptive.tex").write_text(latex, encoding="utf-8")
    (destination / "table1_generation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return summary, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-cohort", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_table1(args.prepared_cohort, args.output_dir, expected_counts=EXPECTED_COUNTS)
    print("Generated aggregate Table 1 files. Review disclosure and historical-value agreement before export.")


if __name__ == "__main__":
    main()
