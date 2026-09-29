import os
import sys
import pandas as pd
import numpy as np

# ============================================================
# STEP 61F
# FINAL ANALYSIS INTEGRITY AUDIT
# ============================================================

folder = os.path.expanduser("~/Documents/AKI_Research")

# ------------------------------------------------------------
# REQUIRED FINAL ARTIFACTS
# ------------------------------------------------------------

required_files = {
    # Core analysis
    "modeling_table":
        "longitudinal_modeling_table.csv",

    "predictors":
        "final_model_predictors.txt",

    "predictor_groups":
        "final_model_predictor_groups.csv",

    "split_manifest":
        "patient_level_split_manifest.csv",

    # Final test predictions
    "test_6h":
        "FINAL_TEST_6h_predictions.csv",

    "test_12h":
        "FINAL_TEST_12h_predictions.csv",

    "test_24h":
        "FINAL_TEST_24h_predictions.csv",

    # Bootstrap
    "bootstrap":
        "step57B_patient_bootstrap_metrics.csv",

    # Final results
    "primary_test":
        "step60_primary_test_performance.csv",

    "calibrated_test":
        "step60_calibrated_test_performance.csv",

    # Final figures/results
    "roc_png":
        "step61A_final_test_ROC.png",

    "roc_pdf":
        "step61A_final_test_ROC.pdf",

    "pr_png":
        "step61A_final_test_PR.png",

    "pr_pdf":
        "step61A_final_test_PR.pdf",

    "curve_points":
        "step61A_curve_points.csv",

    "calibration_original_png":
        "step61B_original_test_calibration.png",

    "calibration_original_pdf":
        "step61B_original_test_calibration.pdf",

    "calibration_recalibrated_png":
        "step61B_validation_calibrated_test_calibration.png",

    "calibration_recalibrated_pdf":
        "step61B_validation_calibrated_test_calibration.pdf",

    "calibration_comparison_png":
        "step61B_original_vs_calibrated_calibration.png",

    "calibration_comparison_pdf":
        "step61B_original_vs_calibrated_calibration.pdf",

    "calibration_points":
        "step61B_calibration_curve_points.csv",

    # Discrimination CI
    "discrimination_ci":
        "step61C_final_discrimination_CI.csv",

    "discrimination_ci_png":
        "step61C_final_discrimination_CI.png",

    "discrimination_ci_pdf":
        "step61C_final_discrimination_CI.pdf",

    # Ablation
    "ablation_summary":
        "step61D_validation_ablation_summary.csv",

    "ablation_png":
        "step61D_validation_ablation_comparison.png",

    "ablation_pdf":
        "step61D_validation_ablation_comparison.pdf",

    # Manuscript table
    "manuscript_table":
        "step61E_manuscript_performance_table.csv",

    "manuscript_report":
        "step61E_manuscript_performance_report.txt",
}

paths = {
    key: os.path.join(folder, filename)
    for key, filename in required_files.items()
}

output_report = os.path.join(
    folder,
    "step61F_final_analysis_integrity_report.txt"
)

output_csv = os.path.join(
    folder,
    "step61F_final_analysis_integrity_summary.csv"
)


# ============================================================
# FROZEN EXPECTED FINAL RESULTS
# ============================================================

expected = {
    "6h": {
        "rows": 64570,
        "patients": 8504,
        "positive": 5544,
        "negative": 59026,
        "auroc": 0.724717,
        "auprc": 0.198494,
        "auroc_low": 0.716089,
        "auroc_high": 0.732488,
        "auprc_low": 0.188588,
        "auprc_high": 0.208395,
    },

    "12h": {
        "rows": 58924,
        "patients": 8231,
        "positive": 9251,
        "negative": 49673,
        "auroc": 0.733356,
        "auprc": 0.328190,
        "auroc_low": 0.725454,
        "auroc_high": 0.742028,
        "auprc_low": 0.315229,
        "auprc_high": 0.341100,
    },

    "24h": {
        "rows": 49367,
        "patients": 7178,
        "positive": 13654,
        "negative": 35713,
        "auroc": 0.731931,
        "auprc": 0.503620,
        "auroc_low": 0.721598,
        "auroc_high": 0.741497,
        "auprc_low": 0.488614,
        "auprc_high": 0.519853,
    },
}

expected_ablation = {
    "6h": {
        "full": (0.734154, 0.209191),
        "vital_context": (0.732899, 0.202766),
        "exposure_context": (0.708511, 0.193789),
    },

    "12h": {
        "full": (0.745764, 0.352170),
        "vital_context": (0.733233, 0.329848),
        "exposure_context": (0.724171, 0.329232),
    },

    "24h": {
        "full": (0.747252, 0.523951),
        "vital_context": (0.738493, 0.508415),
        "exposure_context": (0.723065, 0.498405),
    },
}


# ============================================================
# HELPERS
# ============================================================

checks = []
report = []


def add_check(name, passed, detail=""):
    passed = bool(passed)

    checks.append({
        "check": name,
        "passed": passed,
        "detail": str(detail)
    })

    status = "PASS" if passed else "FAIL"

    line = f"{status}: {name}"

    if detail:
        line += f" | {detail}"

    print(line)
    report.append(line)

    return passed


def close(a, b, tol=5e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except Exception:
        return False


def normalize_horizon(value):
    s = str(value).strip().lower()

    mapping = {
        "6": "6h",
        "6.0": "6h",
        "6h": "6h",

        "12": "12h",
        "12.0": "12h",
        "12h": "12h",

        "24": "24h",
        "24.0": "24h",
        "24h": "24h",
    }

    return mapping.get(s)


def find_column(df, candidates):
    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("STEP 61F — FINAL ANALYSIS INTEGRITY AUDIT")
print("=" * 80)

print("")
print("READ-ONLY AUDIT OF THE COMPLETED ANALYSIS.")
print("NO MODEL WILL BE RETRAINED.")
print("NO CALIBRATION MODEL WILL BE FIT.")
print("NO THRESHOLD WILL BE SELECTED OR CHANGED.")
print("NO TEST PREDICTION WILL BE MODIFIED.")
print("")

report.extend([
    "STEP 61F — FINAL ANALYSIS INTEGRITY AUDIT",
    "=" * 80,
    "",
    "READ-ONLY AUDIT OF THE COMPLETED ANALYSIS.",
    ""
])


# ============================================================
# 1. REQUIRED FILE EXISTENCE
# ============================================================

print("")
print("=" * 80)
print("1. REQUIRED FINAL ARTIFACTS")
print("=" * 80)

report.extend([
    "",
    "1. REQUIRED FINAL ARTIFACTS",
    "-" * 80
])

for key, path in paths.items():

    exists = os.path.isfile(path)

    size = (
        os.path.getsize(path)
        if exists
        else 0
    )

    add_check(
        f"{key} exists",
        exists,
        path
    )

    if exists:
        add_check(
            f"{key} non-empty",
            size > 0,
            f"{size:,} bytes"
        )


# Stop before deeper analysis if required artifacts missing.

missing = [
    key
    for key, path in paths.items()
    if not os.path.isfile(path)
]

if missing:

    report.append("")
    report.append(
        "SAFETY STOP — REQUIRED ARTIFACTS MISSING:"
    )

    for item in missing:
        report.append(item)

    pd.DataFrame(checks).to_csv(
        output_csv,
        index=False
    )

    with open(
        output_report,
        "w",
        encoding="utf-8"
    ) as f:
        f.write("\n".join(report))

    print("")
    print("SAFETY STOP.")
    print("Missing:")
    for item in missing:
        print(item)

    raise RuntimeError(
        "STEP 61F stopped because required final artifacts are missing."
    )


# ============================================================
# 2. FINAL PREDICTION FILE INTEGRITY
# ============================================================

print("")
print("=" * 80)
print("2. FINAL TEST PREDICTION INTEGRITY")
print("=" * 80)

report.extend([
    "",
    "2. FINAL TEST PREDICTION INTEGRITY",
    "-" * 80
])

prediction_paths = {
    "6h": paths["test_6h"],
    "12h": paths["test_12h"],
    "24h": paths["test_24h"],
}

prediction_summary = {}

for horizon, path in prediction_paths.items():

    df = pd.read_csv(path)

    exp = expected[horizon]

    print("")
    print(horizon)

    add_check(
        f"{horizon} prediction row count",
        len(df) == exp["rows"],
        f"{len(df):,} expected {exp['rows']:,}"
    )

    subject_col = find_column(
        df,
        ["subject_id", "patient_id"]
    )

    if subject_col is None:
        add_check(
            f"{horizon} patient identifier present",
            False,
            "subject_id/patient_id not found"
        )
        patients = None

    else:
        patients = df[subject_col].nunique()

        add_check(
            f"{horizon} unique patient count",
            patients == exp["patients"],
            f"{patients:,} expected {exp['patients']:,}"
        )

    label_col = find_column(
        df,
        [
            "label",
            "y_true",
            "target",
            "outcome",
            f"aki_within_{horizon}"
        ]
    )

    if label_col is None:

        # Try any AKI target column
        aki_cols = [
            c for c in df.columns
            if str(c).lower().startswith("aki_within_")
        ]

        if len(aki_cols) == 1:
            label_col = aki_cols[0]

    add_check(
        f"{horizon} outcome column identified",
        label_col is not None,
        str(label_col)
    )

    prob_col = find_column(
        df,
        [
            "predicted_probability",
            "probability",
            "prediction",
            "y_prob",
            "prob"
        ]
    )

    if prob_col is None:

        probability_candidates = [
            c for c in df.columns
            if (
                "prob" in str(c).lower()
                or "pred" in str(c).lower()
            )
        ]

        numeric_candidates = []

        for c in probability_candidates:
            values = pd.to_numeric(
                df[c],
                errors="coerce"
            )

            if (
                values.notna().sum() > 0
                and values.dropna().between(0, 1).all()
            ):
                numeric_candidates.append(c)

        if len(numeric_candidates) == 1:
            prob_col = numeric_candidates[0]

    add_check(
        f"{horizon} probability column identified",
        prob_col is not None,
        str(prob_col)
    )

    if label_col is None or prob_col is None:
        continue

    y = pd.to_numeric(
        df[label_col],
        errors="coerce"
    )

    p = pd.to_numeric(
        df[prob_col],
        errors="coerce"
    )

    add_check(
        f"{horizon} labels nonmissing",
        y.notna().all(),
        f"missing={int(y.isna().sum())}"
    )

    add_check(
        f"{horizon} labels binary",
        y.dropna().isin([0, 1]).all(),
        f"unique={sorted(y.dropna().unique().tolist())}"
    )

    add_check(
        f"{horizon} probabilities nonmissing",
        p.notna().all(),
        f"missing={int(p.isna().sum())}"
    )

    add_check(
        f"{horizon} probabilities finite",
        np.isfinite(p).all(),
        ""
    )

    add_check(
        f"{horizon} probabilities within [0,1]",
        p.between(0, 1).all(),
        f"min={p.min():.8f}; max={p.max():.8f}"
    )

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    add_check(
        f"{horizon} positive count",
        positives == exp["positive"],
        f"{positives:,} expected {exp['positive']:,}"
    )

    add_check(
        f"{horizon} negative count",
        negatives == exp["negative"],
        f"{negatives:,} expected {exp['negative']:,}"
    )

    prediction_summary[horizon] = {
        "rows": len(df),
        "patients": patients,
        "positive": positives,
        "negative": negatives,
    }


# ============================================================
# 3. BOOTSTRAP INTEGRITY
# ============================================================

print("")
print("=" * 80)
print("3. PATIENT-LEVEL BOOTSTRAP INTEGRITY")
print("=" * 80)

report.extend([
    "",
    "3. PATIENT-LEVEL BOOTSTRAP INTEGRITY",
    "-" * 80
])

boot = pd.read_csv(paths["bootstrap"])

add_check(
    "Bootstrap total rows",
    len(boot) == 3000,
    f"{len(boot):,} expected 3,000"
)

hcol = find_column(
    boot,
    ["horizon"]
)

auroc_col = find_column(
    boot,
    ["AUROC", "auroc"]
)

auprc_col = find_column(
    boot,
    ["AUPRC", "auprc"]
)

add_check(
    "Bootstrap horizon column",
    hcol is not None,
    str(hcol)
)

add_check(
    "Bootstrap AUROC column",
    auroc_col is not None,
    str(auroc_col)
)

add_check(
    "Bootstrap AUPRC column",
    auprc_col is not None,
    str(auprc_col)
)

if (
    hcol is not None
    and auroc_col is not None
    and auprc_col is not None
):

    boot["normalized_horizon"] = (
        boot[hcol].apply(normalize_horizon)
    )

    add_check(
        "All bootstrap horizons recognized",
        boot["normalized_horizon"].notna().all(),
        ""
    )

    for horizon in ["6h", "12h", "24h"]:

        sub = boot.loc[
            boot["normalized_horizon"] == horizon
        ].copy()

        add_check(
            f"{horizon} bootstrap replicate count",
            len(sub) == 1000,
            f"{len(sub):,} expected 1,000"
        )

        auroc = pd.to_numeric(
            sub[auroc_col],
            errors="coerce"
        )

        auprc = pd.to_numeric(
            sub[auprc_col],
            errors="coerce"
        )

        add_check(
            f"{horizon} bootstrap AUROC finite",
            (
                auroc.notna().all()
                and np.isfinite(auroc).all()
            ),
            ""
        )

        add_check(
            f"{horizon} bootstrap AUPRC finite",
            (
                auprc.notna().all()
                and np.isfinite(auprc).all()
            ),
            ""
        )

        if len(sub) == 1000:

            al, ah = np.percentile(
                auroc,
                [2.5, 97.5]
            )

            pl, ph = np.percentile(
                auprc,
                [2.5, 97.5]
            )

            exp = expected[horizon]

            add_check(
                f"{horizon} AUROC CI lower",
                close(al, exp["auroc_low"]),
                f"{al:.6f} expected {exp['auroc_low']:.6f}"
            )

            add_check(
                f"{horizon} AUROC CI upper",
                close(ah, exp["auroc_high"]),
                f"{ah:.6f} expected {exp['auroc_high']:.6f}"
            )

            add_check(
                f"{horizon} AUPRC CI lower",
                close(pl, exp["auprc_low"]),
                f"{pl:.6f} expected {exp['auprc_low']:.6f}"
            )

            add_check(
                f"{horizon} AUPRC CI upper",
                close(ph, exp["auprc_high"]),
                f"{ph:.6f} expected {exp['auprc_high']:.6f}"
            )


# ============================================================
# 4. STEP 61C DISCRIMINATION SUMMARY
# ============================================================

print("")
print("=" * 80)
print("4. FINAL DISCRIMINATION CI TABLE")
print("=" * 80)

report.extend([
    "",
    "4. FINAL DISCRIMINATION CI TABLE",
    "-" * 80
])

ci_df = pd.read_csv(
    paths["discrimination_ci"]
)

add_check(
    "Step 61C CI table has 3 rows",
    len(ci_df) == 3,
    f"rows={len(ci_df)}"
)

print("Columns:", list(ci_df.columns))
report.append(
    "Columns: " + str(list(ci_df.columns))
)


# ============================================================
# 5. ABLATION SUMMARY INTEGRITY
# ============================================================

print("")
print("=" * 80)
print("5. VALIDATION ABLATION INTEGRITY")
print("=" * 80)

report.extend([
    "",
    "5. VALIDATION ABLATION INTEGRITY",
    "-" * 80
])

abl = pd.read_csv(
    paths["ablation_summary"]
)

add_check(
    "Ablation summary has 9 rows",
    len(abl) == 9,
    f"rows={len(abl)}"
)

hcol = find_column(
    abl,
    ["horizon"]
)

mcol = find_column(
    abl,
    ["model"]
)

acol = find_column(
    abl,
    ["AUROC", "auroc"]
)

pcol = find_column(
    abl,
    ["AUPRC", "auprc"]
)

for name, col in [
    ("horizon", hcol),
    ("model", mcol),
    ("AUROC", acol),
    ("AUPRC", pcol),
]:
    add_check(
        f"Ablation {name} column identified",
        col is not None,
        str(col)
    )

if all(
    x is not None
    for x in [hcol, mcol, acol, pcol]
):

    abl["_h"] = abl[hcol].apply(
        normalize_horizon
    )

    abl["_model"] = (
        abl[mcol]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    for horizon in [
        "6h",
        "12h",
        "24h"
    ]:

        for model in [
            "full",
            "vital_context",
            "exposure_context"
        ]:

            sub = abl.loc[
                (abl["_h"] == horizon)
                & (abl["_model"] == model)
            ]

            add_check(
                f"{horizon} {model} row unique",
                len(sub) == 1,
                f"rows={len(sub)}"
            )

            if len(sub) == 1:

                observed_auroc = float(
                    sub.iloc[0][acol]
                )

                observed_auprc = float(
                    sub.iloc[0][pcol]
                )

                expected_auroc, expected_auprc = (
                    expected_ablation[horizon][model]
                )

                add_check(
                    f"{horizon} {model} AUROC",
                    close(
                        observed_auroc,
                        expected_auroc
                    ),
                    (
                        f"{observed_auroc:.6f} "
                        f"expected {expected_auroc:.6f}"
                    )
                )

                add_check(
                    f"{horizon} {model} AUPRC",
                    close(
                        observed_auprc,
                        expected_auprc
                    ),
                    (
                        f"{observed_auprc:.6f} "
                        f"expected {expected_auprc:.6f}"
                    )
                )


# ============================================================
# 6. MANUSCRIPT TABLE INTEGRITY
# ============================================================

print("")
print("=" * 80)
print("6. MANUSCRIPT PERFORMANCE TABLE INTEGRITY")
print("=" * 80)

report.extend([
    "",
    "6. MANUSCRIPT PERFORMANCE TABLE INTEGRITY",
    "-" * 80
])

man = pd.read_csv(
    paths["manuscript_table"]
)

add_check(
    "Manuscript table has 3 rows",
    len(man) == 3,
    f"rows={len(man)}"
)

required_manuscript_columns = [
    "Prediction horizon",
    "Test landmarks",
    "Unique patients",
    "AKI events",
    "Non-events",
    "AKI prevalence (%)",
    "AUROC",
    "AUROC 95% CI lower",
    "AUROC 95% CI upper",
    "AUPRC",
    "AUPRC 95% CI lower",
    "AUPRC 95% CI upper",
]

for col in required_manuscript_columns:

    add_check(
        f"Manuscript column: {col}",
        col in man.columns,
        ""
    )

if all(
    col in man.columns
    for col in required_manuscript_columns
):

    for horizon in [
        "6h",
        "12h",
        "24h"
    ]:

        sub = man.loc[
            man["Prediction horizon"]
            .astype(str)
            .str.strip()
            .str.lower()
            == horizon
        ]

        add_check(
            f"{horizon} manuscript row unique",
            len(sub) == 1,
            f"rows={len(sub)}"
        )

        if len(sub) != 1:
            continue

        row = sub.iloc[0]
        exp = expected[horizon]

        manuscript_checks = [
            (
                "Test landmarks",
                int(row["Test landmarks"]),
                exp["rows"],
                0
            ),
            (
                "Unique patients",
                int(row["Unique patients"]),
                exp["patients"],
                0
            ),
            (
                "AKI events",
                int(row["AKI events"]),
                exp["positive"],
                0
            ),
            (
                "Non-events",
                int(row["Non-events"]),
                exp["negative"],
                0
            ),
            (
                "AUROC",
                row["AUROC"],
                exp["auroc"],
                5e-6
            ),
            (
                "AUPRC",
                row["AUPRC"],
                exp["auprc"],
                5e-6
            ),
            (
                "AUROC CI lower",
                row["AUROC 95% CI lower"],
                exp["auroc_low"],
                5e-6
            ),
            (
                "AUROC CI upper",
                row["AUROC 95% CI upper"],
                exp["auroc_high"],
                5e-6
            ),
            (
                "AUPRC CI lower",
                row["AUPRC 95% CI lower"],
                exp["auprc_low"],
                5e-6
            ),
            (
                "AUPRC CI upper",
                row["AUPRC 95% CI upper"],
                exp["auprc_high"],
                5e-6
            ),
        ]

        for label, observed, expected_value, tolerance in manuscript_checks:

            if tolerance == 0:
                passed = observed == expected_value
            else:
                passed = close(
                    observed,
                    expected_value,
                    tolerance
                )

            add_check(
                f"{horizon} manuscript {label}",
                passed,
                (
                    f"{observed} "
                    f"expected {expected_value}"
                )
            )


# ============================================================
# 7. PREDICTOR SPECIFICATION INTEGRITY
# ============================================================

print("")
print("=" * 80)
print("7. FINAL PREDICTOR SPECIFICATION")
print("=" * 80)

report.extend([
    "",
    "7. FINAL PREDICTOR SPECIFICATION",
    "-" * 80
])

with open(
    paths["predictors"],
    "r",
    encoding="utf-8"
) as f:
    predictors = [
        line.strip()
        for line in f
        if line.strip()
    ]

add_check(
    "Final predictor count",
    len(predictors) == 252,
    f"{len(predictors)} expected 252"
)

add_check(
    "Final predictors unique",
    len(set(predictors)) == len(predictors),
    (
        f"{len(set(predictors))} unique / "
        f"{len(predictors)} total"
    )
)

for forbidden in [
    "aki_within_6h",
    "aki_within_12h",
    "aki_within_24h",
    "definitive_aki_time",
    "hours_to_aki",
]:

    add_check(
        f"Forbidden predictor excluded: {forbidden}",
        forbidden not in predictors,
        ""
    )


groups = pd.read_csv(
    paths["predictor_groups"]
)

pred_col = find_column(
    groups,
    ["predictor"]
)

add_check(
    "Predictor-group predictor column",
    pred_col is not None,
    str(pred_col)
)

if pred_col is not None:

    group_predictors = (
        groups[pred_col]
        .astype(str)
        .str.strip()
        .tolist()
    )

    add_check(
        "Predictor-group row count",
        len(group_predictors) == 252,
        f"{len(group_predictors)} expected 252"
    )

    add_check(
        "Every master predictor represented in group file",
        set(predictors) == set(group_predictors),
        (
            f"master={len(set(predictors))}; "
            f"group={len(set(group_predictors))}"
        )
    )


# ============================================================
# 8. FINAL GLOBAL VERDICT
# ============================================================

print("")
print("=" * 80)
print("8. FINAL GLOBAL INTEGRITY VERDICT")
print("=" * 80)

report.extend([
    "",
    "8. FINAL GLOBAL INTEGRITY VERDICT",
    "-" * 80
])

check_df = pd.DataFrame(checks)

total_checks = len(check_df)

passed_checks = int(
    check_df["passed"].sum()
)

failed_checks = (
    total_checks - passed_checks
)

all_pass = (
    failed_checks == 0
)

print(
    f"Total checks: {total_checks}"
)

print(
    f"Passed: {passed_checks}"
)

print(
    f"Failed: {failed_checks}"
)

report.extend([
    f"Total checks: {total_checks}",
    f"Passed: {passed_checks}",
    f"Failed: {failed_checks}",
    ""
])

if all_pass:

    verdict = "PASS"

    completion_statement = (
        "The finalized primary analysis artifacts are internally "
        "consistent with the frozen held-out test results, "
        "patient-level bootstrap confidence intervals, validation "
        "ablation results, predictor specification, and manuscript "
        "performance table."
    )

else:

    verdict = "FAIL"

    completion_statement = (
        "At least one final integrity check failed. "
        "The failed check(s) must be investigated before the "
        "analysis is declared complete."
    )

report.append(
    f"FINAL STEP 61F VERDICT: {verdict}"
)

report.append("")
report.append(
    completion_statement
)

report.append("")
report.extend([
    "NO MODEL WAS RETRAINED.",
    "NO CALIBRATION MODEL WAS FIT.",
    "NO THRESHOLD WAS SELECTED OR CHANGED.",
    "NO TEST PREDICTION WAS MODIFIED.",
    "NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED.",
])


# ============================================================
# WRITE NEW AUDIT OUTPUTS ONLY
# ============================================================

check_df.to_csv(
    output_csv,
    index=False
)

with open(
    output_report,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(report))


# ============================================================
# FINAL CONSOLE
# ============================================================

print("")
print("=" * 80)
print(
    f"FINAL STEP 61F VERDICT: {verdict}"
)
print("=" * 80)

print("")
print(completion_statement)

print("")
print("Integrity summary CSV:")
print(output_csv)

print("")
print("Integrity report:")
print(output_report)

print("")
print("NO MODEL WAS RETRAINED.")
print("NO CALIBRATION MODEL WAS FIT.")
print("NO THRESHOLD WAS SELECTED OR CHANGED.")
print("NO TEST PREDICTION WAS MODIFIED.")
print("NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED.")

if not all_pass:

    print("")
    print("FAILED CHECKS:")

    failed = check_df.loc[
        ~check_df["passed"]
    ]

    for _, row in failed.iterrows():
        print(
            "-",
            row["check"],
            "|",
            row["detail"]
        )

    raise RuntimeError(
        "STEP 61F integrity audit failed. "
        "Review failed checks above."
    )
