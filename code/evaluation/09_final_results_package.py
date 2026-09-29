import os
import re
import csv
import numpy as np
import pandas as pd


# =============================================================================
# STEP 60 — FINAL MANUSCRIPT RESULTS PACKAGE
# =============================================================================
#
# PURPOSE
# -------
# Consolidate the completed locked-model analyses into manuscript-ready
# summary tables WITHOUT retraining, retuning, recalibrating, or modifying
# any prediction/model/research file.
#
# PRIMARY RESULTS:
#   Original locked held-out test predictions.
#
# SECONDARY CALIBRATION RESULTS:
#   Validation-derived calibration parameters applied unchanged to test.
#
# READ-ONLY INPUTS:
#   FINAL_TEST_summary.txt
#   step57B_patient_bootstrap_CI.txt
#   step58C_calibrated_bootstrap_CI.txt
#
# OPTIONAL SUPPORTING INPUTS:
#   step58B_validation_calibration_parameters.csv
#   step59C_coefficient_stability_report.txt
#
# OUTPUTS:
#   step60_primary_test_performance.csv
#   step60_calibrated_test_performance.csv
#   step60_final_results_report.txt
#
# IMPORTANT:
#   Original held-out test performance remains the PRIMARY model evaluation.
#   Validation-derived recalibration is a SECONDARY post-hoc analysis.
#
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")


# =============================================================================
# INPUT PATHS
# =============================================================================

final_test_summary_path = os.path.join(
    folder,
    "FINAL_TEST_summary.txt"
)

bootstrap_path = os.path.join(
    folder,
    "step57B_patient_bootstrap_CI.txt"
)

calibrated_bootstrap_path = os.path.join(
    folder,
    "step58C_calibrated_bootstrap_CI.txt"
)

calibration_parameter_path = os.path.join(
    folder,
    "step58B_validation_calibration_parameters.csv"
)

stability_report_path = os.path.join(
    folder,
    "step59C_coefficient_stability_report.txt"
)


# =============================================================================
# OUTPUT PATHS
# =============================================================================

primary_output_path = os.path.join(
    folder,
    "step60_primary_test_performance.csv"
)

calibrated_output_path = os.path.join(
    folder,
    "step60_calibrated_test_performance.csv"
)

report_output_path = os.path.join(
    folder,
    "step60_final_results_report.txt"
)


# =============================================================================
# EXPECTED LOCKED VALUES
#
# These values were already established during the locked one-time test
# evaluation. They are used ONLY as integrity checks.
# =============================================================================

expected = {

    "6h": {
        "rows": 64570,
        "patients": 8504,
        "threshold": 0.496171,
        "auroc": 0.724717,
        "auprc": 0.198494,
        "brier": 0.217222,
        "log_loss": 0.635285,
        "sensitivity": 0.681999,
        "specificity": 0.650154,
        "ppv": 0.154762,
        "npv": 0.956078,
        "f1": 0.252277,
    },

    "12h": {
        "rows": 58924,
        "patients": 8231,
        "threshold": 0.512416,
        "auroc": 0.733356,
        "auprc": 0.328190,
        "brier": 0.208371,
        "log_loss": 0.616046,
        "sensitivity": 0.641012,
        "specificity": 0.703742,
        "ppv": 0.287223,
        "npv": 0.913240,
        "f1": 0.396695,
    },

    "24h": {
        "rows": 49367,
        "patients": 7178,
        "threshold": 0.466857,
        "auroc": 0.731931,
        "auprc": 0.503620,
        "brier": 0.211482,
        "log_loss": 0.617875,
        "sensitivity": 0.715248,
        "specificity": 0.620278,
        "ppv": 0.418656,
        "npv": 0.850691,
        "f1": 0.528163,
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def read_text(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def normalize_metric_name(name):

    name = name.strip().lower()

    mapping = {
        "auroc": "auroc",
        "auprc": "auprc",
        "brier": "brier",
        "brier score": "brier",
        "log loss": "log_loss",
        "sensitivity": "sensitivity",
        "specificity": "specificity",
        "ppv": "ppv",
        "positive predictive value": "ppv",
        "npv": "npv",
        "negative predictive value": "npv",
        "f1": "f1",
        "f1 score": "f1",
    }

    return mapping.get(name)


def extract_ci_sections(text):

    """
    Extract metric point estimates + 95% CIs from the Step 57B report.

    Expected line style:
      AUROC: 0.724717 (95% CI 0.716089 to 0.732488)
    """

    results = {
        "6h": {},
        "12h": {},
        "24h": {},
    }

    current_horizon = None

    horizon_pattern = re.compile(
        r"^\s*(6H|12H|24H)\b",
        re.IGNORECASE
    )

    metric_pattern = re.compile(
        r"^\s*"
        r"(AUROC|AUPRC|Brier(?: score)?|Log loss|"
        r"Sensitivity|Specificity|PPV|NPV|F1(?: score)?)"
        r"\s*:\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*"
        r"\(95%\s*CI\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*(?:to|[-–—])\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\)",
        re.IGNORECASE
    )

    for line in text.splitlines():

        horizon_match = horizon_pattern.match(line)

        if horizon_match:

            token = horizon_match.group(1).lower()

            if token == "6h":
                current_horizon = "6h"
            elif token == "12h":
                current_horizon = "12h"
            elif token == "24h":
                current_horizon = "24h"

        metric_match = metric_pattern.match(line)

        if metric_match and current_horizon is not None:

            metric = normalize_metric_name(
                metric_match.group(1)
            )

            if metric is None:
                continue

            results[current_horizon][metric] = {
                "estimate": float(metric_match.group(2)),
                "ci_lower": float(metric_match.group(3)),
                "ci_upper": float(metric_match.group(4)),
            }

    return results


def extract_calibrated_ci_sections(text):

    """
    Extract calibrated test metrics from Step 58C.

    Expected examples:
      Observed rate: 0.085860 (95% CI ...)
      Mean predicted probability: ...
      E/O ratio: ...
      Calibration-in-the-large: ...
      Calibration intercept: ...
      Calibration slope: ...
      Brier: ...
      Log loss: ...
    """

    results = {
        "6h": {},
        "12h": {},
        "24h": {},
    }

    current_horizon = None

    horizon_pattern = re.compile(
        r"^\s*(6H|12H|24H)\b",
        re.IGNORECASE
    )

    metric_pattern = re.compile(
        r"^\s*"
        r"(Observed rate|Mean predicted probability|E/O ratio|"
        r"Calibration-in-the-large|Calibration intercept|"
        r"Calibration slope|Brier|Log loss)"
        r"\s*:\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*"
        r"\(95%\s*CI\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\s*(?:to|[-–—])\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
        r"\)",
        re.IGNORECASE
    )

    name_map = {
        "observed rate": "observed_rate",
        "mean predicted probability": "mean_predicted_probability",
        "e/o ratio": "expected_observed_ratio",
        "calibration-in-the-large": "calibration_in_the_large",
        "calibration intercept": "calibration_intercept",
        "calibration slope": "calibration_slope",
        "brier": "brier",
        "log loss": "log_loss",
    }

    for line in text.splitlines():

        horizon_match = horizon_pattern.match(line)

        if horizon_match:

            current_horizon = horizon_match.group(1).lower()

        metric_match = metric_pattern.match(line)

        if metric_match and current_horizon is not None:

            raw_name = metric_match.group(1).lower()

            metric = name_map[raw_name]

            results[current_horizon][metric] = {
                "estimate": float(metric_match.group(2)),
                "ci_lower": float(metric_match.group(3)),
                "ci_upper": float(metric_match.group(4)),
            }

    return results


def close_enough(a, b, tolerance=5e-6):

    return bool(
        abs(float(a) - float(b))
        <= tolerance
    )


def format_ci(value, lower, upper):

    return (
        f"{value:.6f} "
        f"(95% CI {lower:.6f} to {upper:.6f})"
    )


# =============================================================================
# START
# =============================================================================

print("")
print("=" * 80)
print("STEP 60 — FINAL MANUSCRIPT RESULTS PACKAGE")
print("=" * 80)

print("")
print("READ-ONLY CONSOLIDATION")
print("NO MODEL WILL BE RETRAINED.")
print("NO THRESHOLD WILL BE CHANGED.")
print("NO TEST PREDICTIONS WILL BE MODIFIED.")
print("")


# =============================================================================
# VERIFY INPUT FILES
# =============================================================================

required_files = [
    final_test_summary_path,
    bootstrap_path,
    calibrated_bootstrap_path,
]

print("=" * 80)
print("INPUT FILE VERIFICATION")
print("=" * 80)

for path in required_files:

    exists = os.path.exists(path)

    print(
        os.path.basename(path),
        "EXISTS:",
        exists
    )

    if not exists:
        raise FileNotFoundError(path)


print("")


# =============================================================================
# READ PRIMARY BOOTSTRAP RESULTS
# =============================================================================

bootstrap_text = read_text(
    bootstrap_path
)

primary_ci = extract_ci_sections(
    bootstrap_text
)


required_primary_metrics = [
    "auroc",
    "auprc",
    "brier",
    "log_loss",
    "sensitivity",
    "specificity",
    "ppv",
    "npv",
    "f1",
]


print("=" * 80)
print("PRIMARY BOOTSTRAP EXTRACTION")
print("=" * 80)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    missing = [
        metric
        for metric in required_primary_metrics
        if metric not in primary_ci[horizon]
    ]

    if missing:

        raise RuntimeError(
            f"{horizon}: missing primary bootstrap metrics: {missing}"
        )

    print(
        horizon.upper(),
        "metrics extracted:",
        len(primary_ci[horizon])
    )


print("")


# =============================================================================
# VERIFY PRIMARY POINT ESTIMATES AGAINST LOCKED RESULTS
# =============================================================================

print("=" * 80)
print("LOCKED PRIMARY RESULT INTEGRITY CHECK")
print("=" * 80)


integrity_pass = True


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    print("")
    print(horizon.upper())

    for metric in required_primary_metrics:

        observed = primary_ci[horizon][metric]["estimate"]

        expected_value = expected[horizon][metric]

        passed = close_enough(
            observed,
            expected_value
        )

        print(
            f"{metric}: "
            f"observed={observed:.6f} "
            f"expected={expected_value:.6f} "
            f"PASS={passed}"
        )

        if not passed:
            integrity_pass = False


if not integrity_pass:

    raise RuntimeError(
        "SAFETY STOP — bootstrap point estimates do not reproduce "
        "the locked final test results."
    )


# =============================================================================
# PRIMARY MANUSCRIPT TABLE
# =============================================================================

primary_rows = []


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    row = {
        "horizon": horizon,
        "test_landmarks": expected[horizon]["rows"],
        "unique_test_patients": expected[horizon]["patients"],
        "locked_threshold": expected[horizon]["threshold"],
    }

    for metric in required_primary_metrics:

        item = primary_ci[horizon][metric]

        row[metric] = item["estimate"]
        row[f"{metric}_ci_lower"] = item["ci_lower"]
        row[f"{metric}_ci_upper"] = item["ci_upper"]

    primary_rows.append(row)


primary_df = pd.DataFrame(
    primary_rows
)


primary_df.to_csv(
    primary_output_path,
    index=False
)


# =============================================================================
# CALIBRATED BOOTSTRAP RESULTS
# =============================================================================

calibrated_text = read_text(
    calibrated_bootstrap_path
)

calibrated_ci = extract_calibrated_ci_sections(
    calibrated_text
)


required_calibrated_metrics = [
    "observed_rate",
    "mean_predicted_probability",
    "expected_observed_ratio",
    "calibration_in_the_large",
    "calibration_intercept",
    "calibration_slope",
    "brier",
    "log_loss",
]


print("")
print("=" * 80)
print("SECONDARY CALIBRATED BOOTSTRAP EXTRACTION")
print("=" * 80)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    missing = [
        metric
        for metric in required_calibrated_metrics
        if metric not in calibrated_ci[horizon]
    ]

    if missing:

        raise RuntimeError(
            f"{horizon}: missing calibrated bootstrap metrics: {missing}"
        )

    print(
        horizon.upper(),
        "metrics extracted:",
        len(calibrated_ci[horizon])
    )


# =============================================================================
# OPTIONAL VALIDATION CALIBRATION PARAMETERS
# =============================================================================

validation_parameters = {}


if os.path.exists(
    calibration_parameter_path
):

    parameter_df = pd.read_csv(
        calibration_parameter_path,
        low_memory=False
    )

    print("")
    print(
        "Validation calibration parameter file found:",
        os.path.basename(
            calibration_parameter_path
        )
    )

    print(
        "Columns:",
        parameter_df.columns.tolist()
    )

    # Keep the source table available in report without assuming column names.
    validation_parameter_text = (
        parameter_df.to_string(
            index=False
        )
    )

else:

    validation_parameter_text = (
        "Validation calibration parameter CSV not found; "
        "calibrated bootstrap results remain available."
    )


# =============================================================================
# CALIBRATED MANUSCRIPT TABLE
# =============================================================================

calibrated_rows = []


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    row = {
        "horizon": horizon,
        "test_landmarks": expected[horizon]["rows"],
        "unique_test_patients": expected[horizon]["patients"],
    }

    for metric in required_calibrated_metrics:

        item = calibrated_ci[horizon][metric]

        row[metric] = item["estimate"]
        row[f"{metric}_ci_lower"] = item["ci_lower"]
        row[f"{metric}_ci_upper"] = item["ci_upper"]

    calibrated_rows.append(row)


calibrated_df = pd.DataFrame(
    calibrated_rows
)


calibrated_df.to_csv(
    calibrated_output_path,
    index=False
)


# =============================================================================
# OPTIONAL COEFFICIENT STABILITY SUMMARY
# =============================================================================

stability_text = None


if os.path.exists(
    stability_report_path
):

    stability_text = read_text(
        stability_report_path
    )


# =============================================================================
# CONSOLE — PRIMARY RESULTS
# =============================================================================

print("")
print("=" * 80)
print("PRIMARY HELD-OUT TEST RESULTS")
print("=" * 80)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    print("")
    print(
        f"{horizon.upper()} — "
        f"{expected[horizon]['rows']} landmarks, "
        f"{expected[horizon]['patients']} patients"
    )

    print(
        "Locked threshold:",
        f"{expected[horizon]['threshold']:.6f}"
    )

    for metric in required_primary_metrics:

        item = primary_ci[horizon][metric]

        print(
            f"{metric}: "
            + format_ci(
                item["estimate"],
                item["ci_lower"],
                item["ci_upper"]
            )
        )


# =============================================================================
# CONSOLE — SECONDARY CALIBRATED RESULTS
# =============================================================================

print("")
print("=" * 80)
print("SECONDARY VALIDATION-DERIVED CALIBRATION RESULTS")
print("=" * 80)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    print("")
    print(horizon.upper())

    for metric in required_calibrated_metrics:

        item = calibrated_ci[horizon][metric]

        print(
            f"{metric}: "
            + format_ci(
                item["estimate"],
                item["ci_lower"],
                item["ci_upper"]
            )
        )


# =============================================================================
# BUILD REPORT
# =============================================================================

report = []

report.append(
    "STEP 60 — FINAL MANUSCRIPT RESULTS PACKAGE"
)

report.append(
    "=" * 80
)

report.append("")

report.append(
    "ANALYSIS STATUS:"
)

report.append(
    "Underlying predictive models were locked before held-out test evaluation."
)

report.append(
    "The held-out test set was evaluated once using frozen preprocessing, "
    "model coefficients, and validation-selected thresholds."
)

report.append(
    "Patient-level bootstrap confidence intervals resampled subject_id."
)

report.append(
    "Original held-out test performance is the PRIMARY evaluation."
)

report.append(
    "Validation-derived post-hoc calibration is SECONDARY and did not "
    "retrain the underlying predictive models."
)

report.append("")


# =============================================================================
# PRIMARY REPORT
# =============================================================================

report.append(
    "PRIMARY HELD-OUT TEST PERFORMANCE"
)

report.append(
    "=" * 80
)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    report.append("")
    report.append(
        f"{horizon.upper()} HORIZON"
    )

    report.append(
        "-" * 80
    )

    report.append(
        f"Assessable test landmarks: {expected[horizon]['rows']}"
    )

    report.append(
        f"Unique test patients: {expected[horizon]['patients']}"
    )

    report.append(
        f"Locked validation-selected threshold: "
        f"{expected[horizon]['threshold']:.6f}"
    )

    for metric in required_primary_metrics:

        item = primary_ci[horizon][metric]

        report.append(
            f"{metric}: "
            + format_ci(
                item["estimate"],
                item["ci_lower"],
                item["ci_upper"]
            )
        )


# =============================================================================
# SECONDARY REPORT
# =============================================================================

report.append("")
report.append("")
report.append(
    "SECONDARY VALIDATION-DERIVED CALIBRATION ANALYSIS"
)

report.append(
    "=" * 80
)

report.append(
    "Calibration parameters were estimated using validation predictions/"
    "outcomes only and then frozen before application to test predictions."
)

report.append(
    "These results should be presented separately from the primary "
    "unrecalibrated held-out test performance."
)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    report.append("")
    report.append(
        f"{horizon.upper()} HORIZON"
    )

    report.append(
        "-" * 80
    )

    for metric in required_calibrated_metrics:

        item = calibrated_ci[horizon][metric]

        report.append(
            f"{metric}: "
            + format_ci(
                item["estimate"],
                item["ci_lower"],
                item["ci_upper"]
            )
        )


# =============================================================================
# VALIDATION CALIBRATION PARAMETERS
# =============================================================================

report.append("")
report.append("")
report.append(
    "VALIDATION-DERIVED CALIBRATION PARAMETER SOURCE"
)

report.append(
    "=" * 80
)

report.append(
    validation_parameter_text
)


# =============================================================================
# STABILITY SUMMARY
# =============================================================================

if stability_text is not None:

    report.append("")
    report.append("")
    report.append(
        "COEFFICIENT STABILITY AUDIT STATUS"
    )

    report.append(
        "=" * 80
    )

    correlation_patterns = [
        r"6h vs 12h:\s*([-+]?\d*\.?\d+)",
        r"6h vs 24h:\s*([-+]?\d*\.?\d+)",
        r"12h vs 24h:\s*([-+]?\d*\.?\d+)",
    ]

    labels = [
        "6h vs 12h",
        "6h vs 24h",
        "12h vs 24h",
    ]

    for label, pattern in zip(
        labels,
        correlation_patterns
    ):

        match = re.search(
            pattern,
            stability_text,
            re.IGNORECASE
        )

        if match:

            report.append(
                f"{label} coefficient correlation: "
                f"{float(match.group(1)):.6f}"
            )

    material_match = re.search(
        r"direction_changing_material:\s*(\d+)",
        stability_text,
        re.IGNORECASE
    )

    if material_match:

        report.append(
            "Material direction-changing predictors: "
            + material_match.group(1)
        )

    report.append(
        "Coefficient interpretation remains predictive/associational, "
        "not causal."
    )


# =============================================================================
# MANUSCRIPT-READY COMPACT TABLE
# =============================================================================

report.append("")
report.append("")
report.append(
    "COMPACT PRIMARY PERFORMANCE TABLE"
)

report.append(
    "=" * 80
)

report.append(
    "Horizon | AUROC (95% CI) | AUPRC (95% CI) | "
    "Brier (95% CI) | Log loss (95% CI)"
)

report.append(
    "-" * 80
)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    auroc = primary_ci[horizon]["auroc"]
    auprc = primary_ci[horizon]["auprc"]
    brier = primary_ci[horizon]["brier"]
    logloss = primary_ci[horizon]["log_loss"]

    line = (
        f"{horizon} | "
        f"{auroc['estimate']:.3f} "
        f"({auroc['ci_lower']:.3f}–{auroc['ci_upper']:.3f}) | "
        f"{auprc['estimate']:.3f} "
        f"({auprc['ci_lower']:.3f}–{auprc['ci_upper']:.3f}) | "
        f"{brier['estimate']:.3f} "
        f"({brier['ci_lower']:.3f}–{brier['ci_upper']:.3f}) | "
        f"{logloss['estimate']:.3f} "
        f"({logloss['ci_lower']:.3f}–{logloss['ci_upper']:.3f})"
    )

    report.append(line)


report.append("")
report.append(
    "COMPACT LOCKED-THRESHOLD CLASSIFICATION TABLE"
)

report.append(
    "=" * 80
)

report.append(
    "Horizon | Sensitivity | Specificity | PPV | NPV | F1"
)

report.append(
    "-" * 80
)


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    values = []

    for metric in [
        "sensitivity",
        "specificity",
        "ppv",
        "npv",
        "f1",
    ]:

        item = primary_ci[horizon][metric]

        values.append(
            f"{item['estimate']:.3f} "
            f"({item['ci_lower']:.3f}–{item['ci_upper']:.3f})"
        )

    report.append(
        f"{horizon} | "
        + " | ".join(values)
    )


# =============================================================================
# FINAL NOTES
# =============================================================================

report.append("")
report.append("")
report.append(
    "INTERPRETATION / REPORTING RULES"
)

report.append(
    "=" * 80
)

report.append(
    "1. Report original locked held-out test results as primary."
)

report.append(
    "2. Report validation-derived recalibration as a secondary analysis."
)

report.append(
    "3. Do not reselect thresholds using the test set."
)

report.append(
    "4. Do not use test outcomes for additional model fitting."
)

report.append(
    "5. Coefficients describe multivariable predictive associations and "
    "must not be presented as causal effects."
)

report.append(
    "6. Patient-level bootstrap confidence intervals account for repeated "
    "landmarks within patients by resampling subject_id."
)

report.append("")
report.append(
    "=" * 80
)

report.append(
    "STEP 60 COMPLETE"
)

report.append(
    "NO MODEL WAS RETRAINED."
)

report.append(
    "NO MODEL WAS RECALIBRATED IN THIS STEP."
)

report.append(
    "NO THRESHOLD WAS CHANGED."
)

report.append(
    "NO TEST PREDICTION WAS MODIFIED."
)

report.append(
    "NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED."
)


with open(
    report_output_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(report)
    )


# =============================================================================
# FINAL VERIFICATION
# =============================================================================

primary_rows_pass = (
    len(primary_df) == 3
)

calibrated_rows_pass = (
    len(calibrated_df) == 3
)

all_primary_finite = bool(
    np.isfinite(
        primary_df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all()
)

all_calibrated_finite = bool(
    np.isfinite(
        calibrated_df.select_dtypes(
            include=[np.number]
        ).to_numpy()
    ).all()
)


final_pass = all([
    integrity_pass,
    primary_rows_pass,
    calibrated_rows_pass,
    all_primary_finite,
    all_calibrated_finite,
])


print("")
print("=" * 80)

print(
    "FINAL STEP 60 VERDICT:",
    "PASS"
    if final_pass
    else "FAIL"
)

print("=" * 80)

print("")
print(
    "Primary test table:"
)

print(
    primary_output_path
)

print("")
print(
    "Secondary calibrated test table:"
)

print(
    calibrated_output_path
)

print("")
print(
    "Final manuscript results report:"
)

print(
    report_output_path
)

print("")
print(
    "NO MODEL WAS RETRAINED."
)

print(
    "NO MODEL WAS RECALIBRATED IN THIS STEP."
)

print(
    "NO THRESHOLD WAS CHANGED."
)

print(
    "NO TEST PREDICTION WAS MODIFIED."
)

print(
    "NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED."
)
