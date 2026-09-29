import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 58B — VALIDATION-DERIVED POST-HOC CALIBRATION
#
# DESIGN:
#
# 1. Fit logistic recalibration using VALIDATION predictions/outcomes ONLY.
#
#       logit(Y) = intercept + slope * logit(original probability)
#
# 2. Freeze intercept and slope.
#
# 3. Apply them unchanged to the already-generated held-out TEST probabilities.
#
# IMPORTANT:
#   - Underlying predictive models are NOT retrained.
#   - Predictor coefficients are NOT changed.
#   - Validation-selected classification thresholds are NOT changed.
#   - Test outcomes are NOT used to fit calibration parameters.
#   - Original test predictions are NOT overwritten.
#
# Outputs NEW calibrated prediction files only.
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")


models = {

    "6h": {
        "target": "aki_within_6h",

        "validation_path": os.path.join(
            folder,
            "baseline_6h_validation_predictions.csv"
        ),

        "test_path": os.path.join(
            folder,
            "FINAL_TEST_6h_predictions.csv"
        ),

        "calibrated_test_path": os.path.join(
            folder,
            "FINAL_TEST_6h_calibrated_predictions.csv"
        ),
    },

    "12h": {
        "target": "aki_within_12h",

        "validation_path": os.path.join(
            folder,
            "baseline_12h_validation_predictions.csv"
        ),

        "test_path": os.path.join(
            folder,
            "FINAL_TEST_12h_predictions.csv"
        ),

        "calibrated_test_path": os.path.join(
            folder,
            "FINAL_TEST_12h_calibrated_predictions.csv"
        ),
    },

    "24h": {
        "target": "aki_within_24h",

        "validation_path": os.path.join(
            folder,
            "baseline_24h_validation_predictions.csv"
        ),

        "test_path": os.path.join(
            folder,
            "FINAL_TEST_24h_predictions.csv"
        ),

        "calibrated_test_path": os.path.join(
            folder,
            "FINAL_TEST_24h_calibrated_predictions.csv"
        ),
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def sigmoid(x):

    x = np.clip(
        x,
        -50.0,
        50.0
    )

    return 1.0 / (
        1.0 + np.exp(-x)
    )


def logit(p):

    p = np.clip(
        p,
        1e-10,
        1.0 - 1e-10
    )

    return np.log(
        p / (1.0 - p)
    )


def fit_logistic_recalibration(
    probabilities,
    outcomes
):

    # -------------------------------------------------------------------------
    # Fits:
    #
    # logit(P(Y=1)) =
    #     intercept + slope * logit(original_probability)
    #
    # using validation data ONLY.
    # -------------------------------------------------------------------------

    lp = logit(
        probabilities
    )

    y = np.asarray(
        outcomes,
        dtype=np.float64
    )

    X = np.column_stack(
        [
            np.ones(
                len(lp),
                dtype=np.float64
            ),
            lp,
        ]
    )

    beta = np.array(
        [
            0.0,
            1.0
        ],
        dtype=np.float64
    )


    for _ in range(100):

        eta = (
            X @ beta
        )

        fitted = sigmoid(
            eta
        )

        gradient = (
            X.T
            @ (
                y - fitted
            )
        )

        weights = (
            fitted
            * (1.0 - fitted)
        )

        hessian = -(
            X.T
            @ (
                X
                * weights[:, None]
            )
        )


        try:

            step = np.linalg.solve(
                hessian,
                gradient
            )

        except np.linalg.LinAlgError:

            raise RuntimeError(
                "Calibration fit failed: singular Hessian."
            )


        beta = (
            beta - step
        )


        if np.max(
            np.abs(step)
        ) < 1e-10:

            break


    return (
        float(beta[0]),
        float(beta[1])
    )


def brier_score(
    y,
    p
):

    return float(
        np.mean(
            (
                p - y
            ) ** 2
        )
    )


def log_loss_score(
    y,
    p
):

    p = np.clip(
        p,
        1e-15,
        1.0 - 1e-15
    )

    return float(
        -np.mean(
            y * np.log(p)
            + (1.0 - y)
            * np.log(
                1.0 - p
            )
        )
    )


def calibration_statistics(
    y,
    p
):

    observed = float(
        np.mean(y)
    )

    predicted = float(
        np.mean(p)
    )

    if observed > 0:

        ratio = (
            predicted
            / observed
        )

    else:

        ratio = np.nan


    lp = logit(
        p
    )


    # ---------------------------------------------------------
    # Calibration-in-the-large:
    #
    # logit(Y) = fixed LP + intercept
    # ---------------------------------------------------------

    intercept_only = 0.0


    for _ in range(100):

        eta = (
            lp
            + intercept_only
        )

        fitted = sigmoid(
            eta
        )

        gradient = float(
            np.sum(
                y - fitted
            )
        )

        hessian = float(
            -np.sum(
                fitted
                * (1.0 - fitted)
            )
        )


        if abs(
            hessian
        ) < 1e-12:

            break


        step = (
            gradient
            / hessian
        )

        intercept_only = (
            intercept_only
            - step
        )


        if abs(
            step
        ) < 1e-10:

            break


    # ---------------------------------------------------------
    # Full calibration intercept + slope
    # ---------------------------------------------------------

    calibration_intercept, calibration_slope = (
        fit_logistic_recalibration(
            p,
            y
        )
    )


    return {
        "observed_rate":
            observed,

        "predicted_rate":
            predicted,

        "expected_observed_ratio":
            ratio,

        "calibration_in_large":
            float(
                intercept_only
            ),

        "calibration_intercept":
            calibration_intercept,

        "calibration_slope":
            calibration_slope,

        "brier":
            brier_score(
                y,
                p
            ),

        "log_loss":
            log_loss_score(
                y,
                p
            ),
    }


# =============================================================================
# START
# =============================================================================

print("")
print("=" * 80)

print(
    "STEP 58B — VALIDATION-DERIVED POST-HOC CALIBRATION"
)

print("=" * 80)

print("")
print(
    "Calibration parameters will be fitted on VALIDATION data only."
)

print(
    "They will then be frozen and applied unchanged to TEST predictions."
)

print("")


report_sections = []
parameter_rows = []


# =============================================================================
# EACH HORIZON
# =============================================================================

for horizon in [
    "6h",
    "12h",
    "24h",
]:

    info = models[
        horizon
    ]


    print("")
    print("=" * 80)

    print(
        f"{horizon.upper()} CALIBRATION"
    )

    print("=" * 80)


    # -------------------------------------------------------------------------
    # REQUIRED FILES
    # -------------------------------------------------------------------------

    for required_path in [
        info["validation_path"],
        info["test_path"],
    ]:

        if not os.path.exists(
            required_path
        ):

            raise FileNotFoundError(
                required_path
            )


    # -------------------------------------------------------------------------
    # VALIDATION DATA
    # -------------------------------------------------------------------------

    validation = pd.read_csv(
        info["validation_path"],
        low_memory=False
    )


    required_validation_columns = [
        info["target"],
        "predicted_probability",
    ]


    missing_validation = [
        column
        for column
        in required_validation_columns
        if column not in validation.columns
    ]


    if missing_validation:

        raise RuntimeError(
            f"{horizon}: validation prediction file "
            f"is missing columns: "
            f"{missing_validation}"
        )


    validation_y = pd.to_numeric(
        validation[
            info["target"]
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    validation_p = pd.to_numeric(
        validation[
            "predicted_probability"
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    if not np.isfinite(
        validation_p
    ).all():

        raise RuntimeError(
            f"{horizon}: nonfinite validation probabilities."
        )


    if not np.isin(
        validation_y,
        [0.0, 1.0]
    ).all():

        raise RuntimeError(
            f"{horizon}: invalid validation outcomes."
        )


    # -------------------------------------------------------------------------
    # FIT CALIBRATION ON VALIDATION ONLY
    # -------------------------------------------------------------------------

    intercept, slope = (
        fit_logistic_recalibration(
            validation_p,
            validation_y
        )
    )


    print(
        "Validation-derived calibration intercept:",
        f"{intercept:.6f}"
    )

    print(
        "Validation-derived calibration slope:",
        f"{slope:.6f}"
    )


    # -------------------------------------------------------------------------
    # TEST DATA
    #
    # IMPORTANT:
    # Test outcomes are loaded only AFTER calibration parameters have
    # already been fitted and frozen from validation.
    # -------------------------------------------------------------------------

    test = pd.read_csv(
        info["test_path"],
        low_memory=False
    )


    required_test_columns = [
        "subject_id",
        "stay_id",
        "landmark_time",
        info["target"],
        "predicted_probability",
    ]


    missing_test = [
        column
        for column
        in required_test_columns
        if column not in test.columns
    ]


    if missing_test:

        raise RuntimeError(
            f"{horizon}: test prediction file "
            f"is missing columns: "
            f"{missing_test}"
        )


    test_y = pd.to_numeric(
        test[
            info["target"]
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    test_original_p = pd.to_numeric(
        test[
            "predicted_probability"
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    if not np.isfinite(
        test_original_p
    ).all():

        raise RuntimeError(
            f"{horizon}: nonfinite test probabilities."
        )


    # -------------------------------------------------------------------------
    # APPLY FROZEN VALIDATION CALIBRATION TO TEST
    # -------------------------------------------------------------------------

    test_lp = logit(
        test_original_p
    )


    calibrated_test_p = sigmoid(
        intercept
        + slope
        * test_lp
    )


    if not np.isfinite(
        calibrated_test_p
    ).all():

        raise RuntimeError(
            f"{horizon}: calibrated probabilities "
            "contain nonfinite values."
        )


    # -------------------------------------------------------------------------
    # ORIGINAL VS CALIBRATED TEST CALIBRATION
    # -------------------------------------------------------------------------

    original_stats = calibration_statistics(
        test_y,
        test_original_p
    )


    calibrated_stats = calibration_statistics(
        test_y,
        calibrated_test_p
    )


    # -------------------------------------------------------------------------
    # SAVE NEW CALIBRATED TEST PREDICTIONS
    #
    # Original probability is preserved.
    # -------------------------------------------------------------------------

    calibrated_output = pd.DataFrame(
        {
            "subject_id":
                test[
                    "subject_id"
                ],

            "stay_id":
                test[
                    "stay_id"
                ],

            "landmark_time":
                test[
                    "landmark_time"
                ],

            info["target"]:
                test_y.astype(
                    "int8"
                ),

            "original_predicted_probability":
                test_original_p,

            "calibrated_predicted_probability":
                calibrated_test_p,

            "validation_calibration_intercept":
                intercept,

            "validation_calibration_slope":
                slope,
        }
    )


    calibrated_output.to_csv(
        info[
            "calibrated_test_path"
        ],
        index=False
    )


    # -------------------------------------------------------------------------
    # PARAMETER TABLE
    # -------------------------------------------------------------------------

    parameter_rows.append(
        {
            "horizon":
                horizon,

            "validation_calibration_intercept":
                intercept,

            "validation_calibration_slope":
                slope,

            "test_observed_rate":
                original_stats[
                    "observed_rate"
                ],

            "test_original_mean_probability":
                original_stats[
                    "predicted_rate"
                ],

            "test_calibrated_mean_probability":
                calibrated_stats[
                    "predicted_rate"
                ],

            "test_original_EO_ratio":
                original_stats[
                    "expected_observed_ratio"
                ],

            "test_calibrated_EO_ratio":
                calibrated_stats[
                    "expected_observed_ratio"
                ],

            "test_original_calibration_intercept":
                original_stats[
                    "calibration_in_large"
                ],

            "test_calibrated_calibration_intercept":
                calibrated_stats[
                    "calibration_in_large"
                ],

            "test_original_calibration_slope":
                original_stats[
                    "calibration_slope"
                ],

            "test_calibrated_calibration_slope":
                calibrated_stats[
                    "calibration_slope"
                ],

            "test_original_brier":
                original_stats[
                    "brier"
                ],

            "test_calibrated_brier":
                calibrated_stats[
                    "brier"
                ],

            "test_original_log_loss":
                original_stats[
                    "log_loss"
                ],

            "test_calibrated_log_loss":
                calibrated_stats[
                    "log_loss"
                ],
        }
    )


    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    lines = [
        f"{horizon.upper()} VALIDATION-DERIVED CALIBRATION",
        "-" * 80,
        "",
        "CALIBRATION FIT:",
        "Source: validation predictions/outcomes only",
        f"Validation rows: {len(validation_y)}",
        f"Calibration intercept: {intercept:.6f}",
        f"Calibration slope: {slope:.6f}",
        "",
        "HELD-OUT TEST:",
        f"Test rows: {len(test_y)}",
        f"Observed event rate: {original_stats['observed_rate']:.6f}",
        "",
        "ORIGINAL TEST PROBABILITIES:",
        f"Mean predicted probability: {original_stats['predicted_rate']:.6f}",
        f"Expected / Observed ratio: {original_stats['expected_observed_ratio']:.6f}",
        f"Calibration-in-the-large intercept: {original_stats['calibration_in_large']:.6f}",
        f"Calibration slope: {original_stats['calibration_slope']:.6f}",
        f"Brier score: {original_stats['brier']:.6f}",
        f"Log loss: {original_stats['log_loss']:.6f}",
        "",
        "VALIDATION-CALIBRATED TEST PROBABILITIES:",
        f"Mean predicted probability: {calibrated_stats['predicted_rate']:.6f}",
        f"Expected / Observed ratio: {calibrated_stats['expected_observed_ratio']:.6f}",
        f"Calibration-in-the-large intercept: {calibrated_stats['calibration_in_large']:.6f}",
        f"Calibration slope: {calibrated_stats['calibration_slope']:.6f}",
        f"Brier score: {calibrated_stats['brier']:.6f}",
        f"Log loss: {calibrated_stats['log_loss']:.6f}",
        "",
        "IMPORTANT:",
        "Underlying predictive model unchanged.",
        "Calibration parameters fitted on validation only.",
        "No calibration parameter was estimated from test outcomes.",
        "Original test prediction file was not overwritten.",
    ]


    section = "\n".join(
        lines
    )


    report_sections.append(
        section
    )


    print("")
    print(
        section
    )


# =============================================================================
# SAVE CALIBRATION PARAMETERS / SUMMARY
# =============================================================================

parameter_df = pd.DataFrame(
    parameter_rows
)


parameter_path = os.path.join(
    folder,
    "step58B_validation_calibration_parameters.csv"
)


parameter_df.to_csv(
    parameter_path,
    index=False
)


# =============================================================================
# SAVE TEXT REPORT
# =============================================================================

report_path = os.path.join(
    folder,
    "step58B_validation_derived_calibration_report.txt"
)


header = [
    "STEP 58B — VALIDATION-DERIVED POST-HOC CALIBRATION",
    "=" * 80,
    "",
    "Calibration method:",
    "Logistic recalibration of original model log-odds.",
    "",
    "Calibration parameters fitted using validation outcomes only.",
    "Parameters then frozen and applied unchanged to held-out test predictions.",
    "",
    "Underlying models were NOT retrained.",
    "Original test predictions were NOT overwritten.",
    "",
]


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            header
        )
    )

    f.write(
        "\n"
    )

    f.write(
        "\n\n".join(
            report_sections
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "=" * 80
    )

    f.write(
        "\nSTEP 58B COMPLETE\n"
    )

    f.write(
        "CALIBRATION PARAMETERS WERE FIT ON VALIDATION ONLY.\n"
    )

    f.write(
        "UNDERLYING LOCKED MODELS WERE NOT MODIFIED.\n"
    )


print("")
print("=" * 80)

print(
    "STEP 58B COMPLETE"
)

print("=" * 80)

print("")
print(
    "Calibration parameter CSV:"
)

print(
    parameter_path
)

print("")
print(
    "Calibration report:"
)

print(
    report_path
)

print("")
print(
    "UNDERLYING MODELS WERE NOT RETRAINED."
)

print(
    "CALIBRATION PARAMETERS WERE FIT ON VALIDATION ONLY."
)

print(
    "ORIGINAL TEST PREDICTIONS WERE NOT OVERWRITTEN."
)
