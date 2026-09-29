import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 58A — FINAL HELD-OUT TEST CALIBRATION ASSESSMENT
#
# PURPOSE:
#   Evaluate calibration of the already-locked final model predictions.
#
# Calculates:
#   - Calibration-in-the-large (intercept)
#   - Calibration slope
#   - Observed vs predicted event rate
#   - Expected / Observed ratio
#   - 10-bin calibration table
#
# IMPORTANT:
#   - NO model retraining
#   - NO recalibration
#   - NO threshold modification
#   - Frozen test predictions only
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

models = {

    "6h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_6h_predictions.csv"
        ),
        "target": "aki_within_6h",
    },

    "12h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_12h_predictions.csv"
        ),
        "target": "aki_within_12h",
    },

    "24h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_24h_predictions.csv"
        ),
        "target": "aki_within_24h",
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def sigmoid(x):

    x = np.clip(
        x,
        -50,
        50
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


def fit_intercept_only(offset, y):

    # ---------------------------------------------------------
    # Fits:
    #
    # logit(P(Y=1)) = offset + intercept
    #
    # Calibration intercept ideal = 0.
    # ---------------------------------------------------------

    intercept = 0.0

    for _ in range(100):

        eta = (
            offset
            + intercept
        )

        prob = sigmoid(
            eta
        )

        gradient = np.sum(
            y - prob
        )

        hessian = -np.sum(
            prob
            * (1.0 - prob)
        )

        if abs(hessian) < 1e-12:
            break

        step = (
            gradient
            / hessian
        )

        intercept = (
            intercept
            - step
        )

        if abs(step) < 1e-10:
            break

    return float(
        intercept
    )


def fit_calibration_model(lp, y):

    # ---------------------------------------------------------
    # Fits:
    #
    # logit(P(Y=1)) = intercept + slope * lp
    #
    # Ideal:
    #   intercept = 0
    #   slope = 1
    # ---------------------------------------------------------

    beta = np.array(
        [
            0.0,
            1.0
        ],
        dtype=np.float64
    )

    X = np.column_stack(
        [
            np.ones(
                len(lp)
            ),
            lp
        ]
    )

    for _ in range(100):

        eta = (
            X @ beta
        )

        prob = sigmoid(
            eta
        )

        gradient = (
            X.T
            @ (
                y - prob
            )
        )

        weights = (
            prob
            * (1.0 - prob)
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
                "Calibration model Hessian was singular."
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


# =============================================================================
# START
# =============================================================================

print("")
print("=" * 80)
print("STEP 58A — FINAL TEST CALIBRATION")
print("=" * 80)

all_bin_tables = []
report_sections = []


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
    # LOAD FROZEN PREDICTIONS
    # -------------------------------------------------------------------------

    df = pd.read_csv(
        info["path"],
        low_memory=False
    )


    y = pd.to_numeric(
        df[
            info["target"]
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    p = pd.to_numeric(
        df[
            "predicted_probability"
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    if not np.isfinite(
        p
    ).all():

        raise RuntimeError(
            f"{horizon}: nonfinite probabilities."
        )


    # -------------------------------------------------------------------------
    # BASIC CALIBRATION
    # -------------------------------------------------------------------------

    observed_rate = float(
        y.mean()
    )

    predicted_rate = float(
        p.mean()
    )

    if observed_rate > 0:

        expected_observed_ratio = (
            predicted_rate
            / observed_rate
        )

    else:

        expected_observed_ratio = np.nan


    lp = logit(
        p
    )


    # Calibration-in-the-large:
    # predicted LP used as fixed offset.

    calibration_intercept = (
        fit_intercept_only(
            lp,
            y
        )
    )


    # Full calibration model.

    calibration_model_intercept, calibration_slope = (
        fit_calibration_model(
            lp,
            y
        )
    )


    # -------------------------------------------------------------------------
    # DECILE CALIBRATION TABLE
    # -------------------------------------------------------------------------

    calibration_df = pd.DataFrame(
        {
            "outcome": y,
            "probability": p,
        }
    )


    calibration_df[
        "calibration_bin"
    ] = pd.qcut(
        calibration_df[
            "probability"
        ],
        q=10,
        labels=False,
        duplicates="drop"
    )


    grouped = (
        calibration_df
        .groupby(
            "calibration_bin",
            observed=True
        )
        .agg(
            n=(
                "outcome",
                "size"
            ),

            observed_events=(
                "outcome",
                "sum"
            ),

            observed_rate=(
                "outcome",
                "mean"
            ),

            mean_predicted_probability=(
                "probability",
                "mean"
            ),

            minimum_predicted_probability=(
                "probability",
                "min"
            ),

            maximum_predicted_probability=(
                "probability",
                "max"
            ),
        )
        .reset_index()
    )


    grouped.insert(
        0,
        "horizon",
        horizon
    )


    all_bin_tables.append(
        grouped
    )


    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    lines = [
        f"{horizon.upper()} FINAL TEST CALIBRATION",
        "-" * 80,
        f"Test landmarks: {len(y)}",
        f"Observed event rate: {observed_rate:.6f}",
        f"Mean predicted probability: {predicted_rate:.6f}",
        f"Expected / Observed ratio: {expected_observed_ratio:.6f}",
        "",
        f"Calibration-in-the-large intercept: {calibration_intercept:.6f}",
        f"Calibration-model intercept: {calibration_model_intercept:.6f}",
        f"Calibration slope: {calibration_slope:.6f}",
        "",
        "IDEAL CALIBRATION VALUES:",
        "Calibration intercept = 0",
        "Calibration slope = 1",
        "Expected / Observed ratio = 1",
        "",
        "DECILE CALIBRATION TABLE:",
        grouped.to_string(
            index=False
        ),
    ]


    section = "\n".join(
        lines
    )

    report_sections.append(
        section
    )

    print(
        section
    )


# =============================================================================
# SAVE DECILE TABLE
# =============================================================================

all_bins = pd.concat(
    all_bin_tables,
    ignore_index=True
)


bin_output_path = os.path.join(
    folder,
    "step58A_test_calibration_bins.csv"
)


all_bins.to_csv(
    bin_output_path,
    index=False
)


# =============================================================================
# SAVE REPORT
# =============================================================================

report_path = os.path.join(
    folder,
    "step58A_test_calibration_report.txt"
)


header = [
    "STEP 58A — FINAL HELD-OUT TEST CALIBRATION ASSESSMENT",
    "=" * 80,
    "",
    "Frozen final test predictions only.",
    "No model retraining.",
    "No recalibration performed.",
    "No threshold modification.",
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
        "\nSTEP 58A COMPLETE\n"
    )

    f.write(
        "NO MODEL WAS RETRAINED.\n"
    )

    f.write(
        "NO MODEL WAS RECALIBRATED.\n"
    )

    f.write(
        "NO THRESHOLD WAS CHANGED.\n"
    )


print("")
print("=" * 80)
print("STEP 58A COMPLETE")
print("=" * 80)

print("")
print(
    "Calibration-bin CSV:"
)

print(
    bin_output_path
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
    "NO MODEL WAS RETRAINED."
)

print(
    "NO MODEL WAS RECALIBRATED."
)

print(
    "NO THRESHOLD WAS CHANGED."
)
