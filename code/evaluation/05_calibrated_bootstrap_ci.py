import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 58C — PATIENT-LEVEL BOOTSTRAP CI FOR VALIDATION-CALIBRATED TEST RISKS
#
# IMPORTANT:
#   - Uses already-created calibrated test predictions.
#   - Calibration parameters remain frozen.
#   - Does NOT recalibrate inside bootstrap samples.
#   - Does NOT retrain predictive models.
#   - Patient (subject_id) is the resampling unit.
#
# Metrics:
#   - Mean predicted probability
#   - Expected / Observed ratio
#   - Calibration-in-the-large
#   - Calibration intercept
#   - Calibration slope
#   - Brier score
#   - Log loss
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

n_bootstrap = 1000
random_seed = 20260929

rng = np.random.default_rng(
    random_seed
)


models = {

    "6h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_6h_calibrated_predictions.csv"
        ),
        "target": "aki_within_6h",
    },

    "12h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_12h_calibrated_predictions.csv"
        ),
        "target": "aki_within_12h",
    },

    "24h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_24h_calibrated_predictions.csv"
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


def fit_calibration_model(p, y):

    lp = logit(
        p
    )

    X = np.column_stack(
        [
            np.ones(len(lp)),
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

            return (
                np.nan,
                np.nan
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


def calibration_in_large(p, y):

    lp = logit(
        p
    )

    intercept = 0.0

    for _ in range(100):

        fitted = sigmoid(
            lp + intercept
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

            return np.nan

        step = (
            gradient
            / hessian
        )

        intercept = (
            intercept
            - step
        )

        if abs(
            step
        ) < 1e-10:

            break

    return float(
        intercept
    )


def calculate_metrics(y, p):

    observed_rate = float(
        np.mean(y)
    )

    predicted_rate = float(
        np.mean(p)
    )

    if observed_rate > 0:

        eo_ratio = (
            predicted_rate
            / observed_rate
        )

    else:

        eo_ratio = np.nan


    citl = calibration_in_large(
        p,
        y
    )


    calibration_intercept, calibration_slope = (
        fit_calibration_model(
            p,
            y
        )
    )


    brier = float(
        np.mean(
            (p - y) ** 2
        )
    )


    clipped = np.clip(
        p,
        1e-15,
        1.0 - 1e-15
    )


    log_loss = float(
        -np.mean(
            y * np.log(clipped)
            + (1.0 - y)
            * np.log(
                1.0 - clipped
            )
        )
    )


    return {
        "Observed rate":
            observed_rate,

        "Mean predicted probability":
            predicted_rate,

        "E/O ratio":
            eo_ratio,

        "Calibration-in-the-large":
            citl,

        "Calibration intercept":
            calibration_intercept,

        "Calibration slope":
            calibration_slope,

        "Brier":
            brier,

        "Log loss":
            log_loss,
    }


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    return (
        float(
            np.percentile(
                values,
                2.5
            )
        ),
        float(
            np.percentile(
                values,
                97.5
            )
        ),
    )


# =============================================================================
# START
# =============================================================================

print("")
print("=" * 80)

print(
    "STEP 58C — CALIBRATED TEST PATIENT-LEVEL BOOTSTRAP"
)

print("=" * 80)

print(
    "Bootstrap replicates:",
    n_bootstrap
)

print(
    "Resampling unit: subject_id"
)

print(
    "Calibration parameters remain frozen."
)

print("")


all_bootstrap_rows = []
report_sections = []


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
        f"{horizon.upper()} CALIBRATED BOOTSTRAP"
    )

    print("=" * 80)


    if not os.path.exists(
        info["path"]
    ):

        raise FileNotFoundError(
            info["path"]
        )


    df = pd.read_csv(
        info["path"],
        low_memory=False
    )


    subject = pd.to_numeric(
        df["subject_id"],
        errors="raise"
    ).astype(
        "int64"
    ).to_numpy()


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
            "calibrated_predicted_probability"
        ],
        errors="raise"
    ).astype(
        "float64"
    ).to_numpy()


    if not np.isfinite(
        p
    ).all():

        raise RuntimeError(
            f"{horizon}: nonfinite calibrated probabilities."
        )


    point = calculate_metrics(
        y,
        p
    )


    unique_subjects = np.unique(
        subject
    )

    n_subjects = len(
        unique_subjects
    )


    patient_indices = {
        patient:
            np.flatnonzero(
                subject == patient
            )

        for patient
        in unique_subjects
    }


    metric_names = list(
        point.keys()
    )


    bootstrap_values = {
        metric: []
        for metric in metric_names
    }


    print(
        "Test landmarks:",
        len(y)
    )

    print(
        "Unique patients:",
        n_subjects
    )


    # -------------------------------------------------------------------------
    # BOOTSTRAP
    # -------------------------------------------------------------------------

    for b in range(
        1,
        n_bootstrap + 1
    ):

        sampled_subjects = rng.choice(
            unique_subjects,
            size=n_subjects,
            replace=True
        )


        sampled_indices = np.concatenate(
            [
                patient_indices[
                    patient
                ]
                for patient
                in sampled_subjects
            ]
        )


        boot_y = y[
            sampled_indices
        ]

        boot_p = p[
            sampled_indices
        ]


        metrics = calculate_metrics(
            boot_y,
            boot_p
        )


        for metric in metric_names:

            bootstrap_values[
                metric
            ].append(
                metrics[
                    metric
                ]
            )


        row = {
            "horizon":
                horizon,

            "bootstrap_replicate":
                b,

            "sampled_patients":
                n_subjects,

            "sampled_landmarks":
                len(
                    sampled_indices
                ),
        }


        for metric in metric_names:

            row[
                metric
            ] = metrics[
                metric
            ]


        all_bootstrap_rows.append(
            row
        )


        if (
            b == 1
            or b % 100 == 0
            or b == n_bootstrap
        ):

            print(
                f"{horizon}: bootstrap "
                f"{b}/{n_bootstrap}"
            )


    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    lines = [
        f"{horizon.upper()} VALIDATION-CALIBRATED TEST RESULTS",
        "-" * 80,
        f"Test landmarks: {len(y)}",
        f"Unique test patients: {n_subjects}",
        f"Bootstrap replicates: {n_bootstrap}",
        "Bootstrap unit: subject_id",
        "",
    ]


    for metric in metric_names:

        lower, upper = percentile_ci(
            bootstrap_values[
                metric
            ]
        )

        lines.append(
            f"{metric}: "
            f"{point[metric]:.6f} "
            f"(95% CI {lower:.6f} to {upper:.6f})"
        )


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
# SAVE BOOTSTRAP REPLICATES
# =============================================================================

bootstrap_df = pd.DataFrame(
    all_bootstrap_rows
)


bootstrap_path = os.path.join(
    folder,
    "step58C_calibrated_bootstrap_metrics.csv"
)


bootstrap_df.to_csv(
    bootstrap_path,
    index=False
)


# =============================================================================
# SAVE REPORT
# =============================================================================

report_path = os.path.join(
    folder,
    "step58C_calibrated_bootstrap_CI.txt"
)


header = [
    "STEP 58C — PATIENT-LEVEL BOOTSTRAP CI FOR VALIDATION-CALIBRATED TEST RISKS",
    "=" * 80,
    "",
    "Validation-derived calibration parameters remained frozen.",
    "No predictive model was retrained.",
    "No calibration model was refitted using test outcomes.",
    "",
    f"Bootstrap replicates: {n_bootstrap}",
    f"Random seed: {random_seed}",
    "Bootstrap unit: subject_id",
    "CI method: percentile bootstrap (2.5th to 97.5th percentiles)",
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
        "\nSTEP 58C COMPLETE\n"
    )

    f.write(
        "NO MODEL WAS RETRAINED.\n"
    )

    f.write(
        "VALIDATION-DERIVED CALIBRATION PARAMETERS REMAINED FROZEN.\n"
    )


print("")
print("=" * 80)

print(
    "STEP 58C COMPLETE"
)

print("=" * 80)

print("")
print(
    "Bootstrap CSV:"
)

print(
    bootstrap_path
)

print("")
print(
    "CI report:"
)

print(
    report_path
)

print("")
print(
    "NO MODEL WAS RETRAINED."
)

print(
    "CALIBRATION PARAMETERS REMAINED FROZEN."
)
