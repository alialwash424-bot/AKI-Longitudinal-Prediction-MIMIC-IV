import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 57B — PATIENT-LEVEL BOOTSTRAP 95% CONFIDENCE INTERVALS
#
# PURPOSE:
#   Calculate uncertainty around the already-frozen held-out test metrics.
#
# IMPORTANT:
#   - Resampling unit = PATIENT (subject_id), not landmark row.
#   - All landmarks from a sampled patient remain together.
#   - Sampling is with replacement.
#   - Locked thresholds remain unchanged.
#   - NO model fitting.
#   - NO threshold selection.
#   - NO preprocessing estimation.
#
# Outputs:
#   step57B_patient_bootstrap_metrics.csv
#   step57B_patient_bootstrap_CI.txt
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

n_bootstrap = 1000
random_seed = 20260929

rng = np.random.default_rng(
    random_seed
)


# =============================================================================
# FROZEN TEST FILES AND EXPECTED POINT ESTIMATES
# =============================================================================

models = {

    "6h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_6h_predictions.csv"
        ),
        "target": "aki_within_6h",
        "threshold": 0.496171,

        "expected": {
            "AUROC": 0.724717,
            "AUPRC": 0.198494,
            "Brier": 0.217222,
            "Log loss": 0.635285,
            "Sensitivity": 0.681999,
            "Specificity": 0.650154,
            "PPV": 0.154762,
            "NPV": 0.956078,
            "F1": 0.252277,
        },
    },

    "12h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_12h_predictions.csv"
        ),
        "target": "aki_within_12h",
        "threshold": 0.512416,

        "expected": {
            "AUROC": 0.733356,
            "AUPRC": 0.328190,
            "Brier": 0.208371,
            "Log loss": 0.616046,
            "Sensitivity": 0.641012,
            "Specificity": 0.703742,
            "PPV": 0.287223,
            "NPV": 0.913240,
            "F1": 0.396695,
        },
    },

    "24h": {
        "path": os.path.join(
            folder,
            "FINAL_TEST_24h_predictions.csv"
        ),
        "target": "aki_within_24h",
        "threshold": 0.466857,

        "expected": {
            "AUROC": 0.731931,
            "AUPRC": 0.503620,
            "Brier": 0.211482,
            "Log loss": 0.617875,
            "Sensitivity": 0.715248,
            "Specificity": 0.620278,
            "PPV": 0.418656,
            "NPV": 0.850691,
            "F1": 0.528163,
        },
    },
}


# =============================================================================
# HELPERS
# =============================================================================

def safe_divide(a, b):

    if b == 0:
        return np.nan

    return a / b


def calculate_auc(y, p):

    y = np.asarray(
        y,
        dtype=np.int8
    )

    p = np.asarray(
        p,
        dtype=np.float64
    )

    positive = (
        y == 1
    )

    negative = (
        y == 0
    )

    n_positive = int(
        positive.sum()
    )

    n_negative = int(
        negative.sum()
    )

    if (
        n_positive == 0
        or n_negative == 0
    ):
        return np.nan

    order = np.argsort(
        p,
        kind="mergesort"
    )

    sorted_p = p[order]

    ranks = np.empty(
        len(p),
        dtype=np.float64
    )

    i = 0

    while i < len(p):

        j = i + 1

        while (
            j < len(p)
            and sorted_p[j] == sorted_p[i]
        ):
            j += 1

        average_rank = (
            (i + 1 + j)
            / 2.0
        )

        ranks[
            order[i:j]
        ] = average_rank

        i = j

    positive_rank_sum = float(
        ranks[positive].sum()
    )

    auc = (
        positive_rank_sum
        - n_positive
        * (n_positive + 1)
        / 2.0
    ) / (
        n_positive
        * n_negative
    )

    return float(
        auc
    )


def calculate_auprc(y, p):

    y = np.asarray(
        y,
        dtype=np.int8
    )

    p = np.asarray(
        p,
        dtype=np.float64
    )

    positive_count = int(
        (y == 1).sum()
    )

    if positive_count == 0:
        return np.nan

    order = np.argsort(
        -p,
        kind="mergesort"
    )

    y_sorted = y[order]

    tp = np.cumsum(
        y_sorted == 1
    )

    fp = np.cumsum(
        y_sorted == 0
    )

    precision = (
        tp
        / (tp + fp)
    )

    average_precision = float(
        precision[
            y_sorted == 1
        ].sum()
        / positive_count
    )

    return average_precision


def calculate_metrics(
    y,
    p,
    threshold
):

    y = np.asarray(
        y,
        dtype=np.int8
    )

    p = np.asarray(
        p,
        dtype=np.float64
    )

    predicted = (
        p >= threshold
    ).astype(
        np.int8
    )

    tp = int(
        (
            (predicted == 1)
            & (y == 1)
        ).sum()
    )

    tn = int(
        (
            (predicted == 0)
            & (y == 0)
        ).sum()
    )

    fp = int(
        (
            (predicted == 1)
            & (y == 0)
        ).sum()
    )

    fn = int(
        (
            (predicted == 0)
            & (y == 1)
        ).sum()
    )

    sensitivity = safe_divide(
        tp,
        tp + fn
    )

    specificity = safe_divide(
        tn,
        tn + fp
    )

    ppv = safe_divide(
        tp,
        tp + fp
    )

    npv = safe_divide(
        tn,
        tn + fn
    )

    f1 = safe_divide(
        2 * tp,
        2 * tp + fp + fn
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
            + (1 - y)
            * np.log(
                1.0 - clipped
            )
        )
    )

    return {
        "AUROC":
            calculate_auc(
                y,
                p
            ),

        "AUPRC":
            calculate_auprc(
                y,
                p
            ),

        "Brier":
            brier,

        "Log loss":
            log_loss,

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "PPV":
            ppv,

        "NPV":
            npv,

        "F1":
            f1,
    }


def percentile_ci(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:

        return (
            np.nan,
            np.nan
        )

    lower = float(
        np.percentile(
            values,
            2.5
        )
    )

    upper = float(
        np.percentile(
            values,
            97.5
        )
    )

    return (
        lower,
        upper
    )


# =============================================================================
# START
# =============================================================================

print("")
print("=" * 80)
print("STEP 57B — PATIENT-LEVEL BOOTSTRAP 95% CI")
print("=" * 80)

print("")
print(
    "Bootstrap replicates:",
    n_bootstrap
)

print(
    "Random seed:",
    random_seed
)

print(
    "Resampling unit: subject_id"
)

print(
    "Thresholds: frozen validation-selected thresholds"
)

print("")


all_bootstrap_results = []

final_reports = []


# =============================================================================
# PROCESS EACH HORIZON
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
        f"{horizon.upper()} BOOTSTRAP"
    )
    print("=" * 80)


    # -------------------------------------------------------------------------
    # LOAD FROZEN TEST PREDICTIONS
    # -------------------------------------------------------------------------

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
        "int8"
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
            f"{horizon}: nonfinite probability detected."
        )


    # -------------------------------------------------------------------------
    # ORIGINAL POINT ESTIMATES
    # -------------------------------------------------------------------------

    point = calculate_metrics(
        y,
        p,
        info["threshold"]
    )


    print("")
    print(
        "Original test rows:",
        len(df)
    )

    print(
        "Original unique patients:",
        len(
            np.unique(
                subject
            )
        )
    )


    # -------------------------------------------------------------------------
    # VERIFY POINT ESTIMATES AGAINST FROZEN FINAL RESULTS
    # -------------------------------------------------------------------------

    print("")
    print(
        "POINT-ESTIMATE VERIFICATION:"
    )

    for metric, expected_value in info[
        "expected"
    ].items():

        observed = point[
            metric
        ]

        difference = abs(
            observed
            - expected_value
        )

        passed = (
            difference
            < 1e-6
        )

        print(
            metric,
            "observed=",
            f"{observed:.6f}",
            "expected=",
            f"{expected_value:.6f}",
            "PASS="
            + str(
                passed
            )
        )

        if not passed:

            raise RuntimeError(
                f"SAFETY STOP — {horizon} {metric} "
                "does not reproduce the frozen final result."
            )


    # -------------------------------------------------------------------------
    # CREATE PATIENT CLUSTERS
    # -------------------------------------------------------------------------

    unique_subjects = np.unique(
        subject
    )

    n_subjects = len(
        unique_subjects
    )


    patient_indices = {}

    for patient in unique_subjects:

        patient_indices[
            patient
        ] = np.flatnonzero(
            subject == patient
        )


    # -------------------------------------------------------------------------
    # BOOTSTRAP
    # -------------------------------------------------------------------------

    bootstrap_metrics = {
        "AUROC": [],
        "AUPRC": [],
        "Brier": [],
        "Log loss": [],
        "Sensitivity": [],
        "Specificity": [],
        "PPV": [],
        "NPV": [],
        "F1": [],
    }


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
            boot_p,
            info["threshold"]
        )


        for metric_name in bootstrap_metrics:

            bootstrap_metrics[
                metric_name
            ].append(
                metrics[
                    metric_name
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


        for metric_name in bootstrap_metrics:

            row[
                metric_name
            ] = metrics[
                metric_name
            ]


        all_bootstrap_results.append(
            row
        )


        if (
            b == 1
            or b % 100 == 0
            or b == n_bootstrap
        ):

            print(
                f"{horizon}: "
                f"bootstrap {b}/{n_bootstrap}"
            )


    # -------------------------------------------------------------------------
    # CONFIDENCE INTERVALS
    # -------------------------------------------------------------------------

    report_lines = []

    report_lines.append(
        f"{horizon.upper()} PATIENT-LEVEL BOOTSTRAP RESULTS"
    )

    report_lines.append(
        "-" * 80
    )

    report_lines.append(
        f"Unique test patients: {n_subjects}"
    )

    report_lines.append(
        f"Original test landmarks: {len(y)}"
    )

    report_lines.append(
        f"Bootstrap replicates: {n_bootstrap}"
    )

    report_lines.append(
        "Bootstrap unit: subject_id"
    )

    report_lines.append(
        f"Locked threshold: {info['threshold']:.6f}"
    )

    report_lines.append("")


    for metric_name in [
        "AUROC",
        "AUPRC",
        "Brier",
        "Log loss",
        "Sensitivity",
        "Specificity",
        "PPV",
        "NPV",
        "F1",
    ]:

        lower, upper = percentile_ci(
            bootstrap_metrics[
                metric_name
            ]
        )

        point_value = point[
            metric_name
        ]

        report_lines.append(
            f"{metric_name}: "
            f"{point_value:.6f} "
            f"(95% CI {lower:.6f} to {upper:.6f})"
        )


    final_reports.append(
        "\n".join(
            report_lines
        )
    )


    print("")
    print(
        "\n".join(
            report_lines
        )
    )


# =============================================================================
# SAVE ALL BOOTSTRAP REPLICATES
# =============================================================================

bootstrap_output = pd.DataFrame(
    all_bootstrap_results
)


bootstrap_csv_path = os.path.join(
    folder,
    "step57B_patient_bootstrap_metrics.csv"
)


bootstrap_output.to_csv(
    bootstrap_csv_path,
    index=False
)


# =============================================================================
# SAVE FINAL CI REPORT
# =============================================================================

report_path = os.path.join(
    folder,
    "step57B_patient_bootstrap_CI.txt"
)


report_header = [
    "STEP 57B — PATIENT-LEVEL BOOTSTRAP 95% CONFIDENCE INTERVALS",
    "=" * 80,
    "",
    "Frozen held-out test predictions only.",
    "No model retraining.",
    "No test-based threshold selection.",
    "No preprocessing refitting.",
    "",
    f"Bootstrap replicates: {n_bootstrap}",
    f"Random seed: {random_seed}",
    "Bootstrap unit: subject_id",
    "Confidence interval method: percentile bootstrap, 2.5th to 97.5th percentiles",
    "",
]


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_header
        )
    )

    f.write(
        "\n"
    )

    f.write(
        "\n\n".join(
            final_reports
        )
    )

    f.write(
        "\n\n"
    )

    f.write(
        "=" * 80
    )

    f.write(
        "\nSTEP 57B COMPLETE\n"
    )

    f.write(
        "THE LOCKED MODELS WERE NOT MODIFIED.\n"
    )

    f.write(
        "THE LOCKED THRESHOLDS WERE NOT MODIFIED.\n"
    )


# =============================================================================
# FINAL CONSOLE REPORT
# =============================================================================

print("")
print("=" * 80)
print("STEP 57B COMPLETE")
print("=" * 80)

print("")
print(
    "Bootstrap CSV:"
)

print(
    bootstrap_csv_path
)

print("")
print(
    "Confidence-interval report:"
)

print(
    report_path
)

print("")
print(
    "NO MODEL WAS RETRAINED."
)

print(
    "NO THRESHOLD WAS CHANGED."
)

print(
    "FINAL TEST PREDICTIONS WERE NOT MODIFIED."
)
