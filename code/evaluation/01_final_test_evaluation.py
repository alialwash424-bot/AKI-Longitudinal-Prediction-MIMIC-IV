import os
import math

import numpy as np
import pandas as pd


# =============================================================================
# STEP 56C — ONE-TIME HELD-OUT TEST EVALUATION
#
# IMPORTANT:
#   - Uses ONLY the already-fitted locked models.
#   - Uses ONLY preprocessing parameters stored in those models.
#   - Uses ONLY validation-selected thresholds locked before test evaluation.
#   - DOES NOT retrain.
#   - DOES NOT select thresholds on test data.
#   - DOES NOT modify models, predictors, split manifest, or modeling table.
#
# This is the one-time held-out test evaluation.
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

modeling_path = os.path.join(
    folder,
    "longitudinal_modeling_table.csv"
)

split_path = os.path.join(
    folder,
    "patient_level_split_manifest.csv"
)

lock_path = os.path.join(
    folder,
    "step56A_model_lock.txt"
)


# =============================================================================
# LOCKED MODELS AND VALIDATION-SELECTED THRESHOLDS
# =============================================================================

models = {

    "6h": {
        "target": "aki_within_6h",

        "model_path": os.path.join(
            folder,
            "baseline_6h_logistic_model.npz"
        ),

        "threshold": 0.496171,

        "output_predictions": os.path.join(
            folder,
            "FINAL_TEST_6h_predictions.csv"
        ),

        "output_report": os.path.join(
            folder,
            "FINAL_TEST_6h_report.txt"
        ),
    },

    "12h": {
        "target": "aki_within_12h",

        "model_path": os.path.join(
            folder,
            "baseline_12h_logistic_model.npz"
        ),

        "threshold": 0.512416,

        "output_predictions": os.path.join(
            folder,
            "FINAL_TEST_12h_predictions.csv"
        ),

        "output_report": os.path.join(
            folder,
            "FINAL_TEST_12h_report.txt"
        ),
    },

    "24h": {
        "target": "aki_within_24h",

        "model_path": os.path.join(
            folder,
            "baseline_24h_logistic_model.npz"
        ),

        "threshold": 0.466857,

        "output_predictions": os.path.join(
            folder,
            "FINAL_TEST_24h_predictions.csv"
        ),

        "output_report": os.path.join(
            folder,
            "FINAL_TEST_24h_report.txt"
        ),
    },
}


# =============================================================================
# SAFETY CHECKS
# =============================================================================

required_paths = [
    modeling_path,
    split_path,
    lock_path,
]

for info in models.values():
    required_paths.append(
        info["model_path"]
    )


missing = [
    path
    for path in required_paths
    if not os.path.exists(path)
]

if missing:

    raise RuntimeError(
        "SAFETY STOP.\n"
        "Required locked files are missing:\n\n"
        + "\n".join(missing)
    )


# Confirm model-lock manifest says PASS.

with open(
    lock_path,
    "r",
    encoding="utf-8"
) as f:

    lock_text = f.read()


if "FINAL MODEL-LOCK VERDICT: PASS" not in lock_text:

    raise RuntimeError(
        "SAFETY STOP.\n"
        "The pre-test model lock does not show PASS.\n"
        "Test evaluation will not proceed."
    )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def sigmoid(z):

    z = np.clip(
        z,
        -50.0,
        50.0
    )

    return 1.0 / (
        1.0 + np.exp(-z)
    )


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
        return float("nan")

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
            (i + 1)
            + j
        ) / 2.0

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

    return float(auc)


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
        return float("nan")

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

    recall = (
        tp
        / positive_count
    )

    # Average precision:
    # sum precision at each positive observation
    # divided by total positives.

    auprc = float(
        precision[
            y_sorted == 1
        ].sum()
        / positive_count
    )

    return auprc


def safe_divide(a, b):

    if b == 0:
        return float("nan")

    return a / b


def evaluate_binary_model(
    y,
    probabilities,
    threshold
):

    y = np.asarray(
        y,
        dtype=np.int8
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    predicted = (
        probabilities >= threshold
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
            (
                probabilities - y
            ) ** 2
        )
    )

    clipped = np.clip(
        probabilities,
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
                probabilities
            ),

        "AUPRC":
            calculate_auprc(
                y,
                probabilities
            ),

        "Brier":
            brier,

        "Log loss":
            log_loss,

        "Threshold":
            threshold,

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

        "TP":
            tp,

        "TN":
            tn,

        "FP":
            fp,

        "FN":
            fn,
    }


# =============================================================================
# LOAD SPLIT MANIFEST
# =============================================================================

print("")
print("=" * 80)
print("STEP 56C — FINAL HELD-OUT TEST EVALUATION")
print("=" * 80)

print("")
print(
    "Loading patient-level split manifest..."
)

split = pd.read_csv(
    split_path
)


required_split_columns = {
    "subject_id",
    "split",
}

if not required_split_columns.issubset(
    split.columns
):

    raise RuntimeError(
        "SAFETY STOP.\n"
        "Split manifest does not contain "
        "subject_id and split."
    )


split["subject_id"] = pd.to_numeric(
    split["subject_id"],
    errors="raise"
).astype(
    "int64"
)


# Strictly identify test patients.

test_subjects = set(
    split.loc[
        split["split"].astype(str)
        == "test",
        "subject_id"
    ].tolist()
)


if len(test_subjects) == 0:

    raise RuntimeError(
        "SAFETY STOP.\n"
        "No test patients found."
    )


print(
    "Locked test patients:",
    len(test_subjects)
)


# =============================================================================
# EVALUATE EACH LOCKED MODEL
# =============================================================================

summary = []

for horizon in [
    "6h",
    "12h",
    "24h",
]:

    info = models[horizon]

    print("")
    print("=" * 80)
    print(
        f"{horizon.upper()} FINAL TEST EVALUATION"
    )
    print("=" * 80)


    # -------------------------------------------------------------------------
    # LOAD LOCKED MODEL
    # -------------------------------------------------------------------------

    model = np.load(
        info["model_path"],
        allow_pickle=True
    )


    stored_target = str(
        model["target"][0]
    )

    if stored_target != info["target"]:

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            f"Model target is {stored_target}, "
            f"expected {info['target']}."
        )


    predictors = [
        str(x)
        for x in model[
            "active_predictors"
        ].tolist()
    ]

    weights = np.asarray(
        model["weights"],
        dtype=np.float64
    )

    bias = float(
        model["bias"][0]
    )

    means = np.asarray(
        model["means"],
        dtype=np.float64
    )

    standard_deviations = np.asarray(
        model[
            "standard_deviations"
        ],
        dtype=np.float64
    )

    exposure_indices = set(
        np.asarray(
            model[
                "exposure_indices"
            ],
            dtype=np.int64
        ).tolist()
    )


    n_predictors = len(
        predictors
    )


    if not (
        len(weights)
        == n_predictors
        == len(means)
        == len(standard_deviations)
    ):

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            "Locked model array dimensions "
            "do not agree."
        )


    if n_predictors != 252:

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            f"Expected 252 active predictors, "
            f"found {n_predictors}."
        )


    if np.any(
        standard_deviations <= 0
    ):

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            "Nonpositive stored standard deviation."
        )


    # -------------------------------------------------------------------------
    # READ ONLY NEEDED MODELING-TABLE COLUMNS
    # -------------------------------------------------------------------------

    required_columns = [
        "subject_id",
        "stay_id",
        "landmark_time",
        info["target"],
    ] + predictors


    print(
        "Reading locked test rows from "
        "longitudinal modeling table..."
    )


    chunks = []

    for chunk in pd.read_csv(
        modeling_path,
        usecols=required_columns,
        chunksize=5000,
        low_memory=False
    ):

        subject_numeric = pd.to_numeric(
            chunk["subject_id"],
            errors="coerce"
        )

        mask = subject_numeric.isin(
            test_subjects
        )

        if mask.any():

            selected = chunk.loc[
                mask
            ].copy()

            chunks.append(
                selected
            )


    if not chunks:

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            "No test rows found in modeling table."
        )


    test = pd.concat(
        chunks,
        ignore_index=True
    )


    del chunks


    # -------------------------------------------------------------------------
    # TARGET ELIGIBILITY
    # -------------------------------------------------------------------------

    labels = pd.to_numeric(
        test[
            info["target"]
        ],
        errors="coerce"
    )


    eligible = labels.isin(
        [0, 1]
    )


    test = test.loc[
        eligible
    ].reset_index(
        drop=True
    )

    labels = labels.loc[
        eligible
    ].astype(
        "int8"
    ).to_numpy()


    if len(test) == 0:

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            "No assessable test landmarks."
        )


    # -------------------------------------------------------------------------
    # EXACT LOCKED PREPROCESSING
    #
    # Exposure variables:
    #   missing -> 0
    #
    # Other variables:
    #   missing -> stored training-derived mean
    #
    # Then standardize using stored training mean/SD.
    # -------------------------------------------------------------------------

    X = np.empty(
        (
            len(test),
            n_predictors
        ),
        dtype=np.float64
    )


    for j, predictor in enumerate(
        predictors
    ):

        values = pd.to_numeric(
            test[predictor],
            errors="coerce"
        ).to_numpy(
            dtype=np.float64
        )


        nonfinite = ~np.isfinite(
            values
        )


        if j in exposure_indices:

            values[
                nonfinite
            ] = 0.0

        else:

            values[
                nonfinite
            ] = means[j]


        X[:, j] = (
            values - means[j]
        ) / standard_deviations[j]


    if not np.isfinite(X).all():

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            "Nonfinite values remain after "
            "locked preprocessing."
        )


    # -------------------------------------------------------------------------
    # LOCKED MODEL PREDICTIONS
    # -------------------------------------------------------------------------

    logits = (
        X @ weights
        + bias
    )

    probabilities = sigmoid(
        logits
    )


    if not np.isfinite(
        probabilities
    ).all():

        raise RuntimeError(
            f"SAFETY STOP — {horizon}.\n"
            "Nonfinite predicted probabilities."
        )


    # -------------------------------------------------------------------------
    # FINAL TEST METRICS
    # -------------------------------------------------------------------------

    metrics = evaluate_binary_model(
        labels,
        probabilities,
        info["threshold"]
    )


    positive = int(
        labels.sum()
    )

    negative = int(
        len(labels) - positive
    )

    prevalence = (
        positive
        / len(labels)
    )


    # -------------------------------------------------------------------------
    # SAVE FINAL TEST PREDICTIONS
    # -------------------------------------------------------------------------

    prediction_output = pd.DataFrame(
        {
            "subject_id":
                pd.to_numeric(
                    test["subject_id"],
                    errors="raise"
                ).astype(
                    "int64"
                ),

            "stay_id":
                pd.to_numeric(
                    test["stay_id"],
                    errors="raise"
                ).astype(
                    "int64"
                ),

            "landmark_time":
                test[
                    "landmark_time"
                ].astype(
                    str
                ),

            info["target"]:
                labels,

            "predicted_probability":
                probabilities,

            "locked_threshold":
                info["threshold"],

            "predicted_class":
                (
                    probabilities
                    >= info["threshold"]
                ).astype(
                    "int8"
                ),
        }
    )


    prediction_output.to_csv(
        info["output_predictions"],
        index=False
    )


    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    report_lines = [
        f"{horizon.upper()} FINAL HELD-OUT TEST REPORT",
        "",
        "MODEL STATUS:",
        "Model locked before test evaluation",
        "No retraining performed",
        "No test-derived preprocessing performed",
        "No test-derived threshold selection performed",
        "",
        f"Target: {info['target']}",
        f"Active predictors: {n_predictors}",
        f"Locked validation-selected threshold: {info['threshold']:.6f}",
        "",
        "TEST POPULATION:",
        f"Assessable test landmarks: {len(labels)}",
        f"Test positives: {positive}",
        f"Test negatives: {negative}",
        f"Test prevalence: {prevalence:.2%}",
        "",
        "FINAL HELD-OUT TEST PERFORMANCE:",
        f"AUROC: {metrics['AUROC']:.6f}",
        f"AUPRC: {metrics['AUPRC']:.6f}",
        f"Brier score: {metrics['Brier']:.6f}",
        f"Log loss: {metrics['Log loss']:.6f}",
        "",
        "PERFORMANCE AT LOCKED VALIDATION-SELECTED THRESHOLD:",
        f"Threshold: {metrics['Threshold']:.6f}",
        f"Sensitivity: {metrics['Sensitivity']:.6f}",
        f"Specificity: {metrics['Specificity']:.6f}",
        f"Positive predictive value: {metrics['PPV']:.6f}",
        f"Negative predictive value: {metrics['NPV']:.6f}",
        f"F1 score: {metrics['F1']:.6f}",
        f"TP: {metrics['TP']}",
        f"TN: {metrics['TN']}",
        f"FP: {metrics['FP']}",
        f"FN: {metrics['FN']}",
        "",
        "IMPORTANT:",
        "These are the final one-time held-out test results.",
        "The test set must not be used for model tuning or threshold reselection.",
    ]


    with open(
        info["output_report"],
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(
                report_lines
            )
        )


    print(
        "\n".join(
            report_lines
        )
    )


    summary.append(
        {
            "Horizon": horizon,
            "Test landmarks": len(labels),
            "Positives": positive,
            "Negatives": negative,
            "Prevalence": prevalence,
            "AUROC": metrics["AUROC"],
            "AUPRC": metrics["AUPRC"],
            "Brier": metrics["Brier"],
            "Log loss": metrics["Log loss"],
            "Threshold": metrics["Threshold"],
            "Sensitivity": metrics["Sensitivity"],
            "Specificity": metrics["Specificity"],
            "PPV": metrics["PPV"],
            "NPV": metrics["NPV"],
            "F1": metrics["F1"],
        }
    )


    del test
    del X
    del labels
    del probabilities
    del logits


# =============================================================================
# SAVE COMBINED FINAL TEST SUMMARY
# =============================================================================

summary_df = pd.DataFrame(
    summary
)

summary_path = os.path.join(
    folder,
    "FINAL_TEST_summary.csv"
)

summary_df.to_csv(
    summary_path,
    index=False
)


summary_report_path = os.path.join(
    folder,
    "FINAL_TEST_summary.txt"
)


lines = [
    "FINAL LOCKED HELD-OUT TEST SUMMARY",
    "=" * 80,
    "",
    "Models were locked before test evaluation.",
    "No retraining was performed.",
    "No test-derived preprocessing was performed.",
    "No threshold was selected using test outcomes.",
    "",
]


for row in summary:

    lines.extend(
        [
            f"{row['Horizon'].upper()}:",
            f"  Test landmarks: {row['Test landmarks']}",
            f"  Positives: {row['Positives']}",
            f"  Negatives: {row['Negatives']}",
            f"  Prevalence: {row['Prevalence']:.2%}",
            f"  AUROC: {row['AUROC']:.6f}",
            f"  AUPRC: {row['AUPRC']:.6f}",
            f"  Brier: {row['Brier']:.6f}",
            f"  Log loss: {row['Log loss']:.6f}",
            f"  Locked threshold: {row['Threshold']:.6f}",
            f"  Sensitivity: {row['Sensitivity']:.6f}",
            f"  Specificity: {row['Specificity']:.6f}",
            f"  PPV: {row['PPV']:.6f}",
            f"  NPV: {row['NPV']:.6f}",
            f"  F1: {row['F1']:.6f}",
            "",
        ]
    )


lines.extend(
    [
        "=" * 80,
        "TEST EVALUATION COMPLETE",
        "",
        "IMPORTANT:",
        "Do not tune or retrain the models in response to these test results.",
    ]
)


with open(
    summary_report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            lines
        )
    )


print("")
print("=" * 80)
print("FINAL HELD-OUT TEST EVALUATION COMPLETE")
print("=" * 80)

print("")
print(
    "Summary CSV:",
    summary_path
)

print(
    "Summary TXT:",
    summary_report_path
)

print("")
print(
    "NO MODEL WAS RETRAINED."
)

print(
    "NO THRESHOLD WAS RESELECTED."
)

print(
    "THE HELD-OUT TEST SET HAS NOW BEEN EVALUATED ONCE."
)
