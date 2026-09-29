import csv
import os

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# PRESPECIFIED BASELINE
# -----------------------------------------------------------------------------

folder = os.path.expanduser("~/Documents/AKI_Research")
source_path = os.path.join(folder, "longitudinal_modeling_table.csv")
split_path = os.path.join(folder, "patient_level_split_manifest.csv")
predictor_path = os.path.join(folder, "final_model_predictors.txt")
group_path = os.path.join(folder, "final_model_predictor_groups.csv")

model_path = os.path.join(folder, "ablation_12h_vital_context_model.npz")
coefficient_path = os.path.join(folder, "ablation_12h_vital_context_coefficients.csv")
prediction_path = os.path.join(folder, "ablation_12h_vital_context_validation_predictions.csv")
report_path = os.path.join(folder, "ablation_12h_vital_context_validation_report.txt")

target = "aki_within_12h"
chunk_size = 5000
mini_batch_size = 1024
epochs = 5
learning_rate = 0.01
l2_penalty = 0.0001
random_seed = 20260927

for path in (source_path, split_path, predictor_path, group_path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required file: {path}")

with open(predictor_path, "r", encoding="utf-8") as predictor_file:
    predictors = [line.strip() for line in predictor_file if line.strip()]

groups = pd.read_csv(group_path)
group_by_predictor = dict(zip(groups["predictor"], groups["group"]))
if set(predictors) != set(group_by_predictor):
    raise ValueError("Predictor list and predictor-group file do not match")

# Ablation design: retain physiological trajectories and the safe landmark-time
# context predictor, while excluding all modifiable-exposure trajectories.
predictors = [
    name
    for name in predictors
    if group_by_predictor[name] in {"vital_trajectory", "landmark_context"}
]
vital_count = sum(group_by_predictor[name] == "vital_trajectory" for name in predictors)
context_count = sum(group_by_predictor[name] == "landmark_context" for name in predictors)
if vital_count != 120 or context_count != 1 or len(predictors) != 121:
    raise ValueError(
        "Unexpected vital/context predictor composition: "
        f"vital={vital_count}, context={context_count}, total={len(predictors)}"
    )

header = pd.read_csv(source_path, nrows=0).columns.tolist()
required_columns = ["subject_id", "stay_id", "landmark_time", target, *predictors]
missing_columns = [name for name in required_columns if name not in header]
if missing_columns:
    raise ValueError(f"Modeling table is missing columns: {missing_columns}")

split_manifest = pd.read_csv(split_path, usecols=["subject_id", "split"])
split_manifest["subject_id"] = pd.to_numeric(
    split_manifest["subject_id"], errors="raise"
).astype("int64")
if split_manifest["subject_id"].duplicated().any():
    raise ValueError("Patient split manifest contains duplicate subject IDs")
split_by_subject = dict(zip(split_manifest["subject_id"], split_manifest["split"]))

exposure_indices = np.array(
    [
        index
        for index, name in enumerate(predictors)
        if group_by_predictor[name] == "modifiable_exposure_trajectory"
    ],
    dtype=int,
)
nonexposure_indices = np.array(
    [index for index in range(len(predictors)) if index not in set(exposure_indices)],
    dtype=int,
)


def numeric_matrix(frame):
    return (
        frame[predictors]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )


def sigmoid(values):
    values = np.clip(values, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def roc_auc(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        average_rank = 0.5 * ((start + 1) + end)
        ranks[order[start:end]] = average_rank
        start = end
    rank_sum = ranks[y_true == 1].sum()
    return float((rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives))


def average_precision(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_y = y_true[order]
    cumulative_positive = np.cumsum(ordered_y)
    precision = cumulative_positive / np.arange(1, len(ordered_y) + 1)
    return float(precision[ordered_y == 1].sum() / positives)


def youden_threshold(y_true, scores):
    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(y_true.sum())
    negatives = len(y_true) - positives
    order = np.argsort(-scores, kind="mergesort")
    ordered_y = y_true[order]
    ordered_scores = scores[order]
    true_positive_rate = np.cumsum(ordered_y) / positives
    false_positive_rate = np.cumsum(1 - ordered_y) / negatives
    index = int(np.argmax(true_positive_rate - false_positive_rate))
    return float(ordered_scores[index])


def classification_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true, dtype=np.int8)
    predicted = np.asarray(scores) >= threshold
    tp = int(((predicted == 1) & (y_true == 1)).sum())
    tn = int(((predicted == 0) & (y_true == 0)).sum())
    fp = int(((predicted == 1) & (y_true == 0)).sum())
    fn = int(((predicted == 0) & (y_true == 1)).sum())
    sensitivity = tp / (tp + fn) if tp + fn else float("nan")
    specificity = tn / (tn + fp) if tn + fp else float("nan")
    ppv = tp / (tp + fp) if tp + fp else float("nan")
    npv = tn / (tn + fn) if tn + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "f1": f1,
    }


# -----------------------------------------------------------------------------
# PASS 1: TRAINING-ONLY PREPROCESSING STATISTICS AND CLASS COUNTS
# -----------------------------------------------------------------------------

feature_count = len(predictors)
feature_sum = np.zeros(feature_count, dtype=np.float64)
feature_sumsq = np.zeros(feature_count, dtype=np.float64)
feature_observed = np.zeros(feature_count, dtype=np.int64)
training_rows = 0
training_positive = 0
training_negative = 0

print("Pass 1: fitting preprocessing statistics on training patients only...")

read_columns = ["subject_id", target, *predictors]
for chunk_number, chunk in enumerate(
    pd.read_csv(source_path, usecols=read_columns, chunksize=chunk_size, low_memory=False),
    start=1,
):
    subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
    labels = pd.to_numeric(chunk[target], errors="coerce")
    split = subject.map(split_by_subject)
    mask = (split == "train") & labels.isin([0, 1])
    if not mask.any():
        continue

    selected = chunk.loc[mask]
    y = labels.loc[mask].to_numpy(dtype=np.int8)
    matrix = numeric_matrix(selected)
    observed = ~np.isnan(matrix)

    # This branch is retained for consistency; exposure indices are empty here.
    if len(exposure_indices):
        observed[:, exposure_indices] = True
    zero_filled = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    feature_sum += zero_filled.sum(axis=0)
    feature_sumsq += np.square(zero_filled).sum(axis=0)
    feature_observed += observed.sum(axis=0)
    training_rows += len(y)
    training_positive += int(y.sum())
    training_negative += int(len(y) - y.sum())

    if chunk_number % 10 == 0:
        print("Statistics chunks:", chunk_number, "Eligible training rows:", training_rows)

if training_positive == 0 or training_negative == 0:
    raise RuntimeError("Training data does not contain both outcome classes")

means = np.divide(
    feature_sum,
    feature_observed,
    out=np.zeros(feature_count, dtype=np.float64),
    where=feature_observed > 0,
)
variances = np.divide(
    feature_sumsq,
    feature_observed,
    out=np.zeros(feature_count, dtype=np.float64),
    where=feature_observed > 0,
) - np.square(means)
variances = np.maximum(variances, 0.0)
standard_deviations = np.sqrt(variances)
active_mask = (feature_observed > 0) & (standard_deviations > 1e-8)
active_indices = np.flatnonzero(active_mask)
active_predictors = [predictors[index] for index in active_indices]

if not len(active_indices):
    raise RuntimeError("No usable predictors remain after training-only screening")


def transform(frame):
    matrix = numeric_matrix(frame)
    if len(exposure_indices):
        exposure_block = matrix[:, exposure_indices]
        matrix[:, exposure_indices] = np.nan_to_num(
            exposure_block, nan=0.0, posinf=0.0, neginf=0.0
        )
    if len(nonexposure_indices):
        block = matrix[:, nonexposure_indices]
        missing = ~np.isfinite(block)
        if missing.any():
            replacement = np.broadcast_to(means[nonexposure_indices], block.shape)
            block[missing] = replacement[missing]
        matrix[:, nonexposure_indices] = block
    active = matrix[:, active_indices]
    active = (active - means[active_indices]) / standard_deviations[active_indices]
    return np.clip(active, -10.0, 10.0).astype(np.float32, copy=False)


# -----------------------------------------------------------------------------
# TRAIN L2-REGULARIZED LOGISTIC REGRESSION WITH MINI-BATCH ADAM
# -----------------------------------------------------------------------------

weights = np.zeros(len(active_indices), dtype=np.float64)
bias = 0.0
first_moment = np.zeros_like(weights)
second_moment = np.zeros_like(weights)
bias_first_moment = 0.0
bias_second_moment = 0.0
adam_step = 0
rng = np.random.default_rng(random_seed)

positive_weight = training_rows / (2.0 * training_positive)
negative_weight = training_rows / (2.0 * training_negative)

print()
print("Training the 12-hour vital/context ablation model...")

for epoch in range(1, epochs + 1):
    epoch_rows = 0
    epoch_batches = 0
    for chunk in pd.read_csv(
        source_path, usecols=read_columns, chunksize=chunk_size, low_memory=False
    ):
        subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
        labels = pd.to_numeric(chunk[target], errors="coerce")
        split = subject.map(split_by_subject)
        mask = (split == "train") & labels.isin([0, 1])
        if not mask.any():
            continue

        selected = chunk.loc[mask]
        y_all = labels.loc[mask].to_numpy(dtype=np.float64)
        x_all = transform(selected)
        order = rng.permutation(len(y_all))

        for start in range(0, len(order), mini_batch_size):
            batch_indices = order[start : start + mini_batch_size]
            x = x_all[batch_indices]
            y = y_all[batch_indices]
            probabilities = sigmoid(x @ weights + bias)
            sample_weights = np.where(y == 1.0, positive_weight, negative_weight)
            errors = (probabilities - y) * sample_weights
            denominator = sample_weights.sum()
            gradient = (x.T @ errors) / denominator + l2_penalty * weights
            bias_gradient = float(errors.sum() / denominator)

            adam_step += 1
            first_moment = 0.9 * first_moment + 0.1 * gradient
            second_moment = 0.999 * second_moment + 0.001 * np.square(gradient)
            bias_first_moment = 0.9 * bias_first_moment + 0.1 * bias_gradient
            bias_second_moment = 0.999 * bias_second_moment + 0.001 * bias_gradient**2

            corrected_first = first_moment / (1.0 - 0.9**adam_step)
            corrected_second = second_moment / (1.0 - 0.999**adam_step)
            corrected_bias_first = bias_first_moment / (1.0 - 0.9**adam_step)
            corrected_bias_second = bias_second_moment / (1.0 - 0.999**adam_step)

            weights -= learning_rate * corrected_first / (
                np.sqrt(corrected_second) + 1e-8
            )
            bias -= learning_rate * corrected_bias_first / (
                np.sqrt(corrected_bias_second) + 1e-8
            )
            epoch_batches += 1

        epoch_rows += len(y_all)

    print(
        "Epoch:", epoch,
        "of", epochs,
        "Training rows:", epoch_rows,
        "Mini-batches:", epoch_batches,
    )


# -----------------------------------------------------------------------------
# VALIDATION ONLY — TEST PATIENTS ARE NOT EVALUATED
# -----------------------------------------------------------------------------

print()
print("Evaluating on validation patients only...")

validation_parts = []
validation_read_columns = [
    "subject_id", "stay_id", "landmark_time", target, *predictors
]
for chunk in pd.read_csv(
    source_path,
    usecols=validation_read_columns,
    chunksize=chunk_size,
    low_memory=False,
):
    subject = pd.to_numeric(chunk["subject_id"], errors="coerce")
    labels = pd.to_numeric(chunk[target], errors="coerce")
    split = subject.map(split_by_subject)
    mask = (split == "validation") & labels.isin([0, 1])
    if not mask.any():
        continue

    selected = chunk.loc[mask]
    probabilities = sigmoid(transform(selected) @ weights + bias)
    validation_parts.append(
        pd.DataFrame(
            {
                "subject_id": pd.to_numeric(selected["subject_id"], errors="raise").astype("int64"),
                "stay_id": pd.to_numeric(selected["stay_id"], errors="raise").astype("int64"),
                "landmark_time": selected["landmark_time"].astype(str),
                "aki_within_12h": labels.loc[mask].astype("int8"),
                "predicted_probability": probabilities,
            }
        )
    )

if not validation_parts:
    raise RuntimeError("No assessable validation rows were found")

validation = pd.concat(validation_parts, ignore_index=True)
y_validation = validation[target].to_numpy(dtype=np.int8)
p_validation = validation["predicted_probability"].to_numpy(dtype=np.float64)

auc = roc_auc(y_validation, p_validation)
auprc = average_precision(y_validation, p_validation)
brier = float(np.mean(np.square(p_validation - y_validation)))
clipped_probability = np.clip(p_validation, 1e-12, 1.0 - 1e-12)
log_loss = float(
    -np.mean(
        y_validation * np.log(clipped_probability)
        + (1 - y_validation) * np.log(1.0 - clipped_probability)
    )
)
threshold = youden_threshold(y_validation, p_validation)
threshold_metrics = classification_metrics(y_validation, p_validation, threshold)
prevalence = float(y_validation.mean())

validation.to_csv(prediction_path, index=False)

with open(coefficient_path, "w", encoding="utf-8", newline="") as coefficient_file:
    writer = csv.writer(coefficient_file)
    writer.writerow(
        ["predictor", "group", "coefficient_per_sd", "odds_ratio_per_sd", "training_mean", "training_sd"]
    )
    for local_index, source_index in sorted(
        enumerate(active_indices), key=lambda item: -abs(weights[item[0]])
    ):
        name = predictors[source_index]
        coefficient = float(weights[local_index])
        writer.writerow(
            [
                name,
                group_by_predictor[name],
                coefficient,
                float(np.exp(np.clip(coefficient, -20.0, 20.0))),
                float(means[source_index]),
                float(standard_deviations[source_index]),
            ]
        )

np.savez_compressed(
    model_path,
    target=np.asarray([target]),
    active_predictors=np.asarray(active_predictors),
    weights=weights,
    bias=np.asarray([bias]),
    source_indices=active_indices,
    means=means,
    standard_deviations=standard_deviations,
    exposure_indices=exposure_indices,
    random_seed=np.asarray([random_seed]),
    epochs=np.asarray([epochs]),
    learning_rate=np.asarray([learning_rate]),
    l2_penalty=np.asarray([l2_penalty]),
)

lines = [
    "TWELVE-HOUR AKI VITAL/CONTEXT ABLATION — VALIDATION REPORT",
    "",
    "DATA PARTITIONING:",
    "Training patients only: preprocessing and coefficient fitting",
    "Validation patients only: current performance evaluation and threshold selection",
    "Test patients: not evaluated",
    "",
    f"Prespecified predictors: {len(predictors)}",
    f"Active nonconstant predictors: {len(active_predictors)}",
    f"Dropped all-missing/constant predictors: {len(predictors) - len(active_predictors)}",
    f"Assessable training landmarks: {training_rows}",
    f"Training positives: {training_positive}",
    f"Training negatives: {training_negative}",
    f"Assessable validation landmarks: {len(validation)}",
    f"Validation positives: {int(y_validation.sum())}",
    f"Validation negatives: {int(len(y_validation) - y_validation.sum())}",
    f"Validation prevalence: {100.0 * prevalence:.2f}%",
    "",
    "MODEL:",
    "L2-regularized logistic regression trained with mini-batch Adam",
    f"Epochs: {epochs}",
    f"Learning rate: {learning_rate}",
    f"L2 penalty: {l2_penalty}",
    "Vital missing values: training-derived mean imputation",
    "Exposure trajectories: excluded by ablation design",
    "Class imbalance: inverse-frequency training weights",
    "",
    "VALIDATION DISCRIMINATION AND CALIBRATION:",
    f"AUROC: {auc:.6f}",
    f"AUPRC: {auprc:.6f}",
    f"Brier score: {brier:.6f}",
    f"Log loss: {log_loss:.6f}",
    "",
    "VALIDATION-SELECTED YOUDEN THRESHOLD:",
    f"Threshold: {threshold:.6f}",
    f"Sensitivity: {threshold_metrics['sensitivity']:.6f}",
    f"Specificity: {threshold_metrics['specificity']:.6f}",
    f"Positive predictive value: {threshold_metrics['ppv']:.6f}",
    f"Negative predictive value: {threshold_metrics['npv']:.6f}",
    f"F1 score: {threshold_metrics['f1']:.6f}",
    f"TP: {threshold_metrics['tp']}",
    f"TN: {threshold_metrics['tn']}",
    f"FP: {threshold_metrics['fp']}",
    f"FN: {threshold_metrics['fn']}",
    "",
    "IMPORTANT:",
    "These are validation results, not final test-set results.",
    "Do not report them as final external or held-out performance.",
    "The test split remains reserved until the complete model is locked.",
    "",
    f"Saved model: {model_path}",
    f"Coefficient table: {coefficient_path}",
    f"Validation predictions: {prediction_path}",
    f"Report: {report_path}",
]

with open(report_path, "w", encoding="utf-8") as report_file:
    report_file.write("\n".join(lines))

print()
print("TWELVE-HOUR VITAL/CONTEXT ABLATION COMPLETE")
print("\n".join(lines))
