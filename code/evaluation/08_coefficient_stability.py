import os
import csv
import numpy as np


# =============================================================================
# STEP 59C — CROSS-HORIZON COEFFICIENT STABILITY AUDIT
# =============================================================================
#
# PURPOSE
# -------
# Quantify how stable the standardized coefficients are across the locked
# 6h, 12h, and 24h logistic models.
#
# READ-ONLY:
# - No model retraining
# - No threshold changes
# - No test prediction changes
# - No research/model files modified
#
# Outputs:
#   step59C_coefficient_stability.csv
#   step59C_coefficient_stability_report.txt
#
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

model_paths = {
    "6h": os.path.join(folder, "baseline_6h_logistic_model.npz"),
    "12h": os.path.join(folder, "baseline_12h_logistic_model.npz"),
    "24h": os.path.join(folder, "baseline_24h_logistic_model.npz"),
}

output_csv = os.path.join(
    folder,
    "step59C_coefficient_stability.csv"
)

output_report = os.path.join(
    folder,
    "step59C_coefficient_stability_report.txt"
)


# =============================================================================
# SETTINGS
# =============================================================================

# A coefficient is treated as "near zero" if its absolute standardized
# coefficient is below this value.
near_zero_threshold = 0.05

# A direction-changing predictor is called "material" only if coefficients
# on BOTH sides of zero reach at least this absolute magnitude.
#
# Example:
#   +0.08, -0.09, -0.10 -> material direction change
#
# But:
#   +0.001, -0.08, -0.09 -> not material; one side is effectively near zero.
material_sign_threshold = 0.05

# Strong stable predictors:
# same nonzero direction across all horizons AND mean absolute coefficient
# at least this value.
strong_mean_abs_threshold = 0.10


# =============================================================================
# HELPERS
# =============================================================================

def decode_strings(array):
    result = []
    for value in np.asarray(array).ravel():
        if isinstance(value, bytes):
            result.append(value.decode("utf-8"))
        else:
            result.append(str(value))
    return result


def safe_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y):
        raise RuntimeError("Correlation vectors have different lengths.")

    if len(x) < 2:
        return np.nan

    if np.std(x) == 0 or np.std(y) == 0:
        return np.nan

    return float(np.corrcoef(x, y)[0, 1])


def sign_label(value):
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def classify_predictor(c6, c12, c24):
    values = np.array([c6, c12, c24], dtype=float)

    positive = values > 0
    negative = values < 0

    same_positive = bool(np.all(positive))
    same_negative = bool(np.all(negative))
    stable_direction = same_positive or same_negative

    mean_abs = float(np.mean(np.abs(values)))
    max_abs = float(np.max(np.abs(values)))
    min_abs = float(np.min(np.abs(values)))

    if stable_direction:
        if mean_abs >= strong_mean_abs_threshold:
            return "stable_direction_strong"
        else:
            return "stable_direction_weaker"

    # Direction-changing predictor.
    positive_values = values[values > 0]
    negative_values = values[values < 0]

    max_positive = (
        float(np.max(positive_values))
        if len(positive_values)
        else 0.0
    )

    max_negative_magnitude = (
        float(np.max(np.abs(negative_values)))
        if len(negative_values)
        else 0.0
    )

    # To call a reversal material, both sides of zero must have
    # coefficients of meaningful magnitude.
    material_reversal = (
        max_positive >= material_sign_threshold
        and max_negative_magnitude >= material_sign_threshold
    )

    if material_reversal:
        return "direction_changing_material"

    return "direction_changing_near_zero_or_weak"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 59C — CROSS-HORIZON COEFFICIENT STABILITY AUDIT")
print("=" * 80)
print()
print("READ-ONLY ANALYSIS")
print("NO MODEL WILL BE RETRAINED.")
print("NO THRESHOLD WILL BE CHANGED.")
print("NO TEST PREDICTIONS WILL BE MODIFIED.")
print()


# =============================================================================
# LOAD LOCKED MODELS
# =============================================================================

models = {}

for horizon, path in model_paths.items():

    print(f"Loading {horizon} locked model:")
    print(path)

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    data = np.load(path, allow_pickle=False)

    required_keys = {
        "active_predictors",
        "weights",
        "target",
    }

    missing_keys = required_keys.difference(data.files)

    if missing_keys:
        raise RuntimeError(
            f"{horizon} model missing keys: {sorted(missing_keys)}"
        )

    predictors = decode_strings(data["active_predictors"])
    weights = np.asarray(data["weights"], dtype=float).ravel()
    target = decode_strings(data["target"])

    if len(predictors) != len(weights):
        raise RuntimeError(
            f"{horizon}: predictor/weight length mismatch."
        )

    if not np.all(np.isfinite(weights)):
        raise RuntimeError(
            f"{horizon}: non-finite coefficient detected."
        )

    models[horizon] = {
        "predictors": predictors,
        "weights": weights,
        "target": target,
    }

    print(f"  Target: {target}")
    print(f"  Predictors: {len(predictors)}")
    print()


# =============================================================================
# VERIFY IDENTICAL PREDICTOR ORDER
# =============================================================================

p6 = models["6h"]["predictors"]
p12 = models["12h"]["predictors"]
p24 = models["24h"]["predictors"]

same_6_12 = p6 == p12
same_6_24 = p6 == p24
same_12_24 = p12 == p24

identical_order = same_6_12 and same_6_24 and same_12_24

print("=" * 80)
print("STRUCTURAL VERIFICATION")
print("=" * 80)
print(f"6h predictors:  {len(p6)}")
print(f"12h predictors: {len(p12)}")
print(f"24h predictors: {len(p24)}")
print(f"Identical predictor order: {identical_order}")
print()

if not identical_order:
    raise RuntimeError(
        "SAFETY STOP: predictor order differs across locked models."
    )


# =============================================================================
# COEFFICIENT VECTORS
# =============================================================================

w6 = models["6h"]["weights"]
w12 = models["12h"]["weights"]
w24 = models["24h"]["weights"]


# =============================================================================
# PAIRWISE CORRELATIONS
# =============================================================================

corr_6_12 = safe_corr(w6, w12)
corr_6_24 = safe_corr(w6, w24)
corr_12_24 = safe_corr(w12, w24)

print("=" * 80)
print("PAIRWISE COEFFICIENT CORRELATIONS")
print("=" * 80)
print(f"6h vs 12h:  {corr_6_12:.6f}")
print(f"6h vs 24h:  {corr_6_24:.6f}")
print(f"12h vs 24h: {corr_12_24:.6f}")
print()


# =============================================================================
# PER-PREDICTOR STABILITY
# =============================================================================

rows = []

for i, predictor in enumerate(p6):

    c6 = float(w6[i])
    c12 = float(w12[i])
    c24 = float(w24[i])

    values = np.array([c6, c12, c24], dtype=float)

    mean_coef = float(np.mean(values))
    mean_abs = float(np.mean(np.abs(values)))
    max_abs = float(np.max(np.abs(values)))
    min_abs = float(np.min(np.abs(values)))
    coefficient_range = float(np.max(values) - np.min(values))

    stable_positive = bool(np.all(values > 0))
    stable_negative = bool(np.all(values < 0))
    direction_changes = not (stable_positive or stable_negative)

    category = classify_predictor(c6, c12, c24)

    rows.append({
        "predictor": predictor,
        "coefficient_6h": c6,
        "coefficient_12h": c12,
        "coefficient_24h": c24,
        "sign_6h": sign_label(c6),
        "sign_12h": sign_label(c12),
        "sign_24h": sign_label(c24),
        "mean_coefficient": mean_coef,
        "mean_absolute_coefficient": mean_abs,
        "maximum_absolute_coefficient": max_abs,
        "minimum_absolute_coefficient": min_abs,
        "coefficient_range": coefficient_range,
        "direction_changes": direction_changes,
        "stability_category": category,
    })


# =============================================================================
# CATEGORY COUNTS
# =============================================================================

categories = [
    "stable_direction_strong",
    "stable_direction_weaker",
    "direction_changing_near_zero_or_weak",
    "direction_changing_material",
]

category_counts = {
    category: sum(
        row["stability_category"] == category
        for row in rows
    )
    for category in categories
}

print("=" * 80)
print("STABILITY CLASSIFICATION")
print("=" * 80)

for category in categories:
    print(f"{category}: {category_counts[category]}")

print()
print(f"Total predictors: {len(rows)}")
print(
    "Classification total:",
    sum(category_counts.values())
)
print()


# =============================================================================
# CONSISTENT DIRECTION COUNTS
# =============================================================================

positive_all = sum(
    row["sign_6h"] == "positive"
    and row["sign_12h"] == "positive"
    and row["sign_24h"] == "positive"
    for row in rows
)

negative_all = sum(
    row["sign_6h"] == "negative"
    and row["sign_12h"] == "negative"
    and row["sign_24h"] == "negative"
    for row in rows
)

changing = sum(row["direction_changes"] for row in rows)

print("=" * 80)
print("DIRECTIONAL SUMMARY")
print("=" * 80)
print(f"Positive at all horizons: {positive_all}")
print(f"Negative at all horizons: {negative_all}")
print(f"Direction changing: {changing}")
print()


# =============================================================================
# STRONG STABLE PREDICTORS
# =============================================================================

stable_strong = [
    row for row in rows
    if row["stability_category"] == "stable_direction_strong"
]

stable_strong = sorted(
    stable_strong,
    key=lambda x: x["mean_absolute_coefficient"],
    reverse=True,
)

print("=" * 80)
print("STRONGEST STABLE-DIRECTION PREDICTORS")
print("=" * 80)

if not stable_strong:
    print("None.")
else:
    for rank, row in enumerate(stable_strong[:25], start=1):
        print(
            f"{rank:02d}. {row['predictor']} | "
            f"6h={row['coefficient_6h']:.6f} | "
            f"12h={row['coefficient_12h']:.6f} | "
            f"24h={row['coefficient_24h']:.6f} | "
            f"mean_abs={row['mean_absolute_coefficient']:.6f}"
        )

print()


# =============================================================================
# MATERIAL DIRECTION CHANGES
# =============================================================================

material_changes = [
    row for row in rows
    if row["stability_category"] == "direction_changing_material"
]

material_changes = sorted(
    material_changes,
    key=lambda x: x["maximum_absolute_coefficient"],
    reverse=True,
)

print("=" * 80)
print("MATERIAL DIRECTION-CHANGING PREDICTORS")
print("=" * 80)

if not material_changes:
    print("None.")
else:
    for rank, row in enumerate(material_changes, start=1):
        print(
            f"{rank:02d}. {row['predictor']} | "
            f"6h={row['coefficient_6h']:.6f} | "
            f"12h={row['coefficient_12h']:.6f} | "
            f"24h={row['coefficient_24h']:.6f} | "
            f"max_abs={row['maximum_absolute_coefficient']:.6f}"
        )

print()


# =============================================================================
# LARGEST CROSS-HORIZON COEFFICIENT RANGES
# =============================================================================

largest_ranges = sorted(
    rows,
    key=lambda x: x["coefficient_range"],
    reverse=True,
)

print("=" * 80)
print("TOP 20 — LARGEST CROSS-HORIZON COEFFICIENT RANGES")
print("=" * 80)

for rank, row in enumerate(largest_ranges[:20], start=1):
    print(
        f"{rank:02d}. {row['predictor']} | "
        f"6h={row['coefficient_6h']:.6f} | "
        f"12h={row['coefficient_12h']:.6f} | "
        f"24h={row['coefficient_24h']:.6f} | "
        f"range={row['coefficient_range']:.6f} | "
        f"{row['stability_category']}"
    )

print()


# =============================================================================
# WRITE CSV
# =============================================================================

fieldnames = [
    "predictor",
    "coefficient_6h",
    "coefficient_12h",
    "coefficient_24h",
    "sign_6h",
    "sign_12h",
    "sign_24h",
    "mean_coefficient",
    "mean_absolute_coefficient",
    "maximum_absolute_coefficient",
    "minimum_absolute_coefficient",
    "coefficient_range",
    "direction_changes",
    "stability_category",
]

with open(output_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


# =============================================================================
# WRITE REPORT
# =============================================================================

report_lines = []

report_lines.append(
    "STEP 59C — CROSS-HORIZON COEFFICIENT STABILITY AUDIT"
)
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append("INTERPRETATION NOTE:")
report_lines.append(
    "Coefficients are standardized multivariable predictive associations."
)
report_lines.append(
    "They must not be interpreted as causal effects or treatment effects."
)
report_lines.append("")

report_lines.append("STRUCTURE:")
report_lines.append(f"Predictors: {len(rows)}")
report_lines.append(
    f"Identical predictor order across horizons: {identical_order}"
)
report_lines.append("")

report_lines.append("PAIRWISE COEFFICIENT CORRELATIONS:")
report_lines.append(f"6h vs 12h: {corr_6_12:.6f}")
report_lines.append(f"6h vs 24h: {corr_6_24:.6f}")
report_lines.append(f"12h vs 24h: {corr_12_24:.6f}")
report_lines.append("")

report_lines.append("DIRECTIONAL SUMMARY:")
report_lines.append(
    f"Positive at all horizons: {positive_all}"
)
report_lines.append(
    f"Negative at all horizons: {negative_all}"
)
report_lines.append(
    f"Direction changing: {changing}"
)
report_lines.append("")

report_lines.append("STABILITY CLASSIFICATION:")
for category in categories:
    report_lines.append(
        f"{category}: {category_counts[category]}"
    )

report_lines.append("")
report_lines.append(
    "CLASSIFICATION DEFINITIONS:"
)
report_lines.append(
    "stable_direction_strong = same coefficient direction at all "
    f"horizons and mean absolute coefficient >= "
    f"{strong_mean_abs_threshold:.2f}"
)
report_lines.append(
    "stable_direction_weaker = same direction at all horizons but "
    "below the strong threshold"
)
report_lines.append(
    "direction_changing_material = sign reversal where coefficients "
    f"on both sides of zero reach absolute magnitude >= "
    f"{material_sign_threshold:.2f}"
)
report_lines.append(
    "direction_changing_near_zero_or_weak = remaining sign changes, "
    "including crossings near zero"
)
report_lines.append("")

report_lines.append(
    "STRONGEST STABLE-DIRECTION PREDICTORS:"
)

if not stable_strong:
    report_lines.append("None.")
else:
    for rank, row in enumerate(stable_strong[:25], start=1):
        report_lines.append(
            f"{rank:02d}. {row['predictor']} | "
            f"6h={row['coefficient_6h']:.6f} | "
            f"12h={row['coefficient_12h']:.6f} | "
            f"24h={row['coefficient_24h']:.6f} | "
            f"mean_abs={row['mean_absolute_coefficient']:.6f}"
        )

report_lines.append("")
report_lines.append(
    "MATERIAL DIRECTION-CHANGING PREDICTORS:"
)

if not material_changes:
    report_lines.append("None.")
else:
    for rank, row in enumerate(material_changes, start=1):
        report_lines.append(
            f"{rank:02d}. {row['predictor']} | "
            f"6h={row['coefficient_6h']:.6f} | "
            f"12h={row['coefficient_12h']:.6f} | "
            f"24h={row['coefficient_24h']:.6f} | "
            f"max_abs={row['maximum_absolute_coefficient']:.6f}"
        )

report_lines.append("")
report_lines.append(
    "TOP 20 — LARGEST CROSS-HORIZON COEFFICIENT RANGES:"
)

for rank, row in enumerate(largest_ranges[:20], start=1):
    report_lines.append(
        f"{rank:02d}. {row['predictor']} | "
        f"6h={row['coefficient_6h']:.6f} | "
        f"12h={row['coefficient_12h']:.6f} | "
        f"24h={row['coefficient_24h']:.6f} | "
        f"range={row['coefficient_range']:.6f} | "
        f"{row['stability_category']}"
    )

report_lines.append("")
report_lines.append("=" * 80)
report_lines.append("STEP 59C COMPLETE")
report_lines.append("NO MODEL WAS RETRAINED.")
report_lines.append("NO TEST PREDICTIONS WERE MODIFIED.")
report_lines.append("NO THRESHOLD WAS CHANGED.")

with open(output_report, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))


# =============================================================================
# FINAL VERIFICATION
# =============================================================================

classification_total = sum(category_counts.values())

all_pass = (
    identical_order
    and len(rows) == 252
    and classification_total == 252
    and positive_all == 99
    and negative_all == 80
    and changing == 73
    and np.isfinite(corr_6_12)
    and np.isfinite(corr_6_24)
    and np.isfinite(corr_12_24)
)


print("=" * 80)

print(
    "FINAL STEP 59C VERDICT:",
    "PASS" if all_pass else "FAIL"
)

print("=" * 80)
print()
print("Coefficient stability CSV:")
print(output_csv)
print()
print("Stability report:")
print(output_report)
print()

print("NO MODEL WAS RETRAINED.")
print("NO TEST PREDICTIONS WERE MODIFIED.")
print("NO THRESHOLD WAS CHANGED.")
