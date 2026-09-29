import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 59B — CROSS-HORIZON STANDARDIZED COEFFICIENT ANALYSIS
#
# READ-ONLY with respect to models/research data.
#
# Reads:
#   baseline_6h_logistic_model.npz
#   baseline_12h_logistic_model.npz
#   baseline_24h_logistic_model.npz
#   final_model_predictor_groups.csv
#
# Creates:
#   step59B_cross_horizon_coefficients.csv
#   step59B_cross_horizon_coefficient_report.txt
#
# IMPORTANT:
#   Coefficients are predictive associations from standardized logistic models.
#   They are NOT causal effects.
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

model_paths = {
    "6h": os.path.join(
        folder,
        "baseline_6h_logistic_model.npz"
    ),
    "12h": os.path.join(
        folder,
        "baseline_12h_logistic_model.npz"
    ),
    "24h": os.path.join(
        folder,
        "baseline_24h_logistic_model.npz"
    ),
}

group_path = os.path.join(
    folder,
    "final_model_predictor_groups.csv"
)

output_csv = os.path.join(
    folder,
    "step59B_cross_horizon_coefficients.csv"
)

output_report = os.path.join(
    folder,
    "step59B_cross_horizon_coefficient_report.txt"
)


print("")
print("=" * 80)
print("STEP 59B — CROSS-HORIZON STANDARDIZED COEFFICIENT ANALYSIS")
print("=" * 80)


# =============================================================================
# LOAD MODELS
# =============================================================================

loaded = {}

for horizon, path in model_paths.items():

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    model = np.load(
        path,
        allow_pickle=False
    )

    predictors = [
        str(x)
        for x in model["active_predictors"].tolist()
    ]

    weights = np.asarray(
        model["weights"],
        dtype=np.float64
    )

    if len(predictors) != 252:
        raise RuntimeError(
            f"{horizon}: expected 252 predictors, got {len(predictors)}"
        )

    if len(weights) != 252:
        raise RuntimeError(
            f"{horizon}: expected 252 weights, got {len(weights)}"
        )

    if not np.isfinite(weights).all():
        raise RuntimeError(
            f"{horizon}: nonfinite weights detected."
        )

    loaded[horizon] = {
        "predictors": predictors,
        "weights": weights,
    }


# =============================================================================
# VERIFY IDENTICAL PREDICTOR ORDER
# =============================================================================

reference_predictors = loaded["6h"]["predictors"]

for horizon in ["12h", "24h"]:

    if loaded[horizon]["predictors"] != reference_predictors:

        raise RuntimeError(
            f"SAFETY STOP — predictor order differs for {horizon}."
        )


print("")
print("Predictor order identical across 6h / 12h / 24h: True")
print("Predictors:", len(reference_predictors))


# =============================================================================
# BUILD COEFFICIENT TABLE
# =============================================================================

table = pd.DataFrame({
    "predictor": reference_predictors,
    "coefficient_6h": loaded["6h"]["weights"],
    "coefficient_12h": loaded["12h"]["weights"],
    "coefficient_24h": loaded["24h"]["weights"],
})


# =============================================================================
# ADD GROUP INFORMATION
# =============================================================================

if not os.path.exists(group_path):
    raise FileNotFoundError(group_path)

groups = pd.read_csv(
    group_path,
    low_memory=False
)

if "predictor" not in groups.columns:
    raise RuntimeError(
        "Predictor group file missing predictor column."
    )

if groups["predictor"].duplicated().any():
    raise RuntimeError(
        "Duplicate predictors in group file."
    )

table = table.merge(
    groups,
    on="predictor",
    how="left",
    validate="one_to_one"
)

if table["group"].isna().any():

    missing = table.loc[
        table["group"].isna(),
        "predictor"
    ].tolist()

    raise RuntimeError(
        "Missing predictor groups: "
        + str(missing)
    )


# =============================================================================
# DERIVED INTERPRETATION VARIABLES
# =============================================================================

coefficient_columns = [
    "coefficient_6h",
    "coefficient_12h",
    "coefficient_24h",
]


table["absolute_coefficient_6h"] = np.abs(
    table["coefficient_6h"]
)

table["absolute_coefficient_12h"] = np.abs(
    table["coefficient_12h"]
)

table["absolute_coefficient_24h"] = np.abs(
    table["coefficient_24h"]
)


table["mean_absolute_coefficient"] = (
    table[
        [
            "absolute_coefficient_6h",
            "absolute_coefficient_12h",
            "absolute_coefficient_24h",
        ]
    ].mean(axis=1)
)


table["maximum_absolute_coefficient"] = (
    table[
        [
            "absolute_coefficient_6h",
            "absolute_coefficient_12h",
            "absolute_coefficient_24h",
        ]
    ].max(axis=1)
)


def sign_label(value):

    if value > 0:
        return "positive"

    if value < 0:
        return "negative"

    return "zero"


table["sign_6h"] = (
    table["coefficient_6h"]
    .map(sign_label)
)

table["sign_12h"] = (
    table["coefficient_12h"]
    .map(sign_label)
)

table["sign_24h"] = (
    table["coefficient_24h"]
    .map(sign_label)
)


def direction_pattern(row):

    signs = [
        row["sign_6h"],
        row["sign_12h"],
        row["sign_24h"],
    ]

    if signs == [
        "positive",
        "positive",
        "positive",
    ]:
        return "positive_all_horizons"

    if signs == [
        "negative",
        "negative",
        "negative",
    ]:
        return "negative_all_horizons"

    if signs == [
        "zero",
        "zero",
        "zero",
    ]:
        return "zero_all_horizons"

    return "direction_changes"


table["direction_pattern"] = table.apply(
    direction_pattern,
    axis=1
)


table["coefficient_range"] = (
    table[coefficient_columns].max(axis=1)
    - table[coefficient_columns].min(axis=1)
)


# =============================================================================
# RANKINGS
# =============================================================================

table["rank_abs_6h"] = (
    table["absolute_coefficient_6h"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)

table["rank_abs_12h"] = (
    table["absolute_coefficient_12h"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)

table["rank_abs_24h"] = (
    table["absolute_coefficient_24h"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)

table["rank_mean_absolute"] = (
    table["mean_absolute_coefficient"]
    .rank(
        method="min",
        ascending=False
    )
    .astype(int)
)


# =============================================================================
# SAVE COMPLETE TABLE
# =============================================================================

table = table.sort_values(
    [
        "rank_mean_absolute",
        "predictor"
    ]
).reset_index(drop=True)


table.to_csv(
    output_csv,
    index=False
)


# =============================================================================
# SUMMARY
# =============================================================================

positive_all = int(
    (
        table["direction_pattern"]
        == "positive_all_horizons"
    ).sum()
)

negative_all = int(
    (
        table["direction_pattern"]
        == "negative_all_horizons"
    ).sum()
)

direction_changes = int(
    (
        table["direction_pattern"]
        == "direction_changes"
    ).sum()
)


print("")
print("DIRECTIONAL CONSISTENCY")
print("-" * 80)

print(
    "Positive at all three horizons:",
    positive_all
)

print(
    "Negative at all three horizons:",
    negative_all
)

print(
    "Direction changes across horizons:",
    direction_changes
)


# =============================================================================
# GROUP COUNTS
# =============================================================================

print("")
print("PREDICTOR GROUP COUNTS")
print("-" * 80)

group_counts = (
    table["group"]
    .value_counts()
)

for group, count in group_counts.items():

    print(
        f"{group}: {count}"
    )


# =============================================================================
# TOP 20 OVERALL
# =============================================================================

top_overall = (
    table
    .sort_values(
        "mean_absolute_coefficient",
        ascending=False
    )
    .head(20)
)


print("")
print("TOP 20 — MEAN ABSOLUTE STANDARDIZED COEFFICIENT")
print("-" * 80)


for rank, (_, row) in enumerate(
    top_overall.iterrows(),
    start=1
):

    print(
        f"{rank:02d}. "
        f"{row['predictor']} | "
        f"group={row['group']} | "
        f"6h={row['coefficient_6h']:.6f} | "
        f"12h={row['coefficient_12h']:.6f} | "
        f"24h={row['coefficient_24h']:.6f} | "
        f"mean_abs={row['mean_absolute_coefficient']:.6f}"
    )


# =============================================================================
# TOP 15 EACH HORIZON
# =============================================================================

horizon_sections = []


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    coefficient_column = (
        f"coefficient_{horizon}"
    )

    absolute_column = (
        f"absolute_coefficient_{horizon}"
    )

    top = (
        table
        .sort_values(
            absolute_column,
            ascending=False
        )
        .head(15)
    )


    lines = [
        f"TOP 15 — {horizon.upper()} ABSOLUTE STANDARDIZED COEFFICIENT",
        "-" * 80,
    ]


    print("")
    print(lines[0])
    print(lines[1])


    for rank, (_, row) in enumerate(
        top.iterrows(),
        start=1
    ):

        line = (
            f"{rank:02d}. "
            f"{row['predictor']} | "
            f"group={row['group']} | "
            f"coefficient={row[coefficient_column]:.8f}"
        )

        lines.append(line)
        print(line)


    horizon_sections.append(
        "\n".join(lines)
    )


# =============================================================================
# DIRECTION-CHANGE PREDICTORS
# =============================================================================

changing = (
    table.loc[
        table["direction_pattern"]
        == "direction_changes"
    ]
    .sort_values(
        "mean_absolute_coefficient",
        ascending=False
    )
)


print("")
print("PREDICTORS CHANGING COEFFICIENT DIRECTION")
print("-" * 80)

print(
    "Count:",
    len(changing)
)


for _, row in changing.head(30).iterrows():

    print(
        f"{row['predictor']} | "
        f"6h={row['coefficient_6h']:.6f} | "
        f"12h={row['coefficient_12h']:.6f} | "
        f"24h={row['coefficient_24h']:.6f}"
    )


# =============================================================================
# REPORT
# =============================================================================

report_lines = [
    "STEP 59B — CROSS-HORIZON STANDARDIZED COEFFICIENT ANALYSIS",
    "=" * 80,
    "",
    "IMPORTANT INTERPRETATION:",
    (
        "These coefficients describe predictive associations in the locked "
        "multivariable logistic models."
    ),
    (
        "They must not be interpreted as causal effects, treatment effects, "
        "or independent biological mechanisms."
    ),
    "",
    "STRUCTURE:",
    f"Predictors: {len(table)}",
    "Horizons: 6h, 12h, 24h",
    "Predictor order identical across locked models: True",
    "",
    "DIRECTIONAL CONSISTENCY:",
    f"Positive at all three horizons: {positive_all}",
    f"Negative at all three horizons: {negative_all}",
    f"Direction changes across horizons: {direction_changes}",
    "",
    "PREDICTOR GROUP COUNTS:",
]


for group, count in group_counts.items():

    report_lines.append(
        f"{group}: {count}"
    )


report_lines.extend([
    "",
    "TOP 20 — MEAN ABSOLUTE STANDARDIZED COEFFICIENT",
    "-" * 80,
])


for rank, (_, row) in enumerate(
    top_overall.iterrows(),
    start=1
):

    report_lines.append(
        f"{rank:02d}. "
        f"{row['predictor']} | "
        f"group={row['group']} | "
        f"6h={row['coefficient_6h']:.8f} | "
        f"12h={row['coefficient_12h']:.8f} | "
        f"24h={row['coefficient_24h']:.8f} | "
        f"mean_abs={row['mean_absolute_coefficient']:.8f}"
    )


report_lines.append("")
report_lines.extend(
    horizon_sections
)


report_lines.extend([
    "",
    "PREDICTORS CHANGING COEFFICIENT DIRECTION",
    "-" * 80,
    f"Count: {len(changing)}",
])


for _, row in changing.iterrows():

    report_lines.append(
        f"{row['predictor']} | "
        f"group={row['group']} | "
        f"6h={row['coefficient_6h']:.8f} | "
        f"12h={row['coefficient_12h']:.8f} | "
        f"24h={row['coefficient_24h']:.8f}"
    )


report_lines.extend([
    "",
    "=" * 80,
    "STEP 59B COMPLETE",
    "NO MODEL WAS RETRAINED.",
    "NO TEST PREDICTIONS WERE MODIFIED.",
    "NO THRESHOLD WAS CHANGED.",
])


with open(
    output_report,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(
            report_lines
        )
    )


# =============================================================================
# FINAL
# =============================================================================

print("")
print("=" * 80)
print("STEP 59B COMPLETE")
print("=" * 80)

print("")
print(
    "Cross-horizon coefficient CSV:"
)

print(
    output_csv
)

print("")
print(
    "Interpretation report:"
)

print(
    output_report
)

print("")
print(
    "NO MODEL WAS RETRAINED."
)

print(
    "NO TEST PREDICTIONS WERE MODIFIED."
)

print(
    "NO THRESHOLD WAS CHANGED."
)
