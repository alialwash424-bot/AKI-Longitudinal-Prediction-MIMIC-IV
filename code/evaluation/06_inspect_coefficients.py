import os
import numpy as np
import pandas as pd


# =============================================================================
# STEP 59A — FINAL LOCKED-MODEL COEFFICIENT AUDIT
#
# CORRECTED VERSION
#
# Uses the LOCKED .npz model files directly.
#
# READ-ONLY:
#   - NO model retraining
#   - NO coefficient modification
#   - NO predictor modification
#   - NO test evaluation
#   - NO threshold modification
#
# The locked model files already contain:
#   target
#   active_predictors
#   weights
#   bias
#
# Output:
#   step59A_final_coefficient_audit.txt
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

master_predictor_path = os.path.join(
    folder,
    "final_model_predictors.txt"
)

group_path = os.path.join(
    folder,
    "final_model_predictor_groups.csv"
)


models = {

    "6h": {
        "target": "aki_within_6h",
        "model_path": os.path.join(
            folder,
            "baseline_6h_logistic_model.npz"
        ),
    },

    "12h": {
        "target": "aki_within_12h",
        "model_path": os.path.join(
            folder,
            "baseline_12h_logistic_model.npz"
        ),
    },

    "24h": {
        "target": "aki_within_24h",
        "model_path": os.path.join(
            folder,
            "baseline_24h_logistic_model.npz"
        ),
    },
}


print("")
print("=" * 80)
print("STEP 59A — FINAL LOCKED-MODEL COEFFICIENT AUDIT")
print("=" * 80)


# =============================================================================
# MASTER PREDICTOR LIST
# =============================================================================

if not os.path.exists(master_predictor_path):

    raise FileNotFoundError(
        master_predictor_path
    )


with open(
    master_predictor_path,
    "r",
    encoding="utf-8"
) as f:

    master_predictors = [
        line.strip()
        for line in f
        if line.strip()
    ]


print("")
print(
    "Master predictor count:",
    len(master_predictors)
)

print(
    "Unique master predictors:",
    len(set(master_predictors))
)


if len(master_predictors) != 252:

    raise RuntimeError(
        "SAFETY STOP — expected 252 master predictors."
    )


if len(set(master_predictors)) != 252:

    raise RuntimeError(
        "SAFETY STOP — duplicate predictor in master list."
    )


# =============================================================================
# PREDICTOR GROUP FILE
# =============================================================================

if not os.path.exists(group_path):

    raise FileNotFoundError(
        group_path
    )


groups = pd.read_csv(
    group_path,
    low_memory=False
)


if "predictor" not in groups.columns:

    raise RuntimeError(
        "SAFETY STOP — predictor-group file has no 'predictor' column."
    )


group_predictors = (
    groups["predictor"]
    .astype(str)
    .tolist()
)


print("")
print(
    "Predictor-group rows:",
    len(groups)
)

print(
    "Predictor-group columns:",
    groups.columns.tolist()
)


if len(groups) != 252:

    raise RuntimeError(
        "SAFETY STOP — expected 252 predictor-group rows."
    )


if len(set(group_predictors)) != 252:

    raise RuntimeError(
        "SAFETY STOP — duplicate predictor in group file."
    )


group_columns = [
    column
    for column in groups.columns
    if column != "predictor"
]


# =============================================================================
# AUDIT EACH LOCKED MODEL
# =============================================================================

all_pass = True
report_sections = []


for horizon in [
    "6h",
    "12h",
    "24h",
]:

    info = models[horizon]

    print("")
    print("=" * 80)
    print(
        f"{horizon.upper()} COEFFICIENT AUDIT"
    )
    print("=" * 80)


    # -------------------------------------------------------------------------
    # MODEL EXISTS
    # -------------------------------------------------------------------------

    if not os.path.exists(
        info["model_path"]
    ):

        raise FileNotFoundError(
            info["model_path"]
        )


    # -------------------------------------------------------------------------
    # LOAD LOCKED MODEL
    # -------------------------------------------------------------------------

    model = np.load(
        info["model_path"],
        allow_pickle=False
    )


    required_keys = [
        "target",
        "active_predictors",
        "weights",
        "bias",
        "source_indices",
        "means",
        "standard_deviations",
        "exposure_indices",
    ]


    missing_keys = [
        key
        for key in required_keys
        if key not in model.files
    ]


    if missing_keys:

        raise RuntimeError(
            f"{horizon}: locked model missing keys: "
            f"{missing_keys}"
        )


    # -------------------------------------------------------------------------
    # EXTRACT LOCKED CONTENT
    # -------------------------------------------------------------------------

    model_target = str(
        model["target"][0]
    )


    active_predictors = [
        str(value)
        for value
        in model[
            "active_predictors"
        ].tolist()
    ]


    weights = np.asarray(
        model["weights"],
        dtype=np.float64
    )


    bias = float(
        np.asarray(
            model["bias"],
            dtype=np.float64
        ).reshape(-1)[0]
    )


    source_indices = np.asarray(
        model["source_indices"]
    )


    means = np.asarray(
        model["means"],
        dtype=np.float64
    )


    standard_deviations = np.asarray(
        model["standard_deviations"],
        dtype=np.float64
    )


    exposure_indices = np.asarray(
        model["exposure_indices"]
    )


    # -------------------------------------------------------------------------
    # STRUCTURAL CHECKS
    # -------------------------------------------------------------------------

    target_pass = (
        model_target
        == info["target"]
    )


    active_count_pass = (
        len(active_predictors)
        == 252
    )


    unique_predictor_pass = (
        len(set(active_predictors))
        == 252
    )


    weight_count_pass = (
        len(weights)
        == len(active_predictors)
    )


    source_index_count_pass = (
        len(source_indices)
        == len(active_predictors)
    )


    mean_count_pass = (
        len(means)
        == len(active_predictors)
    )


    sd_count_pass = (
        len(standard_deviations)
        == len(active_predictors)
    )


    master_order_pass = (
        active_predictors
        == master_predictors
    )


    master_set_pass = (
        set(active_predictors)
        == set(master_predictors)
    )


    group_set_pass = (
        set(active_predictors)
        == set(group_predictors)
    )


    finite_weights_pass = bool(
        np.isfinite(weights).all()
    )


    finite_bias_pass = bool(
        np.isfinite(bias)
    )


    finite_means_pass = bool(
        np.isfinite(means).all()
    )


    finite_sd_pass = bool(
        np.isfinite(
            standard_deviations
        ).all()
    )


    positive_sd_pass = bool(
        (
            standard_deviations > 0
        ).all()
    )


    exposure_index_pass = bool(
        (
            (exposure_indices >= 0)
            & (
                exposure_indices
                < len(active_predictors)
            )
        ).all()
    )


    # -------------------------------------------------------------------------
    # GROUP COVERAGE
    # -------------------------------------------------------------------------

    predictors_missing_group = [
        predictor
        for predictor in active_predictors
        if predictor not in set(
            group_predictors
        )
    ]


    group_coverage_pass = (
        len(
            predictors_missing_group
        )
        == 0
    )


    # -------------------------------------------------------------------------
    # COEFFICIENT DISTRIBUTION
    # -------------------------------------------------------------------------

    positive_count = int(
        (
            weights > 0
        ).sum()
    )


    negative_count = int(
        (
            weights < 0
        ).sum()
    )


    zero_count = int(
        (
            weights == 0
        ).sum()
    )


    absolute_weights = np.abs(
        weights
    )


    ranking = np.argsort(
        -absolute_weights
    )


    top_n = min(
        20,
        len(ranking)
    )


    top_rows = []


    for rank_number, index in enumerate(
        ranking[:top_n],
        start=1
    ):

        predictor = (
            active_predictors[index]
        )

        coefficient = float(
            weights[index]
        )


        matching_group = groups.loc[
            groups[
                "predictor"
            ].astype(str)
            == predictor
        ]


        group_description = ""


        if len(
            matching_group
        ) > 0:

            values = []

            for column in group_columns:

                value = (
                    matching_group
                    .iloc[0][column]
                )

                if pd.notna(value):

                    values.append(
                        f"{column}={value}"
                    )


            group_description = (
                "; ".join(values)
            )


        top_rows.append(
            (
                rank_number,
                predictor,
                coefficient,
                group_description,
            )
        )


    # -------------------------------------------------------------------------
    # HORIZON VERDICT
    # -------------------------------------------------------------------------

    horizon_pass = all(
        [
            target_pass,
            active_count_pass,
            unique_predictor_pass,
            weight_count_pass,
            source_index_count_pass,
            mean_count_pass,
            sd_count_pass,
            master_set_pass,
            group_set_pass,
            finite_weights_pass,
            finite_bias_pass,
            finite_means_pass,
            finite_sd_pass,
            positive_sd_pass,
            exposure_index_pass,
            group_coverage_pass,
        ]
    )


    if not horizon_pass:

        all_pass = False


    # -------------------------------------------------------------------------
    # CONSOLE OUTPUT
    # -------------------------------------------------------------------------

    print(
        "Model file:",
        info["model_path"]
    )

    print(
        "Target:",
        model_target
    )

    print(
        "Target check:",
        target_pass
    )

    print(
        "Active predictors:",
        len(active_predictors)
    )

    print(
        "Unique active predictors:",
        len(
            set(active_predictors)
        )
    )

    print(
        "Weights:",
        len(weights)
    )

    print(
        "Source indices:",
        len(source_indices)
    )

    print(
        "Means:",
        len(means)
    )

    print(
        "Standard deviations:",
        len(standard_deviations)
    )

    print(
        "Exposure indices:",
        len(exposure_indices)
    )

    print(
        "Master predictor order identical:",
        master_order_pass
    )

    print(
        "Master predictor set identical:",
        master_set_pass
    )

    print(
        "Predictor-group set identical:",
        group_set_pass
    )

    print(
        "All weights finite:",
        finite_weights_pass
    )

    print(
        "Bias finite:",
        finite_bias_pass
    )

    print(
        "All means finite:",
        finite_means_pass
    )

    print(
        "All standard deviations finite:",
        finite_sd_pass
    )

    print(
        "All standard deviations > 0:",
        positive_sd_pass
    )

    print(
        "Exposure indices valid:",
        exposure_index_pass
    )

    print(
        "Predictors missing group assignment:",
        len(
            predictors_missing_group
        )
    )

    print(
        "Bias:",
        f"{bias:.12f}"
    )


    print("")
    print(
        "COEFFICIENT SIGN DISTRIBUTION:"
    )

    print(
        "Positive:",
        positive_count
    )

    print(
        "Negative:",
        negative_count
    )

    print(
        "Exactly zero:",
        zero_count
    )


    print("")
    print(
        "TOP 20 BY ABSOLUTE STANDARDIZED COEFFICIENT:"
    )


    for (
        rank_number,
        predictor,
        coefficient,
        group_description,
    ) in top_rows:

        print(
            f"{rank_number:02d}. "
            f"{predictor} | "
            f"coefficient={coefficient:.8f}"
        )

        if group_description:

            print(
                "    "
                + group_description
            )


    print("")
    print(
        f"{horizon.upper()} VERDICT:",
        "PASS"
        if horizon_pass
        else "FAIL"
    )


    # -------------------------------------------------------------------------
    # REPORT
    # -------------------------------------------------------------------------

    lines = [
        f"{horizon.upper()} FINAL LOCKED-MODEL COEFFICIENT AUDIT",
        "-" * 80,
        f"Model file: {info['model_path']}",
        f"Target: {model_target}",
        f"Target check: {target_pass}",
        f"Active predictors: {len(active_predictors)}",
        f"Unique active predictors: {len(set(active_predictors))}",
        f"Weights: {len(weights)}",
        f"Source indices: {len(source_indices)}",
        f"Means: {len(means)}",
        f"Standard deviations: {len(standard_deviations)}",
        f"Exposure indices: {len(exposure_indices)}",
        f"Master predictor order identical: {master_order_pass}",
        f"Master predictor set identical: {master_set_pass}",
        f"Predictor-group set identical: {group_set_pass}",
        f"All weights finite: {finite_weights_pass}",
        f"Bias finite: {finite_bias_pass}",
        f"All means finite: {finite_means_pass}",
        f"All standard deviations finite: {finite_sd_pass}",
        f"All standard deviations > 0: {positive_sd_pass}",
        f"Exposure indices valid: {exposure_index_pass}",
        f"Predictors missing group assignment: {len(predictors_missing_group)}",
        f"Bias: {bias:.12f}",
        "",
        "COEFFICIENT SIGN DISTRIBUTION:",
        f"Positive: {positive_count}",
        f"Negative: {negative_count}",
        f"Exactly zero: {zero_count}",
        "",
        "TOP 20 BY ABSOLUTE STANDARDIZED COEFFICIENT:",
    ]


    for (
        rank_number,
        predictor,
        coefficient,
        group_description,
    ) in top_rows:

        lines.append(
            f"{rank_number:02d}. "
            f"{predictor} | "
            f"coefficient={coefficient:.8f}"
        )

        if group_description:

            lines.append(
                f"    {group_description}"
            )


    lines.extend(
        [
            "",
            f"{horizon.upper()} VERDICT: "
            + (
                "PASS"
                if horizon_pass
                else "FAIL"
            ),
        ]
    )


    report_sections.append(
        "\n".join(lines)
    )


# =============================================================================
# SAVE REPORT
# =============================================================================

report_path = os.path.join(
    folder,
    "step59A_final_coefficient_audit.txt"
)


header = [
    "STEP 59A — FINAL LOCKED-MODEL COEFFICIENT AUDIT",
    "=" * 80,
    "",
    "CORRECTED VERSION — locked NPZ models inspected directly.",
    "No standalone coefficient CSV is required.",
    "",
    "READ-ONLY.",
    "No model retraining.",
    "No predictor modification.",
    "No test evaluation.",
    "No threshold modification.",
    "",
]


with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(header)
    )

    f.write("\n")

    f.write(
        "\n\n".join(
            report_sections
        )
    )

    f.write("\n\n")

    f.write(
        "=" * 80
    )

    f.write(
        "\nFINAL STEP 59A VERDICT: "
    )

    f.write(
        "PASS"
        if all_pass
        else "FAIL"
    )

    f.write("\n")

    f.write(
        "NO MODEL OR RESEARCH FILE WAS MODIFIED.\n"
    )


# =============================================================================
# FINAL
# =============================================================================

print("")
print("=" * 80)

print(
    "FINAL STEP 59A VERDICT:",
    "PASS"
    if all_pass
    else "FAIL"
)

print("=" * 80)

print("")
print(
    "Report:"
)

print(
    report_path
)

print("")
print(
    "NO MODEL OR RESEARCH FILE WAS MODIFIED."
)
