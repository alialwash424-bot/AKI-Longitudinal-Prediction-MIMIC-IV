import os
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# STEP 64D5
# HELD-OUT TEST SUBGROUP PERFORMANCE
#
# Exploratory descriptive subgroup evaluation of LOCKED test predictions.
#
# - AUROC
# - AUPRC
# - patient-level bootstrap 95% CIs
#
# No retraining.
# No recalibration.
# No threshold changes.
# No prediction changes.
# No hypothesis testing between demographic groups.
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")

split_path = os.path.join(
    folder,
    "patient_level_split_manifest.csv"
)

cohort_path = os.path.join(
    folder,
    "definitive_kdigo_cohort_VALIDATED.csv"
)

patients_path = os.path.join(
    folder,
    "patients.csv.gz"
)

admissions_path = os.path.join(
    folder,
    "admissions.csv.gz"
)

feasibility_path = os.path.join(
    folder,
    "step64D4_subgroup_feasibility.csv"
)

prediction_files = {
    "6h": os.path.join(folder, "FINAL_TEST_6h_predictions.csv"),
    "12h": os.path.join(folder, "FINAL_TEST_12h_predictions.csv"),
    "24h": os.path.join(folder, "FINAL_TEST_24h_predictions.csv"),
}

output_csv = os.path.join(
    folder,
    "step64D5_subgroup_test_performance.csv"
)

audit_path = os.path.join(
    folder,
    "step64D5_subgroup_test_performance_audit.txt"
)

BOOTSTRAPS = 1000
SEED = 20260930


# =============================================================================
# REQUIRED FILES
# =============================================================================

required = [
    split_path,
    cohort_path,
    patients_path,
    admissions_path,
    feasibility_path,
    *prediction_files.values(),
]

for path in required:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# =============================================================================
# METRICS
# =============================================================================

def roc_auc(y_true, scores):

    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    positives = int(y_true.sum())
    negatives = len(y_true) - positives

    if positives == 0 or negatives == 0:
        return np.nan

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]

    ranks = np.empty(len(scores), dtype=np.float64)

    i = 0

    while i < len(scores):

        j = i + 1

        while (
            j < len(scores)
            and sorted_scores[j] == sorted_scores[i]
        ):
            j += 1

        average_rank = (
            (i + 1) + j
        ) / 2.0

        ranks[
            order[i:j]
        ] = average_rank

        i = j

    positive_rank_sum = ranks[
        y_true == 1
    ].sum()

    return float(
        (
            positive_rank_sum
            - positives * (positives + 1) / 2.0
        )
        / (positives * negatives)
    )


def average_precision(y_true, scores):

    y_true = np.asarray(y_true, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)

    positives = int(y_true.sum())

    if positives == 0:
        return np.nan

    order = np.argsort(
        -scores,
        kind="mergesort"
    )

    ordered_y = y_true[order]

    cumulative_positive = np.cumsum(
        ordered_y
    )

    precision = (
        cumulative_positive
        / np.arange(
            1,
            len(ordered_y) + 1
        )
    )

    return float(
        precision[
            ordered_y == 1
        ].sum()
        / positives
    )


# =============================================================================
# LOAD TEST PATIENTS
# =============================================================================

split = pd.read_csv(
    split_path,
    usecols=[
        "subject_id",
        "split"
    ]
)

split["subject_id"] = pd.to_numeric(
    split["subject_id"],
    errors="raise"
).astype("int64")

test_subjects = set(
    split.loc[
        split["split"] == "test",
        "subject_id"
    ]
)

if len(test_subjects) != 8583:
    raise RuntimeError(
        f"Expected 8583 test patients; found {len(test_subjects)}"
    )


# =============================================================================
# PRIMARY COHORT
# =============================================================================

cohort = pd.read_csv(
    cohort_path,
    usecols=[
        "subject_id",
        "hadm_id",
        "analysis_included"
    ]
)

cohort["subject_id"] = pd.to_numeric(
    cohort["subject_id"],
    errors="raise"
).astype("int64")

cohort["hadm_id"] = pd.to_numeric(
    cohort["hadm_id"],
    errors="coerce"
)

flag = cohort["analysis_included"]

if flag.dtype == bool:

    cohort = cohort.loc[
        flag
    ].copy()

else:

    normalized = (
        flag.astype(str)
        .str.strip()
        .str.lower()
    )

    cohort = cohort.loc[
        normalized.isin(
            ["true", "1", "yes"]
        )
    ].copy()

cohort = cohort.loc[
    cohort["subject_id"].isin(
        test_subjects
    )
].copy()


# =============================================================================
# DEMOGRAPHICS
# =============================================================================

patients = pd.read_csv(
    patients_path,
    compression="gzip",
    usecols=[
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year"
    ]
)

patients["subject_id"] = pd.to_numeric(
    patients["subject_id"],
    errors="raise"
).astype("int64")

patients = patients.loc[
    patients["subject_id"].isin(
        test_subjects
    )
].copy()


admissions = pd.read_csv(
    admissions_path,
    compression="gzip",
    usecols=[
        "subject_id",
        "hadm_id",
        "admittime",
        "race"
    ]
)

admissions["subject_id"] = pd.to_numeric(
    admissions["subject_id"],
    errors="raise"
).astype("int64")

admissions["hadm_id"] = pd.to_numeric(
    admissions["hadm_id"],
    errors="coerce"
)

admissions["admittime"] = pd.to_datetime(
    admissions["admittime"],
    errors="coerce"
)

admissions = admissions.loc[
    admissions["subject_id"].isin(
        test_subjects
    )
].copy()


included_admissions = (
    cohort[
        [
            "subject_id",
            "hadm_id"
        ]
    ]
    .dropna()
    .drop_duplicates()
    .merge(
        admissions,
        on=[
            "subject_id",
            "hadm_id"
        ],
        how="left",
        validate="many_to_one"
    )
)

included_admissions = (
    included_admissions
    .sort_values(
        [
            "subject_id",
            "admittime",
            "hadm_id"
        ],
        na_position="last"
    )
)

index_admission = (
    included_admissions
    .drop_duplicates(
        "subject_id",
        keep="first"
    )
)


demo = (
    patients
    .merge(
        index_admission[
            [
                "subject_id",
                "admittime",
                "race"
            ]
        ],
        on="subject_id",
        how="left",
        validate="one_to_one"
    )
)


# =============================================================================
# AGE
# =============================================================================

demo["anchor_age"] = pd.to_numeric(
    demo["anchor_age"],
    errors="coerce"
)

demo["anchor_year"] = pd.to_numeric(
    demo["anchor_year"],
    errors="coerce"
)

demo["admission_year"] = (
    demo["admittime"].dt.year
)

demo["age"] = (
    demo["anchor_age"]
    + (
        demo["admission_year"]
        - demo["anchor_year"]
    )
)

demo["age_group"] = pd.cut(
    demo["age"],
    bins=[
        -np.inf,
        49,
        64,
        79,
        np.inf
    ],
    labels=[
        "<50",
        "50-64",
        "65-79",
        ">=80"
    ]
).astype("object")

demo["age_group"] = (
    demo["age_group"]
    .fillna("Unknown")
)


# =============================================================================
# SEX
# =============================================================================

demo["sex_group"] = (
    demo["gender"]
    .fillna("Unknown")
    .astype(str)
    .str.strip()
)

demo.loc[
    ~demo["sex_group"].isin(
        ["M", "F"]
    ),
    "sex_group"
] = "Unknown"


# =============================================================================
# RACE / ETHNICITY
# =============================================================================

def broad_race(value):

    if pd.isna(value):
        return "Unknown / not reported"

    text = str(value).strip().upper()

    if (
        text == ""
        or "UNKNOWN" in text
        or "UNABLE TO OBTAIN" in text
        or "DECLINED" in text
    ):
        return "Unknown / not reported"

    if (
        text.startswith("WHITE")
        or text == "PORTUGUESE"
    ):
        return "White"

    if text.startswith("BLACK"):
        return "Black"

    if text.startswith("ASIAN"):
        return "Asian"

    if (
        text.startswith("HISPANIC")
        or text.startswith("SOUTH AMERICAN")
    ):
        return "Hispanic / Latino"

    if (
        "AMERICAN INDIAN" in text
        or "ALASKA NATIVE" in text
        or "PACIFIC ISLANDER" in text
        or "HAWAIIAN" in text
    ):
        return (
            "American Indian / Alaska Native / "
            "Native Hawaiian / Pacific Islander"
        )

    return "Other / multiple"


demo["race_group"] = (
    demo["race"].apply(
        broad_race
    )
)


if demo["subject_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate demographic patient."
    )

if len(demo) != 8583:
    raise RuntimeError(
        f"Expected 8583 demographic rows; found {len(demo)}"
    )


# =============================================================================
# FEASIBILITY TABLE
# =============================================================================

feasibility = pd.read_csv(
    feasibility_path
)

feasibility[
    "descriptive_performance_feasible"
] = (
    feasibility[
        "descriptive_performance_feasible"
    ]
    .astype(str)
    .str.lower()
    .eq("true")
)


# =============================================================================
# COLUMN FINDER
# =============================================================================

def find_column(columns, candidates):

    mapping = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in mapping:
            return mapping[
                candidate.lower()
            ]

    return None


# =============================================================================
# PATIENT-LEVEL BOOTSTRAP
# =============================================================================

def patient_bootstrap(
    frame,
    subject_col,
    outcome_col,
    probability_col,
    seed
):

    subjects = frame[
        subject_col
    ].drop_duplicates().to_numpy()

    grouped = {
        sid: group.index.to_numpy()
        for sid, group in frame.groupby(
            subject_col,
            sort=False
        )
    }

    rng = np.random.default_rng(
        seed
    )

    auc_values = []
    ap_values = []

    attempts = 0
    maximum_attempts = BOOTSTRAPS * 3

    while (
        len(auc_values) < BOOTSTRAPS
        and attempts < maximum_attempts
    ):

        attempts += 1

        sampled_subjects = rng.choice(
            subjects,
            size=len(subjects),
            replace=True
        )

        sampled_indices = np.concatenate(
            [
                grouped[sid]
                for sid in sampled_subjects
            ]
        )

        sample = frame.loc[
            sampled_indices
        ]

        y = sample[
            outcome_col
        ].to_numpy(
            dtype=np.int8
        )

        p = sample[
            probability_col
        ].to_numpy(
            dtype=np.float64
        )

        if (
            y.sum() == 0
            or y.sum() == len(y)
        ):
            continue

        auc_values.append(
            roc_auc(y, p)
        )

        ap_values.append(
            average_precision(y, p)
        )

    if len(auc_values) < BOOTSTRAPS:
        raise RuntimeError(
            "Unable to obtain requested number "
            "of valid bootstrap replicates."
        )

    return (
        np.percentile(
            auc_values,
            [2.5, 97.5]
        ),
        np.percentile(
            ap_values,
            [2.5, 97.5]
        ),
    )


# =============================================================================
# SUBGROUP ANALYSIS
# =============================================================================

rows = []

group_columns = {
    "Sex": "sex_group",
    "Age": "age_group",
    "Race/ethnicity": "race_group",
}


for horizon_index, (
    horizon,
    path
) in enumerate(
    prediction_files.items()
):

    pred = pd.read_csv(path)

    subject_col = find_column(
        pred.columns,
        ["subject_id"]
    )

    outcome_col = find_column(
        pred.columns,
        [
            "outcome",
            "target",
            "y_true",
            "label",
            f"aki_within_{horizon}"
        ]
    )

    probability_col = find_column(
        pred.columns,
        [
            "predicted_probability",
            "probability",
            "prediction",
            "predicted_risk"
        ]
    )

    if (
        subject_col is None
        or outcome_col is None
        or probability_col is None
    ):
        raise RuntimeError(
            f"{horizon}: required prediction "
            f"columns not found: {list(pred.columns)}"
        )


    pred[subject_col] = pd.to_numeric(
        pred[subject_col],
        errors="raise"
    ).astype("int64")

    pred[outcome_col] = pd.to_numeric(
        pred[outcome_col],
        errors="coerce"
    )

    pred[probability_col] = pd.to_numeric(
        pred[probability_col],
        errors="coerce"
    )

    pred = pred.loc[
        pred[outcome_col].isin(
            [0, 1]
        )
        & pred[
            probability_col
        ].notna()
    ].copy()


    merged = pred.merge(
        demo[
            [
                "subject_id",
                "sex_group",
                "age_group",
                "race_group"
            ]
        ],
        left_on=subject_col,
        right_on="subject_id",
        how="left",
        validate="many_to_one"
    )


    for variable, group_col in group_columns.items():

        feasible_rows = feasibility.loc[
            (
                feasibility["horizon"]
                .astype(str)
                == horizon
            )
            & (
                feasibility[
                    "subgroup_variable"
                ]
                == variable
            )
            & (
                feasibility[
                    "descriptive_performance_feasible"
                ]
            )
        ]


        for subgroup in feasible_rows[
            "subgroup"
        ].astype(str):

            group = merged.loc[
                merged[
                    group_col
                ].astype(str)
                == subgroup
            ].copy()

            if group.empty:
                raise RuntimeError(
                    f"{horizon} {variable} "
                    f"{subgroup}: no rows."
                )


            y = group[
                outcome_col
            ].to_numpy(
                dtype=np.int8
            )

            p = group[
                probability_col
            ].to_numpy(
                dtype=np.float64
            )


            auc = roc_auc(
                y,
                p
            )

            auprc = average_precision(
                y,
                p
            )


            bootstrap_seed = (
                SEED
                + horizon_index * 10000
                + len(rows) * 101
            )


            auc_ci, auprc_ci = patient_bootstrap(
                group,
                subject_col,
                outcome_col,
                probability_col,
                bootstrap_seed
            )


            rows.append({
                "horizon": horizon,
                "subgroup_variable": variable,
                "subgroup": subgroup,
                "patients": int(
                    group[
                        subject_col
                    ].nunique()
                ),
                "landmarks": int(
                    len(group)
                ),
                "events": int(
                    y.sum()
                ),
                "prevalence_percent": float(
                    100.0 * y.mean()
                ),
                "auroc": float(auc),
                "auroc_ci_lower": float(
                    auc_ci[0]
                ),
                "auroc_ci_upper": float(
                    auc_ci[1]
                ),
                "auprc": float(auprc),
                "auprc_ci_lower": float(
                    auprc_ci[0]
                ),
                "auprc_ci_upper": float(
                    auprc_ci[1]
                ),
                "bootstrap_unit": "patient",
                "bootstrap_replicates": BOOTSTRAPS,
            })


results = pd.DataFrame(
    rows
)

if len(results) != 36:
    raise RuntimeError(
        f"Expected 36 feasible subgroup results; "
        f"generated {len(results)}."
    )


results.to_csv(
    output_csv,
    index=False
)


# =============================================================================
# AUDIT
# =============================================================================

audit = []

audit.append("=" * 100)
audit.append(
    "STEP 64D5 — HELD-OUT TEST SUBGROUP PERFORMANCE"
)
audit.append("=" * 100)

audit.append("")
audit.append(
    "Timestamp: "
    + datetime.now().isoformat(
        timespec="seconds"
    )
)

audit.append("")
audit.append(
    "Analysis type: exploratory descriptive subgroup evaluation"
)

audit.append(
    "Prediction source: previously locked held-out test probabilities"
)

audit.append(
    f"Bootstrap: {BOOTSTRAPS} patient-level replicates"
)

audit.append(
    "Confidence interval: percentile 95% CI"
)

audit.append(
    "No between-group hypothesis tests were performed."
)

audit.append("")
audit.append(
    "Sparse subgroup excluded:"
)

audit.append(
    "American Indian / Alaska Native / Native Hawaiian / "
    "Pacific Islander at all three horizons."
)

audit.append("")
audit.append(
    "SUBGROUP PERFORMANCE:"
)


for horizon in [
    "6h",
    "12h",
    "24h"
]:

    audit.append("")
    audit.append(
        horizon.upper()
    )

    subset = results.loc[
        results["horizon"] == horizon
    ]

    for _, row in subset.iterrows():

        audit.append(
            f"{row['subgroup_variable']} | "
            f"{row['subgroup']} | "
            f"patients={int(row['patients'])} | "
            f"landmarks={int(row['landmarks'])} | "
            f"events={int(row['events'])} | "
            f"AUROC={row['auroc']:.3f} "
            f"({row['auroc_ci_lower']:.3f}-"
            f"{row['auroc_ci_upper']:.3f}) | "
            f"AUPRC={row['auprc']:.3f} "
            f"({row['auprc_ci_lower']:.3f}-"
            f"{row['auprc_ci_upper']:.3f})"
        )


audit.append("")
audit.append(
    "INTERPRETATION SAFETY:"
)

audit.append(
    "These analyses are exploratory and descriptive."
)

audit.append(
    "Differences in AUROC or AUPRC between demographic groups "
    "must not be interpreted as statistically significant without "
    "formal comparative testing."
)

audit.append(
    "AUPRC is sensitive to subgroup outcome prevalence and therefore "
    "should not be compared across groups without considering prevalence."
)

audit.append("")
audit.append(
    "LOCK PRESERVATION:"
)

audit.append(
    "No model was retrained."
)

audit.append(
    "No coefficient was changed."
)

audit.append(
    "No prediction was recalculated."
)

audit.append(
    "No threshold was changed."
)

audit.append(
    "No calibration model was changed."
)

audit.append(
    "No patient split was changed."
)

audit.append(
    "No manuscript was changed."
)

audit.append(
    "No HTE analysis was performed."
)

audit.append("")
audit.append(
    f"CSV created: {output_csv}"
)

audit.append("")
audit.append("=" * 100)
audit.append(
    "STEP 64D5 PASS"
)
audit.append("=" * 100)


with open(
    audit_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(audit)
        + "\n"
    )


# =============================================================================
# TERMINAL
# =============================================================================

print("=" * 80)
print("STEP 64D5 — COMPLETE")
print("=" * 80)

print()
print(
    f"Subgroup-horizon analyses: {len(results)}"
)

print(
    f"Patient-level bootstrap replicates: {BOOTSTRAPS}"
)

print()
print("CREATED:")
print(output_csv)
print(audit_path)

print()
print("EXPLORATORY HELD-OUT ANALYSIS ONLY:")
print("No model was retrained.")
print("No prediction was recalculated.")
print("No threshold/calibration was changed.")
print("No manuscript was changed.")

print()
print("=" * 80)
print("STEP 64D5 PASS")
print("=" * 80)
