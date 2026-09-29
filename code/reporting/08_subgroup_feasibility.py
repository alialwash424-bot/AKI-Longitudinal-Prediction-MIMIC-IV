import os
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# STEP 64D4
# TEST-SET SUBGROUP PERFORMANCE FEASIBILITY AUDIT
#
# PURPOSE
# -------
# Determine whether demographic subgroup performance analyses are statistically
# supportable using the LOCKED held-out test predictions.
#
# THIS STEP DOES NOT:
# - retrain a model
# - recalculate model predictions
# - change thresholds
# - change calibration
# - modify the patient split
# - calculate subgroup AUROC/AUPRC
# - modify the manuscript
#
# It only counts test patients, landmarks, and outcome events by candidate
# demographic subgroup.
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

prediction_files = {
    "6h": os.path.join(
        folder,
        "FINAL_TEST_6h_predictions.csv"
    ),
    "12h": os.path.join(
        folder,
        "FINAL_TEST_12h_predictions.csv"
    ),
    "24h": os.path.join(
        folder,
        "FINAL_TEST_24h_predictions.csv"
    ),
}

output_csv = os.path.join(
    folder,
    "step64D4_subgroup_feasibility.csv"
)

audit_path = os.path.join(
    folder,
    "step64D4_subgroup_feasibility_audit.txt"
)


# =============================================================================
# CHECK FILES
# =============================================================================

required = [
    split_path,
    cohort_path,
    patients_path,
    admissions_path,
    *prediction_files.values(),
]

for path in required:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# =============================================================================
# LOAD LOCKED TEST PATIENTS
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
    ].tolist()
)

if len(test_subjects) != 8583:

    raise RuntimeError(
        f"Expected 8583 test patients; found {len(test_subjects)}"
    )


# =============================================================================
# LOAD PRIMARY COHORT
# =============================================================================

cohort = pd.read_csv(
    cohort_path,
    usecols=[
        "subject_id",
        "hadm_id",
        "analysis_included",
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


# Keep included cohort

flag = cohort[
    "analysis_included"
]

if flag.dtype == bool:

    cohort = cohort.loc[
        flag
    ].copy()

else:

    normalized = (
        flag
        .astype(str)
        .str.strip()
        .str.lower()
    )

    cohort = cohort.loc[
        normalized.isin(
            [
                "true",
                "1",
                "yes"
            ]
        )
    ].copy()


cohort = cohort.loc[
    cohort["subject_id"].isin(
        test_subjects
    )
].copy()


# =============================================================================
# PATIENT DEMOGRAPHICS
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


# =============================================================================
# ADMISSIONS
# =============================================================================

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


# =============================================================================
# INDEX ADMISSION
# =============================================================================

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


# =============================================================================
# BUILD DEMOGRAPHIC TABLE
# =============================================================================

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

demo["anchor_age"] = pd.to_numeric(
    demo["anchor_age"],
    errors="coerce"
)

demo["anchor_year"] = pd.to_numeric(
    demo["anchor_year"],
    errors="coerce"
)

demo["admission_year"] = (
    demo["admittime"]
    .dt.year
)

demo["age"] = (
    demo["anchor_age"]
    + (
        demo["admission_year"]
        - demo["anchor_year"]
    )
)


# =============================================================================
# AGE GROUPS
# =============================================================================

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

demo["age_group"] = demo[
    "age_group"
].fillna(
    "Unknown"
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
        [
            "M",
            "F"
        ]
    ),
    "sex_group"
] = "Unknown"


# =============================================================================
# BROAD RACE / ETHNICITY GROUP
# =============================================================================

def broad_race(value):

    if pd.isna(value):

        return "Unknown / not reported"

    text = str(
        value
    ).strip().upper()

    if (
        text == ""
        or "UNKNOWN" in text
        or "UNABLE TO OBTAIN" in text
        or "DECLINED" in text
    ):

        return "Unknown / not reported"

    if text.startswith(
        "WHITE"
    ) or text == "PORTUGUESE":

        return "White"

    if text.startswith(
        "BLACK"
    ):

        return "Black"

    if text.startswith(
        "ASIAN"
    ):

        return "Asian"

    if (
        text.startswith(
            "HISPANIC"
        )
        or text.startswith(
            "SOUTH AMERICAN"
        )
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

    if (
        "MULTIPLE" in text
        or text == "OTHER"
    ):

        return "Other / multiple"

    return "Other / multiple"


demo[
    "race_group"
] = demo[
    "race"
].apply(
    broad_race
)


# =============================================================================
# VERIFY DEMOGRAPHIC PATIENT COUNT
# =============================================================================

if demo["subject_id"].duplicated().any():

    raise RuntimeError(
        "Duplicate subject_id in demographic table."
    )

if len(demo) != 8583:

    raise RuntimeError(
        f"Expected 8583 test demographic rows; found {len(demo)}"
    )


# =============================================================================
# IDENTIFY PREDICTION COLUMN NAMES
# =============================================================================

def find_column(
    columns,
    candidates
):

    lower_map = {
        str(c).lower(): c
        for c in columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:

            return lower_map[
                candidate.lower()
            ]

    return None


# =============================================================================
# FEASIBILITY COUNTS
# =============================================================================

rows = []

prediction_summaries = []


for horizon, path in prediction_files.items():

    pred = pd.read_csv(
        path
    )

    subject_col = find_column(
        pred.columns,
        [
            "subject_id"
        ]
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

    if subject_col is None:

        raise RuntimeError(
            f"{horizon}: subject_id column not found."
        )

    if outcome_col is None:

        raise RuntimeError(
            f"{horizon}: outcome column not found. "
            f"Columns={list(pred.columns)}"
        )

    if probability_col is None:

        raise RuntimeError(
            f"{horizon}: probability column not found. "
            f"Columns={list(pred.columns)}"
        )


    pred[subject_col] = pd.to_numeric(
        pred[subject_col],
        errors="raise"
    ).astype("int64")

    pred[outcome_col] = pd.to_numeric(
        pred[outcome_col],
        errors="coerce"
    )

    pred = pred.loc[
        pred[outcome_col].isin(
            [
                0,
                1
            ]
        )
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


    missing_demo = int(
        merged[
            "sex_group"
        ].isna().sum()
    )

    if missing_demo != 0:

        raise RuntimeError(
            f"{horizon}: {missing_demo} prediction rows "
            "lack demographic linkage."
        )


    prediction_summaries.append({
        "horizon": horizon,
        "landmarks": len(merged),
        "patients": merged[
            subject_col
        ].nunique(),
        "events": int(
            (
                merged[outcome_col]
                == 1
            ).sum()
        ),
        "nonevents": int(
            (
                merged[outcome_col]
                == 0
            ).sum()
        ),
    })


    subgroup_specs = [
        (
            "Sex",
            "sex_group"
        ),
        (
            "Age",
            "age_group"
        ),
        (
            "Race/ethnicity",
            "race_group"
        ),
    ]


    for variable_name, column in subgroup_specs:

        for level, group in merged.groupby(
            column,
            dropna=False
        ):

            patient_count = int(
                group[
                    subject_col
                ].nunique()
            )

            landmark_count = int(
                len(group)
            )

            event_count = int(
                (
                    group[outcome_col]
                    == 1
                ).sum()
            )

            nonevent_count = int(
                (
                    group[outcome_col]
                    == 0
                ).sum()
            )

            prevalence = (
                100.0
                * event_count
                / landmark_count
                if landmark_count
                else np.nan
            )

            # Conservative feasibility flag.
            #
            # This is NOT a statistical power calculation.
            # It is simply a screen to avoid calculating discrimination
            # estimates in extremely sparse groups.

            feasible = (
                patient_count >= 100
                and event_count >= 50
                and nonevent_count >= 50
            )

            rows.append({
                "horizon": horizon,
                "subgroup_variable": variable_name,
                "subgroup": str(level),
                "patients": patient_count,
                "landmarks": landmark_count,
                "events": event_count,
                "nonevents": nonevent_count,
                "event_prevalence_percent": prevalence,
                "descriptive_performance_feasible": feasible,
            })


# =============================================================================
# SAVE
# =============================================================================

results = pd.DataFrame(
    rows
)

results.to_csv(
    output_csv,
    index=False
)


# =============================================================================
# AUDIT REPORT
# =============================================================================

audit = []

audit.append(
    "=" * 95
)

audit.append(
    "STEP 64D4 — TEST-SET SUBGROUP FEASIBILITY AUDIT"
)

audit.append(
    "=" * 95
)

audit.append("")

audit.append(
    "Timestamp: "
    + datetime.now().isoformat(
        timespec="seconds"
    )
)

audit.append("")

audit.append(
    "This is a descriptive feasibility audit only."
)

audit.append(
    "No subgroup AUROC, AUPRC, calibration, threshold, "
    "or fairness metric was calculated."
)

audit.append("")

audit.append(
    "LOCKED TEST PREDICTION COUNTS:"
)

for item in prediction_summaries:

    audit.append(
        f"{item['horizon']}: "
        f"patients={item['patients']}, "
        f"landmarks={item['landmarks']}, "
        f"events={item['events']}, "
        f"nonevents={item['nonevents']}"
    )


audit.append("")

audit.append(
    "FEASIBILITY RULE:"
)

audit.append(
    "Descriptive subgroup performance considered feasible only if "
    "the subgroup contains >=100 patients, >=50 event landmarks, "
    "and >=50 non-event landmarks."
)

audit.append(
    "This threshold is an analytic safeguard, not a formal "
    "sample-size or power calculation."
)

audit.append("")

audit.append(
    "SUBGROUP COUNTS:"
)


for horizon in [
    "6h",
    "12h",
    "24h"
]:

    audit.append("")
    audit.append(
        f"{horizon.upper()}:"
    )

    subset = results.loc[
        results[
            "horizon"
        ] == horizon
    ]

    for _, row in subset.iterrows():

        audit.append(
            f"{row['subgroup_variable']} | "
            f"{row['subgroup']} | "
            f"patients={int(row['patients'])} | "
            f"landmarks={int(row['landmarks'])} | "
            f"events={int(row['events'])} | "
            f"nonevents={int(row['nonevents'])} | "
            f"prevalence={row['event_prevalence_percent']:.2f}% | "
            f"feasible={bool(row['descriptive_performance_feasible'])}"
        )


audit.append("")
audit.append(
    "SAFETY:"
)

audit.append(
    "No model was retrained."
)

audit.append(
    "No prediction was recalculated."
)

audit.append(
    "No probability was changed."
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
    "No HTE cohort was used."
)

audit.append("")
audit.append(
    f"CSV created: {output_csv}"
)

audit.append("")
audit.append(
    "=" * 95
)

audit.append(
    "STEP 64D4 PASS"
)

audit.append(
    "=" * 95
)


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
print("STEP 64D4 — COMPLETE")
print("=" * 80)

print()
print("Locked test predictions inspected:")

for item in prediction_summaries:

    print(
        f"{item['horizon']}: "
        f"{item['patients']} patients, "
        f"{item['landmarks']} landmarks, "
        f"{item['events']} events"
    )

print()

print(
    "Feasible subgroup rows:",
    int(
        results[
            "descriptive_performance_feasible"
        ].sum()
    ),
    "/",
    len(results)
)

print()
print("CREATED:")
print(output_csv)
print(audit_path)

print()
print("FEASIBILITY ONLY:")
print("No subgroup performance metric was calculated.")
print("No model or prediction was changed.")
print("No manuscript was changed.")

print()
print("=" * 80)
print("STEP 64D4 PASS")
print("=" * 80)
