import os
import numpy as np
import pandas as pd
from datetime import datetime


# =============================================================================
# STEP 64D3
# PRIMARY PREDICTION COHORT CHARACTERISTICS
#
# PURPOSE
# -------
# Construct a descriptive patient-level Table 1 for the PRIMARY prediction
# cohort using:
#
#   patient_level_split_manifest.csv
#   definitive_kdigo_cohort_VALIDATED.csv
#   patients.csv.gz
#   admissions.csv.gz
#
# This is DESCRIPTIVE ONLY.
#
# It does NOT:
# - retrain any model
# - change predictions
# - change outcomes
# - change the patient split
# - use the HTE cohort
# - perform subgroup model-performance analysis
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

table_path = os.path.join(
    folder,
    "step64D3_primary_cohort_characteristics.csv"
)

audit_path = os.path.join(
    folder,
    "step64D3_primary_cohort_characteristics_audit.txt"
)


# =============================================================================
# REQUIRED FILES
# =============================================================================

required = [
    split_path,
    cohort_path,
    patients_path,
    admissions_path,
]

for path in required:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}"
        )


# =============================================================================
# LOAD SPLIT MANIFEST
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

if split["subject_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate subject_id found in split manifest."
    )

expected_splits = {
    "train",
    "validation",
    "test"
}

observed_splits = set(
    split["split"].dropna().astype(str)
)

if observed_splits != expected_splits:
    raise RuntimeError(
        f"Unexpected split labels: {observed_splits}"
    )


# =============================================================================
# LOAD PRIMARY COHORT
# =============================================================================

cohort = pd.read_csv(
    cohort_path,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "analysis_included",
        "analysis_status",
        "definitive_incident_aki",
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

cohort["stay_id"] = pd.to_numeric(
    cohort["stay_id"],
    errors="coerce"
)


# Keep the validated included prediction cohort only.

included = cohort.copy()

if "analysis_included" in included.columns:

    value = included[
        "analysis_included"
    ]

    if value.dtype == bool:
        included = included.loc[value].copy()

    else:
        normalized = (
            value
            .astype(str)
            .str.strip()
            .str.lower()
        )

        included = included.loc[
            normalized.isin(
                [
                    "true",
                    "1",
                    "yes"
                ]
            )
        ].copy()


# =============================================================================
# VERIFY PATIENT UNIVERSE
# =============================================================================

cohort_subjects = set(
    included["subject_id"].unique()
)

split_subjects = set(
    split["subject_id"].unique()
)

missing_from_split = (
    cohort_subjects - split_subjects
)

split_without_cohort = (
    split_subjects - cohort_subjects
)

if missing_from_split:
    raise RuntimeError(
        f"{len(missing_from_split)} included cohort patients "
        "are missing from split manifest."
    )

if split_without_cohort:
    raise RuntimeError(
        f"{len(split_without_cohort)} split-manifest patients "
        "are absent from included cohort."
    )


# =============================================================================
# LOAD MIMIC PATIENT DEMOGRAPHICS
# =============================================================================

patients = pd.read_csv(
    patients_path,
    compression="gzip",
    usecols=[
        "subject_id",
        "gender",
        "anchor_age",
        "anchor_year",
        "anchor_year_group",
    ]
)

patients["subject_id"] = pd.to_numeric(
    patients["subject_id"],
    errors="raise"
).astype("int64")

if patients["subject_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate subject_id found in patients.csv.gz"
    )


# Restrict early to study patients.

patients = patients.loc[
    patients["subject_id"].isin(split_subjects)
].copy()


# =============================================================================
# LOAD ADMISSION DEMOGRAPHICS
# =============================================================================

admissions = pd.read_csv(
    admissions_path,
    compression="gzip",
    usecols=[
        "subject_id",
        "hadm_id",
        "admittime",
        "admission_type",
        "race",
        "insurance",
        "marital_status",
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

admissions = admissions.loc[
    admissions["subject_id"].isin(
        split_subjects
    )
].copy()


# =============================================================================
# BUILD PATIENT-LEVEL PRIMARY COHORT
# =============================================================================

# A patient can have multiple included ICU stays/admissions.
# Table 1 should not count the same patient multiple times.
#
# We select one index included admission per patient:
# the earliest included hospital admission available in the cohort.
#
# This selection is used ONLY for descriptive admission-level variables.
# It does not affect modeling, outcomes, or predictions.


included_admissions = (
    included[
        [
            "subject_id",
            "hadm_id"
        ]
    ]
    .dropna()
    .drop_duplicates()
)

included_admissions = included_admissions.merge(
    admissions,
    on=[
        "subject_id",
        "hadm_id"
    ],
    how="left",
    validate="many_to_one"
)

included_admissions[
    "admittime"
] = pd.to_datetime(
    included_admissions["admittime"],
    errors="coerce"
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
# PATIENT-LEVEL AKI STATUS
# =============================================================================

aki_numeric = pd.to_numeric(
    included["definitive_incident_aki"],
    errors="coerce"
)

included = included.assign(
    definitive_incident_aki_numeric=aki_numeric
)

patient_aki = (
    included
    .groupby(
        "subject_id"
    )[
        "definitive_incident_aki_numeric"
    ]
    .max()
    .rename(
        "any_incident_aki"
    )
    .reset_index()
)


# =============================================================================
# MERGE PATIENT TABLE
# =============================================================================

patient_table = (
    split
    .merge(
        patients,
        on="subject_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        index_admission[
            [
                "subject_id",
                "hadm_id",
                "admittime",
                "admission_type",
                "race",
                "insurance",
                "marital_status",
            ]
        ],
        on="subject_id",
        how="left",
        validate="one_to_one"
    )
    .merge(
        patient_aki,
        on="subject_id",
        how="left",
        validate="one_to_one"
    )
)


# =============================================================================
# AGE
# =============================================================================

patient_table[
    "anchor_age"
] = pd.to_numeric(
    patient_table["anchor_age"],
    errors="coerce"
)

patient_table[
    "anchor_year"
] = pd.to_numeric(
    patient_table["anchor_year"],
    errors="coerce"
)

patient_table[
    "admission_year"
] = patient_table[
    "admittime"
].dt.year


# MIMIC-IV age approximation at admission:
# anchor_age + (admission year - anchor_year)

patient_table[
    "age_at_index_admission"
] = (
    patient_table["anchor_age"]
    + (
        patient_table["admission_year"]
        - patient_table["anchor_year"]
    )
)

# Do not manufacture ages if source values are unavailable.

patient_table.loc[
    ~np.isfinite(
        patient_table[
            "age_at_index_admission"
        ]
    ),
    "age_at_index_admission"
] = np.nan


# =============================================================================
# CLEAN CATEGORICAL VALUES
# =============================================================================

for col in [
    "gender",
    "race",
    "insurance",
    "marital_status",
    "admission_type",
]:

    patient_table[col] = (
        patient_table[col]
        .fillna("Missing")
        .astype(str)
        .str.strip()
    )

    patient_table.loc[
        patient_table[col] == "",
        col
    ] = "Missing"


# =============================================================================
# SUMMARY HELPERS
# =============================================================================

def n_pct(count, denominator):

    if denominator == 0:
        return "0 (NA)"

    pct = 100.0 * count / denominator

    return f"{count} ({pct:.1f}%)"


def summarize_numeric(series):

    x = pd.to_numeric(
        series,
        errors="coerce"
    ).dropna()

    if len(x) == 0:
        return {
            "value": "NA",
            "n_nonmissing": 0
        }

    median = float(
        x.median()
    )

    q1 = float(
        x.quantile(0.25)
    )

    q3 = float(
        x.quantile(0.75)
    )

    return {
        "value": (
            f"{median:.1f} "
            f"[{q1:.1f}, {q3:.1f}]"
        ),
        "n_nonmissing": len(x)
    }


# =============================================================================
# CREATE TABLE 1
# =============================================================================

groups = [
    ("Overall", patient_table),
    (
        "Train",
        patient_table.loc[
            patient_table["split"] == "train"
        ]
    ),
    (
        "Validation",
        patient_table.loc[
            patient_table["split"] == "validation"
        ]
    ),
    (
        "Test",
        patient_table.loc[
            patient_table["split"] == "test"
        ]
    ),
]

rows = []


def add_row(
    characteristic,
    level,
    values
):

    row = {
        "characteristic": characteristic,
        "level": level,
    }

    row.update(values)

    rows.append(row)


# N

values = {}

for label, frame in groups:
    values[label] = str(
        len(frame)
    )

add_row(
    "Patients",
    "N",
    values
)


# Age

values = {}

for label, frame in groups:

    summary = summarize_numeric(
        frame["age_at_index_admission"]
    )

    values[label] = summary[
        "value"
    ]

add_row(
    "Age at index admission, years",
    "Median [Q1, Q3]",
    values
)


# Categorical variables

categorical_variables = [
    (
        "Sex",
        "gender"
    ),
    (
        "Race",
        "race"
    ),
    (
        "Insurance",
        "insurance"
    ),
    (
        "Marital status",
        "marital_status"
    ),
    (
        "Admission type",
        "admission_type"
    ),
]


for display_name, column in categorical_variables:

    levels = sorted(
        patient_table[column]
        .dropna()
        .astype(str)
        .unique()
    )

    for level in levels:

        values = {}

        for label, frame in groups:

            count = int(
                (
                    frame[column]
                    .astype(str)
                    == level
                ).sum()
            )

            values[label] = n_pct(
                count,
                len(frame)
            )

        add_row(
            display_name,
            level,
            values
        )


# Patient-level incident AKI

values = {}

for label, frame in groups:

    aki = pd.to_numeric(
        frame["any_incident_aki"],
        errors="coerce"
    )

    denominator = int(
        aki.notna().sum()
    )

    positive = int(
        (aki == 1).sum()
    )

    values[label] = n_pct(
        positive,
        denominator
    )

add_row(
    "Incident AKI during included ICU stay(s)",
    "Yes",
    values
)


table = pd.DataFrame(
    rows
)

table.to_csv(
    table_path,
    index=False
)


# =============================================================================
# AUDIT
# =============================================================================

audit = []

audit.append("=" * 95)
audit.append(
    "STEP 64D3 — PRIMARY COHORT CHARACTERISTICS AUDIT"
)
audit.append("=" * 95)

audit.append("")
audit.append(
    "Timestamp: "
    + datetime.now().isoformat(
        timespec="seconds"
    )
)

audit.append("")
audit.append(
    f"Patient-level rows: {len(patient_table)}"
)

audit.append(
    f"Unique patients: "
    f"{patient_table['subject_id'].nunique()}"
)

audit.append("")
audit.append("SPLIT COUNTS:")

for split_name in [
    "train",
    "validation",
    "test"
]:

    count = int(
        (
            patient_table["split"]
            == split_name
        ).sum()
    )

    audit.append(
        f"{split_name}: {count}"
    )


audit.append("")
audit.append("DEMOGRAPHIC COMPLETENESS:")

for col in [
    "age_at_index_admission",
    "gender",
    "race",
    "insurance",
    "marital_status",
    "admission_type",
]:

    if col == "age_at_index_admission":

        missing = int(
            patient_table[col]
            .isna()
            .sum()
        )

    else:

        missing = int(
            (
                patient_table[col]
                == "Missing"
            ).sum()
        )

    audit.append(
        f"{col}: missing={missing} "
        f"({100.0 * missing / len(patient_table):.2f}%)"
    )


audit.append("")
audit.append(
    "Index-admission rule:"
)

audit.append(
    "For patients with multiple included admissions, "
    "the earliest included hospital admission was selected "
    "for descriptive admission-level characteristics only."
)

audit.append("")
audit.append(
    "Age was approximated using MIMIC-IV anchor_age + "
    "(index admission year - anchor_year)."
)

audit.append("")
audit.append(
    "No demographic variable was used to retrain or modify "
    "the prediction models."
)

audit.append(
    "No HTE cohort was used."
)

audit.append(
    "No subgroup performance was calculated in this step."
)

audit.append(
    "No existing model, prediction, outcome, split, or manuscript "
    "file was modified."
)

audit.append("")
audit.append(
    f"Table created: {table_path}"
)

audit.append("")
audit.append("=" * 95)
audit.append(
    "STEP 64D3 PASS"
)
audit.append("=" * 95)


with open(
    audit_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "\n".join(audit) + "\n"
    )


# =============================================================================
# TERMINAL
# =============================================================================

print("=" * 80)
print("STEP 64D3 — COMPLETE")
print("=" * 80)

print()
print(
    f"Primary prediction patients: "
    f"{len(patient_table)}"
)

print()
print("Split counts:")

for split_name in [
    "train",
    "validation",
    "test"
]:

    count = int(
        (
            patient_table["split"]
            == split_name
        ).sum()
    )

    print(
        f"{split_name}: {count}"
    )

print()
print("CREATED:")
print(table_path)
print(audit_path)

print()
print("DESCRIPTIVE ONLY:")
print("No model was retrained.")
print("No prediction was changed.")
print("No patient split was changed.")
print("No manuscript was changed.")
print("No HTE cohort was used.")

print()
print("=" * 80)
print("STEP 64D3 PASS")
print("=" * 80)
