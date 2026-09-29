import os
import numpy as np
import pandas as pd
import file_system

folder = file_system.pick_directory()

urine_path = os.path.join(
    folder,
    "kdigo_urine_stays_validated.csv"
)

creatinine_path = os.path.join(
    folder,
    "kdigo_creatinine_stays.csv"
)

rrt_path = os.path.join(
    folder,
    "rrt_stays.csv"
)

output_path = os.path.join(
    folder,
    "definitive_kdigo_cohort.csv"
)

audit_path = os.path.join(
    folder,
    "definitive_kdigo_audit.txt"
)

for path in [urine_path, creatinine_path, rrt_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            os.path.basename(path) + " was not found"
        )


def convert_boolean(series):
    if str(series.dtype) == "bool":
        return series.fillna(False)

    return (
        series.fillna(False)
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes"])
    )


print("Loading validated urine-output results...")

urine = pd.read_csv(urine_path, low_memory=False)

urine_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "weight_kg",
    "weight_source",
    "urine_events",
    "assessable_uo",
    "first_uo_aki_time",
    "uo_onset_hours",
    "first_uo_aki_stage",
    "maximum_uo_stage",
    "incident_uo_aki_after_6h"
]

urine = urine[urine_columns].drop_duplicates("stay_id")


print("Loading creatinine results...")

creatinine = pd.read_csv(
    creatinine_path,
    low_memory=False
)

creatinine = creatinine[
    [
        "stay_id",
        "creatinine_measurements",
        "pre_icu_measurements",
        "icu_measurements",
        "baseline_pre_icu_min",
        "peak_icu_creatinine",
        "assessable_icu",
        "pre_icu_aki",
        "first_icu_aki_time",
        "aki_onset_hours",
        "first_icu_aki_stage",
        "maximum_icu_stage",
        "prevalent_aki",
        "incident_aki_after_6h"
    ]
].drop_duplicates("stay_id")


print("Loading RRT results...")

rrt = pd.read_csv(
    rrt_path,
    low_memory=False
)

rrt = rrt[
    [
        "stay_id",
        "first_rrt_start",
        "last_rrt_end",
        "active_rrt_records",
        "rrt_types",
        "all_rrt_related_records",
        "supporting_records",
        "received_active_rrt"
    ]
].drop_duplicates("stay_id")


# --------------------------------------------------
# MERGE COMPONENTS
# --------------------------------------------------

final = urine.merge(
    creatinine,
    on="stay_id",
    how="left"
)

final = final.merge(
    rrt,
    on="stay_id",
    how="left"
)

print("Merged ICU stays:", len(final))


# --------------------------------------------------
# CONVERT DATES
# --------------------------------------------------

date_columns = [
    "intime",
    "outtime",
    "first_uo_aki_time",
    "first_icu_aki_time",
    "first_rrt_start",
    "last_rrt_end"
]

for column in date_columns:
    final[column] = pd.to_datetime(
        final[column],
        errors="coerce"
    )


# --------------------------------------------------
# CONVERT BOOLEAN COLUMNS
# --------------------------------------------------

boolean_columns = [
    "assessable_uo",
    "incident_uo_aki_after_6h",
    "assessable_icu",
    "pre_icu_aki",
    "prevalent_aki",
    "incident_aki_after_6h",
    "received_active_rrt"
]

for column in boolean_columns:
    final[column] = convert_boolean(final[column])


# --------------------------------------------------
# NUMERIC STAGES
# --------------------------------------------------

stage_columns = [
    "first_uo_aki_stage",
    "maximum_uo_stage",
    "first_icu_aki_stage",
    "maximum_icu_stage"
]

for column in stage_columns:
    final[column] = (
        pd.to_numeric(
            final[column],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )
rrt_flag_text = (
    final["received_active_rrt"]
    .astype(str)
    .str.strip()
    .str.lower()
)

final["received_active_rrt"] = (
    rrt_flag_text.isin(["1", "1.0", "true"])
    | final["first_rrt_start"].notna()
)



final["active_rrt_records"] = (
    pd.to_numeric(
        final["active_rrt_records"],
        errors="coerce"
    )
    .fillna(0)
    .astype(int)
)


# --------------------------------------------------
# SIX-HOUR LANDMARK
# --------------------------------------------------

final["landmark_time"] = (
    final["intime"] +
    pd.Timedelta(hours=6)
)

final["icu_duration_hours"] = (
    final["outtime"] -
    final["intime"]
).dt.total_seconds() / 3600.0

final["followup_after_landmark_hours"] = (
    final["outtime"] -
    final["landmark_time"]
).dt.total_seconds() / 3600.0


# --------------------------------------------------
# IDENTIFY EARLY OR PREVALENT AKI
# --------------------------------------------------

early_creatinine = (
    final["prevalent_aki"]
    |
    (
        final["first_icu_aki_time"].notna()
        &
        (
            final["first_icu_aki_time"]
            <= final["landmark_time"]
        )
    )
)

early_urine = (
    final["first_uo_aki_time"].notna()
    &
    (
        final["first_uo_aki_time"]
        <= final["landmark_time"]
    )
)

early_rrt = (
    final["received_active_rrt"]
    &
    final["first_rrt_start"].notna()
    &
    (
        final["first_rrt_start"]
        <= final["landmark_time"]
    )
)

final["early_or_prevalent_aki"] = (
    early_creatinine
    | early_urine
    | early_rrt
)


# --------------------------------------------------
# INCIDENT COMPONENTS AFTER SIX HOURS
# --------------------------------------------------

creatinine_after_6h = (
    final["first_icu_aki_time"].notna()
    &
    (
        final["first_icu_aki_time"]
        > final["landmark_time"]
    )
)

urine_after_6h = (
    final["first_uo_aki_time"].notna()
    &
    (
        final["first_uo_aki_time"]
        > final["landmark_time"]
    )
)

rrt_after_6h = (
    final["received_active_rrt"]
    &
    final["first_rrt_start"].notna()
    &
    (
        final["first_rrt_start"]
        > final["landmark_time"]
    )
)

eligible_at_landmark = (
    ~final["early_or_prevalent_aki"]
    &
    (final["icu_duration_hours"] > 6)
)

final["incident_creatinine_aki"] = (
    eligible_at_landmark
    & creatinine_after_6h
)

final["incident_urine_aki"] = (
    eligible_at_landmark
    & urine_after_6h
)

final["incident_rrt_aki"] = (
    eligible_at_landmark
    & rrt_after_6h
)

final["definitive_incident_aki"] = (
    final["incident_creatinine_aki"]
    | final["incident_urine_aki"]
    | final["incident_rrt_aki"]
)


# --------------------------------------------------
# DEFINITIVE ONSET TIME
# --------------------------------------------------

candidate_times = pd.DataFrame(
    {
        "creatinine": final["first_icu_aki_time"].where(
            final["incident_creatinine_aki"]
        ),
        "urine": final["first_uo_aki_time"].where(
            final["incident_urine_aki"]
        ),
        "rrt": final["first_rrt_start"].where(
            final["incident_rrt_aki"]
        )
    }
)

final["definitive_aki_time"] = candidate_times.min(axis=1)

final["definitive_aki_onset_hours"] = (
    final["definitive_aki_time"] -
    final["intime"]
).dt.total_seconds() / 3600.0

final["hours_from_landmark_to_aki"] = (
    final["definitive_aki_time"] -
    final["landmark_time"]
).dt.total_seconds() / 3600.0


# --------------------------------------------------
# DEFINITIVE STAGE
# --------------------------------------------------

creatinine_stage = np.where(
    final["incident_creatinine_aki"],
    final["maximum_icu_stage"],
    0
)

urine_stage = np.where(
    final["incident_urine_aki"],
    final["maximum_uo_stage"],
    0
)

rrt_stage = np.where(
    final["incident_rrt_aki"],
    3,
    0
)

final["definitive_aki_stage"] = np.maximum.reduce(
    [
        creatinine_stage,
        urine_stage,
        rrt_stage
    ]
).astype(int)


# --------------------------------------------------
# OUTCOME MECHANISM
# --------------------------------------------------

mechanism = np.full(
    len(final),
    "",
    dtype=object
)

component_information = [
    (
        "creatinine",
        final["incident_creatinine_aki"].to_numpy()
    ),
    (
        "urine",
        final["incident_urine_aki"].to_numpy()
    ),
    (
        "RRT",
        final["incident_rrt_aki"].to_numpy()
    )
]

for label, mask in component_information:
    mechanism = np.where(
        mask,
        np.where(
            mechanism == "",
            label,
            mechanism + "+" + label
        ),
        mechanism
    )

mechanism = np.where(
    mechanism == "",
    "none",
    mechanism
)

final["aki_mechanism"] = mechanism


# --------------------------------------------------
# ANALYSIS STATUS
# --------------------------------------------------

final["any_kdigo_assessment"] = (
    final["assessable_uo"]
    | final["assessable_icu"]
    | final["received_active_rrt"]
)

final["analysis_status"] = "unassessable"

final.loc[
    final["icu_duration_hours"] <= 6,
    "analysis_status"
] = "ICU_stay_6h_or_less"

final.loc[
    final["early_or_prevalent_aki"],
    "analysis_status"
] = "early_or_prevalent_AKI"

control_mask = (
    eligible_at_landmark
    & final["any_kdigo_assessment"]
    & ~final["definitive_incident_aki"]
)

case_mask = (
    eligible_at_landmark
    & final["definitive_incident_aki"]
)

final.loc[
    control_mask,
    "analysis_status"
] = "eligible_control"

final.loc[
    case_mask,
    "analysis_status"
] = "incident_AKI_case"

final["analysis_included"] = final[
    "analysis_status"
].isin(
    [
        "eligible_control",
        "incident_AKI_case"
    ]
)


# --------------------------------------------------
# SAVE RESULTS
# --------------------------------------------------

final.to_csv(
    output_path,
    index=False
)

status_counts = final[
    "analysis_status"
].value_counts()

stage_counts = (
    final.loc[
        final["definitive_incident_aki"],
        "definitive_aki_stage"
    ]
    .value_counts()
    .sort_index()
)

mechanism_counts = (
    final.loc[
        final["definitive_incident_aki"],
        "aki_mechanism"
    ]
    .value_counts()
)

included = final[final["analysis_included"]]
cases = final[final["definitive_incident_aki"]]

audit_lines = [
    "DEFINITIVE KDIGO COHORT",
    "",
    f"Total ICU stays: {len(final)}",
    f"Included analysis stays: {len(included)}",
    f"Incident AKI cases: {len(cases)}",
    f"Eligible controls: {int(control_mask.sum())}",
    f"Unique included patients: {included['subject_id'].nunique()}",
    f"Unique AKI patients: {cases['subject_id'].nunique()}",
    "",
    "ANALYSIS STATUS:"
]

for status, count in status_counts.items():
    audit_lines.append(f"{status}: {count}")

audit_lines.extend(
    [
        "",
        "DEFINITIVE AKI STAGE:"
    ]
)

for stage, count in stage_counts.items():
    audit_lines.append(f"Stage {stage}: {count}")

audit_lines.extend(
    [
        "",
        "AKI MECHANISM:"
    ]
)

for mechanism_name, count in mechanism_counts.items():
    audit_lines.append(
        f"{mechanism_name}: {count}"
    )

with open(
    audit_path,
    "w",
    encoding="utf-8",
    errors="replace"
) as file:
    file.write("\n".join(audit_lines))


print()
print("DEFINITIVE KDIGO MERGE COMPLETE")
print("Total ICU stays:", len(final))
print("Included analysis stays:", len(included))
print("Incident AKI cases:", len(cases))
print("Eligible controls:", int(control_mask.sum()))
print("Unique included patients:", included["subject_id"].nunique())
print()
print("Analysis status:")
print(status_counts)
print()
print("Definitive AKI stages:")
print(stage_counts)
print()
print("AKI mechanisms:")
print(mechanism_counts)
print()
print("Output:", output_path)
print("Audit:", audit_path)
