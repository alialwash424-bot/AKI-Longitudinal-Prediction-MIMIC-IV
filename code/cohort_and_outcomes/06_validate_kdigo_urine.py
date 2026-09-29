import os
import numpy as np
import pandas as pd
import file_system

folder = file_system.pick_directory()

old_path = os.path.join(folder, "kdigo_urine_stays.csv")
urine_path = os.path.join(folder, "urine_output_events.csv")

output_path = os.path.join(
    folder,
    "kdigo_urine_stays_validated.csv"
)

audit_path = os.path.join(
    folder,
    "kdigo_urine_validated_audit.txt"
)

for path in [old_path, urine_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(os.path.basename(path) + " not found")

print("Loading ICU cohort and assigned weights...")

cohort = pd.read_csv(
    old_path,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "weight_kg",
        "weight_source"
    ],
    low_memory=False
)

cohort["intime"] = pd.to_datetime(
    cohort["intime"],
    errors="coerce"
)

cohort["outtime"] = pd.to_datetime(
    cohort["outtime"],
    errors="coerce"
)

cohort["weight_kg"] = pd.to_numeric(
    cohort["weight_kg"],
    errors="coerce"
)

cohort = cohort.drop_duplicates("stay_id")

print("ICU stays:", len(cohort))
print("Stays with usable weight:", cohort["weight_kg"].notna().sum())


# ---------------------------------------------
# READ AND COLLAPSE URINE EVENTS
# ---------------------------------------------

print("Reading urine-output events...")

parts = []
raw_rows = 0
chunk_number = 0

for chunk in pd.read_csv(
    urine_path,
    usecols=["stay_id", "charttime", "urine_ml"],
    chunksize=300000,
    low_memory=False
):
    chunk_number += 1
    raw_rows += len(chunk)

    chunk["charttime"] = pd.to_datetime(
        chunk["charttime"],
        errors="coerce"
    )

    chunk["urine_ml"] = pd.to_numeric(
        chunk["urine_ml"],
        errors="coerce"
    )

    chunk = chunk.dropna(
        subset=["stay_id", "charttime", "urine_ml"]
    )

    collapsed = (
        chunk.groupby(
            ["stay_id", "charttime"],
            as_index=False
        )["urine_ml"].sum()
    )

    parts.append(collapsed)

    print(
        "Chunks:",
        chunk_number,
        "| Raw rows:",
        raw_rows
    )

urine = pd.concat(parts, ignore_index=True)
del parts

urine = (
    urine.groupby(
        ["stay_id", "charttime"],
        as_index=False
    )["urine_ml"].sum()
)

before_filter = len(urine)

urine = urine[
    urine["urine_ml"].between(0, 5000)
].copy()

excluded_values = before_filter - len(urine)

urine = urine.merge(
    cohort[
        [
            "stay_id",
            "intime",
            "outtime",
            "weight_kg"
        ]
    ],
    on="stay_id",
    how="inner"
)

urine = urine[
    (urine["charttime"] >= urine["intime"])
    &
    (urine["charttime"] <= urine["outtime"])
].copy()

urine = urine.sort_values(
    ["stay_id", "charttime"]
)

print("Usable urine records:", len(urine))


# ---------------------------------------------
# VALIDATED ROLLING-WINDOW CALCULATION
# ---------------------------------------------

HOUR_NS = 3600 * 1_000_000_000
results = []
processed = 0

for stay_id, group in urine.groupby("stay_id", sort=False):
    processed += 1

    group = (
        group.groupby(
            ["stay_id", "charttime"],
            as_index=False
        )
        .agg(
            urine_ml=("urine_ml", "sum"),
            intime=("intime", "first"),
            weight_kg=("weight_kg", "first")
        )
        .sort_values("charttime")
    )

    weight = group["weight_kg"].iloc[0]
    intime = group["intime"].iloc[0]

    result = {
        "stay_id": stay_id,
        "urine_events": len(group),
        "assessable_uo": False,
        "first_uo_aki_time": pd.NaT,
        "uo_onset_hours": np.nan,
        "first_uo_aki_stage": 0,
        "maximum_uo_stage": 0
    }

    if pd.isna(weight) or pd.isna(intime):
        results.append(result)
        continue

    times = group["charttime"].astype("int64").to_numpy()
    values = group["urine_ml"].to_numpy(dtype=float)

    cumulative = np.concatenate(
        ([0.0], np.cumsum(values))
    )

    indexes = np.arange(len(times))

    # MIMIC treats each charted amount as representing
    # approximately the preceding hour. Therefore:
    # six hours = current record plus the previous five hours.
    left_6 = np.searchsorted(
        times,
        times - (5 * HOUR_NS),
        side="left"
    )

    left_12 = np.searchsorted(
        times,
        times - (11 * HOUR_NS),
        side="left"
    )

    left_24 = np.searchsorted(
        times,
        times - (23 * HOUR_NS),
        side="left"
    )

    sum_6 = cumulative[indexes + 1] - cumulative[left_6]
    sum_12 = cumulative[indexes + 1] - cumulative[left_12]
    sum_24 = cumulative[indexes + 1] - cumulative[left_24]

    duration_6 = (
        (times - times[left_6]) / HOUR_NS
    ) + 1.0

    duration_12 = (
        (times - times[left_12]) / HOUR_NS
    ) + 1.0

    duration_24 = (
        (times - times[left_24]) / HOUR_NS
    ) + 1.0

    rate_6 = sum_6 / (weight * duration_6)
    rate_12 = sum_12 / (weight * duration_12)
    rate_24 = sum_24 / (weight * duration_24)

    elapsed_hours = (
        times - pd.Timestamp(intime).value
    ) / HOUR_NS

    stages = np.zeros(len(times), dtype=np.int8)

    valid_6 = (
        (elapsed_hours > 6)
        & (duration_6 >= 6)
    )

    valid_12 = (
        (elapsed_hours > 6)
        & (duration_12 >= 12)
    )

    valid_24 = (
        (elapsed_hours > 6)
        & (duration_24 >= 24)
    )

    stages[
        valid_6
        & (rate_6 < 0.5)
    ] = 1

    stages[
        valid_12
        & (rate_12 < 0.5)
    ] = 2

    stages[
        (
            valid_24
            & (rate_24 < 0.3)
        )
        |
        (
            valid_12
            & (sum_12 <= 0)
        )
    ] = 3

    result["assessable_uo"] = bool(np.any(valid_6))

    aki_positions = np.where(stages > 0)[0]

    if len(aki_positions) > 0:
        first_position = aki_positions[0]
        first_time = pd.to_datetime(times[first_position])

        result["first_uo_aki_time"] = first_time
        result["uo_onset_hours"] = (
            first_time - intime
        ).total_seconds() / 3600.0

        result["first_uo_aki_stage"] = int(
            stages[first_position]
        )

        result["maximum_uo_stage"] = int(
            stages.max()
        )

    results.append(result)

    if processed % 5000 == 0:
        print("Validated stays processed:", processed)

urine_results = pd.DataFrame(results)

final = cohort.merge(
    urine_results,
    on="stay_id",
    how="left"
)

final["urine_events"] = (
    final["urine_events"]
    .fillna(0)
    .astype(int)
)

final["assessable_uo"] = (
    final["assessable_uo"]
    .fillna(False)
    .astype(bool)
)

final["first_uo_aki_stage"] = (
    final["first_uo_aki_stage"]
    .fillna(0)
    .astype(int)
)

final["maximum_uo_stage"] = (
    final["maximum_uo_stage"]
    .fillna(0)
    .astype(int)
)

final["incident_uo_aki_after_6h"] = (
    final["assessable_uo"]
    & final["first_uo_aki_time"].notna()
    & (final["uo_onset_hours"] > 6)
)

final.to_csv(output_path, index=False)

assessable = int(final["assessable_uo"].sum())
incident = int(final["incident_uo_aki_after_6h"].sum())

stage_counts = (
    final.loc[
        final["incident_uo_aki_after_6h"],
        "maximum_uo_stage"
    ]
    .value_counts()
    .sort_index()
)

audit_lines = [
    "VALIDATED KDIGO URINE CALCULATION",
    "",
    f"Total ICU stays: {len(final)}",
    f"Stays with usable weight: {final['weight_kg'].notna().sum()}",
    f"Assessable urine-output stays: {assessable}",
    f"Incident urine-output AKI after 6 hours: {incident}",
    f"Raw urine rows: {raw_rows}",
    f"Usable urine records: {len(urine)}",
    f"Excluded implausible totals: {excluded_values}",
    "",
    "MAXIMUM STAGE:"
]

for stage, count in stage_counts.items():
    audit_lines.append(f"Stage {stage}: {count}")

with open(
    audit_path,
    "w",
    encoding="utf-8",
    errors="replace"
) as file:
    file.write("\n".join(audit_lines))

print()
print("VALIDATED KDIGO URINE CALCULATION COMPLETE")
print("Total ICU stays:", len(final))
print("Stays with usable weight:", final["weight_kg"].notna().sum())
print("Assessable urine-output stays:", assessable)
print("Incident urine-output AKI after 6 hours:", incident)
print()
print("Maximum stages:")
print(stage_counts)
print()
print("Output:", output_path)
print("Audit:", audit_path)
