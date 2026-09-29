import os
import numpy as np
import pandas as pd
import file_system

folder = file_system.pick_directory()

icu_path = os.path.join(folder, "icustays.csv.gz")
urine_path = os.path.join(folder, "urine_output_events.csv")
chart_weight_path = os.path.join(folder, "weight_events.csv")
input_weight_path = os.path.join(folder, "inputevent_weights.csv")

output_path = os.path.join(folder, "kdigo_urine_stays.csv")
audit_path = os.path.join(folder, "kdigo_urine_audit.txt")

required = [
    icu_path,
    urine_path,
    chart_weight_path,
    input_weight_path
]

for path in required:
    if not os.path.exists(path):
        raise FileNotFoundError(os.path.basename(path) + " was not found")

print("Loading ICU stays...")

icu = pd.read_csv(
    icu_path,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime"
    ],
    low_memory=False
)

icu["intime"] = pd.to_datetime(icu["intime"], errors="coerce")
icu["outtime"] = pd.to_datetime(icu["outtime"], errors="coerce")

icu = (
    icu.dropna(subset=["stay_id", "intime", "outtime"])
       .drop_duplicates("stay_id")
)

print("ICU stays:", len(icu))


# --------------------------------------------------
# SELECT ADMISSION WEIGHTS
# --------------------------------------------------

print("Loading charted weights...")

chart_weight = pd.read_csv(
    chart_weight_path,
    usecols=["stay_id", "charttime", "weight_kg"],
    low_memory=False
)

chart_weight["charttime"] = pd.to_datetime(
    chart_weight["charttime"],
    errors="coerce"
)

chart_weight["weight_kg"] = pd.to_numeric(
    chart_weight["weight_kg"],
    errors="coerce"
)

chart_weight = chart_weight[
    chart_weight["weight_kg"].between(20, 300)
].dropna()

chart_weight = chart_weight.merge(
    icu[["stay_id", "intime", "outtime"]],
    on="stay_id",
    how="inner"
)

chart_weight = chart_weight[
    (chart_weight["charttime"] >=
     chart_weight["intime"] - pd.Timedelta(hours=24))
    &
    (chart_weight["charttime"] <=
     chart_weight["intime"] + pd.Timedelta(hours=24))
].copy()

chart_weight["distance"] = (
    chart_weight["charttime"] -
    chart_weight["intime"]
).abs()

chart_choice = (
    chart_weight
    .sort_values(["stay_id", "distance", "charttime"])
    .drop_duplicates("stay_id")
    [["stay_id", "weight_kg"]]
    .rename(columns={"weight_kg": "chart_weight_kg"})
)

print("Stays with charted admission weight:", len(chart_choice))


print("Loading input-event weights...")

input_weight = pd.read_csv(
    input_weight_path,
    usecols=["stay_id", "starttime", "inputevent_weight_kg"],
    low_memory=False
)

input_weight["starttime"] = pd.to_datetime(
    input_weight["starttime"],
    errors="coerce"
)

input_weight["inputevent_weight_kg"] = pd.to_numeric(
    input_weight["inputevent_weight_kg"],
    errors="coerce"
)

input_weight = input_weight[
    input_weight["inputevent_weight_kg"].between(20, 300)
].dropna()

input_weight = input_weight.merge(
    icu[["stay_id", "intime", "outtime"]],
    on="stay_id",
    how="inner"
)

input_weight = input_weight[
    (input_weight["starttime"] >=
     input_weight["intime"] - pd.Timedelta(hours=24))
    &
    (input_weight["starttime"] <=
     input_weight["outtime"])
].copy()

input_weight["distance"] = (
    input_weight["starttime"] -
    input_weight["intime"]
).abs()

input_choice = (
    input_weight
    .sort_values(["stay_id", "distance", "starttime"])
    .drop_duplicates("stay_id")
    [["stay_id", "inputevent_weight_kg"]]
)

cohort = icu.merge(
    chart_choice,
    on="stay_id",
    how="left"
)

cohort = cohort.merge(
    input_choice,
    on="stay_id",
    how="left"
)

cohort["weight_kg"] = cohort["chart_weight_kg"].fillna(
    cohort["inputevent_weight_kg"]
)

cohort["weight_source"] = np.select(
    [
        cohort["chart_weight_kg"].notna(),
        cohort["inputevent_weight_kg"].notna()
    ],
    [
        "chartevents",
        "inputevents"
    ],
    default="missing"
)

print("Stays with usable weight:", cohort["weight_kg"].notna().sum())


# --------------------------------------------------
# LOAD AND COLLAPSE URINE EVENTS
# --------------------------------------------------

print("Processing urine-output events...")

parts = []
chunk_number = 0
raw_rows = 0

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
        "Urine chunks processed:",
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
print("Implausible totals excluded:", excluded_values)


# --------------------------------------------------
# CALCULATE ROLLING KDIGO URINE STAGES
# --------------------------------------------------

HOUR_NS = 3600 * 1_000_000_000
results = []
processed = 0

for stay_id, group in urine.groupby("stay_id", sort=False):
    processed += 1

    group = group.sort_values("charttime")

    weight = group["weight_kg"].iloc[0]
    intime = group["intime"].iloc[0]

    base = {
        "stay_id": stay_id,
        "urine_events": len(group),
        "assessable_uo": False,
        "first_uo_aki_time": pd.NaT,
        "uo_onset_hours": np.nan,
        "first_uo_aki_stage": 0,
        "maximum_uo_stage": 0
    }

    if pd.isna(weight) or pd.isna(intime):
        results.append(base)
        continue

    event_data = (
        group.groupby("charttime", as_index=False)
             ["urine_ml"].sum()
             .sort_values("charttime")
    )

    times = event_data["charttime"].astype("int64").to_numpy()
    values = event_data["urine_ml"].to_numpy(dtype=float)

    cumulative = np.concatenate(
        ([0.0], np.cumsum(values))
    )

    indexes = np.arange(len(times))

    left_6 = np.searchsorted(
        times,
        times - (6 * HOUR_NS),
        side="right"
    )

    left_12 = np.searchsorted(
        times,
        times - (12 * HOUR_NS),
        side="right"
    )

    left_24 = np.searchsorted(
        times,
        times - (24 * HOUR_NS),
        side="right"
    )

    sum_6 = cumulative[indexes + 1] - cumulative[left_6]
    sum_12 = cumulative[indexes + 1] - cumulative[left_12]
    sum_24 = cumulative[indexes + 1] - cumulative[left_24]

    intime_ns = pd.Timestamp(intime).value

    elapsed_hours = (
        times - intime_ns
    ) / HOUR_NS

    rate_6 = sum_6 / (weight * 6.0)
    rate_12 = sum_12 / (weight * 12.0)
    rate_24 = sum_24 / (weight * 24.0)

    stages = np.zeros(len(times), dtype=np.int8)

    stage_1 = (
        (elapsed_hours >= 6)
        & (rate_6 < 0.5)
    )

    stage_2 = (
        (elapsed_hours >= 12)
        & (rate_12 < 0.5)
    )

    stage_3 = (
        (
            (elapsed_hours >= 24)
            & (rate_24 < 0.3)
        )
        |
        (
            (elapsed_hours >= 12)
            & (sum_12 <= 0)
        )
    )

    stages[stage_1] = 1
    stages[stage_2] = 2
    stages[stage_3] = 3

    assessable = bool(np.any(elapsed_hours >= 6))
    aki_positions = np.where(stages > 0)[0]

    base["assessable_uo"] = assessable

    if len(aki_positions) > 0:
        first_position = aki_positions[0]
        first_time = pd.to_datetime(times[first_position])

        base["first_uo_aki_time"] = first_time
        base["uo_onset_hours"] = (
            first_time - intime
        ).total_seconds() / 3600

        base["first_uo_aki_stage"] = int(
            stages[first_position]
        )

        base["maximum_uo_stage"] = int(
            stages.max()
        )

    results.append(base)

    if processed % 5000 == 0:
        print("Urine-output stays processed:", processed)

urine_results = pd.DataFrame(results)

final = cohort.merge(
    urine_results,
    on="stay_id",
    how="left"
)

final["urine_events"] = final["urine_events"].fillna(0).astype(int)
final["assessable_uo"] = final["assessable_uo"].fillna(False)
final["first_uo_aki_stage"] = (
    final["first_uo_aki_stage"].fillna(0).astype(int)
)
final["maximum_uo_stage"] = (
    final["maximum_uo_stage"].fillna(0).astype(int)
)

final["incident_uo_aki_after_6h"] = (
    final["assessable_uo"]
    & final["first_uo_aki_time"].notna()
    & (final["uo_onset_hours"] >= 6)
)

final.to_csv(output_path, index=False)

assessable_count = int(final["assessable_uo"].sum())
aki_count = int(final["incident_uo_aki_after_6h"].sum())

stage_counts = (
    final.loc[
        final["incident_uo_aki_after_6h"],
        "maximum_uo_stage"
    ]
    .value_counts()
    .sort_index()
)

audit_lines = [
    "KDIGO URINE-OUTPUT CALCULATION",
    "",
    f"Total ICU stays: {len(final)}",
    f"Stays with usable weight: {final['weight_kg'].notna().sum()}",
    f"Urine-output assessable stays: {assessable_count}",
    f"Incident urine-output AKI after 6 hours: {aki_count}",
    f"Raw urine rows examined: {raw_rows}",
    f"Usable urine records: {len(urine)}",
    f"Implausible totals excluded: {excluded_values}",
    "",
    "MAXIMUM URINE STAGE:"
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
print("KDIGO URINE CALCULATION COMPLETE")
print("Total ICU stays:", len(final))
print("Stays with usable weight:", final["weight_kg"].notna().sum())
print("Assessable urine-output stays:", assessable_count)
print("Incident urine-output AKI after 6 hours:", aki_count)
print()
print("Maximum stage among incident urine AKI:")
print(stage_counts)
print()
print("Output:", output_path)
print("Audit:", audit_path)
