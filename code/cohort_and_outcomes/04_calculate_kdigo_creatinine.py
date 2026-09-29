import os
import pandas as pd
import numpy as np
import file_system
from collections import deque

folder = file_system.pick_directory()

source = os.path.join(folder, "icu_creatinine_events.csv")
output = os.path.join(folder, "kdigo_creatinine_stays.csv")

df = pd.read_csv(
    source,
    usecols=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "charttime",
        "valuenum"
    ],
    low_memory=False
)

for column in ["intime", "outtime", "charttime"]:
    df[column] = pd.to_datetime(df[column], errors="coerce")

df["valuenum"] = pd.to_numeric(df["valuenum"], errors="coerce")

df = df.dropna(
    subset=[
        "subject_id",
        "hadm_id",
        "stay_id",
        "intime",
        "outtime",
        "charttime",
        "valuenum"
    ]
)

df = df[df["valuenum"] > 0].copy()

for column in ["subject_id", "hadm_id", "stay_id"]:
    df[column] = df[column].astype(int)

# Combine duplicate measurements recorded at the same time
df = (
    df.groupby(
        [
            "subject_id",
            "hadm_id",
            "stay_id",
            "intime",
            "outtime",
            "charttime"
        ],
        as_index=False
    )["valuenum"]
    .median()
)

df = df.sort_values(["stay_id", "charttime"])

results = []
processed = 0

for stay_id, group in df.groupby("stay_id", sort=False):
    group = group.sort_values("charttime")

    subject_id = int(group["subject_id"].iloc[0])
    hadm_id = int(group["hadm_id"].iloc[0])
    intime = group["intime"].iloc[0]
    outtime = group["outtime"].iloc[0]

    minimum_48h = deque()
    minimum_7d = deque()

    pre_icu_values = []
    icu_values = []

    pre_icu_aki = False
    assessable_icu = False

    first_icu_aki_time = pd.NaT
    first_icu_aki_stage = 0
    maximum_icu_stage = 0

    for row in group.itertuples(index=False):
        time = row.charttime
        value = float(row.valuenum)

        while (
            minimum_48h
            and minimum_48h[0][0] < time - pd.Timedelta(hours=48)
        ):
            minimum_48h.popleft()

        while (
            minimum_7d
            and minimum_7d[0][0] < time - pd.Timedelta(days=7)
        ):
            minimum_7d.popleft()

        baseline_48h = (
            minimum_48h[0][1] if minimum_48h else np.nan
        )

        baseline_7d = (
            minimum_7d[0][1] if minimum_7d else np.nan
        )

        delta_48h = (
            value - baseline_48h
            if not np.isnan(baseline_48h)
            else np.nan
        )

        ratio_7d = (
            value / baseline_7d
            if not np.isnan(baseline_7d) and baseline_7d > 0
            else np.nan
        )

        stage = 0

        if (
            (not np.isnan(ratio_7d) and ratio_7d >= 3.0)
            or (
                value >= 4.0
                and not np.isnan(delta_48h)
                and delta_48h >= 0.3
            )
        ):
            stage = 3

        elif not np.isnan(ratio_7d) and ratio_7d >= 2.0:
            stage = 2

        elif (
            (not np.isnan(ratio_7d) and ratio_7d >= 1.5)
            or (
                not np.isnan(delta_48h)
                and delta_48h >= 0.3
            )
        ):
            stage = 1

        if time < intime:
            pre_icu_values.append(value)

            if stage > 0:
                pre_icu_aki = True

        elif time <= outtime:
            icu_values.append(value)

            if not np.isnan(baseline_48h) or not np.isnan(baseline_7d):
                assessable_icu = True

            maximum_icu_stage = max(maximum_icu_stage, stage)

            if stage > 0 and pd.isna(first_icu_aki_time):
                first_icu_aki_time = time
                first_icu_aki_stage = stage

        # Add current value only after evaluating it
        while minimum_48h and minimum_48h[-1][1] >= value:
            minimum_48h.pop()
        minimum_48h.append((time, value))

        while minimum_7d and minimum_7d[-1][1] >= value:
            minimum_7d.pop()
        minimum_7d.append((time, value))

    if pd.notna(first_icu_aki_time):
        onset_hours = (
            first_icu_aki_time - intime
        ).total_seconds() / 3600
    else:
        onset_hours = np.nan

    prevalent_aki = bool(
        pre_icu_aki
        or (
            not np.isnan(onset_hours)
            and onset_hours <= 6
        )
    )

    incident_aki_after_6h = bool(
        not pre_icu_aki
        and not np.isnan(onset_hours)
        and onset_hours > 6
    )

    results.append(
        {
            "subject_id": subject_id,
            "hadm_id": hadm_id,
            "stay_id": int(stay_id),
            "intime": intime,
            "outtime": outtime,
            "creatinine_measurements": len(group),
            "pre_icu_measurements": len(pre_icu_values),
            "icu_measurements": len(icu_values),
            "baseline_pre_icu_min": (
                min(pre_icu_values) if pre_icu_values else np.nan
            ),
            "peak_icu_creatinine": (
                max(icu_values) if icu_values else np.nan
            ),
            "assessable_icu": assessable_icu,
            "pre_icu_aki": pre_icu_aki,
            "first_icu_aki_time": first_icu_aki_time,
            "aki_onset_hours": onset_hours,
            "first_icu_aki_stage": first_icu_aki_stage,
            "maximum_icu_stage": maximum_icu_stage,
            "prevalent_aki": prevalent_aki,
            "incident_aki_after_6h": incident_aki_after_6h
        }
    )

    processed += 1

    if processed % 5000 == 0:
        print("ICU stays processed:", processed)

result = pd.DataFrame(results)
result.to_csv(output, index=False)

assessable = result["assessable_icu"].sum()
prevalent = result["prevalent_aki"].sum()
incident = result["incident_aki_after_6h"].sum()

print("\nKDIGO CREATININE CALCULATION COMPLETE")
print("Total ICU stays:", len(result))
print("Assessable ICU stays:", int(assessable))
print("Prevalent/early AKI:", int(prevalent))
print("Incident AKI after 6 hours:", int(incident))

incident_rows = result[result["incident_aki_after_6h"]]

print("\nMaximum stage among incident AKI:")
print(
    incident_rows["maximum_icu_stage"]
    .value_counts()
    .sort_index()
)

print("\nOutput:", output)
