import os
import csv
import pandas as pd

# --------------------------------------------------
# FILE PATHS
# --------------------------------------------------

folder = os.path.expanduser("~/Documents/AKI_Research")

source = os.path.join(
    folder,
    "definitive_kdigo_cohort_VALIDATED.csv"
)

output = os.path.join(
    folder,
    "analysis_landmark_grid.csv"
)

audit_output = os.path.join(
    folder,
    "analysis_landmark_grid_audit.txt"
)

# --------------------------------------------------
# LOAD THE VALIDATED COHORT
# --------------------------------------------------

columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "intime",
    "outtime",
    "definitive_aki_time",
    "analysis_status"
]

print("Loading validated definitive cohort...")

data = pd.read_csv(
    source,
    usecols=columns,
    low_memory=False
)

for column in [
    "intime",
    "outtime",
    "definitive_aki_time"
]:
    data[column] = pd.to_datetime(
        data[column],
        errors="coerce"
    )

# Include only the final analysis population
data = data[
    data["analysis_status"].isin(
        [
            "incident_AKI_case",
            "eligible_control"
        ]
    )
].copy()

print("Eligible ICU stays:", len(data))

# --------------------------------------------------
# OUTPUT COLUMNS
# --------------------------------------------------

output_columns = [
    "subject_id",
    "hadm_id",
    "stay_id",
    "analysis_status",
    "intime",
    "outtime",
    "landmark_time",
    "landmark_number",
    "hours_since_icu_admission",
    "definitive_aki_time",
    "hours_to_aki",
    "aki_within_6h",
    "aki_within_12h",
    "aki_within_24h"
]

total_landmarks = 0
case_stays = 0
control_stays = 0
skipped_stays = 0

positive_6h = 0
positive_12h = 0
positive_24h = 0

evaluable_6h = 0
evaluable_12h = 0
evaluable_24h = 0

# --------------------------------------------------
# FUNCTION FOR HORIZON LABELS
# --------------------------------------------------

def make_label(
    is_case,
    hours_to_event,
    available_followup,
    horizon
):
    if (
        is_case
        and hours_to_event is not None
        and hours_to_event > 0
        and hours_to_event <= horizon
    ):
        return 1

    if available_followup >= horizon:
        return 0

    # Blank means insufficient follow-up
    return ""

# --------------------------------------------------
# BUILD SIX-HOUR LANDMARKS
# --------------------------------------------------

print("Building six-hour landmark grid...")

with open(
    output,
    "w",
    newline="",
    encoding="utf-8"
) as output_file:

    writer = csv.DictWriter(
        output_file,
        fieldnames=output_columns
    )

    writer.writeheader()

    for row in data.itertuples(index=False):

        if pd.isna(row.intime) or pd.isna(row.outtime):
            skipped_stays += 1
            continue

        is_case = (
            row.analysis_status
            == "incident_AKI_case"
        )

        if is_case:
            case_stays += 1

            if pd.isna(row.definitive_aki_time):
                skipped_stays += 1
                continue

            if row.definitive_aki_time > row.outtime:
                skipped_stays += 1
                continue

            endpoint = row.definitive_aki_time

        else:
            control_stays += 1
            endpoint = row.outtime

        landmark_time = (
            row.intime
            + pd.Timedelta(hours=6)
        )

        landmark_number = 1

        while landmark_time < endpoint:

            available_followup = (
                endpoint - landmark_time
            ).total_seconds() / 3600.0

            if is_case:
                hours_to_aki = (
                    row.definitive_aki_time
                    - landmark_time
                ).total_seconds() / 3600.0
            else:
                hours_to_aki = None

            label_6h = make_label(
                is_case,
                hours_to_aki,
                available_followup,
                6
            )

            label_12h = make_label(
                is_case,
                hours_to_aki,
                available_followup,
                12
            )

            label_24h = make_label(
                is_case,
                hours_to_aki,
                available_followup,
                24
            )

            if label_6h != "":
                evaluable_6h += 1
                positive_6h += int(label_6h)

            if label_12h != "":
                evaluable_12h += 1
                positive_12h += int(label_12h)

            if label_24h != "":
                evaluable_24h += 1
                positive_24h += int(label_24h)

            hours_since_icu = (
                landmark_time - row.intime
            ).total_seconds() / 3600.0

            writer.writerow(
                {
                    "subject_id": int(row.subject_id),
                    "hadm_id": int(row.hadm_id),
                    "stay_id": int(row.stay_id),
                    "analysis_status": row.analysis_status,
                    "intime": row.intime,
                    "outtime": row.outtime,
                    "landmark_time": landmark_time,
                    "landmark_number": landmark_number,
                    "hours_since_icu_admission":
                        round(hours_since_icu, 2),
                    "definitive_aki_time":
                        row.definitive_aki_time
                        if is_case else "",
                    "hours_to_aki":
                        round(hours_to_aki, 2)
                        if hours_to_aki is not None
                        else "",
                    "aki_within_6h": label_6h,
                    "aki_within_12h": label_12h,
                    "aki_within_24h": label_24h
                }
            )

            total_landmarks += 1

            if total_landmarks % 100000 == 0:
                print(
                    "Landmarks written:",
                    total_landmarks
                )

            landmark_time += pd.Timedelta(hours=6)
            landmark_number += 1

# --------------------------------------------------
# SAVE AUDIT REPORT
# --------------------------------------------------

audit_lines = [
    "LONGITUDINAL LANDMARK GRID AUDIT",
    "",
    f"Eligible ICU stays loaded: {len(data)}",
    f"Incident AKI stays: {case_stays}",
    f"Eligible control stays: {control_stays}",
    f"Skipped invalid stays: {skipped_stays}",
    f"Total six-hour landmarks: {total_landmarks}",
    "",
    f"Evaluable 6-hour landmarks: {evaluable_6h}",
    f"Positive 6-hour labels: {positive_6h}",
    "",
    f"Evaluable 12-hour landmarks: {evaluable_12h}",
    f"Positive 12-hour labels: {positive_12h}",
    "",
    f"Evaluable 24-hour landmarks: {evaluable_24h}",
    f"Positive 24-hour labels: {positive_24h}",
    "",
    f"Grid output: {output}",
    f"Audit output: {audit_output}"
]

with open(
    audit_output,
    "w",
    encoding="utf-8"
) as audit_file:
    audit_file.write("\n".join(audit_lines))

print()
print("LANDMARK GRID COMPLETE")
print("\n".join(audit_lines))
