import os
import pandas as pd

folder = os.path.expanduser("~/Documents/AKI_Research")

source = os.path.join(
    folder,
    "analysis_landmark_grid.csv"
)

report_path = os.path.join(
    folder,
    "analysis_landmark_grid_validation.txt"
)

print("Loading landmark grid...")

data = pd.read_csv(
    source,
    low_memory=False
)

time_columns = [
    "intime",
    "outtime",
    "landmark_time",
    "definitive_aki_time"
]

for column in time_columns:
    data[column] = pd.to_datetime(
        data[column],
        errors="coerce"
    )

label_columns = [
    "aki_within_6h",
    "aki_within_12h",
    "aki_within_24h"
]

for column in label_columns:
    data[column] = pd.to_numeric(
        data[column],
        errors="coerce"
    )

case_mask = (
    data["analysis_status"]
    == "incident_AKI_case"
)

control_mask = (
    data["analysis_status"]
    == "eligible_control"
)

duplicate_rows = data.duplicated(
    subset=["stay_id", "landmark_time"]
).sum()

before_six_hours = (
    data["landmark_time"]
    < data["intime"] + pd.Timedelta(hours=6)
).sum()

at_or_after_discharge = (
    data["landmark_time"]
    >= data["outtime"]
).sum()

at_or_after_aki = (
    case_mask
    & (
        data["landmark_time"]
        >= data["definitive_aki_time"]
    )
).sum()

invalid_labels = {}

for column in label_columns:
    observed = data[column].dropna()

    invalid_labels[column] = (
        ~observed.isin([0, 1])
    ).sum()

six_implies_twelve_error = (
    (data["aki_within_6h"] == 1)
    & (data["aki_within_12h"] != 1)
).sum()

twelve_implies_twentyfour_error = (
    (data["aki_within_12h"] == 1)
    & (data["aki_within_24h"] != 1)
).sum()

positive_six = data[
    data["aki_within_6h"] == 1
]

positive_counts_by_case = (
    positive_six
    .groupby("stay_id")
    .size()
)

case_stays = set(
    data.loc[case_mask, "stay_id"].unique()
)

positive_case_stays = set(
    positive_six["stay_id"].unique()
)

cases_without_positive_six = len(
    case_stays - positive_case_stays
)

cases_with_multiple_positive_six = (
    positive_counts_by_case > 1
).sum()

control_positive_labels = (
    (
        control_mask
        & (
            (data["aki_within_6h"] == 1)
            | (data["aki_within_12h"] == 1)
            | (data["aki_within_24h"] == 1)
        )
    )
).sum()

structural_errors = (
    duplicate_rows
    + before_six_hours
    + at_or_after_discharge
    + at_or_after_aki
    + sum(invalid_labels.values())
    + six_implies_twelve_error
    + twelve_implies_twentyfour_error
    + cases_without_positive_six
    + cases_with_multiple_positive_six
    + control_positive_labels
)

result = "PASS" if structural_errors == 0 else "CHECK REQUIRED"

lines = [
    "LANDMARK GRID VALIDATION",
    "",
    f"Validation result: {result}",
    f"Total rows: {len(data)}",
    f"Unique ICU stays: {data['stay_id'].nunique()}",
    f"Incident AKI stays: {len(case_stays)}",
    f"Control stays: {data.loc[control_mask, 'stay_id'].nunique()}",
    "",
    f"Duplicate stay-time rows: {duplicate_rows}",
    f"Landmarks before 6 hours: {before_six_hours}",
    f"Landmarks at/after discharge: {at_or_after_discharge}",
    f"Case landmarks at/after AKI: {at_or_after_aki}",
    f"Invalid 6-hour labels: {invalid_labels['aki_within_6h']}",
    f"Invalid 12-hour labels: {invalid_labels['aki_within_12h']}",
    f"Invalid 24-hour labels: {invalid_labels['aki_within_24h']}",
    f"6h-to-12h nesting errors: {six_implies_twelve_error}",
    f"12h-to-24h nesting errors: {twelve_implies_twentyfour_error}",
    f"Cases without positive 6h label: {cases_without_positive_six}",
    f"Cases with multiple positive 6h labels: {cases_with_multiple_positive_six}",
    f"Positive labels among controls: {control_positive_labels}",
    "",
    f"Positive 6-hour labels: {int((data['aki_within_6h'] == 1).sum())}",
    f"Positive 12-hour labels: {int((data['aki_within_12h'] == 1).sum())}",
    f"Positive 24-hour labels: {int((data['aki_within_24h'] == 1).sum())}",
    "",
    f"Total structural errors: {structural_errors}"
]

with open(
    report_path,
    "w",
    encoding="utf-8"
) as report_file:
    report_file.write("\n".join(lines))

print()
print("\n".join(lines))
print()
print("Report:", report_path)
