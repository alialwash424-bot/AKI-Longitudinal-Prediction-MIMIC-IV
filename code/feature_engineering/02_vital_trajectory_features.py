import os

import numpy as np
import pandas as pd


folder = os.environ.get(
    "AKI_RESEARCH_FOLDER",
    os.path.expanduser("~/Documents/AKI_Research"),
)
source_path = os.path.join(folder, "analysis_landmark_vitals.csv")
output_path = os.path.join(folder, "vital_trajectory_features.csv")
temporary_path = os.path.join(folder, "vital_trajectory_features.tmp.csv")
backup_path = os.path.join(folder, "vital_trajectory_features_before_rebuild.csv")
audit_path = os.path.join(folder, "vital_trajectory_features_audit.txt")

variables = ["hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c", "glucose"]
key_columns = ["subject_id", "hadm_id", "stay_id", "landmark_time"]
input_statistics = ["n", "mean", "std", "min", "max"]
input_columns = key_columns + [
    f"{variable}_{statistic}"
    for variable in variables
    for statistic in input_statistics
]

if not os.path.exists(source_path):
    raise FileNotFoundError(source_path)

header = list(pd.read_csv(source_path, nrows=0).columns)
missing_columns = sorted(set(input_columns).difference(header))
if missing_columns:
    raise ValueError(f"Source file is missing columns: {missing_columns}")

print("Loading landmark-level vital features...")
data = pd.read_csv(source_path, usecols=input_columns, low_memory=False)
data["stay_id"] = pd.to_numeric(data["stay_id"], errors="coerce")
data["landmark_time"] = pd.to_datetime(data["landmark_time"], errors="coerce")

missing_keys = int(data[["stay_id", "landmark_time"]].isna().any(axis=1).sum())
if missing_keys:
    raise ValueError(f"Missing landmark keys: {missing_keys}")

data["stay_id"] = data["stay_id"].astype("int64")
data = data.sort_values(["stay_id", "landmark_time"], kind="mergesort").reset_index(drop=True)
duplicate_keys = int(data.duplicated(["stay_id", "landmark_time"]).sum())
if duplicate_keys:
    raise ValueError(f"Duplicate landmark keys: {duplicate_keys}")

time_difference = data.groupby("stay_id", sort=False)["landmark_time"].diff()
nonpositive_time_steps = int((time_difference.dt.total_seconds() <= 0).sum())
if nonpositive_time_steps:
    raise ValueError(f"Non-increasing landmark times: {nonpositive_time_steps}")

first_time = data.groupby("stay_id", sort=False)["landmark_time"].transform("min")
elapsed_hours = (data["landmark_time"] - first_time).dt.total_seconds() / 3600.0
stay_ids = data["stay_id"]

trajectory_features = {}
trajectory_columns = []
ordering_errors = 0
negative_standard_deviations = 0
nonfinite_values = 0


def rolling_sum(frame, window):
    result = (
        frame.groupby(stay_ids, sort=False)
        .rolling(window=window, min_periods=1)
        .sum()
    )
    result.index = result.index.droplevel(0)
    return result.reindex(data.index)


def rolling_min(series, window):
    result = (
        series.groupby(stay_ids, sort=False)
        .rolling(window=window, min_periods=1)
        .min()
    )
    result.index = result.index.droplevel(0)
    return result.reindex(data.index)


def rolling_max(series, window):
    result = (
        series.groupby(stay_ids, sort=False)
        .rolling(window=window, min_periods=1)
        .max()
    )
    result.index = result.index.droplevel(0)
    return result.reindex(data.index)


for variable_number, variable in enumerate(variables, start=1):
    print(f"Building trajectories for {variable} ({variable_number}/{len(variables)})...")

    count = pd.to_numeric(data[f"{variable}_n"], errors="coerce").fillna(0.0)
    mean = pd.to_numeric(data[f"{variable}_mean"], errors="coerce")
    standard_deviation = pd.to_numeric(data[f"{variable}_std"], errors="coerce")
    minimum = pd.to_numeric(data[f"{variable}_min"], errors="coerce")
    maximum = pd.to_numeric(data[f"{variable}_max"], errors="coerce")

    observed = (count > 0) & mean.notna()
    safe_count = count.where(observed, 0.0)
    safe_mean = mean.where(observed, 0.0)
    safe_standard_deviation = standard_deviation.where(observed, 0.0)

    weighted_sum = safe_count * safe_mean
    weighted_sumsq = safe_count * (
        np.square(safe_standard_deviation) + np.square(safe_mean)
    )

    previous_mean = mean.groupby(stay_ids, sort=False).shift(1)
    delta_name = f"{variable}_delta_6h"
    trajectory_features[delta_name] = mean - previous_mean
    trajectory_columns.append(delta_name)

    rolling_base = pd.DataFrame({
        "measurements": safe_count,
        "observed_bins": observed.astype("float64"),
        "weighted_sum": weighted_sum,
        "weighted_sumsq": weighted_sumsq,
        "x": elapsed_hours.where(observed, 0.0),
        "x2": np.square(elapsed_hours).where(observed, 0.0),
        "xy": (elapsed_hours * safe_mean).where(observed, 0.0),
    })

    for window, hours in ((2, 12), (4, 24)):
        totals = rolling_sum(rolling_base, window)
        rolling_minimum = rolling_min(minimum.where(observed), window)
        rolling_maximum = rolling_max(maximum.where(observed), window)

        total_measurements = totals["measurements"]
        observed_bins = totals["observed_bins"]
        rolling_mean = totals["weighted_sum"] / total_measurements.replace(0, np.nan)
        variance = (
            totals["weighted_sumsq"] / total_measurements.replace(0, np.nan)
            - np.square(rolling_mean)
        ).clip(lower=0)
        rolling_standard_deviation = np.sqrt(variance)

        # Use bin means, rather than measurement counts, for the temporal slope.
        y_for_slope = mean.where(observed, 0.0)
        slope_frame = pd.DataFrame({
            "n": observed.astype("float64"),
            "x": elapsed_hours.where(observed, 0.0),
            "x2": np.square(elapsed_hours).where(observed, 0.0),
            "y": y_for_slope,
            "xy": (elapsed_hours * y_for_slope).where(observed, 0.0),
        })
        slope_totals = rolling_sum(slope_frame, window)
        slope_denominator = (
            slope_totals["n"] * slope_totals["x2"]
            - np.square(slope_totals["x"])
        )
        slope = (
            slope_totals["n"] * slope_totals["xy"]
            - slope_totals["x"] * slope_totals["y"]
        ) / slope_denominator.replace(0, np.nan)
        slope = slope.where(slope_totals["n"] >= 2)

        values = {
            f"{variable}_measurements_{hours}h": total_measurements,
            f"{variable}_observed_bins_{hours}h": observed_bins,
            f"{variable}_mean_{hours}h": rolling_mean,
            f"{variable}_std_{hours}h": rolling_standard_deviation,
            f"{variable}_min_{hours}h": rolling_minimum,
            f"{variable}_max_{hours}h": rolling_maximum,
            f"{variable}_slope_per_hour_{hours}h": slope,
        }
        for name, value in values.items():
            trajectory_features[name] = value
            trajectory_columns.append(name)

        ordering_errors += int(
            (
                rolling_mean.notna()
                & (
                    (rolling_minimum > rolling_mean + 1e-9)
                    | (rolling_mean > rolling_maximum + 1e-9)
                )
            ).sum()
        )
        negative_standard_deviations += int((rolling_standard_deviation < 0).sum())

for column in trajectory_columns:
    numeric = pd.to_numeric(trajectory_features[column], errors="coerce")
    nonfinite_values += int(np.isinf(numeric.to_numpy(dtype="float64", na_value=np.nan)).sum())

trajectory = pd.concat(
    [
        data[key_columns].copy(),
        pd.DataFrame(trajectory_features, index=data.index),
    ],
    axis=1,
)

rows_written = len(trajectory)
expected_rows = len(data)
output_duplicate_keys = int(trajectory.duplicated(["stay_id", "landmark_time"]).sum())
structural_errors = (
    output_duplicate_keys
    + ordering_errors
    + negative_standard_deviations
    + nonfinite_values
)

if rows_written != expected_rows:
    raise RuntimeError(f"Row-count mismatch: {rows_written} versus {expected_rows}")
if structural_errors:
    raise RuntimeError(
        "Trajectory validation failed: "
        f"duplicates={output_duplicate_keys}, ordering={ordering_errors}, "
        f"negative_std={negative_standard_deviations}, nonfinite={nonfinite_values}"
    )

if os.path.exists(temporary_path):
    os.remove(temporary_path)

print("Writing trajectory feature file...")
trajectory.to_csv(
    temporary_path,
    index=False,
    date_format="%Y-%m-%d %H:%M:%S",
    float_format="%.8g",
)

if os.path.exists(output_path) and not os.path.exists(backup_path):
    os.replace(output_path, backup_path)
os.replace(temporary_path, output_path)

lines = [
    "VITAL TRAJECTORY FEATURE AUDIT",
    "",
    f"Input landmark rows: {expected_rows}",
    f"Output trajectory rows: {rows_written}",
    f"Trajectory feature columns: {len(trajectory_columns)}",
    f"Duplicate output keys: {output_duplicate_keys}",
    f"Rolling ordering errors: {ordering_errors}",
    f"Negative rolling standard deviations: {negative_standard_deviations}",
    f"Infinite numeric values: {nonfinite_values}",
    "Future information used: 0",
    "Windows: current-and-prior bins only",
    "",
    "Trajectory validation: PASS",
    f"Output: {output_path}",
]

with open(audit_path, "w", encoding="utf-8") as audit_file:
    audit_file.write("\n".join(lines))

print()
print("\n".join(lines))
print("Audit:", audit_path)
