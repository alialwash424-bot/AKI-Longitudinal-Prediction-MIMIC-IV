import os

import numpy as np
import pandas as pd


folder = os.environ.get(
    "AKI_RESEARCH_FOLDER",
    os.path.expanduser("~/Documents/AKI_Research"),
)
source_path = os.path.join(folder, "modifiable_exposure_6h_features.csv")
output_path = os.path.join(folder, "modifiable_exposure_trajectory_features.csv")
temporary_path = output_path + ".temporary"
audit_path = os.path.join(folder, "modifiable_exposure_trajectory_audit.txt")

chunk_size = int(os.environ.get("EXPOSURE_TRAJECTORY_CHUNK_SIZE", "10000"))
key_columns = ["stay_id", "landmark_time"]
window_bins = {"history12h": 2, "history24h": 4}

if not os.path.exists(source_path):
    raise FileNotFoundError(source_path)

header = pd.read_csv(source_path, nrows=0).columns.tolist()
missing_keys = [name for name in key_columns if name not in header]
if missing_keys:
    raise ValueError(f"Missing key columns: {missing_keys}")

numeric_candidates = [
    name for name in header
    if name not in key_columns and name.endswith("_prior6h")
]
sum_columns = [
    name for name in numeric_candidates
    if (
        name.startswith("fluid_")
        and (
            name.endswith("_ml_prior6h")
            or name in ("fluid_records_prior6h", "fluid_ml_per_kg_prior6h")
        )
    )
    or name.endswith("_minutes_prior6h")
    or name in (
        "vasoactive_records_prior6h",
        "vasoactive_total_agent_minutes_prior6h",
    )
]
flag_columns = [
    name for name in numeric_candidates if name.endswith("_any_prior6h")
]
maximum_columns = [
    name for name in numeric_candidates
    if "_max_rate_" in name or name == "vasoactive_agent_count_prior6h"
]

required_columns = [
    "fluid_total_ml_prior6h",
    "fluid_any_prior6h",
    "vasoactive_total_agent_minutes_prior6h",
    "vasoactive_any_prior6h",
]
missing_required = [name for name in required_columns if name not in header]
if missing_required:
    raise ValueError(f"Missing required exposure columns: {missing_required}")


def history_name(column, label):
    return column[: -len("prior6h")] + label


generated_columns = []
for label in window_bins:
    generated_columns.extend(history_name(name, label) for name in sum_columns)
    generated_columns.extend(history_name(name, label) for name in flag_columns)
    generated_columns.extend(history_name(name, label) for name in maximum_columns)
    generated_columns.extend([
        f"fluid_exposed_bins_{label}",
        f"vasoactive_exposed_bins_{label}",
    ])
generated_columns.extend([
    "fluid_total_ml_change_from_previous6h",
    "vasoactive_total_agent_minutes_change_from_previous6h",
])


def rolling_values(frame, column, bins, operation):
    rolling = frame.groupby("stay_id", sort=False)[column].rolling(
        window=bins, min_periods=1
    )
    if operation == "sum":
        values = rolling.sum()
    elif operation == "max":
        values = rolling.max()
    else:
        raise ValueError(operation)
    return values.reset_index(level=0, drop=True).to_numpy()


if os.path.exists(temporary_path):
    os.remove(temporary_path)

rows_read = 0
rows_written = 0
chunks_processed = 0
duplicate_input_keys = 0
duplicate_output_keys = 0
irregular_six_hour_steps = 0
first_bin_history_mismatches = 0
history_smaller_than_current = 0
negative_history_values = 0
infinite_numeric_values = 0
first_write = True
carry = None
last_input_stay = None
last_input_time = None

print("Building low-memory causal exposure trajectories...")

for chunk_number, chunk in enumerate(
    pd.read_csv(source_path, chunksize=chunk_size, low_memory=False), start=1
):
    chunk["stay_id"] = pd.to_numeric(chunk["stay_id"], errors="coerce")
    chunk["landmark_time"] = pd.to_datetime(
        chunk["landmark_time"], errors="coerce"
    )
    if chunk[key_columns].isna().any().any():
        raise ValueError(f"Missing/invalid landmark key in chunk {chunk_number}")

    chunk["stay_id"] = chunk["stay_id"].astype("int64")
    for name in numeric_candidates:
        chunk[name] = pd.to_numeric(chunk[name], errors="coerce")

    if not pd.MultiIndex.from_frame(chunk[key_columns]).is_monotonic_increasing:
        raise ValueError(f"Input is not sorted in chunk {chunk_number}")

    duplicate_input_keys += int(chunk.duplicated(key_columns).sum())
    same_stay = chunk["stay_id"].eq(chunk["stay_id"].shift(1))
    time_step = chunk["landmark_time"].diff()
    irregular_six_hour_steps += int(
        (same_stay & (time_step != pd.Timedelta(hours=6))).sum()
    )

    first_stay = int(chunk["stay_id"].iloc[0])
    first_time = chunk["landmark_time"].iloc[0]
    if last_input_stay is not None:
        if first_stay < last_input_stay:
            raise ValueError("stay_id order decreases across chunks")
        if first_stay == last_input_stay:
            if first_time == last_input_time:
                duplicate_input_keys += 1
            elif first_time - last_input_time != pd.Timedelta(hours=6):
                irregular_six_hour_steps += 1

    last_input_stay = int(chunk["stay_id"].iloc[-1])
    last_input_time = chunk["landmark_time"].iloc[-1]
    rows_read += len(chunk)

    if duplicate_input_keys:
        raise ValueError(f"Duplicate input landmark keys: {duplicate_input_keys}")
    if irregular_six_hour_steps:
        raise ValueError(
            "Exposure input is not a continuous six-hour grid: "
            f"{irregular_six_hour_steps} irregular steps"
        )

    carry_length = 0 if carry is None else len(carry)
    combined = (
        chunk.reset_index(drop=True)
        if carry is None
        else pd.concat([carry, chunk], ignore_index=True)
    )

    generated = {}
    for label, bins in window_bins.items():
        for column in sum_columns:
            generated[history_name(column, label)] = rolling_values(
                combined, column, bins, "sum"
            )
        for column in flag_columns:
            generated[history_name(column, label)] = rolling_values(
                combined, column, bins, "max"
            ).astype("uint8")
        for column in maximum_columns:
            generated[history_name(column, label)] = rolling_values(
                combined, column, bins, "max"
            )
        generated[f"fluid_exposed_bins_{label}"] = rolling_values(
            combined, "fluid_any_prior6h", bins, "sum"
        ).astype("int16")
        generated[f"vasoactive_exposed_bins_{label}"] = rolling_values(
            combined, "vasoactive_any_prior6h", bins, "sum"
        ).astype("int16")

    groups = combined.groupby("stay_id", sort=False)
    generated["fluid_total_ml_change_from_previous6h"] = (
        combined["fluid_total_ml_prior6h"]
        - groups["fluid_total_ml_prior6h"].shift(1).fillna(0.0)
    ).to_numpy()
    generated["vasoactive_total_agent_minutes_change_from_previous6h"] = (
        combined["vasoactive_total_agent_minutes_prior6h"]
        - groups["vasoactive_total_agent_minutes_prior6h"].shift(1).fillna(0.0)
    ).to_numpy()

    generated_frame = pd.DataFrame(generated, index=combined.index)
    output_chunk = pd.concat([combined, generated_frame], axis=1).iloc[
        carry_length:
    ].copy()

    duplicate_output_keys += int(output_chunk.duplicated(key_columns).sum())
    numeric_output = output_chunk.select_dtypes(include=[np.number])
    infinite_numeric_values += int(np.isinf(numeric_output.to_numpy()).sum())

    for label in window_bins:
        for column in sum_columns:
            output_column = history_name(column, label)
            history_smaller_than_current += int((
                output_chunk[output_column].fillna(0.0) + 1e-10
                < output_chunk[column].fillna(0.0)
            ).sum())

    for column in generated_columns:
        if column.endswith("change_from_previous6h"):
            continue
        negative_history_values += int((
            pd.to_numeric(output_chunk[column], errors="coerce") < -1e-10
        ).sum())

    combined_position = combined.groupby("stay_id", sort=False).cumcount()
    first_positions = combined.index[
        (combined_position == 0) & (combined.index >= carry_length)
    ]
    for label in window_bins:
        for column in sum_columns:
            output_column = history_name(column, label)
            first_bin_history_mismatches += int((~np.isclose(
                combined.loc[first_positions, column].fillna(0.0),
                generated_frame.loc[first_positions, output_column].fillna(0.0),
                rtol=1e-10,
                atol=1e-10,
            )).sum())

    output_chunk["landmark_time"] = output_chunk["landmark_time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    output_chunk.to_csv(
        temporary_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )
    first_write = False
    rows_written += len(output_chunk)
    chunks_processed += 1

    last_stay = int(combined["stay_id"].iloc[-1])
    carry = combined.loc[combined["stay_id"] == last_stay, header].tail(3).copy()
    del combined, generated_frame, output_chunk, numeric_output, chunk

    if chunk_number % 5 == 0:
        print("Chunks processed:", chunk_number, "Rows:", rows_written)


validation_passed = (
    rows_read == rows_written
    and rows_written > 0
    and duplicate_input_keys == 0
    and duplicate_output_keys == 0
    and irregular_six_hour_steps == 0
    and first_bin_history_mismatches == 0
    and history_smaller_than_current == 0
    and negative_history_values == 0
    and infinite_numeric_values == 0
)
if not validation_passed:
    raise RuntimeError("Exposure trajectory validation failed; output not replaced")

os.replace(temporary_path, output_path)

audit_lines = [
    "MODIFIABLE EXPOSURE TRAJECTORY FEATURE AUDIT",
    "",
    f"Input landmark rows: {rows_read}",
    f"Output trajectory rows: {rows_written}",
    f"Chunks processed: {chunks_processed}",
    f"Original columns retained: {len(header)}",
    f"Trajectory columns generated: {len(generated_columns)}",
    f"Summed exposure variables: {len(sum_columns)}",
    f"Indicator variables: {len(flag_columns)}",
    f"Maximum variables: {len(maximum_columns)}",
    "",
    "TEMPORAL DESIGN:",
    "12-hour history: current and immediately preceding six-hour bin",
    "24-hour history: current and three preceding six-hour bins",
    "Stay boundaries respected: yes",
    "Future information used: 0",
    "Incomplete early histories: available prior bins only",
    "",
    "VALIDATION:",
    f"Duplicate input keys: {duplicate_input_keys}",
    f"Duplicate output keys: {duplicate_output_keys}",
    f"Irregular six-hour steps: {irregular_six_hour_steps}",
    f"First-bin history mismatches: {first_bin_history_mismatches}",
    f"Histories smaller than current-bin component: {history_smaller_than_current}",
    f"Negative non-difference history values: {negative_history_values}",
    f"Infinite numeric values: {infinite_numeric_values}",
    f"Validation result: {'PASS' if validation_passed else 'FAIL'}",
    "",
    f"Output: {output_path}",
    f"Audit: {audit_path}",
]

with open(audit_path, "w", encoding="utf-8", errors="replace") as audit_file:
    audit_file.write("\n".join(audit_lines))

print()
print("MODIFIABLE EXPOSURE TRAJECTORY FEATURES COMPLETE")
print("\n".join(audit_lines))
