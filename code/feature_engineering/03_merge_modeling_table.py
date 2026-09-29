import itertools
import os
import sqlite3

import numpy as np
import pandas as pd


folder = os.environ.get(
    "AKI_RESEARCH_FOLDER",
    os.path.expanduser("~/Documents/AKI_Research"),
)
grid_path = os.path.join(folder, "analysis_landmark_grid.csv")
vital_path = os.path.join(folder, "vital_trajectory_features.csv")
exposure_path = os.path.join(
    folder, "modifiable_exposure_trajectory_features.csv"
)
output_path = os.path.join(folder, "longitudinal_modeling_table.csv")
temporary_path = output_path + ".temporary"
audit_path = os.path.join(folder, "longitudinal_modeling_table_audit.txt")
database_path = os.path.join(folder, "landmark_grid_sort.temporary.sqlite")

chunk_size = int(os.environ.get("MODELING_MERGE_CHUNK_SIZE", "5000"))
key_columns = ["stay_id", "landmark_time"]
identifier_columns = ["subject_id", "hadm_id", "stay_id", "landmark_time"]

for path in (grid_path, vital_path, exposure_path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

headers = {
    "grid": pd.read_csv(grid_path, nrows=0).columns.tolist(),
    "vital": pd.read_csv(vital_path, nrows=0).columns.tolist(),
    "exposure": pd.read_csv(exposure_path, nrows=0).columns.tolist(),
}

for name, header in headers.items():
    missing = [column for column in key_columns if column not in header]
    if missing:
        raise ValueError(f"{name} file is missing key columns: {missing}")

vital_feature_columns = [
    name for name in headers["vital"] if name not in identifier_columns
]
exposure_feature_columns = [
    name for name in headers["exposure"] if name not in identifier_columns
]

vital_collisions = sorted(set(headers["grid"]).intersection(vital_feature_columns))
exposure_collisions = sorted(
    (set(headers["grid"]) | set(vital_feature_columns))
    .intersection(exposure_feature_columns)
)
if vital_collisions or exposure_collisions:
    raise ValueError(
        "Feature-name collisions: "
        f"vital={vital_collisions}, exposure={exposure_collisions}"
    )

if os.path.exists(temporary_path):
    os.remove(temporary_path)

if os.path.exists(database_path):
    os.remove(database_path)

print("Staging landmark grid for a low-memory disk sort...")
connection = sqlite3.connect(database_path)
staged_grid_rows = 0
for staging_number, staging_chunk in enumerate(
    pd.read_csv(grid_path, chunksize=chunk_size, low_memory=False), start=1
):
    staging_chunk["stay_id"] = pd.to_numeric(
        staging_chunk["stay_id"], errors="coerce"
    )
    staging_chunk["landmark_time"] = pd.to_datetime(
        staging_chunk["landmark_time"], errors="coerce"
    )
    if staging_chunk[key_columns].isna().any().any():
        raise ValueError(f"Invalid grid key in staging chunk {staging_number}")
    staging_chunk["stay_id"] = staging_chunk["stay_id"].astype("int64")
    staging_chunk["landmark_time"] = staging_chunk["landmark_time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    staging_chunk.to_sql(
        "landmark_grid",
        connection,
        if_exists="replace" if staging_number == 1 else "append",
        index=False,
    )
    staged_grid_rows += len(staging_chunk)
    if staging_number % 10 == 0:
        print("Grid staging chunks:", staging_number, "Rows:", staged_grid_rows)

connection.execute(
    "CREATE INDEX landmark_grid_key_index "
    "ON landmark_grid (stay_id, landmark_time)"
)
connection.commit()

quoted_columns = ", ".join(
    '"' + name.replace('"', '""') + '"' for name in headers["grid"]
)
grid_reader = pd.read_sql_query(
    f"SELECT {quoted_columns} FROM landmark_grid "
    "ORDER BY stay_id, landmark_time",
    connection,
    chunksize=chunk_size,
)
vital_reader = pd.read_csv(vital_path, chunksize=chunk_size, low_memory=False)
exposure_reader = pd.read_csv(
    exposure_path, chunksize=chunk_size, low_memory=False
)

rows_written = 0
chunks_processed = 0
duplicate_keys = 0
misaligned_vital_keys = 0
misaligned_exposure_keys = 0
identifier_mismatches = 0
infinite_numeric_values = 0
first_write = True
last_stay = None
last_time = None

print("Building low-memory longitudinal modeling table...")

for chunk_number, triple in enumerate(
    itertools.zip_longest(grid_reader, vital_reader, exposure_reader), start=1
):
    grid_chunk, vital_chunk, exposure_chunk = triple
    if grid_chunk is None or vital_chunk is None or exposure_chunk is None:
        raise RuntimeError("Input files have different numbers of rows")
    if not (len(grid_chunk) == len(vital_chunk) == len(exposure_chunk)):
        raise RuntimeError(
            f"Chunk {chunk_number} row mismatch: "
            f"grid={len(grid_chunk)}, vital={len(vital_chunk)}, "
            f"exposure={len(exposure_chunk)}"
        )

    frames = [grid_chunk, vital_chunk, exposure_chunk]
    for frame in frames:
        frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="coerce")
        frame["landmark_time"] = pd.to_datetime(
            frame["landmark_time"], errors="coerce"
        )
        if frame[key_columns].isna().any().any():
            raise ValueError(f"Invalid key in chunk {chunk_number}")
        frame["stay_id"] = frame["stay_id"].astype("int64")

    grid_keys = pd.MultiIndex.from_frame(grid_chunk[key_columns])
    vital_keys = pd.MultiIndex.from_frame(vital_chunk[key_columns])
    exposure_keys = pd.MultiIndex.from_frame(exposure_chunk[key_columns])

    if not grid_keys.is_monotonic_increasing:
        raise ValueError(f"Grid keys are not sorted in chunk {chunk_number}")

    duplicate_keys += int(grid_keys.duplicated().sum())
    misaligned_vital_keys += int((grid_keys != vital_keys).sum())
    misaligned_exposure_keys += int((grid_keys != exposure_keys).sum())

    first_stay = int(grid_chunk["stay_id"].iloc[0])
    first_time = grid_chunk["landmark_time"].iloc[0]
    if last_stay is not None:
        if first_stay < last_stay or (
            first_stay == last_stay and first_time <= last_time
        ):
            duplicate_keys += 1
    last_stay = int(grid_chunk["stay_id"].iloc[-1])
    last_time = grid_chunk["landmark_time"].iloc[-1]

    for identifier in ("subject_id", "hadm_id"):
        if identifier in grid_chunk and identifier in vital_chunk:
            identifier_mismatches += int((
                pd.to_numeric(grid_chunk[identifier], errors="coerce").to_numpy()
                != pd.to_numeric(vital_chunk[identifier], errors="coerce").to_numpy()
            ).sum())
        if identifier in grid_chunk and identifier in exposure_chunk:
            identifier_mismatches += int((
                pd.to_numeric(grid_chunk[identifier], errors="coerce").to_numpy()
                != pd.to_numeric(exposure_chunk[identifier], errors="coerce").to_numpy()
            ).sum())

    if duplicate_keys or misaligned_vital_keys or misaligned_exposure_keys:
        raise RuntimeError(
            "Key validation failed: "
            f"duplicates={duplicate_keys}, vital_mismatch={misaligned_vital_keys}, "
            f"exposure_mismatch={misaligned_exposure_keys}"
        )
    if identifier_mismatches:
        raise RuntimeError(f"Identifier mismatches: {identifier_mismatches}")

    result = pd.concat(
        [
            grid_chunk.reset_index(drop=True),
            vital_chunk[vital_feature_columns].reset_index(drop=True),
            exposure_chunk[exposure_feature_columns].reset_index(drop=True),
        ],
        axis=1,
    )
    numeric_result = result.select_dtypes(include=[np.number])
    infinite_numeric_values += int(np.isinf(numeric_result.to_numpy()).sum())

    result["landmark_time"] = result["landmark_time"].dt.strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    result.to_csv(
        temporary_path,
        mode="w" if first_write else "a",
        header=first_write,
        index=False,
    )
    first_write = False
    rows_written += len(result)
    chunks_processed += 1

    del result, numeric_result, grid_chunk, vital_chunk, exposure_chunk

    if chunk_number % 10 == 0:
        print("Chunks processed:", chunk_number, "Rows:", rows_written)

connection.close()
os.remove(database_path)

validation_passed = (
    rows_written > 0
    and duplicate_keys == 0
    and misaligned_vital_keys == 0
    and misaligned_exposure_keys == 0
    and identifier_mismatches == 0
    and infinite_numeric_values == 0
)
if not validation_passed:
    raise RuntimeError("Modeling-table validation failed; output not replaced")

os.replace(temporary_path, output_path)

output_columns = (
    len(headers["grid"])
    + len(vital_feature_columns)
    + len(exposure_feature_columns)
)
audit_lines = [
    "LONGITUDINAL MODELING TABLE AUDIT",
    "",
    f"Output rows: {rows_written}",
    f"Landmark-grid rows staged: {staged_grid_rows}",
    f"Output columns: {output_columns}",
    f"Chunks processed: {chunks_processed}",
    f"Landmark-grid columns: {len(headers['grid'])}",
    f"Vital-trajectory columns added: {len(vital_feature_columns)}",
    f"Exposure-trajectory columns added: {len(exposure_feature_columns)}",
    "",
    "VALIDATION:",
    f"Duplicate landmark keys: {duplicate_keys}",
    f"Misaligned vital keys: {misaligned_vital_keys}",
    f"Misaligned exposure keys: {misaligned_exposure_keys}",
    f"Identifier mismatches: {identifier_mismatches}",
    f"Infinite numeric values: {infinite_numeric_values}",
    f"Validation result: {'PASS' if validation_passed else 'FAIL'}",
    "",
    "IMPORTANT MODELING NOTE:",
    "Outcome and post-landmark columns are retained for labeling/audit only.",
    "They must not be supplied to a prediction model as input features.",
    "",
    f"Output: {output_path}",
    f"Audit: {audit_path}",
]

with open(audit_path, "w", encoding="utf-8", errors="replace") as audit_file:
    audit_file.write("\n".join(audit_lines))

print()
print("LONGITUDINAL MODELING TABLE MERGE COMPLETE")
print("\n".join(audit_lines))
