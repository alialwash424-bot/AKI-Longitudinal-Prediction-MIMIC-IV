import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# FILE PATHS AND SETTINGS
# -----------------------------------------------------------------------------

folder = os.path.expanduser("~/Documents/AKI_Research")
source_path = os.path.join(folder, "longitudinal_modeling_table.csv")
manifest_path = os.path.join(folder, "patient_level_split_manifest.csv")
audit_path = os.path.join(folder, "patient_level_split_audit.txt")

targets = ["aki_within_6h", "aki_within_12h", "aki_within_24h"]
split_names = ("train", "validation", "test")
seed = 20260927
chunk_size = 10000


# -----------------------------------------------------------------------------
# CHECK INPUT
# -----------------------------------------------------------------------------

if not os.path.exists(source_path):
    raise FileNotFoundError(f"Missing input file: {source_path}")

header = pd.read_csv(source_path, nrows=0).columns.tolist()
required = ["subject_id", *targets]
missing_columns = [name for name in required if name not in header]
if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")


# -----------------------------------------------------------------------------
# BUILD ONE OUTCOME/OBSERVABILITY STRATUM PER PATIENT
# P = at least one positive landmark
# N = observed landmark(s), but no positive landmark
# M = no assessable landmark for that horizon
# -----------------------------------------------------------------------------

print("Reading patient-level outcome patterns in low-memory chunks...")

patient_states = {}
rows_scanned = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        source_path,
        usecols=required,
        chunksize=chunk_size,
        low_memory=False,
    ),
    start=1,
):
    chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce")
    chunk = chunk.dropna(subset=["subject_id"])
    chunk["subject_id"] = chunk["subject_id"].astype("int64")
    rows_scanned += len(chunk)

    for target in targets:
        chunk[target] = pd.to_numeric(chunk[target], errors="coerce")

    grouped = chunk.groupby("subject_id", sort=False)[targets].agg(["count", "max"])

    for subject_id, values in grouped.iterrows():
        state = patient_states.setdefault(
            int(subject_id),
            {target: {"observed": False, "positive": False} for target in targets},
        )
        for target in targets:
            if int(values[(target, "count")]) > 0:
                state[target]["observed"] = True
                if float(values[(target, "max")]) == 1.0:
                    state[target]["positive"] = True

    if chunk_number % 10 == 0:
        print("Pattern chunks:", chunk_number, "Rows:", rows_scanned)


def state_code(state):
    codes = []
    for target in targets:
        if state[target]["positive"]:
            codes.append("P")
        elif state[target]["observed"]:
            codes.append("N")
        else:
            codes.append("M")
    return "".join(codes)


strata = defaultdict(list)
for subject_id, state in patient_states.items():
    strata[state_code(state)].append(subject_id)


# -----------------------------------------------------------------------------
# STRATIFIED 70/15/15 PATIENT SPLIT
# -----------------------------------------------------------------------------

rng = np.random.default_rng(seed)
subject_to_split = {}
stratum_split_counts = {}

for stratum in sorted(strata):
    subject_ids = np.asarray(sorted(strata[stratum]), dtype="int64")
    rng.shuffle(subject_ids)
    number = len(subject_ids)

    train_end = int(round(number * 0.70))
    validation_end = train_end + int(round(number * 0.15))

    # These safeguards matter only for very small strata.
    train_end = min(train_end, number)
    validation_end = min(validation_end, number)

    assignments = {
        "train": subject_ids[:train_end],
        "validation": subject_ids[train_end:validation_end],
        "test": subject_ids[validation_end:],
    }

    stratum_split_counts[stratum] = {
        name: len(assignments[name]) for name in split_names
    }
    for split_name, ids in assignments.items():
        for subject_id in ids:
            subject_to_split[int(subject_id)] = split_name

if len(subject_to_split) != len(patient_states):
    raise RuntimeError("Not every patient received exactly one split")

manifest = pd.DataFrame(
    {
        "subject_id": sorted(subject_to_split),
        "split": [subject_to_split[sid] for sid in sorted(subject_to_split)],
        "outcome_stratum": [
            state_code(patient_states[sid]) for sid in sorted(subject_to_split)
        ],
    }
)

if manifest["subject_id"].duplicated().any():
    raise RuntimeError("Duplicate patients found in split manifest")

manifest.to_csv(manifest_path, index=False)


# -----------------------------------------------------------------------------
# VERIFY ROW-LEVEL DISTRIBUTIONS WITHOUT COPYING THE 265-COLUMN TABLE
# -----------------------------------------------------------------------------

print("Verifying row-level split and target distributions...")

row_counts = Counter()
target_counts = {
    split_name: {
        target: Counter({"positive": 0, "negative": 0, "missing": 0, "invalid": 0})
        for target in targets
    }
    for split_name in split_names
}
unassigned_rows = 0

for chunk_number, chunk in enumerate(
    pd.read_csv(
        source_path,
        usecols=required,
        chunksize=chunk_size,
        low_memory=False,
    ),
    start=1,
):
    chunk["subject_id"] = pd.to_numeric(chunk["subject_id"], errors="coerce")
    chunk["split"] = chunk["subject_id"].map(subject_to_split)
    unassigned_rows += int(chunk["split"].isna().sum())

    for split_name in split_names:
        subset = chunk.loc[chunk["split"] == split_name]
        row_counts[split_name] += len(subset)
        for target in targets:
            values = pd.to_numeric(subset[target], errors="coerce")
            target_counts[split_name][target]["positive"] += int((values == 1).sum())
            target_counts[split_name][target]["negative"] += int((values == 0).sum())
            target_counts[split_name][target]["missing"] += int(values.isna().sum())
            target_counts[split_name][target]["invalid"] += int(
                (values.notna() & ~values.isin([0, 1])).sum()
            )

    if chunk_number % 10 == 0:
        print("Verification chunks:", chunk_number)

patient_counts = manifest["split"].value_counts().to_dict()
overlap_count = len(manifest) - manifest["subject_id"].nunique()
invalid_target_values = sum(
    target_counts[split_name][target]["invalid"]
    for split_name in split_names
    for target in targets
)
validation_pass = (
    overlap_count == 0
    and unassigned_rows == 0
    and invalid_target_values == 0
    and sum(row_counts.values()) == rows_scanned
)


# -----------------------------------------------------------------------------
# AUDIT REPORT
# -----------------------------------------------------------------------------

lines = [
    "PATIENT-LEVEL TRAIN/VALIDATION/TEST SPLIT AUDIT",
    "",
    f"Source rows: {rows_scanned}",
    f"Unique patients: {len(patient_states)}",
    f"Random seed: {seed}",
    "Requested allocation: train 70%, validation 15%, test 15%",
    "Split unit: subject_id (all admissions, stays, and landmarks remain together)",
    "Stratification: joint patient-level P/N/M pattern across 6h, 12h, and 24h outcomes",
    "",
    "P = at least one positive landmark for the patient",
    "N = assessable landmark(s), but no positive landmark for the patient",
    "M = no assessable landmark for the patient at that horizon",
    "",
    "PATIENT AND ROW COUNTS:",
]

for split_name in split_names:
    patient_number = int(patient_counts.get(split_name, 0))
    patient_pct = 100.0 * patient_number / len(patient_states)
    row_number = int(row_counts[split_name])
    row_pct = 100.0 * row_number / rows_scanned
    lines.append(
        f"{split_name}: patients={patient_number} ({patient_pct:.2f}%), "
        f"rows={row_number} ({row_pct:.2f}%)"
    )

lines.extend(["", "ROW-LEVEL TARGET DISTRIBUTIONS:"])
for split_name in split_names:
    lines.append(f"{split_name.upper()}:")
    for target in targets:
        counts = target_counts[split_name][target]
        assessable = counts["positive"] + counts["negative"]
        prevalence = (
            100.0 * counts["positive"] / assessable if assessable else float("nan")
        )
        lines.append(
            f"  {target}: positive={counts['positive']}, negative={counts['negative']}, "
            f"missing={counts['missing']}, invalid={counts['invalid']}, "
            f"prevalence_among_assessable={prevalence:.2f}%"
        )

lines.extend(["", "PATIENT STRATA BY SPLIT:"])
for stratum in sorted(stratum_split_counts):
    counts = stratum_split_counts[stratum]
    lines.append(
        f"{stratum}: train={counts['train']}, "
        f"validation={counts['validation']}, test={counts['test']}"
    )

lines.extend(
    [
        "",
        "LEAKAGE AND STRUCTURAL CHECKS:",
        f"Patients appearing in more than one split: {overlap_count}",
        f"Rows without a split assignment: {unassigned_rows}",
        f"Invalid nonmissing target values: {invalid_target_values}",
        f"Rows accounted for exactly once: {sum(row_counts.values()) == rows_scanned}",
        f"Validation result: {'PASS' if validation_pass else 'FAIL'}",
        "",
        "IMPORTANT:",
        "Use only rows with a nonmissing value for the target being modeled.",
        "Never convert a missing outcome label to zero.",
        "Fit preprocessing, imputation, feature selection, and model parameters on train only.",
        "Use validation for model selection and test only once for final performance.",
        "",
        f"Split manifest: {manifest_path}",
        f"Audit report: {audit_path}",
    ]
)

with open(audit_path, "w", encoding="utf-8", errors="replace") as audit_file:
    audit_file.write("\n".join(lines))

print()
print("PATIENT-LEVEL SPLIT COMPLETE")
print("\n".join(lines))

if not validation_pass:
    raise RuntimeError("Patient-level split validation failed; inspect the audit report")
