import csv
import os

import pandas as pd


folder = os.path.expanduser("~/Documents/AKI_Research")
source_path = os.path.join(folder, "longitudinal_modeling_table.csv")
predictor_path = os.path.join(folder, "final_model_predictors.txt")
group_path = os.path.join(folder, "final_model_predictor_groups.csv")
exclusion_path = os.path.join(folder, "final_model_exclusions.txt")
audit_path = os.path.join(folder, "final_model_predictor_audit.txt")

if not os.path.exists(source_path):
    raise FileNotFoundError(f"Missing input file: {source_path}")

header = pd.read_csv(source_path, nrows=0).columns.tolist()
if len(header) != len(set(header)):
    raise ValueError("Duplicate columns exist in the modeling table")

vital_prefixes = (
    "hr_", "sbp_", "dbp_", "map_", "rr_", "spo2_", "temp_c_", "glucose_"
)
exposure_prefixes = (
    "fluid_", "vasoactive_", "norepinephrine_", "phenylephrine_",
    "epinephrine_", "dopamine_", "dobutamine_", "milrinone_", "vasopressin_",
)

vital_predictors = [name for name in header if name.startswith(vital_prefixes)]
exposure_predictors = [name for name in header if name.startswith(exposure_prefixes)]
context_predictors = [
    name for name in ("hours_since_icu_admission",) if name in header
]
final_predictors = vital_predictors + exposure_predictors + context_predictors

if "hours_since_icu_admission" not in header:
    raise ValueError("Required safe timing predictor hours_since_icu_admission is missing")
if len(final_predictors) != len(set(final_predictors)):
    raise ValueError("Predictor groups overlap")

explicit_reasons = {
    "subject_id": "patient identifier; split/grouping only",
    "hadm_id": "admission identifier; not a clinical predictor",
    "stay_id": "ICU-stay identifier; not a clinical predictor",
    "intime": "absolute admission timestamp; identifier/time-origin information",
    "outtime": "future ICU discharge timestamp; post-landmark leakage",
    "landmark_time": "absolute timestamp; key only",
    "landmark_number": "redundant with hours_since_icu_admission",
    "hours_to_aki": "calculated from future AKI time; direct outcome leakage",
    "analysis_status": "cohort/outcome-derived audit status",
    "definitive_aki_time": "future outcome timing; direct leakage",
    "definitive_aki_onset_hours": "future outcome timing; direct leakage",
    "hours_from_landmark_to_aki": "future outcome timing; direct leakage",
    "definitive_incident_aki": "outcome definition",
    "definitive_aki_stage": "future outcome severity",
    "aki_mechanism": "outcome mechanism derived after assessment",
    "any_kdigo_assessment": "outcome-assessment audit field",
    "analysis_included": "cohort-selection field",
    "aki_within_6h": "prediction target",
    "aki_within_12h": "prediction target",
    "aki_within_24h": "prediction target",
}

excluded = []
for name in header:
    if name not in final_predictors:
        reason = explicit_reasons.get(name)
        if reason is None:
            lower = name.lower()
            if lower.startswith(("definitive_", "future_")):
                reason = "future/outcome-derived field"
            elif lower.endswith(("_outcome", "_target")):
                reason = "outcome/target field"
            else:
                reason = "not in the prespecified trajectory predictor set"
        excluded.append((name, reason))

for forbidden in (
    "hours_to_aki", "outtime", "intime", "landmark_number",
    "definitive_aki_time", "aki_within_6h", "aki_within_12h", "aki_within_24h",
):
    if forbidden in final_predictors:
        raise RuntimeError(f"Leakage-control failure: {forbidden} entered predictors")

with open(predictor_path, "w", encoding="utf-8") as predictor_file:
    predictor_file.write("\n".join(final_predictors))

with open(group_path, "w", encoding="utf-8", newline="") as group_file:
    writer = csv.writer(group_file)
    writer.writerow(["predictor", "group", "planned_missing_value_handling"])
    for name in vital_predictors:
        writer.writerow([name, "vital_trajectory", "training-derived imputation"])
    for name in exposure_predictors:
        writer.writerow([name, "modifiable_exposure_trajectory", "zero when no exposure history"])
    for name in context_predictors:
        writer.writerow([name, "landmark_context", "no future information"])

with open(exclusion_path, "w", encoding="utf-8") as exclusion_file:
    for name, reason in excluded:
        exclusion_file.write(f"{name}\t{reason}\n")

validation_pass = (
    len(vital_predictors) == 120
    and len(exposure_predictors) == 131
    and len(context_predictors) == 1
    and len(final_predictors) == 252
)

lines = [
    "FINAL LEAKAGE-SAFE MODEL PREDICTOR AUDIT",
    "",
    f"Total source columns: {len(header)}",
    f"Vital trajectory predictors retained: {len(vital_predictors)}",
    f"Exposure trajectory predictors retained: {len(exposure_predictors)}",
    f"Safe landmark-context predictors retained: {len(context_predictors)}",
    f"Final predictor count: {len(final_predictors)}",
    f"Excluded columns: {len(excluded)}",
    "",
    "SAFE CONTEXT PREDICTORS:",
    *context_predictors,
    "",
    "CRITICAL EXCLUSIONS:",
]

for name in (
    "intime", "outtime", "landmark_time", "landmark_number", "hours_to_aki",
    "definitive_aki_time", "analysis_status",
    "aki_within_6h", "aki_within_12h", "aki_within_24h",
):
    if name in header:
        lines.append(f"{name}: {explicit_reasons[name]}")

lines.extend(
    [
        "",
        "LEAKAGE CONTROL:",
        "All identifiers, absolute timestamps, discharge information, outcome labels,",
        "future AKI timing, and outcome-derived audit fields are excluded.",
        "hours_since_icu_admission is retained because it is available at landmark_time.",
        "landmark_number is excluded because it duplicates the same six-hour progression.",
        "",
        f"Validation result: {'PASS' if validation_pass else 'FAIL'}",
        "",
        f"Predictors: {predictor_path}",
        f"Predictor groups: {group_path}",
        f"Exclusions: {exclusion_path}",
        f"Audit: {audit_path}",
    ]
)

with open(audit_path, "w", encoding="utf-8") as audit_file:
    audit_file.write("\n".join(lines))

print()
print("FINAL MODEL PREDICTOR MANIFEST COMPLETE")
print("\n".join(lines))

if not validation_pass:
    raise RuntimeError("Unexpected predictor counts; inspect the audit before modeling")
