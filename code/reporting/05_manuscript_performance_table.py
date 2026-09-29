import os
import pandas as pd
import numpy as np

# ============================================================
# STEP 61E
# FINAL MANUSCRIPT-READY MODEL PERFORMANCE TABLE
# ============================================================

folder = os.path.expanduser("~/Documents/AKI_Research")

bootstrap_path = os.path.join(
    folder,
    "step57B_patient_bootstrap_metrics.csv"
)

ablation_path = os.path.join(
    folder,
    "step61D_validation_ablation_summary.csv"
)

output_csv = os.path.join(
    folder,
    "step61E_manuscript_performance_table.csv"
)

output_txt = os.path.join(
    folder,
    "step61E_manuscript_performance_report.txt"
)

# Frozen final held-out test point estimates
# These are the verified Step 60 / Step 61A results.

test_results = {
    "6h": {
        "n": 64570,
        "patients": 8504,
        "positives": 5544,
        "negatives": 59026,
        "prevalence": 0.085860,
        "auroc": 0.724717,
        "auprc": 0.198494,
    },
    "12h": {
        "n": 58924,
        "patients": 8231,
        "positives": 9251,
        "negatives": 49673,
        "prevalence": 0.156999,
        "auroc": 0.733356,
        "auprc": 0.328190,
    },
    "24h": {
        "n": 49367,
        "patients": 7178,
        "positives": 13654,
        "negatives": 35713,
        "prevalence": 0.276582,
        "auroc": 0.731931,
        "auprc": 0.503620,
    },
}

expected_bootstrap_replicates = 1000


# ============================================================
# SAFETY / INPUT CHECKS
# ============================================================

print("=" * 80)
print("STEP 61E — FINAL MANUSCRIPT PERFORMANCE TABLE")
print("=" * 80)

print("")
print("REPORTING-ONLY STEP")
print("NO MODEL WILL BE RETRAINED.")
print("NO CALIBRATION MODEL WILL BE FIT.")
print("NO THRESHOLD WILL BE SELECTED OR CHANGED.")
print("NO TEST PREDICTION WILL BE MODIFIED.")
print("")

for path in [bootstrap_path, ablation_path]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

boot = pd.read_csv(bootstrap_path)
abl = pd.read_csv(ablation_path)

print("Bootstrap file:")
print(bootstrap_path)
print("Rows:", len(boot))
print("Columns:", list(boot.columns))

print("")
print("Ablation file:")
print(ablation_path)
print("Rows:", len(abl))
print("Columns:", list(abl.columns))


# ============================================================
# NORMALIZE BOOTSTRAP HORIZON
# ============================================================

if "horizon" not in boot.columns:
    raise RuntimeError(
        "Bootstrap file does not contain required column: horizon"
    )

if "AUROC" not in boot.columns or "AUPRC" not in boot.columns:
    raise RuntimeError(
        "Bootstrap file must contain AUROC and AUPRC."
    )

boot["normalized_horizon"] = (
    boot["horizon"]
    .astype(str)
    .str.strip()
    .str.lower()
)

mapping = {
    "6": "6h",
    "6h": "6h",
    "6.0": "6h",
    "12": "12h",
    "12h": "12h",
    "12.0": "12h",
    "24": "24h",
    "24h": "24h",
    "24.0": "24h",
}

boot["normalized_horizon"] = (
    boot["normalized_horizon"]
    .map(mapping)
)

if boot["normalized_horizon"].isna().any():
    bad = boot.loc[
        boot["normalized_horizon"].isna(),
        "horizon"
    ].unique()

    raise RuntimeError(
        f"Unrecognized bootstrap horizons: {bad}"
    )


# ============================================================
# CALCULATE PATIENT-BOOTSTRAP 95% CI
# ============================================================

ci_results = {}

print("")
print("=" * 80)
print("FINAL HELD-OUT TEST DISCRIMINATION WITH 95% CI")
print("=" * 80)

for horizon in ["6h", "12h", "24h"]:

    sub = boot.loc[
        boot["normalized_horizon"] == horizon
    ].copy()

    if len(sub) != expected_bootstrap_replicates:
        raise RuntimeError(
            f"{horizon}: expected "
            f"{expected_bootstrap_replicates} bootstrap replicates, "
            f"found {len(sub)}."
        )

    auroc_values = pd.to_numeric(
        sub["AUROC"],
        errors="coerce"
    ).dropna()

    auprc_values = pd.to_numeric(
        sub["AUPRC"],
        errors="coerce"
    ).dropna()

    if len(auroc_values) != expected_bootstrap_replicates:
        raise RuntimeError(
            f"{horizon}: incomplete AUROC bootstrap values."
        )

    if len(auprc_values) != expected_bootstrap_replicates:
        raise RuntimeError(
            f"{horizon}: incomplete AUPRC bootstrap values."
        )

    auroc_low, auroc_high = np.percentile(
        auroc_values,
        [2.5, 97.5]
    )

    auprc_low, auprc_high = np.percentile(
        auprc_values,
        [2.5, 97.5]
    )

    ci_results[horizon] = {
        "auroc_low": float(auroc_low),
        "auroc_high": float(auroc_high),
        "auprc_low": float(auprc_low),
        "auprc_high": float(auprc_high),
    }

    point = test_results[horizon]

    auroc_inside = (
        auroc_low
        <= point["auroc"]
        <= auroc_high
    )

    auprc_inside = (
        auprc_low
        <= point["auprc"]
        <= auprc_high
    )

    print("")
    print(horizon)
    print(
        f"AUROC: {point['auroc']:.6f} "
        f"(95% CI {auroc_low:.6f} to {auroc_high:.6f})"
    )
    print(
        f"AUPRC: {point['auprc']:.6f} "
        f"(95% CI {auprc_low:.6f} to {auprc_high:.6f})"
    )
    print(
        "Point estimates inside CI:",
        "PASS"
        if auroc_inside and auprc_inside
        else "FAIL"
    )

    if not (auroc_inside and auprc_inside):
        raise RuntimeError(
            f"{horizon}: point estimate outside bootstrap CI."
        )


# ============================================================
# BUILD PRIMARY MANUSCRIPT TABLE
# ============================================================

rows = []

for horizon in ["6h", "12h", "24h"]:

    r = test_results[horizon]
    ci = ci_results[horizon]

    rows.append({
        "Prediction horizon": horizon,
        "Test landmarks": r["n"],
        "Unique patients": r["patients"],
        "AKI events": r["positives"],
        "Non-events": r["negatives"],
        "AKI prevalence (%)": r["prevalence"] * 100.0,

        "AUROC": r["auroc"],
        "AUROC 95% CI lower": ci["auroc_low"],
        "AUROC 95% CI upper": ci["auroc_high"],

        "AUPRC": r["auprc"],
        "AUPRC 95% CI lower": ci["auprc_low"],
        "AUPRC 95% CI upper": ci["auprc_high"],
    })

table = pd.DataFrame(rows)

table.to_csv(
    output_csv,
    index=False
)


# ============================================================
# FORMAT MANUSCRIPT TEXT
# ============================================================

report_lines = []

report_lines.append(
    "STEP 61E — FINAL MANUSCRIPT PERFORMANCE REPORT"
)
report_lines.append("=" * 80)
report_lines.append("")

report_lines.append(
    "FINAL HELD-OUT TEST DISCRIMINATION"
)
report_lines.append("-" * 80)

for _, row in table.iterrows():

    report_lines.append("")
    report_lines.append(
        f"{row['Prediction horizon']}"
    )

    report_lines.append(
        f"Landmarks: {int(row['Test landmarks']):,}"
    )

    report_lines.append(
        f"Unique patients: {int(row['Unique patients']):,}"
    )

    report_lines.append(
        f"AKI events: {int(row['AKI events']):,}"
    )

    report_lines.append(
        f"AKI prevalence: "
        f"{row['AKI prevalence (%)']:.2f}%"
    )

    report_lines.append(
        f"AUROC: {row['AUROC']:.3f} "
        f"(95% CI "
        f"{row['AUROC 95% CI lower']:.3f}–"
        f"{row['AUROC 95% CI upper']:.3f})"
    )

    report_lines.append(
        f"AUPRC: {row['AUPRC']:.3f} "
        f"(95% CI "
        f"{row['AUPRC 95% CI lower']:.3f}–"
        f"{row['AUPRC 95% CI upper']:.3f})"
    )


# ============================================================
# ADD VALIDATION ABLATION RESULTS
# ============================================================

report_lines.append("")
report_lines.append("")
report_lines.append(
    "VALIDATION ABLATION ANALYSIS"
)
report_lines.append("-" * 80)

# We do not assume exact column capitalization.
lower_lookup = {
    str(c).strip().lower(): c
    for c in abl.columns
}

required_semantic = [
    "horizon",
    "model",
    "auroc",
    "auprc",
]

missing = [
    x for x in required_semantic
    if x not in lower_lookup
]

if missing:
    raise RuntimeError(
        "Could not identify required ablation columns: "
        + ", ".join(missing)
    )

hcol = lower_lookup["horizon"]
mcol = lower_lookup["model"]
acol = lower_lookup["auroc"]
pcol = lower_lookup["auprc"]

for horizon in ["6h", "12h", "24h"]:

    report_lines.append("")
    report_lines.append(horizon)

    normalized = (
        abl[hcol]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    candidates = {
        "6h": ["6h", "6", "6.0"],
        "12h": ["12h", "12", "12.0"],
        "24h": ["24h", "24", "24.0"],
    }[horizon]

    sub = abl.loc[
        normalized.isin(candidates)
    ].copy()

    if sub.empty:
        raise RuntimeError(
            f"No ablation rows found for {horizon}."
        )

    for _, row in sub.iterrows():

        model_name = str(row[mcol])

        auroc = float(row[acol])
        auprc = float(row[pcol])

        report_lines.append(
            f"{model_name}: "
            f"AUROC={auroc:.6f}; "
            f"AUPRC={auprc:.6f}"
        )


# ============================================================
# MANUSCRIPT-READY RESULTS PARAGRAPH
# ============================================================

report_lines.append("")
report_lines.append("")
report_lines.append(
    "MANUSCRIPT-READY RESULTS TEXT"
)
report_lines.append("-" * 80)
report_lines.append("")

r6 = table.iloc[0]
r12 = table.iloc[1]
r24 = table.iloc[2]

paragraph = (
    "In the held-out test set, the prespecified full model "
    f"achieved an AUROC of {r6['AUROC']:.3f} "
    f"(95% CI {r6['AUROC 95% CI lower']:.3f}–"
    f"{r6['AUROC 95% CI upper']:.3f}) and an AUPRC of "
    f"{r6['AUPRC']:.3f} "
    f"(95% CI {r6['AUPRC 95% CI lower']:.3f}–"
    f"{r6['AUPRC 95% CI upper']:.3f}) "
    "for prediction within 6 hours. "
    f"For the 12-hour horizon, AUROC was {r12['AUROC']:.3f} "
    f"(95% CI {r12['AUROC 95% CI lower']:.3f}–"
    f"{r12['AUROC 95% CI upper']:.3f}) and AUPRC was "
    f"{r12['AUPRC']:.3f} "
    f"(95% CI {r12['AUPRC 95% CI lower']:.3f}–"
    f"{r12['AUPRC 95% CI upper']:.3f}). "
    f"For the 24-hour horizon, AUROC was {r24['AUROC']:.3f} "
    f"(95% CI {r24['AUROC 95% CI lower']:.3f}–"
    f"{r24['AUROC 95% CI upper']:.3f}) and AUPRC was "
    f"{r24['AUPRC']:.3f} "
    f"(95% CI {r24['AUPRC 95% CI lower']:.3f}–"
    f"{r24['AUPRC 95% CI upper']:.3f})."
)

report_lines.append(paragraph)

report_lines.append("")
report_lines.append(
    "Confidence intervals were obtained from the completed "
    "patient-level bootstrap analysis."
)

report_lines.append("")
report_lines.append(
    "Ablation comparisons were performed on the validation set "
    "and are reported descriptively; they are not interpreted "
    "as formal statistical tests of superiority."
)


# ============================================================
# WRITE REPORT
# ============================================================

with open(
    output_txt,
    "w",
    encoding="utf-8"
) as f:
    f.write("\n".join(report_lines))


# ============================================================
# FINAL CONSOLE SUMMARY
# ============================================================

print("")
print("=" * 80)
print("FINAL MANUSCRIPT TABLE")
print("=" * 80)

for _, row in table.iterrows():

    print("")
    print(row["Prediction horizon"])

    print(
        f"N={int(row['Test landmarks']):,}; "
        f"patients={int(row['Unique patients']):,}; "
        f"events={int(row['AKI events']):,}; "
        f"prevalence={row['AKI prevalence (%)']:.2f}%"
    )

    print(
        f"AUROC "
        f"{row['AUROC']:.3f} "
        f"(95% CI "
        f"{row['AUROC 95% CI lower']:.3f}–"
        f"{row['AUROC 95% CI upper']:.3f})"
    )

    print(
        f"AUPRC "
        f"{row['AUPRC']:.3f} "
        f"(95% CI "
        f"{row['AUPRC 95% CI lower']:.3f}–"
        f"{row['AUPRC 95% CI upper']:.3f})"
    )


print("")
print("=" * 80)
print("STEP 61E COMPLETE")
print("=" * 80)

print("")
print("Manuscript table CSV:")
print(output_csv)

print("")
print("Manuscript report:")
print(output_txt)

print("")
print("=" * 80)
print("FINAL STEP 61E VERDICT: PASS")
print("=" * 80)

print("")
print("NO MODEL WAS RETRAINED.")
print("NO CALIBRATION MODEL WAS FIT.")
print("NO THRESHOLD WAS SELECTED OR CHANGED.")
print("NO TEST PREDICTION WAS MODIFIED.")
print("NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED.")
