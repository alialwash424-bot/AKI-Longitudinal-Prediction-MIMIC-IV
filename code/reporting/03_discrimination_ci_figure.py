import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 61C — FINAL DISCRIMINATION + 95% CI FIGURE
# ============================================================

folder = os.path.expanduser("~/Documents/AKI_Research")

print("=" * 80)
print("STEP 61C — FINAL HELD-OUT TEST DISCRIMINATION WITH 95% CI")
print("=" * 80)
print()
print("READ-ONLY FIGURE GENERATION")
print("NO MODEL WILL BE RETRAINED.")
print("NO CALIBRATION WILL BE FIT.")
print("NO THRESHOLD WILL BE SELECTED OR CHANGED.")
print("NO TEST PREDICTION WILL BE MODIFIED.")
print()


# ============================================================
# EXPECTED FINAL TEST RESULTS
# ============================================================

# Point estimates are the already-finalized held-out-test results.
# These are used only as safety checks.

expected = {
    "6h": {
        "auroc": 0.724717,
        "auprc": 0.198494,
    },
    "12h": {
        "auroc": 0.733356,
        "auprc": 0.328190,
    },
    "24h": {
        "auroc": 0.731931,
        "auprc": 0.503620,
    },
}


# ============================================================
# FIND STEP 57B BOOTSTRAP CSV
# ============================================================

print("=" * 80)
print("LOCATING STEP 57B BOOTSTRAP RESULTS")
print("=" * 80)

patterns = [
    os.path.join(folder, "step57B*.csv"),
    os.path.join(folder, "*57B*bootstrap*.csv"),
    os.path.join(folder, "*patient*bootstrap*.csv"),
]

candidates = []

for pattern in patterns:
    candidates.extend(glob.glob(pattern))

# remove duplicates
candidates = sorted(set(candidates))

print()
print("Candidate CSV files found:")

for path in candidates:
    print(" -", path)

if len(candidates) == 0:
    raise FileNotFoundError(
        "No Step 57B/bootstrap CSV was found in "
        "~/Documents/AKI_Research"
    )


# ============================================================
# IDENTIFY THE CORRECT BOOTSTRAP FILE
# ============================================================

bootstrap_path = None
bootstrap_df = None

for path in candidates:

    try:
        temp = pd.read_csv(path)
    except Exception:
        continue

    lower_columns = {
        str(c).lower()
        for c in temp.columns
    }

    # We need a file containing bootstrap results,
    # horizon information, and AUROC/AUPRC information.

    has_horizon = any(
        "horizon" in c
        for c in lower_columns
    )

    has_auroc = any(
        ("auroc" in c) or ("roc_auc" in c)
        for c in lower_columns
    )

    has_auprc = any(
        ("auprc" in c)
        or ("average_precision" in c)
        or ("pr_auc" in c)
        for c in lower_columns
    )

    if has_horizon and has_auroc and has_auprc:
        bootstrap_path = path
        bootstrap_df = temp
        break


if bootstrap_path is None:
    raise RuntimeError(
        "Bootstrap CSV candidates were found, but none "
        "contained recognizable horizon, AUROC, and AUPRC columns."
    )


print()
print("Selected bootstrap file:")
print(bootstrap_path)

print()
print("Rows:", len(bootstrap_df))
print("Columns:")
print(list(bootstrap_df.columns))


# ============================================================
# COLUMN IDENTIFICATION
# ============================================================

def find_column(columns, possibilities):

    lower_map = {
        str(c).lower(): c
        for c in columns
    }

    # exact first
    for name in possibilities:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # partial second
    for col in columns:

        lc = str(col).lower()

        for name in possibilities:

            if name.lower() in lc:
                return col

    return None


horizon_col = find_column(
    bootstrap_df.columns,
    [
        "horizon",
        "prediction_horizon",
    ],
)

auroc_col = find_column(
    bootstrap_df.columns,
    [
        "auroc",
        "roc_auc",
    ],
)

auprc_col = find_column(
    bootstrap_df.columns,
    [
        "auprc",
        "average_precision",
        "pr_auc",
    ],
)


if horizon_col is None:
    raise RuntimeError(
        "Could not identify horizon column."
    )

if auroc_col is None:
    raise RuntimeError(
        "Could not identify AUROC column."
    )

if auprc_col is None:
    raise RuntimeError(
        "Could not identify AUPRC column."
    )


print()
print("Detected columns:")
print("Horizon:", horizon_col)
print("AUROC:", auroc_col)
print("AUPRC:", auprc_col)


# ============================================================
# NORMALIZE HORIZON
# ============================================================

def normalize_horizon(value):

    text = str(value).strip().lower()

    if text in {"6", "6h", "6.0"}:
        return "6h"

    if text in {"12", "12h", "12.0"}:
        return "12h"

    if text in {"24", "24h", "24.0"}:
        return "24h"

    if "6h" in text:
        return "6h"

    if "12h" in text:
        return "12h"

    if "24h" in text:
        return "24h"

    return None


bootstrap_df["_normalized_horizon"] = (
    bootstrap_df[horizon_col]
    .map(normalize_horizon)
)


# ============================================================
# NUMERIC CLEANING
# ============================================================

bootstrap_df["_auroc"] = pd.to_numeric(
    bootstrap_df[auroc_col],
    errors="coerce",
)

bootstrap_df["_auprc"] = pd.to_numeric(
    bootstrap_df[auprc_col],
    errors="coerce",
)

usable = bootstrap_df[
    bootstrap_df["_normalized_horizon"].isin(
        ["6h", "12h", "24h"]
    )
    &
    bootstrap_df["_auroc"].notna()
    &
    bootstrap_df["_auprc"].notna()
].copy()


print()
print("Usable bootstrap rows:", len(usable))


# ============================================================
# REQUIRE BOOTSTRAP REPLICATES
# ============================================================

counts = (
    usable
    .groupby("_normalized_horizon")
    .size()
)

print()
print("Bootstrap rows per horizon:")
print(counts)


for horizon in ["6h", "12h", "24h"]:

    if horizon not in counts.index:
        raise RuntimeError(
            f"No bootstrap results found for {horizon}."
        )

    if counts.loc[horizon] < 100:
        raise RuntimeError(
            f"Too few bootstrap replicates for {horizon}: "
            f"{counts.loc[horizon]}"
        )


# ============================================================
# CALCULATE PATIENT-BOOTSTRAP 95% CI
# ============================================================

results = []

for horizon in ["6h", "12h", "24h"]:

    d = usable[
        usable["_normalized_horizon"] == horizon
    ]

    auroc_values = (
        d["_auroc"]
        .dropna()
        .to_numpy()
    )

    auprc_values = (
        d["_auprc"]
        .dropna()
        .to_numpy()
    )

    auroc_low = float(
        np.percentile(
            auroc_values,
            2.5,
        )
    )

    auroc_high = float(
        np.percentile(
            auroc_values,
            97.5,
        )
    )

    auprc_low = float(
        np.percentile(
            auprc_values,
            2.5,
        )
    )

    auprc_high = float(
        np.percentile(
            auprc_values,
            97.5,
        )
    )

    results.append({
        "horizon": horizon,
        "auroc": expected[horizon]["auroc"],
        "auroc_ci_low": auroc_low,
        "auroc_ci_high": auroc_high,
        "auprc": expected[horizon]["auprc"],
        "auprc_ci_low": auprc_low,
        "auprc_ci_high": auprc_high,
        "bootstrap_replicates": len(d),
    })


summary = pd.DataFrame(results)


# ============================================================
# SAFETY CHECKS
# ============================================================

print()
print("=" * 80)
print("FINAL DISCRIMINATION CI SUMMARY")
print("=" * 80)

all_pass = True

for _, row in summary.iterrows():

    horizon = row["horizon"]

    auroc_inside = (
        row["auroc_ci_low"]
        <= row["auroc"]
        <= row["auroc_ci_high"]
    )

    auprc_inside = (
        row["auprc_ci_low"]
        <= row["auprc"]
        <= row["auprc_ci_high"]
    )

    valid_bounds = (
        0 <= row["auroc_ci_low"] <= 1
        and
        0 <= row["auroc_ci_high"] <= 1
        and
        0 <= row["auprc_ci_low"] <= 1
        and
        0 <= row["auprc_ci_high"] <= 1
    )

    horizon_pass = (
        auroc_inside
        and auprc_inside
        and valid_bounds
    )

    all_pass = (
        all_pass
        and horizon_pass
    )

    print()
    print(horizon.upper())

    print(
        "AUROC:",
        f"{row['auroc']:.6f}",
        f"(95% CI "
        f"{row['auroc_ci_low']:.6f} "
        f"to "
        f"{row['auroc_ci_high']:.6f})",
    )

    print(
        "AUPRC:",
        f"{row['auprc']:.6f}",
        f"(95% CI "
        f"{row['auprc_ci_low']:.6f} "
        f"to "
        f"{row['auprc_ci_high']:.6f})",
    )

    print(
        "Bootstrap replicates:",
        int(row["bootstrap_replicates"]),
    )

    print(
        "Point estimates inside CI:",
        "PASS" if horizon_pass else "FAIL",
    )


if not all_pass:

    raise RuntimeError(
        "STEP 61C safety verification failed. "
        "No figure was generated."
    )


# ============================================================
# OUTPUT PATHS
# ============================================================

summary_csv = os.path.join(
    folder,
    "step61C_final_discrimination_CI.csv",
)

png_path = os.path.join(
    folder,
    "step61C_final_discrimination_CI.png",
)

pdf_path = os.path.join(
    folder,
    "step61C_final_discrimination_CI.pdf",
)

report_path = os.path.join(
    folder,
    "step61C_final_discrimination_CI_report.txt",
)


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

summary.to_csv(
    summary_csv,
    index=False,
)


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(12, 5.5),
)

horizons = ["6h", "12h", "24h"]

y_positions = np.arange(
    len(horizons)
)


# ------------------------------------------------------------
# AUROC PANEL
# ------------------------------------------------------------

ax = axes[0]

values = np.array([
    summary.loc[
        summary["horizon"] == h,
        "auroc",
    ].iloc[0]
    for h in horizons
])

low = np.array([
    summary.loc[
        summary["horizon"] == h,
        "auroc_ci_low",
    ].iloc[0]
    for h in horizons
])

high = np.array([
    summary.loc[
        summary["horizon"] == h,
        "auroc_ci_high",
    ].iloc[0]
    for h in horizons
])

errors = np.vstack([
    values - low,
    high - values,
])

ax.errorbar(
    values,
    y_positions,
    xerr=errors,
    fmt="o",
    capsize=5,
    markersize=8,
    linewidth=2,
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    ["6-hour", "12-hour", "24-hour"]
)

ax.invert_yaxis()

ax.set_xlabel(
    "AUROC (95% CI)"
)

ax.set_title(
    "Area Under the ROC Curve"
)

ax.set_xlim(
    0.65,
    0.80,
)

ax.grid(
    True,
    axis="x",
    alpha=0.3,
)

for i, value in enumerate(values):

    ax.text(
        value + 0.004,
        i,
        f"{value:.3f}",
        va="center",
    )


# ------------------------------------------------------------
# AUPRC PANEL
# ------------------------------------------------------------

ax = axes[1]

values = np.array([
    summary.loc[
        summary["horizon"] == h,
        "auprc",
    ].iloc[0]
    for h in horizons
])

low = np.array([
    summary.loc[
        summary["horizon"] == h,
        "auprc_ci_low",
    ].iloc[0]
    for h in horizons
])

high = np.array([
    summary.loc[
        summary["horizon"] == h,
        "auprc_ci_high",
    ].iloc[0]
    for h in horizons
])

errors = np.vstack([
    values - low,
    high - values,
])

ax.errorbar(
    values,
    y_positions,
    xerr=errors,
    fmt="o",
    capsize=5,
    markersize=8,
    linewidth=2,
)

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    ["6-hour", "12-hour", "24-hour"]
)

ax.invert_yaxis()

ax.set_xlabel(
    "AUPRC (95% CI)"
)

ax.set_title(
    "Area Under the Precision–Recall Curve"
)

ax.set_xlim(
    0.10,
    0.60,
)

ax.grid(
    True,
    axis="x",
    alpha=0.3,
)

for i, value in enumerate(values):

    ax.text(
        value + 0.012,
        i,
        f"{value:.3f}",
        va="center",
    )


# ------------------------------------------------------------
# FIGURE TITLE
# ------------------------------------------------------------

fig.suptitle(
    "Final Held-Out Test Discrimination "
    "with Patient-Level Bootstrap 95% Confidence Intervals",
    fontsize=14,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.93]
)


# ============================================================
# SAVE FIGURE
# ============================================================

plt.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# VERIFY FIGURES
# ============================================================

for path in [
    png_path,
    pdf_path,
]:

    if not os.path.exists(path):
        raise RuntimeError(
            f"Output figure missing: {path}"
        )

    if os.path.getsize(path) <= 0:
        raise RuntimeError(
            f"Output figure is empty: {path}"
        )


# ============================================================
# WRITE REPORT
# ============================================================

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "STEP 61C — FINAL HELD-OUT TEST "
        "DISCRIMINATION WITH 95% CI\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        f"Bootstrap source: {bootstrap_path}\n\n"
    )

    f.write(
        "Bootstrap confidence intervals were derived "
        "from the existing Step 57B bootstrap results.\n"
    )

    f.write(
        "Point estimates are the frozen final "
        "held-out-test AUROC/AUPRC values.\n\n"
    )

    for _, row in summary.iterrows():

        f.write(
            f"{row['horizon'].upper()}\n"
        )

        f.write(
            f"AUROC: {row['auroc']:.6f} "
            f"(95% CI "
            f"{row['auroc_ci_low']:.6f} "
            f"to "
            f"{row['auroc_ci_high']:.6f})\n"
        )

        f.write(
            f"AUPRC: {row['auprc']:.6f} "
            f"(95% CI "
            f"{row['auprc_ci_low']:.6f} "
            f"to "
            f"{row['auprc_ci_high']:.6f})\n"
        )

        f.write(
            f"Bootstrap replicates: "
            f"{int(row['bootstrap_replicates'])}\n\n"
        )

    f.write(
        "NO MODEL WAS RETRAINED.\n"
    )

    f.write(
        "NO CALIBRATION MODEL WAS FIT.\n"
    )

    f.write(
        "NO THRESHOLD WAS SELECTED OR CHANGED.\n"
    )

    f.write(
        "NO TEST PREDICTION WAS MODIFIED.\n"
    )


# ============================================================
# FINAL CONSOLE
# ============================================================

print()
print("=" * 80)
print("STEP 61C COMPLETE")
print("=" * 80)

print()
print("Summary CSV:")
print(summary_csv)

print()
print("Figure PNG:")
print(png_path)

print()
print("Figure PDF:")
print(pdf_path)

print()
print("Report:")
print(report_path)

print()
print("=" * 80)
print("FINAL STEP 61C VERDICT: PASS")
print("=" * 80)

print()
print("NO MODEL WAS RETRAINED.")
print("NO CALIBRATION MODEL WAS FIT.")
print("NO THRESHOLD WAS SELECTED OR CHANGED.")
print("NO TEST PREDICTION WAS MODIFIED.")
print("NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED.")
