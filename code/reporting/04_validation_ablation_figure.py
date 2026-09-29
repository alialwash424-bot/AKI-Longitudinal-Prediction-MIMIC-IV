import os
import re
import csv
import math

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# STEP 61D — VALIDATION ABLATION COMPARISON FIGURE
# =============================================================================
#
# PURPOSE
# -------
# Create a manuscript-ready comparison of:
#
#   1. Full model
#   2. Vital + context ablation model
#   3. Exposure + context ablation model
#
# at:
#   - 6 hours
#   - 12 hours
#   - 24 hours
#
# IMPORTANT
# ---------
# VALIDATION RESULTS ONLY.
#
# NO MODEL IS RETRAINED.
# NO TEST DATA ARE READ.
# NO TEST PREDICTIONS ARE READ.
# NO CALIBRATION IS FIT.
# NO THRESHOLD IS SELECTED OR CHANGED.
# NO EXISTING RESEARCH/MODEL FILE IS MODIFIED.
#
# =============================================================================


folder = os.path.expanduser("~/Documents/AKI_Research")


# =============================================================================
# OUTPUTS
# =============================================================================

summary_csv = os.path.join(
    folder,
    "step61D_validation_ablation_summary.csv"
)

figure_png = os.path.join(
    folder,
    "step61D_validation_ablation_comparison.png"
)

figure_pdf = os.path.join(
    folder,
    "step61D_validation_ablation_comparison.pdf"
)

report_path = os.path.join(
    folder,
    "step61D_validation_ablation_report.txt"
)


# =============================================================================
# EXPECTED RESULTS FROM THE COMPLETED VALIDATION RUNS
#
# These are used ONLY as verification anchors.
# The script still searches the completed report files and extracts the
# validation metrics from them.
# =============================================================================

expected = {
    "6h": {
        "full": {
            "auroc": 0.734154,
            "auprc": 0.209191,
        },
        "vital_context": {
            "auroc": 0.732899,
            "auprc": 0.202766,
        },
        "exposure_context": {
            "auroc": 0.708511,
            "auprc": 0.193789,
        },
    },

    "12h": {
        "full": {
            "auroc": 0.745764,
            "auprc": 0.352170,
        },
        "vital_context": {
            "auroc": 0.733233,
            "auprc": 0.329848,
        },
        "exposure_context": {
            "auroc": 0.724171,
            "auprc": 0.329232,
        },
    },

    "24h": {
        "full": {
            "auroc": 0.747252,
            "auprc": 0.523951,
        },
        "vital_context": {
            "auroc": 0.738493,
            "auprc": 0.508415,
        },
        "exposure_context": {
            "auroc": None,
            "auprc": None,
        },
    },
}


# =============================================================================
# CANDIDATE REPORT FILES
# =============================================================================

report_candidates = {
    ("6h", "full"): [
        "baseline_6h_logistic_validation_report.txt",
        "baseline_6h_validation_report.txt",
        "baseline_6h_logistic_report.txt",
    ],

    ("12h", "full"): [
        "baseline_12h_logistic_validation_report.txt",
        "baseline_12h_validation_report.txt",
        "baseline_12h_logistic_report.txt",
    ],

    ("24h", "full"): [
        "baseline_24h_logistic_validation_report.txt",
        "baseline_24h_validation_report.txt",
        "baseline_24h_logistic_report.txt",
    ],

    ("6h", "vital_context"): [
        "ablation_6h_vital_context_validation_report.txt",
        "ablation_6h_vital_context_report.txt",
    ],

    ("12h", "vital_context"): [
        "ablation_12h_vital_context_validation_report.txt",
        "ablation_12h_vital_context_report.txt",
    ],

    ("24h", "vital_context"): [
        "ablation_24h_vital_context_validation_report.txt",
        "ablation_24h_vital_context_report.txt",
    ],

    ("6h", "exposure_context"): [
        "ablation_6h_exposure_context_validation_report.txt",
        "ablation_6h_exposure_context_report.txt",
    ],

    ("12h", "exposure_context"): [
        "ablation_12h_exposure_context_validation_report.txt",
        "ablation_12h_exposure_context_report.txt",
    ],

    ("24h", "exposure_context"): [
        "ablation_24h_exposure_context_validation_report.txt",
        "ablation_24h_exposure_context_report.txt",
    ],
}


# =============================================================================
# HELPERS
# =============================================================================

def read_text(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def find_report(candidates):
    for name in candidates:
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path

    return None


def extract_metric(text, metric):
    """
    Extract validation metric from completed report.

    Handles common formats such as:
        AUROC: 0.734154
        AUROC = 0.734154
        Validation AUROC: 0.734154
    """

    patterns = [
        rf"Validation\s+{metric}\s*[:=]\s*([0-9]*\.?[0-9]+)",
        rf"{metric}\s*[:=]\s*([0-9]*\.?[0-9]+)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if matches:
            return float(matches[-1])

    return None


def close(a, b, tol=5e-6):
    if a is None or b is None:
        return False

    return abs(float(a) - float(b)) <= tol


def fmt(x):
    if x is None:
        return "NA"

    return f"{x:.6f}"


# =============================================================================
# HEADER
# =============================================================================

print("=" * 80)
print("STEP 61D — VALIDATION ABLATION COMPARISON FIGURE")
print("=" * 80)
print("")
print("READ-ONLY ANALYSIS OF COMPLETED VALIDATION RESULTS")
print("NO MODEL WILL BE RETRAINED.")
print("NO TEST DATA OR TEST PREDICTIONS WILL BE READ.")
print("NO CALIBRATION WILL BE FIT.")
print("NO THRESHOLD WILL BE SELECTED OR CHANGED.")
print("NO EXISTING RESEARCH OR MODEL FILE WILL BE MODIFIED.")
print("")


# =============================================================================
# LOCATE AND READ REPORTS
# =============================================================================

results = {}

all_pass = True

print("=" * 80)
print("LOCATING COMPLETED VALIDATION REPORTS")
print("=" * 80)

for key, candidates in report_candidates.items():

    horizon, model = key

    path = find_report(candidates)

    print("")
    print(f"{horizon.upper()} — {model}")

    if path is None:
        print("Report found: NO")

        # For all previously established metrics except the still-unknown
        # 24h exposure/context result, a missing report is a safety stop.
        if expected[horizon][model]["auroc"] is not None:
            all_pass = False

        results[key] = {
            "path": None,
            "auroc": None,
            "auprc": None,
        }

        continue

    print("Report:")
    print(path)

    text = read_text(path)

    auroc = extract_metric(text, "AUROC")
    auprc = extract_metric(text, "AUPRC")

    results[key] = {
        "path": path,
        "auroc": auroc,
        "auprc": auprc,
    }

    print("AUROC:", auroc)
    print("AUPRC:", auprc)

    if auroc is None or auprc is None:
        print("Metric extraction: FAIL")
        all_pass = False
    else:
        print("Metric extraction: PASS")


# =============================================================================
# VERIFY AGAINST ALREADY ESTABLISHED RESULTS
# =============================================================================

print("")
print("=" * 80)
print("VALIDATION METRIC VERIFICATION")
print("=" * 80)

for horizon in ["6h", "12h", "24h"]:

    print("")
    print(horizon.upper())

    for model in [
        "full",
        "vital_context",
        "exposure_context",
    ]:

        observed = results[(horizon, model)]
        exp = expected[horizon][model]

        print("")
        print(model)

        if exp["auroc"] is None:
            if observed["auroc"] is None:
                print(
                    "No previously fixed expected value; "
                    "report result unavailable."
                )
            else:
                print(
                    "AUROC:",
                    fmt(observed["auroc"]),
                    "(accepted from completed validation report)"
                )
                print(
                    "AUPRC:",
                    fmt(observed["auprc"]),
                    "(accepted from completed validation report)"
                )

            continue

        auroc_pass = close(
            observed["auroc"],
            exp["auroc"]
        )

        auprc_pass = close(
            observed["auprc"],
            exp["auprc"]
        )

        print(
            "AUROC:",
            fmt(observed["auroc"]),
            "expected:",
            fmt(exp["auroc"]),
            "PASS=",
            auroc_pass
        )

        print(
            "AUPRC:",
            fmt(observed["auprc"]),
            "expected:",
            fmt(exp["auprc"]),
            "PASS=",
            auprc_pass
        )

        if not auroc_pass or not auprc_pass:
            all_pass = False


# =============================================================================
# REQUIRE ALL NINE MODEL/HORIZON RESULTS
# =============================================================================

missing = []

for horizon in ["6h", "12h", "24h"]:
    for model in [
        "full",
        "vital_context",
        "exposure_context",
    ]:

        r = results[(horizon, model)]

        if r["auroc"] is None or r["auprc"] is None:
            missing.append(
                f"{horizon}:{model}"
            )


if missing:
    print("")
    print("=" * 80)
    print("STEP 61D SAFETY STOP")
    print("=" * 80)
    print("")
    print("Missing completed validation results:")
    for x in missing:
        print(" -", x)

    print("")
    print(
        "No figure was generated because the ablation comparison "
        "must contain all three model variants at all three horizons."
    )
    print("")
    print("NO MODEL OR PREDICTION FILE WAS MODIFIED.")

    raise RuntimeError(
        "STEP 61D stopped because one or more completed "
        "validation ablation reports could not be located/read."
    )


# =============================================================================
# BUILD SUMMARY TABLE
# =============================================================================

rows = []

for horizon in ["6h", "12h", "24h"]:

    full_auroc = results[(horizon, "full")]["auroc"]
    full_auprc = results[(horizon, "full")]["auprc"]

    for model in [
        "full",
        "vital_context",
        "exposure_context",
    ]:

        auroc = results[(horizon, model)]["auroc"]
        auprc = results[(horizon, model)]["auprc"]

        rows.append({
            "horizon": horizon,
            "model": model,
            "AUROC": auroc,
            "AUPRC": auprc,
            "AUROC_difference_vs_full": (
                auroc - full_auroc
            ),
            "AUPRC_difference_vs_full": (
                auprc - full_auprc
            ),
            "source_report": results[(horizon, model)]["path"],
        })


# =============================================================================
# WRITE NEW STEP 61D SUMMARY CSV
# =============================================================================

with open(
    summary_csv,
    "w",
    newline="",
    encoding="utf-8"
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "horizon",
            "model",
            "AUROC",
            "AUPRC",
            "AUROC_difference_vs_full",
            "AUPRC_difference_vs_full",
            "source_report",
        ]
    )

    writer.writeheader()
    writer.writerows(rows)


# =============================================================================
# CONSOLE SUMMARY
# =============================================================================

print("")
print("=" * 80)
print("FINAL VALIDATION ABLATION SUMMARY")
print("=" * 80)

for horizon in ["6h", "12h", "24h"]:

    print("")
    print(horizon.upper())

    full_auroc = results[(horizon, "full")]["auroc"]
    full_auprc = results[(horizon, "full")]["auprc"]

    for model in [
        "full",
        "vital_context",
        "exposure_context",
    ]:

        auroc = results[(horizon, model)]["auroc"]
        auprc = results[(horizon, model)]["auprc"]

        print(
            f"{model:18s}",
            f"AUROC={auroc:.6f}",
            f"AUPRC={auprc:.6f}",
            f"ΔAUROC={auroc - full_auroc:+.6f}",
            f"ΔAUPRC={auprc - full_auprc:+.6f}",
        )


# =============================================================================
# FIGURE DATA
# =============================================================================

horizons = ["6h", "12h", "24h"]

model_order = [
    "full",
    "vital_context",
    "exposure_context",
]

display_names = {
    "full": "Full model",
    "vital_context": "Vital + context",
    "exposure_context": "Exposure + context",
}

x = np.arange(len(horizons))

width = 0.24


# =============================================================================
# CREATE FIGURE
# =============================================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.5)
)


# -----------------------------------------------------------------------------
# AUROC
# -----------------------------------------------------------------------------

ax = axes[0]

for i, model in enumerate(model_order):

    values = [
        results[(h, model)]["auroc"]
        for h in horizons
    ]

    positions = x + (i - 1) * width

    bars = ax.bar(
        positions,
        values,
        width,
        label=display_names[model],
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_title("AUROC")
ax.set_ylabel("AUROC")
ax.set_xticks(x)
ax.set_xticklabels(["6 h", "12 h", "24 h"])
ax.set_ylim(0.65, 0.78)
ax.grid(
    axis="y",
    alpha=0.25
)


# -----------------------------------------------------------------------------
# AUPRC
# -----------------------------------------------------------------------------

ax = axes[1]

for i, model in enumerate(model_order):

    values = [
        results[(h, model)]["auprc"]
        for h in horizons
    ]

    positions = x + (i - 1) * width

    bars = ax.bar(
        positions,
        values,
        width,
        label=display_names[model],
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

ax.set_title("AUPRC")
ax.set_ylabel("AUPRC")
ax.set_xticks(x)
ax.set_xticklabels(["6 h", "12 h", "24 h"])

max_auprc = max(
    results[(h, m)]["auprc"]
    for h in horizons
    for m in model_order
)

ax.set_ylim(
    0,
    min(
        1.0,
        max_auprc + 0.10
    )
)

ax.grid(
    axis="y",
    alpha=0.25
)


# =============================================================================
# SHARED LEGEND / TITLE
# =============================================================================

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=3,
    frameon=False,
    bbox_to_anchor=(0.5, -0.01),
)

fig.suptitle(
    "Validation Ablation Analysis Across Prediction Horizons",
    fontsize=15,
)

fig.text(
    0.5,
    0.015,
    (
        "Ablation models retain either vital trajectories + landmark context "
        "or modifiable exposure trajectories + landmark context."
    ),
    ha="center",
    fontsize=9,
)

plt.tight_layout(
    rect=[0, 0.09, 1, 0.93]
)


# =============================================================================
# SAVE FIGURE
# =============================================================================

fig.savefig(
    figure_png,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    figure_pdf,
    bbox_inches="tight"
)

plt.close(fig)


# =============================================================================
# VERIFY OUTPUTS
# =============================================================================

output_checks = {
    "summary_csv": os.path.exists(summary_csv),
    "figure_png": os.path.exists(figure_png),
    "figure_pdf": os.path.exists(figure_pdf),
}

for name, passed in output_checks.items():
    print("")
    print(
        name,
        "PASS=",
        passed
    )

    if not passed:
        all_pass = False


# =============================================================================
# WRITE REPORT
# =============================================================================

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "STEP 61D — VALIDATION ABLATION COMPARISON\n"
    )
    f.write("=" * 80 + "\n\n")

    f.write(
        "This analysis used completed validation results only.\n"
    )
    f.write(
        "No held-out test prediction was used for ablation comparison.\n"
    )
    f.write(
        "No model was retrained.\n"
    )
    f.write(
        "No calibration model was fit.\n"
    )
    f.write(
        "No threshold was selected or changed.\n"
    )
    f.write(
        "No existing research/model file was modified.\n\n"
    )

    for horizon in horizons:

        f.write(
            horizon.upper() + "\n"
        )
        f.write("-" * 80 + "\n")

        full_auroc = results[
            (horizon, "full")
        ]["auroc"]

        full_auprc = results[
            (horizon, "full")
        ]["auprc"]

        for model in model_order:

            r = results[(horizon, model)]

            f.write(
                f"{display_names[model]}\n"
            )

            f.write(
                f"AUROC: {r['auroc']:.6f}\n"
            )

            f.write(
                f"AUPRC: {r['auprc']:.6f}\n"
            )

            f.write(
                "AUROC difference vs full: "
                f"{r['auroc'] - full_auroc:+.6f}\n"
            )

            f.write(
                "AUPRC difference vs full: "
                f"{r['auprc'] - full_auprc:+.6f}\n"
            )

            f.write(
                f"Source: {r['path']}\n\n"
            )

    f.write("=" * 80 + "\n")

    if all_pass:
        f.write(
            "FINAL STEP 61D VERDICT: PASS\n"
        )
    else:
        f.write(
            "FINAL STEP 61D VERDICT: FAIL\n"
        )

    f.write("=" * 80 + "\n")


# =============================================================================
# FINAL SAFETY CHECK
# =============================================================================

if not all_pass:

    print("")
    print("=" * 80)
    print("FINAL STEP 61D VERDICT: FAIL")
    print("=" * 80)

    raise RuntimeError(
        "STEP 61D verification failed. "
        "Review the diagnostics above."
    )


# =============================================================================
# FINAL CONSOLE
# =============================================================================

print("")
print("=" * 80)
print("STEP 61D COMPLETE")
print("=" * 80)

print("")
print("Summary CSV:")
print(summary_csv)

print("")
print("Figure PNG:")
print(figure_png)

print("")
print("Figure PDF:")
print(figure_pdf)

print("")
print("Report:")
print(report_path)

print("")
print("=" * 80)
print("FINAL STEP 61D VERDICT: PASS")
print("=" * 80)

print("")
print("NO MODEL WAS RETRAINED.")
print("NO TEST DATA OR TEST PREDICTIONS WERE USED.")
print("NO CALIBRATION MODEL WAS FIT.")
print("NO THRESHOLD WAS SELECTED OR CHANGED.")
print("NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED.")
