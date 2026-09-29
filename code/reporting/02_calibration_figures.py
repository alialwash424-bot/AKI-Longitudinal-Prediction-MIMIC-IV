import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# STEP 61B — FINAL HELD-OUT TEST CALIBRATION FIGURES
# ============================================================

folder = os.path.expanduser("~/Documents/AKI_Research")

print("=" * 80)
print("STEP 61B — FINAL HELD-OUT TEST CALIBRATION FIGURES")
print("=" * 80)
print()
print("READ-ONLY ANALYSIS OF FROZEN FINAL TEST PREDICTIONS")
print("NO MODEL WILL BE RETRAINED.")
print("NO CALIBRATION MODEL WILL BE REFIT.")
print("NO THRESHOLD WILL BE SELECTED OR CHANGED.")
print("NO TEST PREDICTION FILE WILL BE MODIFIED.")
print()

# ------------------------------------------------------------
# INPUT FILES
# ------------------------------------------------------------

configs = {
    "6h": {
        "original": os.path.join(
            folder,
            "FINAL_TEST_6h_predictions.csv"
        ),
        "calibrated": os.path.join(
            folder,
            "FINAL_TEST_6h_calibrated_predictions.csv"
        ),
        "target": "aki_within_6h",
        "expected_rows": 64570,
        "expected_positive": 5544,
        "expected_negative": 59026,
    },

    "12h": {
        "original": os.path.join(
            folder,
            "FINAL_TEST_12h_predictions.csv"
        ),
        "calibrated": os.path.join(
            folder,
            "FINAL_TEST_12h_calibrated_predictions.csv"
        ),
        "target": "aki_within_12h",
        "expected_rows": 58924,
        "expected_positive": 9251,
        "expected_negative": 49673,
    },

    "24h": {
        "original": os.path.join(
            folder,
            "FINAL_TEST_24h_predictions.csv"
        ),
        "calibrated": os.path.join(
            folder,
            "FINAL_TEST_24h_calibrated_predictions.csv"
        ),
        "target": "aki_within_24h",
        "expected_rows": 49367,
        "expected_positive": 13654,
        "expected_negative": 35713,
    },
}

# ------------------------------------------------------------
# OUTPUT FILES
# ------------------------------------------------------------

original_png = os.path.join(
    folder,
    "step61B_original_test_calibration.png"
)

original_pdf = os.path.join(
    folder,
    "step61B_original_test_calibration.pdf"
)

calibrated_png = os.path.join(
    folder,
    "step61B_validation_calibrated_test_calibration.png"
)

calibrated_pdf = os.path.join(
    folder,
    "step61B_validation_calibrated_test_calibration.pdf"
)

comparison_png = os.path.join(
    folder,
    "step61B_original_vs_calibrated_calibration.png"
)

comparison_pdf = os.path.join(
    folder,
    "step61B_original_vs_calibrated_calibration.pdf"
)

table_path = os.path.join(
    folder,
    "step61B_calibration_curve_points.csv"
)

report_path = os.path.join(
    folder,
    "step61B_calibration_figure_report.txt"
)

# ------------------------------------------------------------
# REQUIRED COLUMN CHECKS
# ------------------------------------------------------------

required_original = {
    "subject_id",
    "stay_id",
    "landmark_time",
    "predicted_probability",
}

required_calibrated = {
    "subject_id",
    "stay_id",
    "landmark_time",
    "original_predicted_probability",
    "calibrated_predicted_probability",
    "validation_calibration_intercept",
    "validation_calibration_slope",
}

# ------------------------------------------------------------
# HELPERS
# ------------------------------------------------------------

def calibration_points(y, p, bins=10):
    """
    Equal-frequency calibration bins.

    Returns one row per calibration bin:
    n
    observed events
    observed event rate
    mean predicted probability
    minimum predicted probability
    maximum predicted probability
    """

    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)

    valid = (
        np.isfinite(y)
        & np.isfinite(p)
    )

    y = y[valid]
    p = p[valid]

    if len(y) == 0:
        raise RuntimeError(
            "No valid rows available for calibration."
        )

    if np.any((p < 0) | (p > 1)):
        raise RuntimeError(
            "Predicted probabilities outside [0, 1]."
        )

    order = np.argsort(p, kind="mergesort")

    y_sorted = y[order]
    p_sorted = p[order]

    index_groups = np.array_split(
        np.arange(len(y_sorted)),
        bins
    )

    rows = []

    for bin_number, idx in enumerate(index_groups):

        yy = y_sorted[idx]
        pp = p_sorted[idx]

        rows.append({
            "calibration_bin": bin_number,
            "n": len(idx),
            "observed_events": int(np.sum(yy)),
            "observed_rate": float(np.mean(yy)),
            "mean_predicted_probability": float(np.mean(pp)),
            "minimum_predicted_probability": float(np.min(pp)),
            "maximum_predicted_probability": float(np.max(pp)),
        })

    return pd.DataFrame(rows)


def verify_identity(original, calibrated, target, horizon):
    """
    Confirm calibrated file corresponds to the same frozen
    test rows as the original prediction file.
    """

    key_cols = [
        "subject_id",
        "stay_id",
        "landmark_time",
    ]

    if len(original) != len(calibrated):
        raise RuntimeError(
            f"{horizon}: original/calibrated row count mismatch."
        )

    for col in key_cols:

        a = original[col].astype(str).to_numpy()
        b = calibrated[col].astype(str).to_numpy()

        if not np.array_equal(a, b):
            raise RuntimeError(
                f"{horizon}: row identity mismatch in {col}."
            )

    y1 = pd.to_numeric(
        original[target],
        errors="coerce"
    ).to_numpy()

    y2 = pd.to_numeric(
        calibrated[target],
        errors="coerce"
    ).to_numpy()

    if not np.array_equal(y1, y2):
        raise RuntimeError(
            f"{horizon}: target values differ between files."
        )

    p_original = pd.to_numeric(
        original["predicted_probability"],
        errors="coerce"
    ).to_numpy()

    p_copy = pd.to_numeric(
        calibrated["original_predicted_probability"],
        errors="coerce"
    ).to_numpy()

    if not np.allclose(
        p_original,
        p_copy,
        rtol=0,
        atol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError(
            f"{horizon}: original probabilities differ "
            "between original and calibrated files."
        )


def make_single_figure(
    point_tables,
    probability_type,
    title,
    png_path,
    pdf_path,
):
    """
    Produce one calibration figure containing all three horizons.
    """

    plt.figure(figsize=(8, 8))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Ideal calibration",
    )

    for horizon in ["6h", "12h", "24h"]:

        df = point_tables[
            (point_tables["horizon"] == horizon)
            & (
                point_tables["probability_type"]
                == probability_type
            )
        ].copy()

        plt.plot(
            df["mean_predicted_probability"],
            df["observed_rate"],
            marker="o",
            linewidth=2,
            markersize=6,
            label=horizon,
        )

    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed event proportion")
    plt.title(title)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.grid(
        True,
        linewidth=0.5,
        alpha=0.3,
    )

    plt.legend()
    plt.tight_layout()

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


# ------------------------------------------------------------
# LOAD + VERIFY
# ------------------------------------------------------------

all_points = []
loaded = {}

print("=" * 80)
print("INPUT VERIFICATION")
print("=" * 80)

for horizon, cfg in configs.items():

    print()
    print("-" * 80)
    print(horizon.upper())
    print("-" * 80)

    if not os.path.exists(cfg["original"]):
        raise FileNotFoundError(
            cfg["original"]
        )

    if not os.path.exists(cfg["calibrated"]):
        raise FileNotFoundError(
            cfg["calibrated"]
        )

    original = pd.read_csv(
        cfg["original"]
    )

    calibrated = pd.read_csv(
        cfg["calibrated"]
    )

    missing_original = (
        required_original
        - set(original.columns)
    )

    missing_calibrated = (
        required_calibrated
        - set(calibrated.columns)
    )

    if cfg["target"] not in original.columns:
        missing_original.add(
            cfg["target"]
        )

    if cfg["target"] not in calibrated.columns:
        missing_calibrated.add(
            cfg["target"]
        )

    if missing_original:
        raise RuntimeError(
            f"{horizon}: missing original columns: "
            f"{sorted(missing_original)}"
        )

    if missing_calibrated:
        raise RuntimeError(
            f"{horizon}: missing calibrated columns: "
            f"{sorted(missing_calibrated)}"
        )

    verify_identity(
        original,
        calibrated,
        cfg["target"],
        horizon,
    )

    y = pd.to_numeric(
        original[cfg["target"]],
        errors="coerce",
    )

    original_probability = pd.to_numeric(
        original["predicted_probability"],
        errors="coerce",
    )

    calibrated_probability = pd.to_numeric(
        calibrated[
            "calibrated_predicted_probability"
        ],
        errors="coerce",
    )

    rows = len(original)

    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())

    row_pass = (
        rows == cfg["expected_rows"]
    )

    positive_pass = (
        positives
        == cfg["expected_positive"]
    )

    negative_pass = (
        negatives
        == cfg["expected_negative"]
    )

    probability_pass = bool(
        original_probability.between(
            0,
            1,
            inclusive="both",
        ).all()
        and
        calibrated_probability.between(
            0,
            1,
            inclusive="both",
        ).all()
    )

    print(
        f"Rows: {rows} "
        f"expected: {cfg['expected_rows']} "
        f"PASS={row_pass}"
    )

    print(
        f"Positives: {positives} "
        f"expected: {cfg['expected_positive']} "
        f"PASS={positive_pass}"
    )

    print(
        f"Negatives: {negatives} "
        f"expected: {cfg['expected_negative']} "
        f"PASS={negative_pass}"
    )

    print(
        "Original/calibrated row identity: PASS"
    )

    print(
        "Original probability identity: PASS"
    )

    print(
        "Probability range [0,1]:",
        "PASS" if probability_pass else "FAIL",
    )

    if not (
        row_pass
        and positive_pass
        and negative_pass
        and probability_pass
    ):
        raise RuntimeError(
            f"{horizon}: safety verification failed."
        )

    # Original curve
    original_points = calibration_points(
        y,
        original_probability,
        bins=10,
    )

    original_points.insert(
        0,
        "probability_type",
        "original",
    )

    original_points.insert(
        0,
        "horizon",
        horizon,
    )

    # Calibrated curve
    calibrated_points = calibration_points(
        y,
        calibrated_probability,
        bins=10,
    )

    calibrated_points.insert(
        0,
        "probability_type",
        "validation_calibrated",
    )

    calibrated_points.insert(
        0,
        "horizon",
        horizon,
    )

    all_points.append(
        original_points
    )

    all_points.append(
        calibrated_points
    )

    loaded[horizon] = {
        "original": original,
        "calibrated": calibrated,
        "y": y,
        "original_probability": original_probability,
        "calibrated_probability": calibrated_probability,
    }


# ------------------------------------------------------------
# COMBINE CALIBRATION POINTS
# ------------------------------------------------------------

points = pd.concat(
    all_points,
    ignore_index=True,
)

if len(points) != 60:
    raise RuntimeError(
        "Expected exactly 60 calibration-point rows "
        "(3 horizons × 2 probability types × 10 bins)."
    )

points.to_csv(
    table_path,
    index=False,
)


# ------------------------------------------------------------
# ORIGINAL CALIBRATION FIGURE
# ------------------------------------------------------------

make_single_figure(
    point_tables=points,
    probability_type="original",
    title=(
        "Final Held-Out Test Calibration — "
        "Original Locked Models"
    ),
    png_path=original_png,
    pdf_path=original_pdf,
)


# ------------------------------------------------------------
# VALIDATION-CALIBRATED FIGURE
# ------------------------------------------------------------

make_single_figure(
    point_tables=points,
    probability_type="validation_calibrated",
    title=(
        "Final Held-Out Test Calibration — "
        "Validation-Derived Recalibration"
    ),
    png_path=calibrated_png,
    pdf_path=calibrated_pdf,
)


# ------------------------------------------------------------
# COMPARISON FIGURE
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(15, 5),
)

for ax, horizon in zip(
    axes,
    ["6h", "12h", "24h"],
):

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        linewidth=1.5,
        label="Ideal",
    )

    original_df = points[
        (points["horizon"] == horizon)
        & (
            points["probability_type"]
            == "original"
        )
    ]

    calibrated_df = points[
        (points["horizon"] == horizon)
        & (
            points["probability_type"]
            == "validation_calibrated"
        )
    ]

    ax.plot(
        original_df[
            "mean_predicted_probability"
        ],
        original_df[
            "observed_rate"
        ],
        marker="o",
        linewidth=2,
        label="Original",
    )

    ax.plot(
        calibrated_df[
            "mean_predicted_probability"
        ],
        calibrated_df[
            "observed_rate"
        ],
        marker="o",
        linewidth=2,
        label="Validation-calibrated",
    )

    ax.set_title(
        horizon.upper()
    )

    ax.set_xlabel(
        "Mean predicted probability"
    )

    ax.set_ylabel(
        "Observed event proportion"
    )

    ax.set_xlim(
        0,
        1,
    )

    ax.set_ylim(
        0,
        1,
    )

    ax.grid(
        True,
        linewidth=0.5,
        alpha=0.3,
    )

    ax.legend()

fig.suptitle(
    "Final Held-Out Test Calibration: "
    "Original vs Validation-Derived Recalibration",
    fontsize=14,
)

plt.tight_layout()

plt.savefig(
    comparison_png,
    dpi=300,
    bbox_inches="tight",
)

plt.savefig(
    comparison_pdf,
    bbox_inches="tight",
)

plt.close()


# ------------------------------------------------------------
# VERIFY OUTPUT FIGURES
# ------------------------------------------------------------

figure_paths = [
    original_png,
    original_pdf,
    calibrated_png,
    calibrated_pdf,
    comparison_png,
    comparison_pdf,
]

for path in figure_paths:

    if not os.path.exists(path):
        raise RuntimeError(
            f"Figure was not generated: {path}"
        )

    if os.path.getsize(path) <= 0:
        raise RuntimeError(
            f"Generated figure is empty: {path}"
        )


# ------------------------------------------------------------
# REPORT
# ------------------------------------------------------------

with open(
    report_path,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "STEP 61B — FINAL HELD-OUT TEST "
        "CALIBRATION FIGURES\n"
    )

    f.write(
        "=" * 80 + "\n\n"
    )

    f.write(
        "Analysis status:\n"
    )

    f.write(
        "Frozen final held-out test predictions only.\n"
    )

    f.write(
        "No predictive model retrained.\n"
    )

    f.write(
        "No calibration model refitted.\n"
    )

    f.write(
        "No threshold selected or changed.\n"
    )

    f.write(
        "No test prediction file modified.\n\n"
    )

    for horizon in [
        "6h",
        "12h",
        "24h",
    ]:

        data = loaded[horizon]

        y = data["y"]

        p_original = data[
            "original_probability"
        ]

        p_calibrated = data[
            "calibrated_probability"
        ]

        observed = float(
            y.mean()
        )

        mean_original = float(
            p_original.mean()
        )

        mean_calibrated = float(
            p_calibrated.mean()
        )

        f.write(
            f"{horizon.upper()}\n"
        )

        f.write(
            "-" * 80 + "\n"
        )

        f.write(
            f"Rows: {len(y)}\n"
        )

        f.write(
            f"Observed event rate: "
            f"{observed:.6f}\n"
        )

        f.write(
            f"Original mean predicted probability: "
            f"{mean_original:.6f}\n"
        )

        f.write(
            f"Validation-calibrated mean predicted "
            f"probability: {mean_calibrated:.6f}\n"
        )

        f.write(
            f"Original E/O ratio: "
            f"{mean_original / observed:.6f}\n"
        )

        f.write(
            f"Validation-calibrated E/O ratio: "
            f"{mean_calibrated / observed:.6f}\n"
        )

        intercept_values = (
            data["calibrated"][
                "validation_calibration_intercept"
            ]
            .dropna()
            .unique()
        )

        slope_values = (
            data["calibrated"][
                "validation_calibration_slope"
            ]
            .dropna()
            .unique()
        )

        if len(intercept_values) != 1:
            raise RuntimeError(
                f"{horizon}: calibration intercept "
                "is not constant."
            )

        if len(slope_values) != 1:
            raise RuntimeError(
                f"{horizon}: calibration slope "
                "is not constant."
            )

        f.write(
            f"Frozen validation calibration intercept: "
            f"{intercept_values[0]:.6f}\n"
        )

        f.write(
            f"Frozen validation calibration slope: "
            f"{slope_values[0]:.6f}\n\n"
        )

    f.write(
        "Calibration curve construction:\n"
    )

    f.write(
        "Equal-frequency deciles calculated separately "
        "within each horizon and probability type.\n"
    )

    f.write(
        "Observed event proportion plotted against "
        "mean predicted probability.\n\n"
    )

    f.write(
        "IMPORTANT:\n"
    )

    f.write(
        "The validation-derived calibration parameters "
        "were already frozen before this step.\n"
    )

    f.write(
        "This script did not use test outcomes to "
        "estimate any model parameter.\n"
    )

    f.write(
        "Test outcomes were used only for final "
        "calibration assessment and plotting.\n"
    )


# ------------------------------------------------------------
# FINAL CONSOLE
# ------------------------------------------------------------

print()
print("=" * 80)
print("STEP 61B COMPLETE")
print("=" * 80)

print()
print("Original calibration PNG:")
print(original_png)

print()
print("Original calibration PDF:")
print(original_pdf)

print()
print("Validation-calibrated calibration PNG:")
print(calibrated_png)

print()
print("Validation-calibrated calibration PDF:")
print(calibrated_pdf)

print()
print("Original-vs-calibrated comparison PNG:")
print(comparison_png)

print()
print("Original-vs-calibrated comparison PDF:")
print(comparison_pdf)

print()
print("Calibration curve-point CSV:")
print(table_path)

print()
print("Figure report:")
print(report_path)

print()
print("=" * 80)
print("FINAL STEP 61B VERDICT: PASS")
print("=" * 80)

print()
print("NO MODEL WAS RETRAINED.")
print("NO CALIBRATION MODEL WAS REFIT.")
print("NO THRESHOLD WAS SELECTED OR CHANGED.")
print("NO TEST PREDICTION WAS MODIFIED.")
print("NO EXISTING RESEARCH OR MODEL FILE WAS MODIFIED.")
