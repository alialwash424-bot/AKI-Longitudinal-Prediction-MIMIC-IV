import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt


BASE = Path(__file__).resolve().parents[2]
CURVE_FILE = BASE / "results" / "curve_points.csv"
OUTPUT_DIR = BASE / "results"

AUROC = {
    "6h": 0.724717,
    "12h": 0.733356,
    "24h": 0.731931,
}

AUPRC = {
    "6h": 0.198494,
    "12h": 0.328190,
    "24h": 0.503620,
}

PREVALENCE = {
    "6h": 0.085860,
    "12h": 0.156999,
    "24h": 0.276582,
}


def load_curves():
    curves = {}

    with open(CURVE_FILE, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        required = {
            "horizon",
            "curve",
            "point_index",
            "x",
            "y",
            "threshold",
        }

        if set(reader.fieldnames or []) != required:
            raise ValueError(
                f"Unexpected curve-point columns: {reader.fieldnames}"
            )

        for row in reader:
            key = (row["horizon"], row["curve"])
            curves.setdefault(key, []).append(
                (
                    int(row["point_index"]),
                    float(row["x"]),
                    float(row["y"]),
                )
            )

    for key in curves:
        curves[key].sort(key=lambda z: z[0])

    return curves


def plot_roc(curves):
    fig, ax = plt.subplots(figsize=(7, 6))

    for horizon in ("6h", "12h", "24h"):
        points = curves[(horizon, "ROC")]
        x = [p[1] for p in points]
        y = [p[2] for p in points]

        ax.plot(
            x,
            y,
            label=f"{horizon} (AUROC = {AUROC[horizon]:.3f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Final Held-Out Test ROC Curves")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "final_test_ROC_reproduced.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "final_test_ROC_reproduced.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_pr(curves):
    fig, ax = plt.subplots(figsize=(7, 6))

    for horizon in ("6h", "12h", "24h"):
        points = curves[(horizon, "PR")]
        x = [p[1] for p in points]
        y = [p[2] for p in points]

        ax.plot(
            x,
            y,
            label=f"{horizon} (AUPRC = {AUPRC[horizon]:.3f})",
        )

        ax.axhline(
            PREVALENCE[horizon],
            linestyle="--",
            alpha=0.35,
        )

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Final Held-Out Test Precision-Recall Curves")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR / "final_test_PR_reproduced.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        OUTPUT_DIR / "final_test_PR_reproduced.pdf",
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    curves = load_curves()

    expected = {
        ("6h", "ROC"),
        ("6h", "PR"),
        ("12h", "ROC"),
        ("12h", "PR"),
        ("24h", "ROC"),
        ("24h", "PR"),
    }

    if set(curves) != expected:
        raise ValueError(
            f"Unexpected curve groups. Found: {sorted(curves)}"
        )

    plot_roc(curves)
    plot_pr(curves)

    print("ROC/PR figures reproduced from locked aggregate curve points.")
    print("No patient-level predictions are required.")


if __name__ == "__main__":
    main()
