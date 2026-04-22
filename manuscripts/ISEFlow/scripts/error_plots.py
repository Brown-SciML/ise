import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# -------------------------------------------------------------------
# File paths
# -------------------------------------------------------------------
dir_ = Path("/oscar/home/pvankatw/research/ise/manuscripts/ISEFlow")

files = {
    ("AIS", "ISEFlow"): dir_ / "iseflow-ais-predictions.csv",
    ("AIS", "GP emulator"): dir_ / "emulandice-ais-predictions.csv",
    ("GrIS", "ISEFlow"): dir_ / "iseflow-gris-predictions.csv",
    ("GrIS", "GP emulator"): dir_ / "emulandice-gris-predictions.csv",
}

panel_order = [
    ("AIS", "ISEFlow"),
    ("AIS", "GP emulator"),
    ("GrIS", "ISEFlow"),
    ("GrIS", "GP emulator"),
]

# -------------------------------------------------------------------
# Helper metrics
# -------------------------------------------------------------------
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

# -------------------------------------------------------------------
# Load data
# Each CSV must contain: year, pred, true
# -------------------------------------------------------------------
data = {}
for key, fp in files.items():
    df = pd.read_csv(fp)

    if "sle" in df.columns and "true" not in df.columns:
        df["true"] = df["sle"]

    required = {"year", "pred", "true"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{fp} is missing columns: {missing}")

    df["error"] = df["pred"] - df["true"]
    data[key] = df.copy()

# -------------------------------------------------------------------
# FIGURE 1: Emulator vs simulator scatter plots
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 9), constrained_layout=True)
axes = axes.flatten()

for ax, key in zip(axes, panel_order):
    sheet, model = key
    df = data[key]

    x = df["true"].values
    y = df["pred"].values

    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    lo -= pad
    hi += pad

    ax.scatter(x, y, alpha=0.05, s=10)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=2, color="red")

    panel_rmse = rmse(x, y)
    panel_mae = mae(x, y)
    panel_r2 = r2(x, y)

    # ax.text(
    #     0.04, 0.96,
    #     f"$R^2$ = {panel_r2:.3f}\nRMSE = {panel_rmse:.3f}\nMAE = {panel_mae:.3f}",
    #     transform=ax.transAxes,
    #     va="top",
    #     ha="left",
    #     bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
    #     fontsize=9,
    # )

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"{sheet} — {model}")
    ax.set_xlabel("Simulator / ISMIP6 SLE anomaly")
    ax.set_ylabel("Emulator SLE anomaly")

fig.suptitle("Emulator vs. simulator validation", fontsize=14)
fig.savefig("validation_scatter.png", dpi=300, bbox_inches="tight")
plt.show()

# -------------------------------------------------------------------
# FIGURE 2: Error histograms
# -------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
axes = axes.flatten()

# Use common x-axis limits for fair comparison
all_errors = np.concatenate([data[key]["error"].values for key in panel_order])
xmax = np.nanpercentile(np.abs(all_errors), 99)
xmax = max(1, np.ceil(xmax * 10) / 10)

bins = np.linspace(-xmax, xmax, 40)

for ax, key in zip(axes, panel_order):
    sheet, model = key
    df = data[key]
    errs = df["error"].values

    ax.hist(errs, bins=bins, density=True, alpha=0.75)
    ax.axvline(0, linestyle="--", linewidth=2, color="red")
    ax.axvline(np.mean(errs), linestyle=":", linewidth=2, color="black")

    bias = np.mean(errs)
    sd = np.std(errs, ddof=1)

    ax.text(
        0.04, 0.96,
        f"Bias = {bias:.3f}\nSD = {sd:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.7"),
        fontsize=9,
    )

    ax.set_title(f"{sheet} — {model}")
    ax.set_xlabel("Prediction error (pred - true)")
    ax.set_ylabel("Density")
    ax.set_xlim(-xmax, xmax)

fig.suptitle("Prediction error distributions", fontsize=14)
fig.savefig("error_histograms.png", dpi=300, bbox_inches="tight")
plt.show()