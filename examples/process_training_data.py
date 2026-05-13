"""Example: build an ISEFlow-ready training dataset from raw ISMIP6 archives.

This script walks through the full data-preparation pipeline used to train
ISEFlow. Starting from raw ISMIP6 forcings, sector grids, and Zenodo-archived
ice sheet model outputs, it produces a model-ready train/val/test split with
scaled features, lag variables, model characteristics, and outlier filtering.

Pipeline
--------
1. ``process_sectors``         — ingest raw NetCDF forcings + ISM outputs,
                                 aggregate to sectors, and emit ``dataset.csv``.
2. ``FeatureEngineer``         — load the sector-aggregated CSV and apply:
     - model-characteristics merge (numerics, stress balance, etc.)
     - feature scaling (saved alongside model weights for inference)
     - lag variable construction (default lag=5 years)
     - outlier removal on the SLE target via quantile clipping
     - train/val/test split (70/15/15, ``random_state=1``)

Outputs
-------
``EXPORT_DIR/dataset.csv``        sector-aggregated, pre-feature-engineering
``EXPORT_DIR/train.csv``          70% — used for model fitting
``EXPORT_DIR/val.csv``            15% — used for early stopping / HP search
``EXPORT_DIR/test.csv``           15% — held out for final evaluation
``EXPORT_DIR/scalers/*.pkl``      fitted scalers; must accompany the model at
                                  inference time so new inputs are scaled
                                  identically

Inputs required
---------------
ISMIP6_FORCINGS   GHub ISMIP6 forcing archive root (per-ice-sheet)
ISMIP6_GRIDS      Sector grid NetCDF (8 km AIS or 5 km GrIS)
ISMIP6_OUTPUTS    Zenodo "Computed Scalars Paper" archive of ISM outputs

Edit the three path constants below to point at your local copies before
running. ``ICE_SHEET`` selects which pipeline to run ("AIS" or "GrIS").
"""

import pandas as pd

from ise.data.feature_engineer import FeatureEngineer
from ise.data.process import process_sectors

# ── 1. Configure paths and ice sheet ──────────────────────────────────────────
#
# Switch ICE_SHEET to "GrIS" to process Greenland instead. The downstream
# steps branch on this value where the two ice sheets differ (e.g. AIS
# drops the ``mrro_anomaly`` column in v1.1.0; GrIS retains all features).

ICE_SHEET = "AIS"  # "AIS" or "GrIS"

ISMIP6_FORCINGS = r"/path/to/ismip6_forcings/GHub-ISMIP6-Forcing/AIS"
ISMIP6_GRIDS = r"/path/to/ismip6_gridfiles/Grid_Files/AIS_sectors_8km.nc"
ISMIP6_OUTPUTS = r"/path/to/ismip6_outputs/Zenodo_Outputs/ComputedScalarsPaper"

EXPORT_DIR = f"path/to/export_dir/{ICE_SHEET}"


# ── 2. Sector aggregation ─────────────────────────────────────────────────────
#
# Reads each NetCDF forcing and ISM output file, computes per-sector
# averages, aligns timestamps to the 86-year 2015-2100 window, and writes
# ``EXPORT_DIR/dataset.csv``.
#
#   overwrite=False   skip files already written from a prior run
#   with_ctrl=False   subtract the control run from each experiment so the
#                     target is the *anomaly* of SLE due to forcing, not the
#                     raw projection

print("=" * 70)
print(f"ISEFlow Training Data Pipeline ({ICE_SHEET})")
print("=" * 70)
print(f"  Forcings: {ISMIP6_FORCINGS}")
print(f"  Grids:    {ISMIP6_GRIDS}")
print(f"  Outputs:  {ISMIP6_OUTPUTS}")
print(f"  Export:   {EXPORT_DIR}")

print("\n[1/6] Aggregating raw NetCDF files into sector-level dataset...")
dataset = process_sectors(
    ice_sheet=ICE_SHEET,
    forcing_directory=ISMIP6_FORCINGS,
    grid_file=ISMIP6_GRIDS,
    zenodo_directory=ISMIP6_OUTPUTS,
    export_directory=EXPORT_DIR,
    overwrite=False,
    with_ctrl=False,
)


# ── 3. Feature engineering ────────────────────────────────────────────────────
#
# Load the sector-aggregated CSV and apply the transformations required to
# match the inputs ISEFlow was trained on. ``split_dataset=False`` defers
# splitting until after all per-row transforms are applied below.

print("\n[2/6] Loading sector-aggregated CSV into FeatureEngineer...")
fe = FeatureEngineer(
    ice_sheet=ICE_SHEET,
    data=pd.read_csv(f"{EXPORT_DIR}/dataset.csv"),
    split_dataset=False,
    output_directory=None,
)
print(f"  Loaded {len(fe.data):,} rows, {fe.data.shape[1]} columns")

# v1.1.0 drops ``mrro_anomaly`` from AIS due to inconsistent availability
# across CMIP models. GrIS retains all features.
if ICE_SHEET == "AIS":
    print("  Dropping `mrro_anomaly` column (v1.1.0 AIS convention)")
    fe.data = fe.data.drop(columns="mrro_anomaly")

# Merge ISM model characteristics (numerics, stress balance, resolution, etc.)
# from the bundled JSON config. These categorical features condition ISEFlow
# on the structural choices of each ice sheet model run.
print("\n[3/6] Merging ISM model characteristics...")
fe.add_model_characteristics()

# Fit and apply feature scaling. The fitted scalers are persisted to disk so
# the same transformation can be re-applied at inference time.
print(f"\n[4/6] Fitting and saving scalers to {EXPORT_DIR}/scalers/...")
fe.scale_data(save_dir=f"{EXPORT_DIR}/scalers/")

# Add 5-year lagged copies of the forcing variables so the model sees recent
# history at each timestep. Matches the ``sequence_length=5`` convention used
# by ``EmulatorDataset``.
print("\n[5/6] Adding 5-year lag variables and removing SLE outliers (0.5% tails)...")
fe.add_lag_variables(lag=5)

# Clip the tails of the SLE distribution (top and bottom 0.5%) to remove
# extreme outliers that would otherwise dominate the loss.
fe.drop_outliers("quantile", "sle", quantiles=[0.005, 1 - 0.005])

# Final 70/15/15 split. ``random_state=1`` is fixed inside FeatureEngineer
# for reproducibility — do not change without retraining all downstream models.
print(f"\n[6/6] Splitting into train/val/test (70/15/15) and writing to {EXPORT_DIR}...")
fe.split_data(
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    output_directory=EXPORT_DIR,
)

print("\nDone. Training-ready files written to:")
print(f"  {EXPORT_DIR}/train.csv")
print(f"  {EXPORT_DIR}/val.csv")
print(f"  {EXPORT_DIR}/test.csv")
print(f"  {EXPORT_DIR}/scalers/")
