"""One-time script to extract sector-averaged climatologies from ISMIP6 forcing files.

Produces three CSV files in ise/data/data_files/:
  - AIS_atmos_climatologies.csv  : pr, evspsbl, smb, ts (and mrro where available)
                                    1995-2014 mean, spatially averaged per AIS sector (1-18)
  - GrIS_atmos_climatologies.csv : SMB, ST
                                    1960-1989 MAR reference mean, per GrIS sector (1-6)

Run once from the repo root:
    python extract_climatologies.py \
        --ais-atmos-dir  /path/to/GHub-ISMIP6-Forcing/AIS/Atmosphere_Forcing \
        --ais-grid       /path/to/Grid_Files/AIS_sectors_8km.nc \
        --gris-atmos-dir /path/to/GHub-ISMIP6-Forcing/GrIS/Atmosphere_Forcing/aSMB_observed/v1 \
        --gris-grid      /path/to/Grid_Files/GrIS_Basins_Rignot_sectors_5km.nc \
        --output-dir     ise/data/data_files

Default paths are hard-coded for the Oscar HPC environment below; override with CLI args.
"""

import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import xarray as xr

# ---------------------------------------------------------------------------
# Default paths (Oscar HPC)
# ---------------------------------------------------------------------------
_FORCING_ROOT = (
    "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/GHub-ISMIP6-Forcing"
)
_DEFAULT_AIS_ATMOS = os.path.join(_FORCING_ROOT, "AIS/Atmosphere_Forcing")
_DEFAULT_AIS_GRID = (
    "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files/AIS_sectors_8km.nc"
)
_DEFAULT_GRIS_ATMOS = os.path.join(
    _FORCING_ROOT, "GrIS/Atmosphere_Forcing/aSMB_observed/v1"
)
_DEFAULT_GRIS_GRID = (
    "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/Grid_Files"
    "/GrIS_Basins_Rignot_sectors_5km.nc"
)
_DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "ise/data/data_files")

# ---------------------------------------------------------------------------
# Canonical AOGCM name mapping
#
# Keys   = directory names as they appear on disk
# Values = canonical names used in ismip6_experiments_updated.csv
#          (lowercase, hyphen-separated model, underscore before scenario)
# ---------------------------------------------------------------------------
AIS_ATMOS_DIR_TO_CANONICAL = {
    "noresm1-m_rcp8.5":      "noresm1-m_rcp85",
    "noresm1-m_rcp2.6":      "noresm1-m_rcp26",
    "miroc-esm-chem_rcp8.5": "miroc-esm-chem_rcp85",
    "miroc-esm-chem_rcp2.6": "miroc-esm-chem_rcp26",
    "ccsm4_rcp8.5":          "ccsm4_rcp85",
    "ccsm4_rcp2.6":          "ccsm4_rcp26",
    "HadGEM2-ES_rcp85":      "hadgem2-es_rcp85",
    "CSIRO-Mk3-6-0_rcp85":   "csiro-mk3.6_rcp85",
    "IPSL-CM5A-MR_rcp85":    "ipsl-cm5-mr_rcp85",
    "IPSL-CM5A-MR_rcp26":    "ipsl-cm5-mr_rcp26",
    "CNRM_CM6_ssp585":       "cnrm-cm6_ssp585",
    "CNRM_CM6_ssp126":       "cnrm-cm6_ssp126",
    "CNRM_ESM2_ssp585":      "cnrm-esm2_ssp585",
    "UKESM1-0-LL":           "ukesm1-0-ll_ssp585",
    "CESM2_ssp585":          "cesm2_ssp585",
}

GRIS_ATMOS_DIR_TO_CANONICAL = {
    "ACCESS1.3-rcp85":   "access1.3_rcp85",
    "CESM2-ssp585":      "cesm2_ssp585",
    "CNRM-CM6-ssp126":   "cnrm-cm6_ssp126",
    "CNRM-CM6-ssp585":   "cnrm-cm6_ssp585",
    "CNRM-ESM2-ssp585":  "cnrm-esm2_ssp585",
    "CSIRO-Mk3.6-rcp85": "csiro-mk3.6_rcp85",
    "HadGEM2-ES-rcp85":  "hadgem2-es_rcp85",
    "IPSL-CM5-MR-rcp85": "ipsl-cm5-mr_rcp85",
    "MIROC5-rcp26":      "miroc5_rcp26",
    "MIROC5-rcp85":      "miroc5_rcp85",
    "NorESM1-rcp85":     "noresm1-m_rcp85",
    "UKESM1-CM6-ssp585": "ukesm1-0-ll_ssp585",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _spatial_mean_by_sector(data_2d: np.ndarray, sectors: np.ndarray, sector_ids: list) -> dict:
    """Return {sector_id: mean_value} for each sector, ignoring NaNs."""
    result = {}
    for s in sector_ids:
        mask = sectors == s
        vals = data_2d[mask]
        vals = vals[~np.isnan(vals)]
        result[s] = float(np.mean(vals)) if len(vals) > 0 else np.nan
    return result


# ---------------------------------------------------------------------------
# AIS atmospheric climatologies
# ---------------------------------------------------------------------------

def extract_ais_atmos(atmos_dir: str, grid_path: str) -> pd.DataFrame:
    """
    Read each AOGCM's *clim_1995-2014*8km*.nc file, sector-average each
    atmospheric variable, and return a DataFrame.

    Columns: aogcm, sector, pr_clim, evspsbl_clim, smb_clim, ts_clim
    mrro_clim is included where available (NaN otherwise).
    """
    grid = xr.open_dataset(grid_path)
    sectors_grid = grid.sectors.values  # (761, 761)
    sector_ids = sorted([int(s) for s in np.unique(sectors_grid) if not np.isnan(s) and s > 0])

    rows = []
    for dir_name, canonical in AIS_ATMOS_DIR_TO_CANONICAL.items():
        clim_dir = os.path.join(atmos_dir, dir_name, "Regridded_8km")
        if not os.path.isdir(clim_dir):
            warnings.warn(f"AIS atmos: directory not found, skipping: {clim_dir}")
            continue

        pattern = os.path.join(clim_dir, "*clim*1995-2014*.nc")
        matches = glob.glob(pattern)
        if not matches:
            warnings.warn(f"AIS atmos: no clim file found for {dir_name}, skipping")
            continue

        clim_path = matches[0]
        print(f"  AIS atmos [{canonical}]: {os.path.basename(clim_path)}")
        ds = xr.open_dataset(clim_path)

        pr      = ds["pr_clim"].values       # (761, 761)
        evspsbl = ds["evspsbl_clim"].values
        smb     = ds["smb_clim"].values
        ts      = ds["ts_clim"].values
        mrro    = ds["mrro_clim"].values if "mrro_clim" in ds else None

        pr_by_s      = _spatial_mean_by_sector(pr,      sectors_grid, sector_ids)
        evspsbl_by_s = _spatial_mean_by_sector(evspsbl, sectors_grid, sector_ids)
        smb_by_s     = _spatial_mean_by_sector(smb,     sectors_grid, sector_ids)
        ts_by_s      = _spatial_mean_by_sector(ts,      sectors_grid, sector_ids)
        mrro_by_s    = (_spatial_mean_by_sector(mrro, sectors_grid, sector_ids)
                        if mrro is not None else {s: np.nan for s in sector_ids})

        for s in sector_ids:
            rows.append({
                "aogcm":        canonical,
                "sector":       s,
                "pr_clim":      pr_by_s[s],
                "evspsbl_clim": evspsbl_by_s[s],
                "smb_clim":     smb_by_s[s],
                "ts_clim":      ts_by_s[s],
                "mrro_clim":    mrro_by_s[s],
            })

        ds.close()

    df = pd.DataFrame(rows)
    print(f"  AIS atmos: {len(df)} rows ({df.aogcm.nunique()} AOGCMs × {len(sector_ids)} sectors)")
    return df


# ---------------------------------------------------------------------------
# GrIS atmospheric climatologies
# ---------------------------------------------------------------------------

def extract_gris_atmos(atmos_dir: str, grid_path: str) -> pd.DataFrame:
    """
    Read each AOGCM's MAR Reference file (1960-1989 long-term mean), downsample
    from 1km to 5km grid, sector-average SMB and ST, and return a tidy DataFrame.

    Columns: aogcm, sector, smb_clim, st_clim
    """
    grid = xr.open_dataset(grid_path)
    sectors_grid = grid.ID.values  # (577, 337)
    sector_ids = sorted([int(s) for s in np.unique(sectors_grid) if not np.isnan(s) and s > 0])

    rows = []
    seen_refs = {}  # ref_filename -> rows already written (avoids re-processing shared refs)

    for dir_name, canonical in GRIS_ATMOS_DIR_TO_CANONICAL.items():
        ref_dir = os.path.join(atmos_dir, dir_name, "Reference")
        if not os.path.isdir(ref_dir):
            warnings.warn(f"GrIS atmos: Reference dir not found, skipping: {ref_dir}")
            continue

        # Find the ltm reference file (not the topg file)
        matches = [f for f in os.listdir(ref_dir) if "ltm" in f and f.endswith(".nc")]
        if not matches:
            warnings.warn(f"GrIS atmos: no reference file found for {dir_name}, skipping")
            continue

        ref_path = os.path.join(ref_dir, matches[0])
        print(f"  GrIS atmos [{canonical}]: {matches[0]}")

        # Multiple pathways share the same reference file (e.g. CNRM-CM6-ssp126/ssp585).
        # We still write a row for each canonical name so lookup is straightforward.
        if ref_path in seen_refs:
            # Re-use pre-computed sector means, just swap the canonical name
            for cached_row in seen_refs[ref_path]:
                rows.append({**cached_row, "aogcm": canonical})
            continue

        ds = xr.open_dataset(ref_path, decode_times=False)

        # Squeeze the single time dimension (30-year mean collapsed to 1 step)
        smb_full = ds["SMB"].squeeze("time").values   # (2881, 1681)
        st_full  = ds["ST"].squeeze("time").values

        # Downsample from ~1km (2881×1681) to 5km (577×337) to match sector grid
        smb_5km = smb_full[::5, ::5]   # (577, 337)
        st_5km  = st_full[::5, ::5]

        smb_by_s = _spatial_mean_by_sector(smb_5km, sectors_grid, sector_ids)
        st_by_s  = _spatial_mean_by_sector(st_5km,  sectors_grid, sector_ids)

        this_ref_rows = []
        for s in sector_ids:
            row = {
                "aogcm":    canonical,
                "sector":   s,
                "smb_clim": smb_by_s[s],
                "st_clim":  st_by_s[s],
            }
            rows.append(row)
            this_ref_rows.append(row)

        seen_refs[ref_path] = this_ref_rows
        ds.close()

    df = pd.DataFrame(rows)
    print(f"  GrIS atmos: {len(df)} rows ({df.aogcm.nunique()} AOGCMs × {len(sector_ids)} sectors)")
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ais-atmos-dir",  default=_DEFAULT_AIS_ATMOS)
    parser.add_argument("--ais-grid",       default=_DEFAULT_AIS_GRID)
    parser.add_argument("--gris-atmos-dir", default=_DEFAULT_GRIS_ATMOS)
    parser.add_argument("--gris-grid",      default=_DEFAULT_GRIS_GRID)
    parser.add_argument("--output-dir",     default=_DEFAULT_OUT)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\n=== Extracting AIS atmospheric climatologies ===")
    ais_atmos = extract_ais_atmos(args.ais_atmos_dir, args.ais_grid)
    ais_out = os.path.join(args.output_dir, "AIS_atmos_climatologies.csv")
    ais_atmos.to_csv(ais_out, index=False)
    print(f"  Saved: {ais_out}")

    print("\n=== Extracting GrIS atmospheric climatologies ===")
    gris_atmos = extract_gris_atmos(args.gris_atmos_dir, args.gris_grid)
    gris_out = os.path.join(args.output_dir, "GrIS_atmos_climatologies.csv")
    gris_atmos.to_csv(gris_out, index=False)
    print(f"  Saved: {gris_out}")

    print("\nDone.")


if __name__ == "__main__":
    main()
