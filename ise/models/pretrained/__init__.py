"""Pretrained ISEFlow weight management.

Weights are hosted on HuggingFace Hub at ``Brown-SciML/ISEFlow`` and are
downloaded automatically on first use via ``huggingface_hub``.  The downloaded
files are cached in the default HuggingFace cache directory
(``~/.cache/huggingface/hub`` or ``$HF_HOME``).

During local development, if the HuggingFace download fails (e.g. no internet
access) and local weights exist under ``ise/models/pretrained/ISEFlow/``, the
loader falls back to those local paths transparently.
"""

import os

from huggingface_hub import snapshot_download

HF_REPO_ID = "Brown-SciML/ISEFlow"

ISEFLOW_LATEST_MODEL_VERSION = "v1.1.0"
"""The most recent ISEFlow pretrained model version. Distinct from the ise-py package version."""

_LOCAL_PRETRAINED_DIR = os.path.dirname(__file__)


def get_model_dir(version: str, ice_sheet: str) -> str:
    """Return the local directory containing weights for a given model version.

    Downloads the weights from HuggingFace Hub if not already cached.  Falls
    back to the bundled local path when HF is unavailable and local weights
    exist (development / air-gapped environments).

    Parameters
    ----------
    version : str
        Model version string, e.g. ``'v1.0.0'`` or ``'v1.1.0'``.
    ice_sheet : str
        Ice sheet identifier — ``'AIS'`` or ``'GrIS'``.

    Returns
    -------
    str
        Absolute path to the directory containing ``deep_ensemble.pth``,
        ``normalizing_flow.pth``, ``scaler_X.pkl``, and ``scaler_y.pkl``.
    """
    subfolder = _subfolder(version, ice_sheet)
    local_fallback = os.path.join(_LOCAL_PRETRAINED_DIR, "ISEFlow", subfolder)

    try:
        local_dir = snapshot_download(
            repo_id=HF_REPO_ID,
            allow_patterns=[f"{subfolder}/*"],
        )
        return os.path.join(local_dir, subfolder)
    except Exception:
        # Fall back to bundled weights (local dev or air-gapped HPC).
        if os.path.isdir(local_fallback):
            return local_fallback
        raise RuntimeError(
            f"Could not download weights from HuggingFace Hub ({HF_REPO_ID}) "
            f"and no local fallback found at {local_fallback}. "
            "Install huggingface_hub and ensure internet access, or place "
            "the weights at the local path."
        )


def _subfolder(version: str, ice_sheet: str) -> str:
    """Return the HuggingFace subfolder path for a given version and ice sheet."""
    tag = version.replace(".", "-")   # e.g. v1.1.0 -> v1-1-0
    return f"{version}/ISEFlow_{ice_sheet}_{tag}"


# ---------------------------------------------------------------------------
# Backward-compat path constants (kept as shims; now resolved via get_model_dir)
# ---------------------------------------------------------------------------

def _lazy_path(version: str, ice_sheet: str) -> str:
    """Resolve a model path, preferring local if it exists (avoids HF call at import)."""
    subfolder = _subfolder(version, ice_sheet)
    local = os.path.join(_LOCAL_PRETRAINED_DIR, "ISEFlow", subfolder)
    if os.path.isdir(local):
        return local
    # Return the expected local path; actual download happens in get_model_dir()
    return local


ISEFlow_AIS_v1_0_0_path = _lazy_path("v1.0.0", "AIS")
ISEFlow_GrIS_v1_0_0_path = _lazy_path("v1.0.0", "GrIS")
ISEFlow_AIS_v1_1_0_path = _lazy_path("v1.1.0", "AIS")
ISEFlow_GrIS_v1_1_0_path = _lazy_path("v1.1.0", "GrIS")


# ---------------------------------------------------------------------------
# Variable lists (unchanged — these define model feature order)
# ---------------------------------------------------------------------------

ISEFlow_AIS_v1_1_0_variables = [
    "year",
    "sector",
    "initial_year",
    "numerics_FD",
    "numerics_FE",
    "numerics_FE/FV",
    "stress_balance_HO",
    "stress_balance_Hybrid",
    "stress_balance_L1L2",
    "stress_balance_SIA_SSA",
    "stress_balance_SSA",
    "stress_balance_Stokes",
    "resolution_16",
    "resolution_20",
    "resolution_32",
    "resolution_4",
    "resolution_8",
    "resolution_variable",
    "init_method_DA",
    "init_method_DA_geom",
    "init_method_DA_relax",
    "init_method_Eq",
    "init_method_SP",
    "init_method_SP_icethickness",
    "melt_Floating_condition",
    "melt_No",
    "melt_Sub-grid",
    "ice_front_Div",
    "ice_front_Fix",
    "ice_front_MH",
    "ice_front_RO",
    "ice_front_StR",
    "open_melt_param_Lin",
    "open_melt_param_Nonlocal_Slope",
    "open_melt_param_PICO",
    "open_melt_param_PICOP",
    "open_melt_param_Plume",
    "open_melt_param_Quad",
    "standard_melt_param_Local",
    "standard_melt_param_Local_anom",
    "standard_melt_param_Nonlocal",
    "standard_melt_param_Nonlocal_anom",
    "Ocean forcing_Open",
    "Ocean forcing_Standard",
    "Ocean sensitivity_High",
    "Ocean sensitivity_Low",
    "Ocean sensitivity_Medium",
    "Ocean sensitivity_PIGL",
    "Ice shelf fracture_False",
    "Ice shelf fracture_True",
    "pr_anomaly",
    "evspsbl_anomaly",
    "smb_anomaly",
    "ts_anomaly",
    "thermal_forcing",
    "salinity",
    "temperature",
    "pr_anomaly.lag1",
    "evspsbl_anomaly.lag1",
    "smb_anomaly.lag1",
    "ts_anomaly.lag1",
    "thermal_forcing.lag1",
    "salinity.lag1",
    "temperature.lag1",
    "pr_anomaly.lag2",
    "evspsbl_anomaly.lag2",
    "smb_anomaly.lag2",
    "ts_anomaly.lag2",
    "thermal_forcing.lag2",
    "salinity.lag2",
    "temperature.lag2",
    "pr_anomaly.lag3",
    "evspsbl_anomaly.lag3",
    "smb_anomaly.lag3",
    "ts_anomaly.lag3",
    "thermal_forcing.lag3",
    "salinity.lag3",
    "temperature.lag3",
    "pr_anomaly.lag4",
    "evspsbl_anomaly.lag4",
    "smb_anomaly.lag4",
    "ts_anomaly.lag4",
    "thermal_forcing.lag4",
    "salinity.lag4",
    "temperature.lag4",
    "pr_anomaly.lag5",
    "evspsbl_anomaly.lag5",
    "smb_anomaly.lag5",
    "ts_anomaly.lag5",
    "thermal_forcing.lag5",
    "salinity.lag5",
    "temperature.lag5",
]

ISEFlow_AIS_v1_0_0_variables = [
    "year",
    "sector",
    "pr_anomaly",
    "evspsbl_anomaly",
    "mrro_anomaly",
    "smb_anomaly",
    "ts_anomaly",
    "thermal_forcing",
    "salinity",
    "temperature",
    "pr_anomaly.lag1",
    "evspsbl_anomaly.lag1",
    "mrro_anomaly.lag1",
    "smb_anomaly.lag1",
    "ts_anomaly.lag1",
    "thermal_forcing.lag1",
    "salinity.lag1",
    "temperature.lag1",
    "pr_anomaly.lag2",
    "evspsbl_anomaly.lag2",
    "mrro_anomaly.lag2",
    "smb_anomaly.lag2",
    "ts_anomaly.lag2",
    "thermal_forcing.lag2",
    "salinity.lag2",
    "temperature.lag2",
    "pr_anomaly.lag3",
    "evspsbl_anomaly.lag3",
    "mrro_anomaly.lag3",
    "smb_anomaly.lag3",
    "ts_anomaly.lag3",
    "thermal_forcing.lag3",
    "salinity.lag3",
    "temperature.lag3",
    "pr_anomaly.lag4",
    "evspsbl_anomaly.lag4",
    "mrro_anomaly.lag4",
    "smb_anomaly.lag4",
    "ts_anomaly.lag4",
    "thermal_forcing.lag4",
    "salinity.lag4",
    "temperature.lag4",
    "pr_anomaly.lag5",
    "evspsbl_anomaly.lag5",
    "mrro_anomaly.lag5",
    "smb_anomaly.lag5",
    "ts_anomaly.lag5",
    "thermal_forcing.lag5",
    "salinity.lag5",
    "temperature.lag5",
    "initial_year",
    "numerics_FD",
    "numerics_FE",
    "numerics_FE/FV",
    "stress_balance_HO",
    "stress_balance_Hybrid",
    "stress_balance_L1L2",
    "stress_balance_SIA_SSA",
    "stress_balance_SSA",
    "stress_balance_Stokes",
    "resolution_16",
    "resolution_20",
    "resolution_32",
    "resolution_4",
    "resolution_8",
    "resolution_variable",
    "init_method_DA",
    "init_method_DA_geom",
    "init_method_DA_relax",
    "init_method_Eq",
    "init_method_SP",
    "init_method_SP_icethickness",
    "melt_Floating_condition",
    "melt_No",
    "melt_Sub-grid",
    "ice_front_Div",
    "ice_front_Fix",
    "ice_front_MH",
    "ice_front_RO",
    "ice_front_StR",
    "open_melt_param_Lin",
    "open_melt_param_Nonlocal_Slope",
    "open_melt_param_PICO",
    "open_melt_param_PICOP",
    "open_melt_param_Plume",
    "open_melt_param_Quad",
    "standard_melt_param_Local",
    "standard_melt_param_Local_anom",
    "standard_melt_param_Nonlocal",
    "standard_melt_param_Nonlocal_anom",
    "Ocean forcing_Open",
    "Ocean forcing_Standard",
    "Ocean sensitivity_High",
    "Ocean sensitivity_Low",
    "Ocean sensitivity_Medium",
    "Ocean sensitivity_PIGL",
    "Ice shelf fracture_False",
    "Ice shelf fracture_True",
]


ISEFlow_GrIS_v1_0_0_variables = []
ISEFlow_GrIS_v1_1_0_variables = [
    "year",
    "sector",
    "initial_year",
    "numerics_FD",
    "numerics_FD_FV5",
    "numerics_FE",
    "numerics_FV",
    "ice_flow_HO",
    "ice_flow_HYB",
    "ice_flow_SIA",
    "ice_flow_SSA",
    "initialization_CYC_DAI",
    "initialization_CYC_NDM",
    "initialization_CYC_NDS",
    "initialization_DAV",
    "initialization_SP_DAI",
    "initialization_SP_DAS",
    "initialization_SP_DAV",
    "initialization_SP_NDM",
    "initialization_SP_NDS",
    "initial_smb_BOX_MAR",
    "initial_smb_BOX_RA3",
    "initial_smb_HIR",
    "initial_smb_ISMB",
    "initial_smb_MAR",
    "initial_smb_RA1",
    "initial_smb_RA3",
    "velocity_J",
    "velocity_RM",
    "bed_B",
    "bed_M",
    "surface_thickness_M",
    "ghf_G",
    "ghf_MIX",
    "ghf_SR",
    "res_min_0.2",
    "res_min_0.25",
    "res_min_0.5",
    "res_min_0.75",
    "res_min_0.9",
    "res_min_1.0",
    "res_min_1.2",
    "res_min_2.0",
    "res_min_3.0",
    "res_min_4.0",
    "res_min_5.0",
    "res_min_8.0",
    "res_min_16.0",
    "res_max_0.9",
    "res_max_2.0",
    "res_max_4.0",
    "res_max_4.8",
    "res_max_5.0",
    "res_max_7.5",
    "res_max_8.0",
    "res_max_14.0",
    "res_max_15.0",
    "res_max_16.0",
    "res_max_20.0",
    "res_max_25.0",
    "res_max_30.0",
    "Ocean forcing_Standard",
    "Ocean sensitivity_High",
    "Ocean sensitivity_Low",
    "Ocean sensitivity_Medium",
    "Ice shelf fracture_False",
    "aSMB",
    "aST",
    "thermal_forcing",
    "basin_runoff",
    "aSMB.lag1",
    "aST.lag1",
    "thermal_forcing.lag1",
    "basin_runoff.lag1",
    "aSMB.lag2",
    "aST.lag2",
    "thermal_forcing.lag2",
    "basin_runoff.lag2",
    "aSMB.lag3",
    "aST.lag3",
    "thermal_forcing.lag3",
    "basin_runoff.lag3",
    "aSMB.lag4",
    "aST.lag4",
    "thermal_forcing.lag4",
    "basin_runoff.lag4",
    "aSMB.lag5",
    "aST.lag5",
    "thermal_forcing.lag5",
    "basin_runoff.lag5",
]
