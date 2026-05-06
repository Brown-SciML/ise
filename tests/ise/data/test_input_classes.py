"""Pytest tests: ISEFlowAISInputs and ISEFlowGrISInputs produce identical feature
tensors to direct test-data inference.

NOTE: These tests require external dataset files and are skipped automatically
when those paths are not present on the current machine.

For each ice sheet we pick held-out projections from test.csv and verify that the
preprocessed feature tensor produced by the inputs class pipeline (Path B) is
numerically identical to the tensor produced by directly loading the already-processed
test CSV (Path A).

Path A: test.csv → get_X_y → numpy array  (reference, what the comparison script does)
Path B: unscale continuous columns → decode ISM config → build inputs dataclass
        → ISEFlow_AIS/GrIS.process() → numpy array
"""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

AIS_DATA_DIR_CHECK = "/oscar/home/pvankatw/research/ise/supplemental/dataset/AIS_slc"
GRIS_DATA_DIR_CHECK = "/oscar/home/pvankatw/research/ise/supplemental/dataset/GrIS_slc"
_datasets_available = os.path.isfile(f"{AIS_DATA_DIR_CHECK}/test.csv") and os.path.isfile(
    f"{GRIS_DATA_DIR_CHECK}/test.csv"
)
pytestmark = pytest.mark.skipif(
    not _datasets_available,
    reason="External dataset files not present on this machine",
)

from ise.data.inputs import ISEFlowAISInputs, ISEFlowGrISInputs
from ise.models.iseflow import ISEFlow_AIS, ISEFlow_GrIS
from ise.models.pretrained import (
    ISEFlow_AIS_v1_1_0_path,
    ISEFlow_AIS_v1_1_0_variables,
    ISEFlow_GrIS_v1_1_0_path,
    ISEFlow_GrIS_v1_1_0_variables,
)
from ise.utils.functions import get_X_y

AIS_DATA_DIR = "/oscar/home/pvankatw/research/ise/supplemental/dataset/AIS_slc"
GRIS_DATA_DIR = "/oscar/home/pvankatw/research/ise/supplemental/dataset/GrIS_slc"
PROJ_LEN = 86
N_PROJ = 5
TOLERANCE = 1e-6


# ── Helpers ────────────────────────────────────────────────────────────────────


def unscale_continuous(proj_df, scaler_path):
    scaler = pickle.load(open(scaler_path, "rb"))
    cols = scaler.get_feature_names_out()
    arr = scaler.inverse_transform(proj_df[cols].values)
    return {col: arr[:, i] for i, col in enumerate(cols)}


def decode_one_hot(proj_df, prefix):
    cols = [c for c in proj_df.columns if c.startswith(prefix + "_")]
    row = proj_df.iloc[0]
    true_cols = [c for c in cols if row[c] is True or row[c] == 1]
    if len(true_cols) == 0:
        return None
    if len(true_cols) > 1:
        raise ValueError(f"Multiple True columns for prefix '{prefix}': {true_cols}")
    return true_cols[0][len(prefix) + 1 :]


def pick_proj_indices(n_rows, n_samples=N_PROJ, seed=1):
    rng = np.random.default_rng(seed)
    n_proj = n_rows // PROJ_LEN
    chosen = rng.choice(n_proj, size=min(n_samples, n_proj), replace=False)
    chosen.sort()
    return [int(i) for i in chosen]


# ── AIS helpers ────────────────────────────────────────────────────────────────

AIS_INVERSE_ARG_MAP = {
    "numerics": {"FD": "fd", "FE": "fe", "FE/FV": "fe/fv"},
    "stress_balance": {
        "HO": "ho",
        "Hybrid": "hybrid",
        "L1L2": "l1l2",
        "SIA_SSA": "sia+ssa",
        "SSA": "ssa",
        "Stokes": "stokes",
    },
    "init_method": {
        "DA": "da",
        "DA_geom": "da*",
        "DA_relax": "da+",
        "Eq": "eq",
        "SP": "sp",
        "SP_icethickness": "sp+",
    },
    "melt": {"Floating_condition": "floating condition", "Sub-grid": "sub-grid", "No": "No"},
    "ice_front": {"StR": "str", "Fix": "fix", "MH": "mh", "RO": "ro", "Div": "div"},
    "Ocean_forcing": {"Open": "open", "Standard": "standard"},
    "Ocean_sensitivity": {"High": "high", "Low": "low", "Medium": "medium", "PIGL": "pigl"},
    "open_melt_param": {
        "Lin": "lin",
        "Quad": "quad",
        "Nonlocal_Slope": "nonlocal+slope",
        "PICO": "pico",
        "PICOP": "picop",
        "Plume": "plume",
    },
    "standard_melt_param": {
        "Local": "local",
        "Nonlocal": "nonlocal",
        "Local_anom": "local anom",
        "Nonlocal_anom": "nonlocal anom",
    },
}


def decode_ais_config(proj_df):
    cfg = {}
    for field, prefix in [
        ("numerics", "numerics"),
        ("stress_balance", "stress_balance"),
        ("init_method", "init_method"),
        ("melt", "melt"),
        ("ice_front", "ice_front"),
        ("open_melt_param", "open_melt_param"),
        ("standard_melt_param", "standard_melt_param"),
    ]:
        suffix = decode_one_hot(proj_df, prefix)
        cfg[field] = AIS_INVERSE_ARG_MAP[field].get(suffix, suffix) if suffix else "None"

    cfg["ocean_forcing_type"] = AIS_INVERSE_ARG_MAP["Ocean_forcing"].get(
        decode_one_hot(proj_df, "Ocean forcing")
    )
    cfg["ocean_sensitivity"] = AIS_INVERSE_ARG_MAP["Ocean_sensitivity"].get(
        decode_one_hot(proj_df, "Ocean sensitivity")
    )
    isf_suffix = decode_one_hot(proj_df, "Ice shelf fracture")
    cfg["ice_shelf_fracture"] = isf_suffix == "True"
    return cfg


def build_ais_inputs(proj_df):
    raw = unscale_continuous(proj_df, f"{ISEFlow_AIS_v1_1_0_path}/scaler_X.pkl")
    cfg = decode_ais_config(proj_df)
    res_suffix = decode_one_hot(proj_df, "resolution")

    return ISEFlowAISInputs(
        year=np.arange(2015, 2101),
        sector=int(round(raw["sector"][0])),
        pr_anomaly=raw["pr_anomaly"],
        evspsbl_anomaly=raw["evspsbl_anomaly"],
        smb_anomaly=raw["smb_anomaly"],
        ts_anomaly=raw["ts_anomaly"],
        ocean_thermal_forcing=raw["thermal_forcing"],
        ocean_salinity=raw["salinity"],
        ocean_temperature=raw["temperature"],
        initial_year=int(round(raw["initial_year"][0])),
        numerics=cfg["numerics"],
        stress_balance=cfg["stress_balance"],
        resolution=res_suffix,
        init_method=cfg["init_method"],
        melt_in_floating_cells=cfg["melt"],
        icefront_migration=cfg["ice_front"],
        ocean_forcing_type=cfg["ocean_forcing_type"],
        ocean_sensitivity=cfg["ocean_sensitivity"],
        ice_shelf_fracture=cfg["ice_shelf_fracture"],
        open_melt_type=cfg["open_melt_param"],
        standard_melt_type=cfg["standard_melt_param"],
    )


# ── GrIS helpers ───────────────────────────────────────────────────────────────

GRIS_INVERSE_ARG_MAP = {
    "numerics": {"FD": "fd", "FD_FV5": "fd/fv", "FE": "fe", "FV": "fv"},
    "ice_flow": {"HO": "ho", "HYB": "hybrid", "SIA": "sia", "SSA": "ssa"},
    "initialization": {
        "CYC_DAI": "cyc/dai",
        "CYC_NDM": "cyc/ndm",
        "CYC_NDS": "cyc/nds",
        "DAV": "dav",
        "SP_DAI": "sp/dai",
        "SP_DAS": "sp/das",
        "SP_DAV": "sp/dav",
        "SP_NDM": "sp/ndm",
        "SP_NDS": "sp/nds",
    },
    "initial_smb": {
        "BOX_MAR": "box/mar",
        "BOX_RA3": "box/ra3",
        "HIR": "hir",
        "ISMB": "ismb",
        "MAR": "mar",
        "RA1": "ra1",
        "RA3": "ra3",
    },
    "velocity": {"J": "joughin", "RM": "rignot"},
    "bed": {"B": "bamber", "M": "morlighem"},
    "surface_thickness": {"M": "morlighem"},
    "ghf": {"G": "g", "MIX": "mix", "SR": "sr"},
    "Ocean_forcing": {"Standard": True, "Open": False},
    "Ocean_sensitivity": {"High": "high", "Low": "low", "Medium": "medium"},
}


def decode_gris_config(proj_df):
    cfg = {}
    for field, prefix in [
        ("numerics", "numerics"),
        ("ice_flow", "ice_flow"),
        ("initialization", "initialization"),
        ("initial_smb", "initial_smb"),
        ("velocity", "velocity"),
        ("bed", "bed"),
        ("ghf", "ghf"),
    ]:
        suffix = decode_one_hot(proj_df, prefix)
        cfg[field] = GRIS_INVERSE_ARG_MAP[field].get(suffix, suffix) if suffix else None

    st_suffix = decode_one_hot(proj_df, "surface_thickness")
    cfg["surface_thickness"] = (
        GRIS_INVERSE_ARG_MAP["surface_thickness"].get(st_suffix, st_suffix) if st_suffix else "None"
    )

    cfg["res_min"] = float(decode_one_hot(proj_df, "res_min") or 1.0)
    cfg["res_max"] = float(decode_one_hot(proj_df, "res_max") or 5.0)

    of_suffix = decode_one_hot(proj_df, "Ocean forcing")
    cfg["standard_ocean_forcing"] = GRIS_INVERSE_ARG_MAP["Ocean_forcing"].get(of_suffix)
    os_suffix = decode_one_hot(proj_df, "Ocean sensitivity")
    cfg["ocean_sensitivity"] = GRIS_INVERSE_ARG_MAP["Ocean_sensitivity"].get(os_suffix, os_suffix)
    isf_suffix = decode_one_hot(proj_df, "Ice shelf fracture")
    cfg["ice_shelf_fracture"] = isf_suffix == "True"
    return cfg


def build_gris_inputs(proj_df):
    raw = unscale_continuous(proj_df, f"{ISEFlow_GrIS_v1_1_0_path}/scaler_X.pkl")
    cfg = decode_gris_config(proj_df)

    return ISEFlowGrISInputs(
        year=np.arange(2015, 2101),
        sector=int(round(raw["sector"][0])),
        aST=raw["aST"],
        aSMB=raw["aSMB"],
        ocean_thermal_forcing=raw["thermal_forcing"],
        basin_runoff=raw["basin_runoff"],
        ice_shelf_fracture=cfg["ice_shelf_fracture"],
        ocean_sensitivity=cfg["ocean_sensitivity"],
        standard_ocean_forcing=cfg["standard_ocean_forcing"],
        initial_year=int(round(raw["initial_year"][0])),
        numerics=cfg["numerics"],
        ice_flow_model=cfg["ice_flow"],
        initialization=cfg["initialization"],
        initial_smb=cfg["initial_smb"],
        velocity=cfg["velocity"],
        bedrock_topography=cfg["bed"],
        surface_thickness=cfg["surface_thickness"],
        geothermal_heat_flux=cfg["ghf"],
        res_min=cfg["res_min"],
        res_max=cfg["res_max"],
    )


# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def ais_model():
    return ISEFlow_AIS(version="v1.1.0")


@pytest.fixture(scope="module")
def gris_model():
    return ISEFlow_GrIS(version="v1.1.0")


@pytest.fixture(scope="module")
def ais_test_df():
    return pd.read_csv(f"{AIS_DATA_DIR}/test.csv")


@pytest.fixture(scope="module")
def gris_test_df():
    return pd.read_csv(f"{GRIS_DATA_DIR}/test.csv")


def ais_proj_indices(test_df):
    return pick_proj_indices(len(test_df), N_PROJ)


def gris_proj_indices(test_df):
    return pick_proj_indices(len(test_df), N_PROJ)


# ── AIS tests ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "proj_idx", pick_proj_indices(len(pd.read_csv(f"{AIS_DATA_DIR}/test.csv")), N_PROJ)
)
def test_ais_inputs_tensor_matches_reference(proj_idx, ais_model, ais_test_df):
    """Feature tensor from ISEFlowAISInputs must match the reference test CSV tensor."""
    start = proj_idx * PROJ_LEN
    proj_df = ais_test_df.iloc[start : start + PROJ_LEN].copy()

    X_a, _ = get_X_y(proj_df, dataset_type="sectors", return_format="numpy")
    inputs = build_ais_inputs(proj_df)
    X_b_df = ais_model.process(inputs)
    X_b = X_b_df.values.astype(float)
    cols_b = list(X_b_df.columns)

    assert cols_b == ISEFlow_AIS_v1_1_0_variables, (
        f"proj {proj_idx}: column mismatch\n"
        f"  only in reference : {sorted(set(ISEFlow_AIS_v1_1_0_variables) - set(cols_b))}\n"
        f"  only in inputs    : {sorted(set(cols_b) - set(ISEFlow_AIS_v1_1_0_variables))}"
    )

    diffs = {}
    for col in ISEFlow_AIS_v1_1_0_variables:
        idx_a = ISEFlow_AIS_v1_1_0_variables.index(col)
        idx_b = cols_b.index(col)
        diff = float(np.max(np.abs(X_a[:, idx_a].astype(float) - X_b[:, idx_b].astype(float))))
        if diff > TOLERANCE:
            diffs[col] = diff

    assert not diffs, (
        f"proj {proj_idx} ({proj_df.model.iloc[0]}): "
        f"feature tensor differs in {len(diffs)} column(s):\n"
        + "\n".join(
            f"  {col}: max_diff={d:.8f}" for col, d in sorted(diffs.items(), key=lambda x: -x[1])
        )
    )


# ── GrIS tests ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "proj_idx", pick_proj_indices(len(pd.read_csv(f"{GRIS_DATA_DIR}/test.csv")), N_PROJ)
)
def test_gris_inputs_tensor_matches_reference(proj_idx, gris_model, gris_test_df):
    """Feature tensor from ISEFlowGrISInputs must match the reference test CSV tensor."""
    start = proj_idx * PROJ_LEN
    proj_df = gris_test_df.iloc[start : start + PROJ_LEN].copy()

    X_a, _ = get_X_y(proj_df, dataset_type="sectors", return_format="numpy")
    inputs = build_gris_inputs(proj_df)
    X_b_df = gris_model.process(inputs)
    X_b = X_b_df.values.astype(float)
    cols_b = list(X_b_df.columns)

    assert cols_b == ISEFlow_GrIS_v1_1_0_variables, (
        f"proj {proj_idx}: column mismatch\n"
        f"  only in reference : {sorted(set(ISEFlow_GrIS_v1_1_0_variables) - set(cols_b))}\n"
        f"  only in inputs    : {sorted(set(cols_b) - set(ISEFlow_GrIS_v1_1_0_variables))}"
    )

    diffs = {}
    for col in ISEFlow_GrIS_v1_1_0_variables:
        idx_a = ISEFlow_GrIS_v1_1_0_variables.index(col)
        idx_b = cols_b.index(col)
        diff = float(np.max(np.abs(X_a[:, idx_a].astype(float) - X_b[:, idx_b].astype(float))))
        if diff > TOLERANCE:
            diffs[col] = diff

    assert not diffs, (
        f"proj {proj_idx} ({proj_df.model.iloc[0]}): "
        f"feature tensor differs in {len(diffs)} column(s):\n"
        + "\n".join(
            f"  {col}: max_diff={d:.8f}" for col, d in sorted(diffs.items(), key=lambda x: -x[1])
        )
    )
