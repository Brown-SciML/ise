"""Example: ISEFlow projection from raw (non-anomaly) forcing values.

This script demonstrates ``ISEFlowAISInputs.from_absolute_forcings()`` and
``ISEFlowGrISInputs.from_absolute_forcings()`` for users who have absolute forcing
values rather than pre-computed anomalies.  Anomaly conversion is handled
automatically using the existing ISMIP6 climatological baselines.

Two construction paths are shown for each ice sheet:
  A) existing ISMIP6 climatology — specify ``aogcm``
  B) Custom climatology         — specify ``custom_climatology`` (e.g. for a CMIP7 model)
"""

import numpy as np

from ise.data.anomaly import AnomalyConverter
from ise.data.inputs import ISEFlowAISInputs, ISEFlowGrISInputs
from ise.models.iseflow import ISEFlow_AIS, ISEFlow_GrIS

years = np.arange(2015, 2101)  # 86 years


# =============================================================================
# AIS — Antarctic Ice Sheet
# =============================================================================

print("=" * 70)
print("ISEFlow Example: Building Inputs from Absolute (non-anomaly) Forcings")
print("=" * 70)
print("\n──── AIS — Antarctic Ice Sheet ────────────────────────────────────")

# ── 1. Raw absolute atmospheric forcing arrays ────────────────────────────────
#
# Illustrative values representative of NorESM1-M RCP8.5, sector 10.
# All atmospheric variables must be in the same units as the ISMIP6 forcing
# files.  from_absolute_forcings() subtracts the 1995-2014 climatological
# baseline and returns anomalies in the same units.

pr_raw = np.full(86, 1.3e-5)  # precipitation            (kg m⁻² s⁻¹)
evspsbl_raw = np.full(86, 4.0e-6)  # evaporation/sublimation  (kg m⁻² s⁻¹)
smb_raw_ais = np.full(86, 9.0e-6)  # surface mass balance     (kg m⁻² s⁻¹)
ts_raw = np.full(86, 255.0)  # surface temperature      (K)

# Ocean variables are absolute values and are passed through unchanged.
otf_ais = np.linspace(2.0, 2.8, 86)  # ocean thermal forcing  (°C)
sal_ais = np.full(86, 34.35)  # ocean salinity         (PSU)
temp_ais = np.linspace(-0.4, 0.1, 86)  # ocean temperature      (°C)


# ── 2A. AIS using a existing ISMIP6 climatology ────────────────────────────────

print("\n[AIS 1/3] Converting absolute forcings via bundled ISMIP6 climatology...")
print("  Available AIS AOGCMs:", AnomalyConverter("AIS").list_aogcms())

inputs_ais_existing = ISEFlowAISInputs.from_absolute_forcings(
    year=years,
    sector=10,
    pr=pr_raw,
    evspsbl=evspsbl_raw,
    smb=smb_raw_ais,
    ts=ts_raw,
    ocean_thermal_forcing=otf_ais,
    ocean_salinity=sal_ais,
    ocean_temperature=temp_ais,
    aogcm="noresm1-m_rcp85",
    numerics="fd",
    stress_balance="hybrid",
    resolution="8",
    init_method="eq",
    initial_year=2005,
    melt_in_floating_cells="sub-grid",
    icefront_migration="str",
    ocean_forcing_type="open",
    ocean_sensitivity="medium",
    ice_shelf_fracture=False,
    open_melt_type="quad",
    standard_melt_type=None,
)

print("\n[AIS — existing climatology]")
print(inputs_ais_existing)


# ── 2B. AIS using a custom climatology ───────────────────────────────────────
#
# Provide the 1995-2014 absolute means for your AOGCM in the same units as the
# raw inputs: kg m⁻² s⁻¹ for pr / evspsbl / smb, K for ts.

print("\n[AIS 2/3] Converting absolute forcings via user-supplied climatology...")
inputs_ais_custom = ISEFlowAISInputs.from_absolute_forcings(
    year=years,
    sector=10,
    pr=pr_raw,
    evspsbl=evspsbl_raw,
    smb=smb_raw_ais,
    ts=ts_raw,
    ocean_thermal_forcing=otf_ais,
    ocean_salinity=sal_ais,
    ocean_temperature=temp_ais,
    custom_climatology={  # 1995-2014 absolute baseline means
        "pr": 1.3e-5,  # kg m⁻² s⁻¹
        "evspsbl": 4.0e-6,  # kg m⁻² s⁻¹
        "smb": 9.0e-6,  # kg m⁻² s⁻¹
        "ts": 253.7,  # K
    },
    numerics="fd",
    stress_balance="hybrid",
    resolution="8",
    init_method="eq",
    initial_year=2005,
    melt_in_floating_cells="sub-grid",
    icefront_migration="str",
    ocean_forcing_type="open",
    ocean_sensitivity="medium",
    ice_shelf_fracture=False,
    open_melt_type="quad",
    standard_melt_type=None,
)

print("\n[AIS — custom climatology]")
print(inputs_ais_custom)


# ── 3. AIS prediction ─────────────────────────────────────────────────────────

print("\n[AIS 3/3] Loading ISEFlow-AIS and predicting...")
model_ais = ISEFlow_AIS(version="v1.1.0")
pred_ais, uq_ais = model_ais.predict(inputs_ais_existing, smoothing_window=0)

pred_ais = np.asarray(pred_ais).squeeze()
ep_ais = np.asarray(uq_ais["epistemic"]).squeeze()
al_ais = np.asarray(uq_ais["aleatoric"]).squeeze()

print("\n── AIS Prediction Summary ─────────────────────────────────────────")
print(f"  Prediction range:           {pred_ais.min():>7.2f} to {pred_ais.max():>7.2f} mm SLE")
print(f"  Mean epistemic uncertainty: {ep_ais.mean():>7.3f} mm")
print(f"  Mean aleatoric uncertainty: {al_ais.mean():>7.3f} mm")


# =============================================================================
# GrIS — Greenland Ice Sheet
# =============================================================================

print("\n──── GrIS — Greenland Ice Sheet ───────────────────────────────────")

# ── 4. Raw absolute atmospheric forcing arrays ────────────────────────────────
#
# Illustrative values representative of HadGEM2-ES RCP8.5, sector 1.
#
# SMB must be in mm w.e. yr⁻¹ and ST in °C — matching the MAR 3.9 Reference
# file convention (the source of the bundled 1960-1989 climatological baseline).
# from_absolute_forcings() subtracts the 1960-1989 MAR baseline and converts
# the SMB anomaly from mm w.e. yr⁻¹ to kg m⁻² s⁻¹ (ISEFlow training units).
#
# NOTE: do NOT pass aSMB/aST values that are already anomalies (e.g. values
# read directly from ISMIP6 aSMB NetCDF files, which are in kg m⁻² s⁻¹).
# Use from_absolute_forcings() only when starting from absolute MAR output.

smb_raw_gris = np.linspace(-200.0, -350.0, 86)  # absolute SMB  (mm w.e. yr⁻¹)
st_raw = np.linspace(-20.0, -17.0, 86)  # absolute surface temperature (°C)

# Ocean variables are passed through unchanged.
# Training data thermal forcing mean ~4.7 °C; sector 1 values typically 2–6 °C.
otf_gris = np.linspace(3.5, 5.5, 86)  # ocean thermal forcing  (°C)
runoff = np.linspace(0.05, 0.20, 86)  # basin runoff           (m yr⁻¹)


# ── 5A. GrIS using a existing ISMIP6 climatology ───────────────────────────────

print("\n[GrIS 1/3] Converting absolute forcings via bundled ISMIP6 climatology...")
print("  Available GrIS AOGCMs:", AnomalyConverter("GrIS").list_aogcms())

inputs_gris_existing = ISEFlowGrISInputs.from_absolute_forcings(
    year=years,
    sector=1,
    smb=smb_raw_gris,
    st=st_raw,
    ocean_thermal_forcing=otf_gris,
    basin_runoff=runoff,
    aogcm="hadgem2-es_rcp85",
    initial_year=1990,
    numerics="fe",
    ice_flow_model="ho",
    initialization="dav",
    initial_smb="ra3",
    velocity="joughin",
    bedrock_topography="morlighem",
    surface_thickness="None",
    geothermal_heat_flux="g",
    res_min=1.0,
    res_max=7.5,
    standard_ocean_forcing=True,
    ocean_sensitivity="medium",
    ice_shelf_fracture=False,
)

print("\n[GrIS — existing climatology]")
print(inputs_gris_existing)


# ── 5B. GrIS using a custom climatology ──────────────────────────────────────
#
# Provide the 1960-1989 MAR absolute baseline means:
#   smb in mm w.e. yr⁻¹  (same units as the raw smb input above)
#   st  in °C             (same units as the raw st input above)

print("\n[GrIS 2/3] Converting absolute forcings via user-supplied climatology...")
inputs_gris_custom = ISEFlowGrISInputs.from_absolute_forcings(
    year=years,
    sector=1,
    smb=smb_raw_gris,
    st=st_raw,
    ocean_thermal_forcing=otf_gris,
    basin_runoff=runoff,
    custom_climatology={  # 1960-1989 MAR absolute baseline means
        "smb": -241.2,  # mm w.e. yr⁻¹  (matches HadGEM2-ES sector 1 baseline)
        "st": -22.8,  # °C
    },
    initial_year=1990,
    numerics="fe",
    ice_flow_model="ho",
    initialization="dav",
    initial_smb="ra3",
    velocity="joughin",
    bedrock_topography="morlighem",
    surface_thickness="None",
    geothermal_heat_flux="g",
    res_min=1.0,
    res_max=7.5,
    standard_ocean_forcing=True,
    ocean_sensitivity="medium",
    ice_shelf_fracture=False,
)

print("\n[GrIS — custom climatology]")
print(inputs_gris_custom)


# ── 6. GrIS prediction ────────────────────────────────────────────────────────

print("\n[GrIS 3/3] Loading ISEFlow-GrIS and predicting...")
model_gris = ISEFlow_GrIS(version="v1.1.0")
pred_gris, uq_gris = model_gris.predict(inputs_gris_existing, smoothing_window=0)

pred_gris = np.asarray(pred_gris).squeeze()
ep_gris = np.asarray(uq_gris["epistemic"]).squeeze()
al_gris = np.asarray(uq_gris["aleatoric"]).squeeze()

print("\n── GrIS Prediction Summary ────────────────────────────────────────")
print(f"  Prediction range:           {pred_gris.min():>7.2f} to {pred_gris.max():>7.2f} mm SLE")
print(f"  Mean epistemic uncertainty: {ep_gris.mean():>7.3f} mm")
print(f"  Mean aleatoric uncertainty: {al_gris.mean():>7.3f} mm")

print("\nDone.")
