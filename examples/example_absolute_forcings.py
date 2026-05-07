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

# ── 1. Raw absolute atmospheric forcing arrays ────────────────────────────────
#
# Illustrative values representative of NorESM1-M RCP8.5, sector 10.
# from_absolute_forcings() subtracts the 1995-2014 ISMIP6 climatological baseline.

pr_raw = np.full(86, 1.3e-5)  # precipitation        (kg m⁻² s⁻¹)
evspsbl_raw = np.full(86, 4.0e-6)  # evaporation          (kg m⁻² s⁻¹)
smb_raw_ais = np.full(86, 9.0e-6)  # surface mass balance (kg m⁻² s⁻¹)
ts_raw = np.full(86, 255.0)  # surface temperature  (K)

# Ocean variables are absolute — passed through unchanged.
otf_ais = np.linspace(2.0, 2.8, 86)  # ocean thermal forcing (°C)
sal_ais = np.full(86, 34.35)  # ocean salinity       (PSU)
temp_ais = np.linspace(-0.4, 0.1, 86)  # ocean temperature    (°C)


# ── 2A. AIS using a existing ISMIP6 climatology ────────────────────────────────

print("Available AIS AOGCMs:", AnomalyConverter("AIS").list_aogcms())

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
    custom_climatology={  # 1995-2014 baseline means for your AOGCM
        "pr": 1.3e-5,
        "evspsbl": 4.0e-6,
        "smb": 9.0e-6,
        "ts": 253.7,
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

model_ais = ISEFlow_AIS(version="v1.1.0")
pred_ais, uq_ais = model_ais.predict(inputs_ais_existing, smoothing_window=0)

pred_ais = np.asarray(pred_ais).squeeze()
ep_ais = np.asarray(uq_ais["epistemic"]).squeeze()
al_ais = np.asarray(uq_ais["aleatoric"]).squeeze()
total_ais = ep_ais + al_ais

print(f"\n[AIS] Prediction range: {pred_ais.min():.2f} – {pred_ais.max():.2f} mm SLE")
print(f"[AIS] Mean epistemic uncertainty: {ep_ais.mean():.3f} mm")
print(f"[AIS] Mean aleatoric uncertainty: {al_ais.mean():.3f} mm")


# =============================================================================
# GrIS — Greenland Ice Sheet
# =============================================================================

# ── 4. Raw absolute atmospheric forcing arrays ────────────────────────────────
#
# from_absolute_forcings() subtracts the 1960-1989 MAR baseline for smb and st.

smb_raw_gris = np.full(86, -200.0)  # raw SMB  (mm w.e. yr⁻¹)
st_raw = np.full(86, -20.0)  # raw surface temperature (°C)

# Ocean variables passed through unchanged.
otf_gris = np.linspace(2.2, 3.5, 86)  # ocean thermal forcing (°C)
runoff = np.linspace(0.01, 0.10, 86)  # basin runoff


# ── 5A. GrIS using a existing ISMIP6 climatology ───────────────────────────────

print("\nAvailable GrIS AOGCMs:", AnomalyConverter("GrIS").list_aogcms())

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

inputs_gris_custom = ISEFlowGrISInputs.from_absolute_forcings(
    year=years,
    sector=1,
    smb=smb_raw_gris,
    st=st_raw,
    ocean_thermal_forcing=otf_gris,
    basin_runoff=runoff,
    custom_climatology={  # 1960-1989 MAR baseline means for your AOGCM
        "smb": -241.2,
        "st": -22.8,
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

model_gris = ISEFlow_GrIS(version="v1.1.0")
pred_gris, uq_gris = model_gris.predict(inputs_gris_existing, smoothing_window=0)

pred_gris = np.asarray(pred_gris).squeeze()
ep_gris = np.asarray(uq_gris["epistemic"]).squeeze()
al_gris = np.asarray(uq_gris["aleatoric"]).squeeze()
print(al_gris)
total_gris = ep_gris + al_gris

print(f"\n[GrIS] Prediction range: {pred_gris.min():.2f} – {pred_gris.max():.2f} mm SLE")
print(f"[GrIS] Mean epistemic uncertainty: {ep_gris.mean():.3f} mm")
print(f"[GrIS] Mean aleatoric uncertainty: {al_gris.mean():.3f} mm")
