"""Example: ISEFlow-GrIS sea level projection with uncertainty.

This script shows how to use ISEFlow to project Greenland Ice Sheet (GrIS)
sea level contributions for a single drainage basin over 2015-2100.

The inputs used here correspond to the AWI_ISSM1 ice sheet model run,
basin 1, with standard ocean forcing and medium ocean sensitivity.

Input categories
----------------
Forcings (86-element arrays, one value per year 2015-2100):
  - aST                  atmospheric surface temperature anomaly (K)
  - aSMB                 atmospheric surface mass balance anomaly (m/yr ice equivalent)
  - ocean_thermal_forcing  ocean thermal forcing (°C)
  - basin_runoff         basin-integrated runoff (m/yr)

Scalar configuration:
  - sector               GrIS drainage basin (1-6)
  - initial_year         model spin-up year
  - numerics             numerical scheme: 'fe', 'fv', 'fd', 'fd/fv'
  - ice_flow_model       ice dynamics: 'ho', 'ssa', 'sia', 'hybrid'
  - initialization       init method: 'dav','cyc/nds','sp/ndm','sp/dav','sp/das',
                                      'cyc/ndm','sp/dai','cyc/dai','sp/nds'
  - initial_smb          SMB product: 'ra3','hir','ismb','box/mar','box/ra3','mar','ra1'
  - velocity             observed velocity dataset: 'joughin', 'rignot', or 'None'
  - bedrock_topography   bed DEM: 'morlighem' or 'bamber'
  - surface_thickness    surface DEM: 'morlighem' or 'None'
  - geothermal_heat_flux GHF dataset: 'g', 'sr', 'mix', or 'None'
  - res_min              minimum grid resolution in km
  - res_max              maximum grid resolution in km
  - standard_ocean_forcing  True (Standard) or False (Open)
  - ocean_sensitivity    'low', 'medium', or 'high'
  - ice_shelf_fracture   True / False
"""

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib.pyplot as plt
import numpy as np

from ise.data.inputs import ISEFlowGrISInputs
from ise.models.iseflow import ISEFlow_GrIS

# ── 1. Define your forcing time series (86 values: 2015-2100) ─────────────────
#
# Source: AWI_ISSM1, basin 1, ISMIP6 GrIS test dataset

years = np.arange(2015, 2101)

aST = np.array(
    [
        0.76599243,
        0.62598964,
        0.97443316,
        0.54339343,
        0.46631682,
        1.31155264,
        1.41823621,
        1.29220294,
        1.12609535,
        0.79429084,
        1.51666977,
        1.16639308,
        1.55040648,
        1.37255715,
        1.54480461,
        0.73161267,
        0.33661897,
        1.57077782,
        1.6363523,
        1.60350161,
        1.96365806,
        2.21425902,
        1.11627203,
        1.43880099,
        2.33085456,
        1.99097072,
        1.88102687,
        1.85378137,
        1.79469155,
        2.46153615,
        2.01769196,
        1.87069637,
        2.00943039,
        2.82217937,
        2.02799516,
        2.56325579,
        1.73240533,
        2.75694412,
        2.16483408,
        3.17131155,
        2.89808011,
        3.42044554,
        2.8598648,
        3.35159891,
        3.68493942,
        3.64544104,
        2.61512617,
        3.55532572,
        3.08746319,
        3.23080656,
        3.72935922,
        3.68444596,
        4.17569873,
        3.40731448,
        4.22645458,
        3.02433179,
        4.40449068,
        4.02495513,
        4.03667852,
        3.79508274,
        4.86480137,
        4.72539509,
        4.79772991,
        4.25674922,
        4.58648866,
        4.35668976,
        4.84960037,
        5.46344384,
        5.45447408,
        4.74452658,
        5.34799779,
        5.25699608,
        5.39890152,
        5.87388835,
        5.45031472,
        5.6064863,
        5.89674521,
        5.74595002,
        4.90281999,
        5.33480315,
        5.81392142,
        5.46294673,
        5.38919949,
        6.1780074,
        6.6019236,
        6.09573585,
    ]
)

aSMB = np.array(
    [
        -4.88244939e-06,
        -4.40702884e-06,
        -1.75151234e-06,
        -1.75158059e-06,
        -5.00861574e-06,
        -8.59235667e-06,
        -7.41719865e-06,
        -5.40757513e-06,
        -3.33339566e-06,
        -6.12911914e-06,
        -2.90992562e-06,
        -6.74864946e-06,
        -4.83555504e-06,
        -9.38951716e-06,
        -2.52792094e-06,
        -2.50017448e-06,
        -1.42386487e-06,
        -6.74075762e-06,
        -9.78877798e-06,
        -7.79506997e-06,
        -1.22454295e-05,
        -3.21926472e-06,
        -7.57656024e-06,
        5.45221391e-07,
        -5.60095980e-06,
        -9.25485199e-07,
        -1.10425755e-05,
        -1.00116779e-05,
        -9.25352937e-06,
        -6.77730991e-06,
        -2.18657589e-07,
        -7.05282006e-06,
        -6.37312167e-06,
        -1.26531844e-05,
        -6.82976660e-06,
        -9.65697827e-06,
        -7.77117504e-06,
        -1.15005900e-05,
        -8.21258186e-06,
        -1.26085122e-05,
        -1.60423516e-05,
        -1.71881640e-05,
        -1.38700008e-05,
        -6.82293195e-06,
        -1.56657834e-05,
        -1.91001011e-05,
        -1.04439103e-05,
        -9.06912308e-06,
        -1.04552702e-05,
        -1.31218101e-05,
        -1.56786498e-05,
        -2.06624806e-05,
        -1.67670988e-05,
        -3.50210127e-06,
        -1.90032096e-05,
        -7.50955532e-06,
        -2.14099791e-05,
        -1.51829744e-05,
        -2.05508336e-05,
        -1.53635816e-05,
        -2.04972035e-05,
        -2.00040142e-05,
        -2.14181653e-05,
        -8.87183153e-06,
        -2.35584828e-05,
        -1.73711029e-05,
        -1.92541153e-05,
        -2.82522782e-05,
        -2.18096965e-05,
        -2.09796613e-05,
        -1.87388348e-05,
        -1.88446967e-05,
        -2.35881028e-05,
        -2.24427946e-05,
        -2.04336070e-05,
        -1.93465226e-05,
        -2.72620351e-05,
        -3.37945841e-05,
        -1.34800934e-05,
        -1.64026853e-05,
        -2.71319206e-05,
        -1.81893441e-05,
        -2.20792668e-05,
        -2.63553600e-05,
        -3.66465232e-05,
        -2.92694384e-05,
    ]
)

ocean_thermal_forcing = np.array(
    [
        2.2557037,
        2.2535386,
        2.2624812,
        2.2598813,
        2.2695706,
        2.2711203,
        2.2564478,
        2.2814674,
        2.2730343,
        2.2837887,
        2.2676625,
        2.267249,
        2.2475677,
        2.2469006,
        2.271228,
        2.270991,
        2.254414,
        2.2532024,
        2.27265,
        2.250156,
        2.2812278,
        2.2756546,
        2.2700915,
        2.3058012,
        2.3290966,
        2.3272586,
        2.2967117,
        2.2934797,
        2.306437,
        2.310772,
        2.3169475,
        2.3057303,
        2.3388257,
        2.3635883,
        2.3537054,
        2.3213995,
        2.3303416,
        2.3498785,
        2.370187,
        2.3862188,
        2.4045875,
        2.4177632,
        2.3829894,
        2.3779984,
        2.4039109,
        2.460082,
        2.4615393,
        2.4497287,
        2.45903,
        2.4536111,
        2.4726045,
        2.4712942,
        2.4842565,
        2.5143049,
        2.5166428,
        2.547697,
        2.5440228,
        2.6089754,
        2.620515,
        2.640196,
        2.625371,
        2.7061782,
        2.6844757,
        2.6669323,
        2.7177262,
        2.7238317,
        2.7771363,
        2.9261715,
        2.8017287,
        2.7980592,
        2.8252933,
        2.8256543,
        2.8531923,
        2.9614332,
        2.9565496,
        2.917681,
        3.0877674,
        3.1308446,
        3.0817225,
        3.1060648,
        3.1927412,
        3.1016414,
        3.247878,
        3.3295996,
        3.5182993,
        3.4760363,
    ]
)

basin_runoff = np.array(
    [
        0.01318234,
        0.01138918,
        0.00739406,
        0.00990163,
        0.01017726,
        0.01585067,
        0.01482083,
        0.01341854,
        0.00770838,
        0.01216469,
        0.01087858,
        0.01126392,
        0.0103262,
        0.01681866,
        0.00949364,
        0.00972788,
        0.0075493,
        0.01304828,
        0.01857127,
        0.01685385,
        0.02147065,
        0.01370934,
        0.01585394,
        0.00585816,
        0.017807,
        0.00983836,
        0.02161748,
        0.0167218,
        0.01866721,
        0.0153286,
        0.00800473,
        0.01640165,
        0.01374282,
        0.02818176,
        0.01428118,
        0.01922181,
        0.01428509,
        0.02549671,
        0.02023242,
        0.02838592,
        0.03033495,
        0.03673435,
        0.02914987,
        0.02009838,
        0.03435063,
        0.03747115,
        0.0239902,
        0.02065792,
        0.02404768,
        0.03011866,
        0.03448204,
        0.04042858,
        0.04111657,
        0.01621638,
        0.0453674,
        0.02035363,
        0.05527696,
        0.03221546,
        0.04225896,
        0.03094193,
        0.05639096,
        0.05228007,
        0.06085691,
        0.03425354,
        0.06061719,
        0.04398923,
        0.05305641,
        0.06149295,
        0.06460306,
        0.04932992,
        0.05861019,
        0.05973655,
        0.07045428,
        0.06048507,
        0.04888317,
        0.05445918,
        0.0741209,
        0.10829171,
        0.03956125,
        0.04903079,
        0.08828158,
        0.04489271,
        0.06072397,
        0.08483346,
        0.10218374,
        0.0862635,
    ]
)


# ── 2. Define ice sheet model configuration ───────────────────────────────────
#
# These settings describe the ISM used to generate the target projection.
# See ISEFlowGrISInputs docstring for all accepted values.

sector = 1  # GrIS drainage basin (1-6)
initial_year = 1990  # model spin-up start year

numerics = "fe"  # numerical scheme
ice_flow_model = "ho"  # ice-dynamics approximation
initialization = "dav"  # initialisation method
initial_smb = "ra3"  # SMB product used during spin-up

velocity = "joughin"  # observed surface velocity dataset
bedrock_topography = "morlighem"  # bedrock DEM
surface_thickness = "None"  # surface thickness DEM (absent for this model)
geothermal_heat_flux = "g"  # geothermal heat flux dataset

res_min = 1.0  # minimum grid resolution (km)
res_max = 7.5  # maximum grid resolution (km)

standard_ocean_forcing = True  # True = Standard, False = Open
ocean_sensitivity = "medium"  # 'low', 'medium', 'high'
ice_shelf_fracture = False


# ── 3. Build the inputs object ────────────────────────────────────────────────

inputs = ISEFlowGrISInputs(
    year=years,
    sector=sector,
    aST=aST,
    aSMB=aSMB,
    ocean_thermal_forcing=ocean_thermal_forcing,
    basin_runoff=basin_runoff,
    initial_year=initial_year,
    numerics=numerics,
    ice_flow_model=ice_flow_model,
    initialization=initialization,
    initial_smb=initial_smb,
    velocity=velocity,
    bedrock_topography=bedrock_topography,
    surface_thickness=surface_thickness,
    geothermal_heat_flux=geothermal_heat_flux,
    res_min=res_min,
    res_max=res_max,
    standard_ocean_forcing=standard_ocean_forcing,
    ocean_sensitivity=ocean_sensitivity,
    ice_shelf_fracture=ice_shelf_fracture,
)

print(inputs)  # inspect the validated, internally-encoded inputs


# ── 4. Load the pretrained model and predict ──────────────────────────────────

model = ISEFlow_GrIS(version="v1.1.0")
pred, uq = model.predict(inputs, smoothing_window=0)

pred = np.asarray(pred).squeeze()
epistemic = np.asarray(uq["epistemic"]).squeeze()
aleatoric = np.asarray(uq["aleatoric"]).squeeze()
total = epistemic + aleatoric

print(f"\nPrediction range: {pred.min():.2f} - {pred.max():.2f} mm SLE")
print(f"Mean epistemic uncertainty: {epistemic.mean():.3f} mm")
print(f"Mean aleatoric uncertainty: {aleatoric.mean():.3f} mm")


# ── 5. Plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(
    years,
    pred - total,
    pred + total,
    alpha=0.20,
    color="#d62728",
    label="Total uncertainty (epistemic + aleatoric)",
)
ax.fill_between(
    years,
    pred - epistemic,
    pred + epistemic,
    alpha=0.45,
    color="#d62728",
    label=r"Epistemic uncertainty (2$\sigma$)",
)
ax.plot(years, pred, color="#d62728", linewidth=2.0, label="ISEFlow prediction")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Sea Level Equivalent (mm SLE)", fontsize=12)
ax.set_title("ISEFlow-GrIS: AWI_ISSM1, Basin 1 (2015-2100)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(years[0], years[-1])

plt.tight_layout()
plt.savefig("example_gris_projection.png", dpi=200, bbox_inches="tight")
print("\nSaved example_gris_projection.png")
plt.show()
plt.close("all")
