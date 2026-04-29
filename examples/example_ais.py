"""Example: ISEFlow-AIS sea level projection with uncertainty.

This script shows how to use ISEFlow to project Antarctic Ice Sheet (AIS)
sea level contributions for a single drainage sector over 2015-2100.

The inputs used here correspond to the AWI_PISM1 ice sheet model run,
sector 10, with open-ocean forcing and medium ocean sensitivity.

Input categories
----------------
Forcings (86-element arrays, one value per year 2015-2100):
  - pr_anomaly           precipitation anomaly (kg/m²/s)
  - evspsbl_anomaly      evaporation anomaly (kg/m²/s)
  - smb_anomaly          surface mass balance anomaly (kg/m²/s)
  - ts_anomaly           surface temperature anomaly (K)
  - ocean_thermal_forcing  ocean thermal forcing (°C)
  - ocean_salinity       ocean salinity (PSU)
  - ocean_temperature    ocean temperature (°C)

Scalar configuration:
  - sector               AIS drainage sector (1-18)
  - initial_year         model spin-up year
  - numerics             numerical scheme: 'fe', 'fd', or 'fe/fv'
  - stress_balance       ice dynamics: 'ho', 'hybrid', 'l1l2', 'sia+ssa', 'ssa', 'stokes'
  - resolution           grid resolution in km: '4', '8', '16', '20', '32', 'variable'
  - init_method          initialisation: 'da', 'da*', 'da+', 'eq', 'sp', 'sp+'
  - melt_in_floating_cells  sub-shelf melt treatment: 'floating condition', 'sub-grid', 'No'
  - icefront_migration   calving scheme: 'str', 'fix', 'mh', 'ro', 'div'
  - ocean_forcing_type   'open' or 'standard'
  - ocean_sensitivity    'low', 'medium', 'high', or 'pigl'
  - ice_shelf_fracture   True / False
  - open_melt_type       (if ocean_forcing_type='open')  'lin','quad','nonlocal+slope','pico','picop','plume'
  - standard_melt_type   (if ocean_forcing_type='standard') 'local','nonlocal','local anom','nonlocal anom'
"""

import numpy as np
import matplotlib.pyplot as plt

from ise.data.inputs import ISEFlowAISInputs
from ise.models.iseflow import ISEFlow_AIS


# ── 1. Define your forcing time series (86 values: 2015-2100) ─────────────────
#
# Source: AWI_PISM1, sector 10, ISMIP6 AIS test dataset

# Note: ISE has the capability to calculate the anomaly from absolute forcing values using an 
# existing climatology already in ISMIP6 or using a custom climatology for each forcing variable.
# see: examples/example_raw_values.py

years = np.arange(2015, 2101)

pr_anomaly = np.array([
    -1.6400566e-06, -2.1999982e-07,  7.6758030e-07,  2.6185293e-07,
    -1.3948036e-06, -1.4264520e-06, -2.6079160e-07, -4.2344456e-07,
     4.0821234e-07, -2.4092176e-06,  2.3274938e-06, -4.6353244e-07,
     1.8525886e-06,  9.8904950e-07, -1.3240235e-07, -1.4854892e-06,
    -2.9465814e-09,  3.8920507e-07,  9.2896990e-07,  7.9101850e-08,
    -1.7960203e-06, -6.0460620e-07, -5.7158990e-07,  1.4028686e-06,
    -2.0658746e-07, -1.1332535e-06, -1.8512807e-06, -5.0665324e-07,
    -1.3608745e-06,  1.8238069e-06,  9.3821353e-07, -8.9124380e-07,
     3.9346392e-07,  7.5351585e-07,  2.7523367e-06,  1.9894949e-06,
     6.7793130e-07,  2.3273110e-06,  6.3021000e-07,  1.5951164e-06,
    -1.6798366e-07, -1.1491868e-06,  1.2750229e-06,  8.7001246e-07,
    -4.6929793e-07,  2.6497670e-07,  1.9915778e-06,  2.5839543e-06,
     2.8216210e-06,  6.2270460e-07, -1.6441803e-07,  5.2135470e-07,
     1.9745640e-06,  2.4688522e-06, -7.8840780e-07,  1.2168525e-06,
    -3.3674546e-08,  2.8960078e-06,  3.2701370e-06,  7.7339380e-07,
     1.3480920e-06,  8.6018540e-07,  1.2928897e-06,  2.7470242e-06,
     3.8249350e-07,  5.2444090e-07,  1.8498152e-06,  3.0102335e-06,
     2.6368334e-06,  1.3224250e-06,  1.8799008e-06,  1.9637523e-06,
     2.7531540e-07,  1.9345719e-06,  4.1641746e-07,  4.0339205e-06,
     3.0471150e-07,  3.1532372e-06,  2.7657197e-06,  2.2105937e-06,
     1.1081444e-07,  1.8135511e-06, -5.5171070e-07,  2.8644738e-06,
     2.8696074e-06,  2.7308383e-06,
])

evspsbl_anomaly = np.array([
     2.9472585e-07, -3.7120305e-08,  1.6690375e-07,  1.0232811e-07,
    -2.2143861e-07, -1.6888119e-07,  2.6765787e-07,  1.3739545e-07,
     4.7752817e-08, -2.6096800e-07, -3.0310582e-07, -3.7657685e-07,
    -8.0825245e-08, -6.9101490e-07,  1.5406216e-07, -2.2880725e-08,
     7.6673750e-08, -1.5699574e-07, -3.8059620e-07, -3.4737300e-07,
    -3.9191298e-07, -2.9903575e-07, -5.8531940e-07, -7.2624135e-07,
    -1.4328502e-07,  6.3565430e-08, -1.9046072e-07, -2.0602312e-07,
     4.8268507e-08,  8.3778640e-07, -7.1713595e-09,  1.0390612e-07,
    -1.4090128e-07, -4.0054890e-08,  2.1894069e-09,  1.8802469e-07,
    -1.8900369e-07, -2.1352450e-07,  2.0693055e-07,  3.7618156e-07,
     2.2717776e-07, -5.1382910e-07, -3.7058786e-07, -4.5135820e-07,
    -2.3209606e-07, -6.1302336e-07, -4.1922390e-07, -3.2677520e-07,
    -6.0491936e-07, -4.8387080e-07, -5.6329170e-07, -4.6965798e-07,
    -3.0161706e-07, -2.5008455e-07, -6.6772260e-07, -9.8716484e-08,
    -2.8605680e-07, -5.1713863e-07,  1.0566101e-07, -9.7980795e-08,
    -2.9422475e-07, -4.2029157e-07, -2.6673862e-07, -1.2897866e-07,
    -3.9477774e-07, -1.8837125e-07, -4.5400574e-07,  8.1054180e-08,
     1.1647152e-07, -5.9752960e-10, -2.7718320e-08, -1.1219626e-07,
     5.5040132e-08, -2.6345091e-07, -1.4753947e-07,  5.2483685e-08,
    -3.1678210e-07, -5.9635870e-08, -3.1209288e-07, -1.3939182e-07,
    -2.3019620e-07, -3.4749040e-07, -1.9907140e-07, -3.0536452e-07,
    -7.5298310e-08, -1.6552540e-07,
])

smb_anomaly = np.array([
    -2.3041382e-06, -3.7996642e-07,  6.0597216e-07,  2.1300339e-07,
    -1.1277913e-06, -1.1584839e-06, -4.8250670e-07, -5.0403960e-07,
     1.6898457e-07, -2.3249304e-06,  2.6999887e-06, -3.7893656e-08,
     1.8800846e-06,  1.6622974e-06, -2.1837637e-07, -1.4143999e-06,
    -5.8160152e-09,  6.6377726e-07,  1.4232135e-06,  4.2521773e-07,
    -1.4118608e-06, -2.5528408e-07,  7.1243576e-08,  2.1215285e-06,
    -1.2339984e-07, -1.3102813e-06, -1.6565400e-06, -2.8771050e-07,
    -1.3443475e-06,  1.0081134e-06,  7.8204630e-07, -9.6348870e-07,
     5.4227410e-07,  6.2559840e-07,  2.7091126e-06,  1.8493159e-06,
     9.2525624e-07,  2.5560320e-06,  2.5610098e-07,  7.2425080e-07,
    -5.7703414e-07, -8.7050420e-07,  1.5913694e-06,  1.2124375e-06,
    -3.0810673e-07,  7.8682490e-07,  2.3451125e-06,  2.8834327e-06,
     3.4311220e-06,  8.3402176e-07,  2.7932245e-07,  6.1085920e-07,
     2.1412063e-06,  2.6238802e-06, -3.8182637e-07,  1.3223269e-06,
     2.7565770e-07,  3.3289070e-06,  2.8733643e-06,  6.0582823e-07,
     1.4209370e-06,  1.2566231e-06,  1.4970857e-06,  2.8474822e-06,
     7.8407743e-07,  7.4587920e-07,  2.3227265e-06,  2.9149319e-06,
     2.3380599e-06,  7.2033450e-07,  1.7711625e-06,  1.9699120e-06,
     1.5028529e-07,  2.0894865e-06,  3.4902380e-07,  3.8564363e-06,
     5.4928570e-07,  3.1554500e-06,  2.9073387e-06,  2.2688305e-06,
     3.2889312e-07,  1.7756694e-06, -7.9513490e-07,  2.7754460e-06,
     2.7803608e-06,  2.8388529e-06,
])

ts_anomaly = np.array([
    -0.08362333, -0.39000508, -0.0056171 ,  0.13071519,  0.7874273 ,
     0.20308678,  0.73929846,  0.589319  ,  0.6784226 , -0.748556  ,
     1.4191982 ,  0.21627003,  0.4959335 , -0.27583057,  0.78854257,
     0.50395536,  1.0756247 ,  0.37411672,  0.5913609 ,  0.38890645,
    -0.24515022,  0.92847174, -0.2882686 ,  0.42179376,  0.6944764 ,
     0.4614343 , -0.36278176, -0.2140003 , -0.15351659,  1.120452  ,
     1.8230755 ,  0.42654493,  1.0635933 ,  0.547486  ,  0.8574605 ,
     1.5399394 ,  0.70799303,  1.3567361 ,  1.139115  ,  1.5344971 ,
     1.5081264 ,  0.64018404,  1.0175732 ,  1.4133884 ,  0.36330426,
     0.39781907,  1.0884428 ,  2.025764  ,  1.6668899 ,  1.1605837 ,
    -0.20198044,  1.3416494 ,  1.6859697 ,  1.9286752 ,  0.9667812 ,
     2.2782655 ,  1.289731  ,  1.4012108 ,  2.2198727 ,  0.96441305,
     1.7107161 ,  2.0374937 ,  1.5118704 ,  2.1128793 ,  1.1588762 ,
     1.149104  ,  1.6590308 ,  2.6382358 ,  2.8511174 ,  2.3122652 ,
     2.1512837 ,  2.4978833 ,  2.5007813 ,  2.9650383 ,  2.068834  ,
     2.6906083 ,  1.5326387 ,  3.0714102 ,  2.6247933 ,  2.5673249 ,
     1.9602505 ,  2.6207566 ,  1.6389945 ,  2.8242822 ,  3.1617858 ,
     2.8438585 ,
])

ocean_thermal_forcing = np.array([
    2.1680412, 2.170162 , 2.1817477, 2.1663392, 2.1708064, 2.1465223,
    2.1699207, 2.1922095, 2.2332432, 2.2265897, 2.2399676, 2.2667139,
    2.2461202, 2.2866673, 2.2708907, 2.2411802, 2.2226512, 2.2160902,
    2.2344103, 2.2297437, 2.2077575, 2.1991105, 2.1713085, 2.178022 ,
    2.1910417, 2.1750712, 2.1884494, 2.2018213, 2.2049217, 2.2295897,
    2.234256 , 2.222563 , 2.2419615, 2.2763286, 2.3009527, 2.3327084,
    2.3409467, 2.3242614, 2.3133354, 2.3115368, 2.3356006, 2.3646083,
    2.3823545, 2.3927965, 2.4041023, 2.4251297, 2.4164407, 2.4401886,
    2.418774 , 2.434123 , 2.4511483, 2.4344945, 2.4622247, 2.4834936,
    2.4654496, 2.4556086, 2.469399 , 2.4630558, 2.4530985, 2.4297872,
    2.4382277, 2.4531705, 2.455593 , 2.466802 , 2.4827576, 2.502902 ,
    2.5106611, 2.5342164, 2.5689902, 2.568011 , 2.5545046, 2.5667593,
    2.5539296, 2.5537274, 2.5625591, 2.5870113, 2.6061954, 2.6174629,
    2.6494575, 2.668021 , 2.6761084, 2.6678755, 2.6543903, 2.6671207,
    2.6807907, 2.6931725,
])

ocean_salinity = np.array([
    34.389538, 34.389057, 34.39486 , 34.385185, 34.392254, 34.371475,
    34.374756, 34.358517, 34.37188 , 34.398067, 34.371258, 34.37017 ,
    34.355804, 34.365692, 34.363487, 34.36715 , 34.371445, 34.33682 ,
    34.343346, 34.33314 , 34.35247 , 34.336525, 34.30353 , 34.306984,
    34.336494, 34.326717, 34.33479 , 34.34624 , 34.32816 , 34.33891 ,
    34.3728  , 34.352306, 34.31656 , 34.30453 , 34.34002 , 34.326824,
    34.337467, 34.326744, 34.305897, 34.29877 , 34.286705, 34.29329 ,
    34.296703, 34.312847, 34.323753, 34.30916 , 34.299652, 34.294422,
    34.303474, 34.269184, 34.2959  , 34.28812 , 34.269283, 34.31771 ,
    34.28669 , 34.27069 , 34.27843 , 34.31067 , 34.32557 , 34.305126,
    34.29193 , 34.25153 , 34.235672, 34.23573 , 34.24447 , 34.25463 ,
    34.260883, 34.252884, 34.250195, 34.258457, 34.248108, 34.2599  ,
    34.25883 , 34.229477, 34.18652 , 34.210697, 34.216785, 34.239853,
    34.2245  , 34.20819 , 34.200573, 34.248245, 34.244026, 34.24515 ,
    34.228817, 34.239204,
])

ocean_temperature = np.array([
    -0.4162523 , -0.41410366, -0.40285054, -0.4177059 , -0.4136424 ,
    -0.43673837, -0.41352758, -0.39031062, -0.3500413 , -0.35819298,
    -0.34328154, -0.316474  , -0.336246  , -0.29626423, -0.31191427,
    -0.3418344 , -0.36060843, -0.36518967, -0.34724262, -0.35132527,
    -0.37441695, -0.38215193, -0.40806693, -0.4015506 , -0.39021823,
    -0.40562946, -0.3927136 , -0.37999615, -0.3758623 , -0.35180902,
    -0.34908056, -0.35960197, -0.33815965, -0.30310515, -0.28051075,
    -0.24800082, -0.24037142, -0.25644347, -0.2661778 , -0.267569  ,
    -0.24281633, -0.2141854 , -0.19663481, -0.18711565, -0.17643382,
    -0.15457262, -0.16271828, -0.13867171, -0.1606034 , -0.14329441,
    -0.12779608, -0.14400613, -0.11519954, -0.09669887, -0.11297009,
    -0.1218965 , -0.10854833, -0.11673342, -0.12754185, -0.14968489,
    -0.14049022, -0.12323911, -0.11991028, -0.10870435, -0.09324841,
    -0.07368457, -0.06628366, -0.04227159, -0.00734476, -0.00879619,
    -0.0217112 , -0.01013045, -0.02289875, -0.02142386, -0.01013852,
     0.01293292,  0.03176869,  0.04171814,  0.0745884 ,  0.0940826 ,
     0.10260472,  0.09164981,  0.07840611,  0.0910715 ,  0.10567508,
     0.11746211,
])


# ── 2. Define ice sheet model configuration ───────────────────────────────────
#
# These settings describe the ISM used to generate the target projection.
# See ISEFlowAISInputs docstring for all accepted values.

sector        = 10          # AIS drainage sector (1-18)

initial_year  = 2005        # model spin-up start year
numerics      = 'fd'        # numerical scheme
stress_balance = 'hybrid'   # ice-dynamics approximation
resolution    = '8'         # grid resolution in km
init_method   = 'eq'        # initialisation method

ocean_forcing_type      = 'open'    # 'open' or 'standard'
ocean_sensitivity       = 'medium'  # 'low', 'medium', 'high', 'pigl'
ice_shelf_fracture      = False

melt_in_floating_cells  = 'sub-grid'  # sub-shelf melt parameterisation
icefront_migration      = 'str'       # calving scheme
open_melt_type          = 'quad'      # melt param for open-ocean forcing
standard_melt_type      = 'nonlocal'  # (unused here; ocean_forcing_type='open')


# ── 3. Build the inputs object ────────────────────────────────────────────────

inputs = ISEFlowAISInputs(
    year=years,
    sector=sector,
    pr_anomaly=pr_anomaly,
    evspsbl_anomaly=evspsbl_anomaly,
    smb_anomaly=smb_anomaly,
    ts_anomaly=ts_anomaly,
    ocean_thermal_forcing=ocean_thermal_forcing,
    ocean_salinity=ocean_salinity,
    ocean_temperature=ocean_temperature,
    initial_year=initial_year,
    numerics=numerics,
    stress_balance=stress_balance,
    resolution=resolution,
    init_method=init_method,
    melt_in_floating_cells=melt_in_floating_cells,
    icefront_migration=icefront_migration,
    ocean_forcing_type=ocean_forcing_type,
    ocean_sensitivity=ocean_sensitivity,
    ice_shelf_fracture=ice_shelf_fracture,
    open_melt_type=open_melt_type,
    standard_melt_type=standard_melt_type,
)

print(inputs)


# ── 4. Load the pretrained model and predict ──────────────────────────────────

model = ISEFlow_AIS(version="v1.1.0")
pred, uq = model.predict(inputs, smoothing_window=0)

pred      = np.asarray(pred).squeeze()
epistemic = np.asarray(uq["epistemic"]).squeeze()
aleatoric = np.asarray(uq["aleatoric"]).squeeze()
total     = epistemic + aleatoric

print(f"\nPrediction range: {pred.min():.2f} - {pred.max():.2f} mm SLE")
print(f"Mean epistemic uncertainty: {epistemic.mean():.3f} mm")
print(f"Mean aleatoric uncertainty: {aleatoric.mean():.3f} mm")


# ── 5. Plot ───────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 6))

ax.fill_between(years, pred - total, pred + total,
                alpha=0.20, color="#1f77b4", label="Total uncertainty (epistemic + aleatoric)")
ax.fill_between(years, pred - epistemic, pred + epistemic,
                alpha=0.45, color="#1f77b4", label=r"Epistemic uncertainty (2$\sigma$)")
ax.plot(years, pred, color="#1f77b4", linewidth=2.0, label="ISEFlow prediction")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Sea Level Equivalent (mm SLE)", fontsize=12)
ax.set_title("ISEFlow-AIS: AWI_PISM1, Sector 10 (2015-2100)", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(years[0], years[-1])

plt.tight_layout()
plt.savefig("example_ais_projection.png", dpi=200, bbox_inches="tight")
print("\nSaved example_ais_projection.png")
plt.show()
plt.close("all")
