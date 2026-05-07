import numpy as np
import torch
from ise.models.iseflow import ISEFlow_GrIS
from ise.data.inputs import ISEFlowGrISInputs

years = np.arange(2015, 2101)
inputs = ISEFlowGrISInputs.from_absolute_forcings(
    year=years, sector=1, smb=np.full(86, -200.0), st=np.full(86, -20.0),
    ocean_thermal_forcing=np.linspace(2.2, 3.5, 86), basin_runoff=np.linspace(0.01, 0.10, 86),
    aogcm='hadgem2-es_rcp85', initial_year=1990, numerics='fe',
    ice_flow_model='ho', initialization='dav', initial_smb='ra3',
    velocity='joughin', bedrock_topography='morlighem',
    surface_thickness='None', geothermal_heat_flux='g',
    res_min=1.0, res_max=7.5, standard_ocean_forcing=True,
    ocean_sensitivity='medium', ice_shelf_fracture=False,
)
model = ISEFlow_GrIS(version='v1.1.0')
data = model.process(inputs)
import pandas as pd
X = torch.tensor(data.to_numpy(dtype=float), dtype=torch.float32)
nf = model.normalizing_flow
nf.eval()
with torch.no_grad():
    # Test log_prob (what training used)
    dummy_y = torch.zeros(X.shape[0], 1)
    lp = nf.flow.log_prob(inputs=dummy_y, context=X)
    print('log_prob NaN:', torch.isnan(lp).sum().item(), '/ inf:', torch.isinf(lp).sum().item())
    print('log_prob min/max:', lp.min().item(), lp.max().item())
    # Test sampling
    samples = nf.flow.sample(10, context=X[:5])
    print('samples shape:', samples.shape)
    print('samples NaN:', torch.isnan(samples).sum().item())
    print('samples min/max:', samples[~torch.isnan(samples)].min().item() if not torch.isnan(samples).all() else 'ALL NaN', samples[~torch.isnan(samples)].max().item() if not torch.isnan(samples).all() else 'ALL NaN')
    # Test latent
    z = nf.get_latent(X)
    print('latent NaN:', torch.isnan(z).sum().item())
    print('latent min/max:', z.min().item(), z.max().item())