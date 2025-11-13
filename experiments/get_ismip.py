import pandas as pd

from experiments.simulation import Simulation, SimulationEnsemble, SimulationIndex

experiments = {
    "exp01": {"model": "NorESM1-M", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp02": {"model": "MIROC-ESM-CHEM", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp03": {"model": "NorESM1-M", "scenario": "ssp126", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp04": {"model": "CCSM4", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp05": {"model": "NorESM1-M", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp06": {"model": "MIROC-ESM-CHEM", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp07": {"model": "NorESM1-M", "scenario": "ssp126", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp08": {"model": "CCSM4", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp09": {"model": "NorESM1-M", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "high", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp10": {"model": "NorESM1-M", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "low", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    "exp11": {"model": "CCSM4", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": True, "tier": "Tier 1 (Core)"},
    "exp12": {"model": "CCSM4", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": True, "tier": "Tier 1 (Core)"},
    "exp13": {"model": "NorESM1-M", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "pigl", "ice_shelf_fracture": False, "tier": "Tier 1 (Core)"},
    
    "expA1": {"model": "HadGEM2-ES", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA2": {"model": "CSIRO-MK3", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA3": {"model": "IPSL-CM5A-MR", "scenario": "ssp585", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA4": {"model": "IPSL-CM5A-MR", "scenario": "ssp126", "ocean_forcing_type": "open", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA5": {"model": "HadGEM2-ES", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA6": {"model": "CSIRO-MK3", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA7": {"model": "IPSL-CM5A-MR", "scenario": "ssp585", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
    "expA8": {"model": "IPSL-CM5A-MR", "scenario": "ssp126", "ocean_forcing_type": "standard", "sensitivity": "medium", "ice_shelf_fracture": False, "tier": "Tier 2"},
}

experiments_df = pd.DataFrame.from_dict(experiments, orient='index')

runs = {
    'AWI_PISM1': ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'expA1', 'expA2', 'expA3', 'expA4', 'expA5', 'expA6', 'expA7', 'expA8'],
    'DOE_MALI': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp12', 'exp13', ],
    'IMAU_IMAUICE1': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp12', 'exp13', ],
    'IMAU_IMAUICE2': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp12', 'exp13', 'expA5', 'expA6', 'expA7', 'expA8'],
    'JPL1_ISSM': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp12', 'exp13', 'expA5', 'expA6', 'expA7', 'expA8'],
    'LSCE_GRISLI': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp12', 'exp13', 'expA5', 'expA6', 'expA7', 'expA8'],
    'NCAR_CISM': ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp13', 'expA1', 'expA2', 'expA3', 'expA4', 'expA5', 'expA6', 'expA7', 'expA8'],
    'PIK_PISM1': ['exp01', 'exp02', 'exp03', 'exp04', ],
    'PIK_PISM2': ['exp01', 'exp02', 'exp03', 'exp04', ],
    'ILTS_PIK_SICOPOLIS': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp12', 'exp13', 'expA5', 'expA6', 'expA7', 'expA8'],
    'UCIJPL_ISSM': ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'expA5', 'expA6', 'expA7', 'expA8'],
    'UTAS_ElmerIce': ['exp05', 'exp06', 'exp13', ],
    'VUB_AISMPALEO': ['exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp13', 'expA5', 'expA6', 'expA7', ],
    'VUW_PISM': ['exp01', 'exp02', 'exp03', 'exp04', ],
    'ULB_FETISH_16km': ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'expA1', 'expA2', 'expA3', 'expA4', 'expA5', 'expA6', 'expA7', 'expA8'],
    'ULB_FETISH_32km': ['exp01', 'exp02', 'exp03', 'exp04', 'exp05', 'exp06', 'exp07', 'exp08', 'exp09', 'exp10', 'exp11', 'exp12', 'exp13', 'expA1', 'expA2', 'expA3', 'expA4', 'expA5', 'expA6', 'expA7', 'expA8'],

}

output_dir = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/CMIP/projections/"
sim_index = SimulationIndex(output_dir)


df = pd.DataFrame()
for model, exp_list in runs.items():
    for exp in exp_list:
        
        if model not in sim_index.load_index().ism.unique():
            raise ValueError(f"Model {model} not found in simulation index.")
        
        sim = sim_index.query_simulations(
            ism=model,
            ocean_forcing_type=experiments[exp]['ocean_forcing_type'],
            sensitivity=experiments[exp]['sensitivity'],
            ice_shelf_fracture=experiments[exp]['ice_shelf_fracture']
        )
        sim = sim[sim.status == "completed"]
        df = pd.concat([df, sim], ignore_index=True)

# some of the experiments are the same if you take out "aogcm", drop dupes
df = df.drop_duplicates()

print(f"[INFO] Total simulations found: {len(df)}")

print(f"[INFO] Generating SimulationEnsemble...")
simulations = []
for sim_id in df.sim_id.unique():
    simulation = Simulation(sim_id, simulation_dir=output_dir)
    simulations.append(simulation)
    
ismip6 = SimulationEnsemble(simulations=simulations)
ismip6_df = ismip6.to_df("supplemental/ismip_weighted_simulations_all.csv")
ismip6_df[ismip6_df.sector == 0.0].to_csv(r'supplemental/ismip_weighted_simulations_total.csv', index=False)
print(f"[INFO] SimulationEnsemble saved to ismip_weighted_simulations.csv")

stop = ''