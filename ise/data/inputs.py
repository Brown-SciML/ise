"""Input dataclasses for ISEFlow-AIS and ISEFlow-GrIS predictions.

This module defines ISEFlowAISInputs and ISEFlowGrISInputs, which encapsulate
climate forcings, experiment configuration, and ice sheet model settings
required for running pretrained ISEFlow emulators.
"""
from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np
import json
import pandas as pd
import warnings

from ise.utils import ismip6_model_configs_path


@dataclass
class ISEFlowAISInputs:
    """Configuration settings for an ISEFlow_AIS model run."""
    # Forcing data
    year: np.ndarray
    sector: np.ndarray
    pr_anomaly: np.ndarray
    evspsbl_anomaly: np.ndarray
    smb_anomaly: np.ndarray
    ts_anomaly: np.ndarray
    ocean_thermal_forcing: np.ndarray
    ocean_salinity: np.ndarray
    ocean_temperature: np.ndarray
    
    # Experiment configuration
    ice_shelf_fracture: bool
    ocean_sensitivity: str
    
    # Version 1.0.0 only
    mrro_anomaly: Optional[np.ndarray] = None

    # Model configuration
    initial_year: Optional[int] = None
    numerics: Optional[str] = None
    stress_balance: Optional[str] = None
    resolution: Optional[str] = None
    init_method: Optional[str] = None
    melt_in_floating_cells: Optional[str] = None
    icefront_migration: Optional[str] = None
    ocean_forcing_type: Optional[str] = None
    open_melt_type: Optional[str] = None
    standard_melt_type: Optional[str] = None
    
    # ISMIP6 model to emulate
    model_configs: Optional[str] = None
    
    # ISEFlow version
    version: str = "v1.1.0"
    
    override_params: dict = None
    

    # Validation logic runs after the object is created
    def __post_init__(self):
        
        if self.model_configs:
            self._load_all_ism_configs()
            
            if self.model_configs not in self.all_ism_configs:
                raise ValueError(f"Model name {self.model_configs} in 'model_configs' not found, must be in {list(self.all_ism_configs.keys())}")
            if self.all_ism_configs[self.model_configs]['ocean_forcing_type'] != self.ocean_forcing_type:
                raise ValueError(f"Model {self.model_configs} has ocean_forcing_type {self.all_ism_configs[self.model_configs]['ocean_forcing_type']}, but received {self.ocean_forcing_type}")
                
            self._assign_model_configs(self.model_configs)
            
        self._check_inputs()
        self._map_args()
        self._convert_arrays()
        self.df = None
        self.all_ism_configs = None if not self.model_configs else self.all_ism_configs
        
    def _check_inputs(self,):
        """Check the validity of input parameters."""
        # This method can be used for additional checks if needed
         # check inputs
         
        if self.year[0] == 2015:
            self.year = self.year - 2015 + 1  # convert 2015–2100 → 1–86 (model encoding)

        if isinstance(self.sector, int):
            self.sector = np.ones_like(self.year) * self.sector

        if not self.model_configs and (not self.numerics or not self.stress_balance or not self.resolution or not self.init_method or not self.initial_year or not self.melt_in_floating_cells or not self.icefront_migration or not self.ocean_forcing_type or not self.ocean_sensitivity or self.ice_shelf_fracture is None):
            raise ValueError("Either 'model_configs' must be provided or all individual configuration parameters must be specified.")
        
        if not isinstance(self.initial_year, int):
            raise ValueError("initial_year must be an integer")

        if str(self.numerics).lower() not in ('fe', 'fd', 'fe/fv'):
            raise ValueError("numerics must be one of 'fe', 'fd', or 'fe/fv'")
        
        if str(self.stress_balance) not in ('ho', 'hybrid', "l1l2", 'sia+ssa', 'ssa', 'stokes'):
            raise ValueError("stress_balance must be one of 'ho', 'hybrid', 'l1l2', 'sia+ssa', 'ssa', or 'stokes'")
        
        if str(self.resolution) not in ('16', '20', '32', '4', '8', 'variable'):
            raise ValueError("resolution must be one of '16', '20', '32', '4', '8', or 'variable'")
        
        if str(self.init_method) not in ('da', 'da*', 'da+', 'eq', 'sp', 'sp+'):
            raise ValueError("init_method must be one of 'da', 'da*', 'da+', 'eq', 'sp', or 'sp+'")
        
        if str(self.melt_in_floating_cells) not in ('floating condition', 'sub-grid', 'None', 'False', "No"):
            raise ValueError("melt_in_floating_cells must be one of 'floating condition', 'sub-grid', 'No', 'None', or 'False'")

        if str(self.icefront_migration) not in ('str', 'fix', 'mh', 'ro', 'div'):
            raise ValueError("icefront_migration must be one of 'str', 'fix', 'mh', 'ro', or 'div'")
        
        if str(self.ocean_forcing_type) not in ('standard', 'open'):
            raise ValueError("ocean_forcing_type must be one of 'standard' or 'open'")
        
        if str(self.ocean_forcing_type) == 'standard' and self.standard_melt_type is None:
            raise ValueError("standard_melt_type must be provided if ocean_forcing_type is 'standard'")
        elif str(self.ocean_forcing_type) == 'standard' and self.standard_melt_type not in ("local", "nonlocal", "local anom", "nonlocal anom", "None",):
            raise ValueError("standard_melt_type must be one of 'local', 'nonlocal', 'local anom', 'nonlocal anom', or None")
        
        if str(self.ocean_forcing_type) == 'open' and self.open_melt_type is None:
            raise ValueError("open_melt_type must be provided if ocean_forcing_type is 'open'")
        elif str(self.ocean_forcing_type) == 'open' and self.open_melt_type not in ("lin", "quad", "nonlocal+slope", "pico", "picop", "plume", "None",):
            raise ValueError("open_melt_type must be one of 'lin', 'quad', 'nonlocal+slope', 'pico', 'picop', 'plume', or None")
        
        if str(self.ocean_sensitivity) not in ('low', 'medium', 'high', 'pigl'):
            raise ValueError("ocean_sensitivity must be one of 'low', 'medium', 'high', or 'pigl'")

        if not isinstance(self.ice_shelf_fracture, bool):
            raise ValueError("ice_shelf_fracture must be a boolean")
        
    def _map_args(self, ):
    
        # map from accepted input to how the model expects variable names
        arg_map = {
            'numerics': {
                'fe': 'FE',
                'fd': 'FD',
                'fe/fv': 'FE/FV',
            },
            'stress_balance': {
                'ho': 'HO',
                'hybrid': 'Hybrid',
                'l1l2': 'L1L2',
                'sia+ssa': 'SIA_SSA',
                'ssa': 'SSA',
                'stokes': 'Stokes',
            },
            "init_method": {
                'da': 'DA',
                'da*': 'DA_geom',
                'da+': 'DA_relax',
                'eq': 'Eq',
                'sp': 'SP',
                'sp+': 'SP_icethickness',
            },
            "melt_in_floating_cells": {
                'floating condition': 'Floating_condition',
                'sub-grid': 'Sub-grid',
                "No": "No",
                'None': None
            },
            'icefront_migration': {
                'str': 'StR',
                'fix': 'Fix',
                'mh': 'MH',
                'ro': 'RO',
                'div': 'Div',
            },
            'ocean_forcing_type': {
                'standard': 'Standard',
                'open': 'Open',
            },
            'ocean_sensitivity': {
                'low': 'Low',
                'medium': 'Medium',
                'high': 'High',
                'pigl': 'PIGL',
            },
            "open_melt_type": {
                'lin': 'Lin',
                'quad': 'Quad',
                'nonlocal+slope': 'Nonlocal_Slope',
                'pico': 'PICO',
                'picop': 'PICOP',
                'plume': 'Plume',
                'None': None,
            },
            "standard_melt_type": {
                'local': 'Local',
                'nonlocal': 'Nonlocal',
                'local anom': 'Local_anom',
                'nonlocal anom': 'Nonlocal_anom',
                'None': None,
            }
        }
        
    
        for key, value in vars(self).items():
            current_value = getattr(self, key)

            if key in arg_map:
                # Normalise Python None to the string 'None' so the lookup succeeds
                lookup_key = 'None' if current_value is None else current_value
                new_value = arg_map[key][lookup_key]
                setattr(self, key, new_value)

        if self.override_params:
            if not isinstance(self.override_params, dict):
                raise ValueError("override_params must be a dictionary")
            
            for key, value in self.override_params.items():
                if key not in arg_map and not hasattr(self, key):
                    raise ValueError(f"Invalid configuration key '{key}' in 'override_params' mapping. Should be one of {list(arg_map.keys())}.")
                
                if value not in arg_map.get(key, {}):
                    raise ValueError(f"Invalid value '{value}' for key '{key}' in 'override_params'. Accepted values are: {list(arg_map.get(key, {}).keys())}")
                
                setattr(self, key, arg_map[key][value])
                

    def _convert_arrays(self):

        forcings = ("year", "pr_anomaly", "evspsbl_anomaly", "smb_anomaly", "ts_anomaly", "ocean_thermal_forcing", "ocean_salinity", "ocean_temperature")
        forcings += ("mrro_anomaly",) if self.version == "v1.0.0" else ()
        
        for arr_name in forcings:
            forcing_array = getattr(self, arr_name)

            try:
                setattr(self, arr_name, np.array(forcing_array))
            except Exception as e:
                raise ValueError(f"Variable {arr_name} must be a numpy array, received {type(forcing_array)}.") from e

    
    def to_df(self):
        """Convert the dataclass to a pandas DataFrame."""
        
        data = {    
            "year": self.year,
            "sector": self.sector,
            "pr_anomaly": self.pr_anomaly,
            "evspsbl_anomaly": self.evspsbl_anomaly,
            "smb_anomaly": self.smb_anomaly,
            "ts_anomaly": self.ts_anomaly,
            "thermal_forcing": self.ocean_thermal_forcing,
            "salinity": self.ocean_salinity,
            "temperature": self.ocean_temperature,
            "initial_year": self.initial_year,
            "numerics": self.numerics,
            "stress_balance": self.stress_balance,
            "resolution": self.resolution,
            "init_method": self.init_method,
            "melt": self.melt_in_floating_cells,
            "ice_front": self.icefront_migration,
            "Ocean sensitivity": self.ocean_sensitivity,
            "Ice shelf fracture": self.ice_shelf_fracture,
            "Ocean forcing": self.ocean_forcing_type,
            "open_melt_param": self.open_melt_type,
            "standard_melt_param": self.standard_melt_type,
        }
        
        if self.version == "v1.0.0":
            data['mrro_anomaly'] = self.mrro_anomaly
        
        self.df = pd.DataFrame(data)
        # self.df = self._order_columns(self.df)
        return self.df
    
    def __str__(self):
        def _arr_summary(arr):
            if arr is None:
                return "None"
            a = np.asarray(arr)
            return f"array(shape={a.shape}, min={a.min():.4g}, max={a.max():.4g}, mean={a.mean():.4g})"

        lines = [
            f"ISEFlowAISInputs (version={self.version})",
            "",
            "  Forcings:",
            f"    year                  : {_arr_summary(self.year)}",
            f"    sector                : {_arr_summary(self.sector)}",
            f"    pr_anomaly            : {_arr_summary(self.pr_anomaly)}",
            f"    evspsbl_anomaly       : {_arr_summary(self.evspsbl_anomaly)}",
            f"    smb_anomaly           : {_arr_summary(self.smb_anomaly)}",
            f"    ts_anomaly            : {_arr_summary(self.ts_anomaly)}",
            f"    ocean_thermal_forcing : {_arr_summary(self.ocean_thermal_forcing)}",
            f"    ocean_salinity        : {_arr_summary(self.ocean_salinity)}",
            f"    ocean_temperature     : {_arr_summary(self.ocean_temperature)}",
        ]
        if self.version == "v1.0.0":
            lines.append(f"    mrro_anomaly          : {_arr_summary(self.mrro_anomaly)}")

        lines += [
            "",
            "  Experiment config:",
            f"    ice_shelf_fracture    : {self.ice_shelf_fracture}",
            f"    ocean_sensitivity     : {self.ocean_sensitivity}",
            f"    ocean_forcing_type    : {self.ocean_forcing_type}",
            f"    standard_melt_type    : {self.standard_melt_type}",
            f"    open_melt_type        : {self.open_melt_type}",
            "",
            "  Model config:",
            f"    model_configs         : {self.model_configs}",
            f"    initial_year          : {self.initial_year}",
            f"    numerics              : {self.numerics}",
            f"    stress_balance        : {self.stress_balance}",
            f"    resolution            : {self.resolution}",
            f"    init_method           : {self.init_method}",
            f"    melt_in_floating_cells: {self.melt_in_floating_cells}",
            f"    icefront_migration    : {self.icefront_migration}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _load_all_ism_configs(self,):
        if not self.model_configs:
            raise ValueError("model_configs must be provided to get ISM characteristics.")
        
        with open(ismip6_model_configs_path, 'r') as file:
            self.all_ism_configs = json.load(file)

        return self.all_ism_configs

    def _assign_model_configs(self, model_name, characteristics_json=ismip6_model_configs_path):
        
        configs_provided = any([self.numerics, self.stress_balance, self.resolution, self.init_method, self.initial_year, self.melt_in_floating_cells, self.icefront_migration, ])
        if configs_provided:
            warnings.warn("Both 'model_configs' and individual configuration parameters are provided. 'model_configs' will take precedence.")
         
        if not self.all_ism_configs:
            self._load_all_ism_configs()
        
        if model_name in self.all_ism_configs:
            model_config = self.all_ism_configs[model_name]
        else:
            raise ValueError(f"Model name {model_name} in 'model_configs' not found, must be in {list(self.all_ism_configs.keys())}")
                
        for key, value in model_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key '{key}' in 'model_configs' mapping.") 
            
            
            
            
            
            
            
            

@dataclass
class ISEFlowGrISInputs:
    """Configuration settings for an ISEFlow_GrIS model run."""
    # Forcing data
    year: np.ndarray
    sector: np.ndarray
    aST: np.ndarray
    aSMB: np.ndarray
    ocean_thermal_forcing: np.ndarray
    basin_runoff: np.ndarray
    
    # Experiment configuration
    ice_shelf_fracture: bool
    ocean_sensitivity: str
    standard_ocean_forcing: bool
    # ['numerics', 'ice_flow', 'initialization', 'initial_smb', 'velocity', 'bed', 'surface_thickness', 'ghf', 'res_min', 'res_max', 'Ocean forcing', 'Ocean sensitivity', 'Ice shelf fracture'], dtype=bool)
    
    # Model configuration
    initial_year: Optional[int] = None
    numerics: Optional[str] = None
    ice_flow_model: Optional[str] = None
    initialization: Optional[str] = None
    initial_smb: Optional[str] = None
    velocity: Optional[str] = None
    bedrock_topography: Optional[str] = None
    surface_thickness: Optional[str] = None
    geothermal_heat_flux: Optional[str] = None
    res_min: Optional[str] = None
    res_max: Optional[str] = None
    
    # ISMIP6 model to emulate
    model_configs: Optional[str] = None
    
    # ISEFlow version
    version: str = "v1.1.0"
    

    # Validation logic runs after the object is created
    def __post_init__(self):
        self._assign_model_configs(self.model_configs) if self.model_configs else None
        self._check_inputs()
        self._map_args()
        self._convert_arrays()
        self.df = None
        
    def _check_inputs(self,):
        """Check the validity of input parameters."""
        # This method can be used for additional checks if needed
         # check inputs
         
        if self.year[0] == 2015:
            self.year = self.year - 2015 + 1  # convert 2015–2100 → 1–86 (model encoding)

        if isinstance(self.sector, int):
            self.sector = np.ones_like(self.year) * self.sector

        if not isinstance(self.initial_year, int):
            raise ValueError("initial_year must be an integer")

        # velocity, surface_thickness, and geothermal_heat_flux are legitimately absent
        # for some models (stored as None/'None'). Only the core ISM config fields are required.
        if not self.model_configs and (not self.numerics or not self.ice_flow_model or not self.initialization or not self.initial_smb or not self.bedrock_topography or not self.res_min or not self.res_max or self.standard_ocean_forcing is None or not self.ocean_sensitivity or self.ice_shelf_fracture is None):
            raise ValueError("Either 'model_configs' must be provided or all individual configuration parameters must be specified.")

        elif self.model_configs and (self.numerics or self.ice_flow_model or self.initialization or self.initial_smb or self.velocity or self.bedrock_topography or self.surface_thickness or self.geothermal_heat_flux or self.res_min or self.res_max or self.standard_ocean_forcing or self.ocean_sensitivity):
            warnings.warn("Both 'model_configs' and individual configuration parameters are provided. 'model_configs' will take precedence.")
         

        if str(self.numerics).lower() not in ('fe', 'fv', 'fd', 'fd/fv'):
            raise ValueError("numerics must be one of 'fe', 'fv', 'fd', or 'fd/fv'")

        if str(self.ice_flow_model) not in ('ho', 'ssa', "sia", 'hybrid',):
            raise ValueError("ice_flow_model must be one of 'ho', 'ssa', 'sia', or 'hybrid'")

        if str(self.initialization) not in ('dav', 'cyc/nds', 'sp/ndm', 'sp/dav', 'sp/das', 'cyc/ndm', 'sp/dai', 'cyc/dai', 'sp/nds'):
            raise ValueError("initialization must be one of 'dav', 'cyc/nds', 'sp/ndm', 'sp/dav', 'sp/das', 'cyc/ndm', 'sp/dai', 'cyc/dai', or 'sp/nds'")

        if str(self.initial_smb) not in ('ra3', 'hir', 'ismb', 'box/mar', 'box/ra3', 'mar', 'ra1'):
            raise ValueError("initial_smb must be one of 'ra3', 'hir', 'ismb', 'box/mar', 'box/ra3', 'mar', or 'ra1'")

        if str(self.bedrock_topography) not in ('morlighem', 'bamber'):
            raise ValueError("bed must be one of 'morlighem' or 'bamber'")

        if str(self.surface_thickness) not in ("None", "morlighem"):
            raise ValueError("surface_thickness must be one of 'None' or 'morlighem'")
        
        if str(self.velocity) not in ('joughin', 'rignot', 'None'):
            raise ValueError("velocity must be one of 'joughin', 'rignot', or 'None'")

        if str(self.geothermal_heat_flux) not in ('g', 'None', 'sr', 'mix'):
            raise ValueError("geothermal_heat_flux must be one of 'g', 'None', 'sr', or 'mix'")

        if float(self.res_min) not in [ 0.2 ,  0.25,  0.5 ,  0.75,  0.9 ,  1.  ,  1.2 ,  2.  ,  3.  ,
        4.  ,  5.  ,  8.  , 16.  ]:
            raise ValueError("res_min must be one of 0.2, 0.25, 0.5, 0.75, 0.9, 1., 1.2, 2., 3., 4., 5., 8., or 16.")

        if float(self.res_max) not in [ 0.9,  2. ,  4. ,  4.8,  5. ,  7.5,  8. , 14. , 15. , 16. , 20. ,
       25. , 30. ]:
            raise ValueError("res_max must be one of 0.9, 2., 4., 4.8, 5., 7.5, 8., 14., 15., 16., 20., 25., or 30.")
        
    

        if not isinstance(self.ice_shelf_fracture, bool):
            raise ValueError("ice_shelf_fracture must be a boolean")
        
        if not isinstance(self.standard_ocean_forcing, bool):
            raise ValueError("standard_ocean_forcing must be a boolean")
        
    def _map_args(self, ):
    
        # map from accepted input to how the model expects variable names
        arg_map = {
            'numerics': {
                'fe': 'FE',
                'fv': 'FV',
                'fd': 'FD',
                'fd/fv': 'FD_FV5',
            },
            'ice_flow_model': {
                'ho': 'HO',
                'ssa': 'SSA',
                'sia': 'SIA',
                'hybrid': 'HYB',
            },
            "initialization": {
                'dav': 'DAV',
                'cyc/nds': 'CYC_NDS',
                'sp/ndm': 'SP_NDM',
                'sp/dav': 'SP_DAV',
                'sp/das': 'SP_DAS',
                'cyc/ndm': 'CYC_NDM',
                'sp/dai': 'SP_DAI',
                'cyc/dai': 'CYC_DAI',
                'sp/nds': 'SP_NDS',
            },
            "initial_smb": {
                'ra3': 'RA3',
                'hir': 'HIR',
                'ismb': 'ISMB',
                'box/mar': 'BOX_MAR',
                'box/ra3': 'BOX_RA3',
                'mar': 'MAR',
                'ra1': 'RA1',
            },
            'bedrock_topography': {
                'morlighem': 'M',
                'bamber': 'B',
            },
            'surface_thickness': {
                'None': None,
                'morlighem': 'M',
            },
            'geothermal_heat_flux': {
                'g': 'G',
                'None': None,
                'sr': 'SR',
                'mix': 'MIX',
            },
            'velocity': {
                'joughin': 'J',
                'rignot': 'RM',
                'None': None,
            },
            'ocean_sensitivity': {
                'low': 'Low',
                'medium': 'Medium',
                'high': 'High',
            },
        }


        for key, value in vars(self).items():
            current_value = getattr(self, key)

            if key == "res_min" or key == "res_max":
                new_value = str(float(current_value))
                setattr(self, key, new_value)

            elif key in arg_map:
                # Normalise Python None to the string 'None' so the lookup succeeds
                lookup_key = 'None' if current_value is None else current_value
                new_value = arg_map[key][lookup_key]
                setattr(self, key, new_value)
            
                
    def _convert_arrays(self):

        forcings = ("year", 'aST', "aSMB", "ocean_thermal_forcing", "basin_runoff")
        
        for arr_name in forcings:
            forcing_array = getattr(self, arr_name)

            try:
                setattr(self, arr_name, np.array(forcing_array))
            except Exception as e:
                raise ValueError(f"Variable {arr_name} must be a numpy array, received {type(forcing_array)}.") from e
    
    def to_df(self):
        """Convert the dataclass to a pandas DataFrame."""
        
        data = {    
            "year": self.year,
            "sector": self.sector,
            "aST": self.aST,
            "aSMB": self.aSMB,
            "thermal_forcing": self.ocean_thermal_forcing,
            "basin_runoff": self.basin_runoff,
            "initial_year": self.initial_year,
            "numerics": self.numerics,
            "ice_flow": self.ice_flow_model,
            "initialization": self.initialization,
            "initial_smb": self.initial_smb,
            "velocity": self.velocity,
            "bed": self.bedrock_topography,
            "surface_thickness": self.surface_thickness,
            "ghf": self.geothermal_heat_flux,
            "res_min": self.res_min,
            "res_max": self.res_max,
            "Ocean forcing": 'Standard' if self.standard_ocean_forcing else 'Open',
            "Ocean sensitivity": self.ocean_sensitivity,
            "Ice shelf fracture": self.ice_shelf_fracture,
        }
        
        
        self.df = pd.DataFrame(data)
        return self.df


    def __str__(self):
        def _arr_summary(arr):
            if arr is None:
                return "None"
            a = np.asarray(arr)
            return f"array(shape={a.shape}, min={a.min():.4g}, max={a.max():.4g}, mean={a.mean():.4g})"

        lines = [
            f"ISEFlowGrISInputs (version={self.version})",
            "",
            "  Forcings:",
            f"    year                  : {_arr_summary(self.year)}",
            f"    sector                : {_arr_summary(self.sector)}",
            f"    aST                   : {_arr_summary(self.aST)}",
            f"    aSMB                  : {_arr_summary(self.aSMB)}",
            f"    ocean_thermal_forcing : {_arr_summary(self.ocean_thermal_forcing)}",
            f"    basin_runoff          : {_arr_summary(self.basin_runoff)}",
            "",
            "  Experiment config:",
            f"    ice_shelf_fracture    : {self.ice_shelf_fracture}",
            f"    ocean_sensitivity     : {self.ocean_sensitivity}",
            f"    standard_ocean_forcing: {self.standard_ocean_forcing}",
            "",
            "  Model config:",
            f"    model_configs         : {self.model_configs}",
            f"    initial_year          : {self.initial_year}",
            f"    numerics              : {self.numerics}",
            f"    ice_flow_model        : {self.ice_flow_model}",
            f"    initialization        : {self.initialization}",
            f"    initial_smb           : {self.initial_smb}",
            f"    velocity              : {self.velocity}",
            f"    bedrock_topography    : {self.bedrock_topography}",
            f"    surface_thickness     : {self.surface_thickness}",
            f"    geothermal_heat_flux  : {self.geothermal_heat_flux}",
            f"    res_min               : {self.res_min}",
            f"    res_max               : {self.res_max}",
        ]
        return "\n".join(lines)

    def __repr__(self):
        return self.__str__()

    def _assign_model_configs(self, model_name, characteristics_json=ismip6_model_configs_path):
        with open(characteristics_json, 'r') as file:
            characteristics = json.load(file)
            
        if model_name in characteristics:
            model_config = characteristics[model_name]
        else:
            raise ValueError(f"Model name {model_name} in 'model_configs' not found, must be in {list(characteristics.keys())}")

        for key, value in model_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key '{key}' in 'model_configs' mapping.") 