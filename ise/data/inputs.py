"""Input dataclasses for ISEFlow-AIS and ISEFlow-GrIS predictions.

This module defines ``ISEFlowAISInputs`` and ``ISEFlowGrISInputs``, which
validate, encode, and package the climate forcing arrays and ice sheet model
(ISM) configuration required by the pretrained ISEFlow emulators.

Both dataclasses perform the following on construction:

1. **Validation** — all parameter values are checked against the enumerated
   sets of allowed options (numerics, stress balance, resolution, etc.).
2. **Encoding** — human-readable strings (e.g. ``'fd'``, ``'hybrid'``) are
   mapped to the internal categorical encodings expected by the model weights
   (e.g. ``'FD'``, ``'Hybrid'``).
3. **Array coercion** — all forcing arrays are cast to ``numpy.ndarray``.
4. **Year encoding** — calendar years 2015-2100 are converted to the
   model-internal 1-86 encoding.

Alternative constructor — raw absolute forcings
-----------------------------------------------
If you have raw (non-anomaly) atmospheric forcing values, use
``from_absolute_forcings()``.  It calls ``AnomalyConverter`` internally to subtract
the ISMIP6 climatological baseline before building the dataclass::

    from ise.data.inputs import ISEFlowAISInputs
    import numpy as np

    inputs = ISEFlowAISInputs.from_absolute_forcings(
        year=np.arange(2015, 2101),
        sector=10,
        pr=pr_array,           # kg m⁻² s⁻¹, raw absolute values
        evspsbl=evspsbl_array,
        smb=smb_array,
        ts=ts_array,           # K
        ocean_thermal_forcing=otf_array,
        ocean_salinity=sal_array,
        ocean_temperature=temp_array,
        aogcm="noresm1-m_rcp85",   # or custom_climatology={...} for new CMIP models
        # ISM configuration:
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

If the ISM configuration matches one of the bundled ISMIP6 models, you can
pass ``model_configs="BISICLES_UBC"`` (or whichever model key appears in
``ismip6_model_configs.json``) instead of specifying all parameters
individually.

Output
------
Call ``inputs.to_df()`` to obtain a ``pandas.DataFrame`` (86 rows × features)
that can be passed directly to ``ISEFlow_AIS.process()`` or
``ISEFlow_GrIS.process()``.  The pretrained wrappers call ``process()``
internally when you invoke ``model.predict(inputs)``.

See also: ``ise.data.anomaly.AnomalyConverter``
"""

import json
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ise.utils import ismip6_model_configs_path


@dataclass
class ISEFlowAISInputs:
    """Inputs for an ISEFlow-AIS prediction.

    Expects pre-computed anomaly arrays (``pr_anomaly``, ``evspsbl_anomaly``,
    ``smb_anomaly``, ``ts_anomaly``).  If you have raw absolute forcing values
    instead, use the alternative constructor::

        inputs = ISEFlowAISInputs.from_absolute_forcings(
            year=..., sector=..., pr=..., evspsbl=..., smb=..., ts=...,
            ocean_thermal_forcing=..., ocean_salinity=..., ocean_temperature=...,
            aogcm="noresm1-m_rcp85",   # or custom_climatology={...}
            **ism_config_kwargs,
        )

    ``from_absolute_forcings()`` subtracts the ISMIP6 1995-2014 climatological
    baseline automatically.  Pass ``aogcm`` for a bundled ISMIP6 model or
    ``custom_climatology`` (dict with keys ``'pr'``, ``'evspsbl'``, ``'smb'``,
    ``'ts'``) for a CMIP model not in the bundled climatology.
    """

    # Forcing data
    year: np.ndarray
    sector: np.ndarray | int
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
    mrro_anomaly: np.ndarray | None = None

    # Model configuration
    initial_year: int | None = None
    numerics: str | None = None
    stress_balance: str | None = None
    resolution: str | None = None
    init_method: str | None = None
    melt_in_floating_cells: str | None = None
    icefront_migration: str | None = None
    ocean_forcing_type: str | None = None
    open_melt_type: str | None = None
    standard_melt_type: str | None = None

    # ISMIP6 model to emulate
    model_configs: str | None = None

    # ISEFlow *model weights* version (distinct from the ise-py package version)
    version: str = "v1.1.0"

    override_params: dict | None = None

    # ------------------------------------------------------------------
    # Alternative constructor: raw (non-anomaly) forcing values
    # ------------------------------------------------------------------

    @classmethod
    def from_absolute_forcings(
        cls,
        year: np.ndarray,
        sector: int,
        pr: np.ndarray,
        evspsbl: np.ndarray,
        smb: np.ndarray,
        ts: np.ndarray,
        ocean_thermal_forcing: np.ndarray,
        ocean_salinity: np.ndarray,
        ocean_temperature: np.ndarray,
        aogcm: str | None = None,
        custom_climatology: dict | None = None,
        mrro: np.ndarray | None = None,
        **kwargs,
    ) -> "ISEFlowAISInputs":
        """Construct ISEFlowAISInputs from raw (non-anomaly) atmospheric forcings.

        Subtracts the ISMIP6 1995-2014 climatological baseline from each
        atmospheric variable to produce the anomaly arrays required by the
        model.  Ocean variables (``ocean_thermal_forcing``, ``ocean_salinity``,
        ``ocean_temperature``) are absolute values and are passed through
        unchanged.

        Exactly one of ``aogcm`` or ``custom_climatology`` must be provided.

        Parameters
        ----------
        year : np.ndarray
            Years corresponding to the time series (86 values, 2015-2100).
        sector : int
            AIS drainage sector (1-18).
        pr : np.ndarray
            Raw precipitation (86 values, kg m⁻² s⁻¹).
        evspsbl : np.ndarray
            Raw evaporation / sublimation (86 values, kg m⁻² s⁻¹).
        smb : np.ndarray
            Raw surface mass balance (86 values, kg m⁻² s⁻¹).
        ts : np.ndarray
            Raw surface temperature (86 values, K).
        ocean_thermal_forcing : np.ndarray
            Ocean thermal forcing (86 values, °C).  Passed through unchanged.
        ocean_salinity : np.ndarray
            Ocean salinity (86 values, PSU).  Passed through unchanged.
        ocean_temperature : np.ndarray
            Ocean temperature (86 values, °C).  Passed through unchanged.
        aogcm : str, optional
            AOGCM name to look up in the bundled ISMIP6 climatology
            (e.g. ``'noresm1-m_rcp85'``).  Common alternate spellings are
            normalised automatically.
        custom_climatology : dict, optional
            Baseline means for a CMIP model not in the bundled climatology.
            Must contain keys ``'pr'``, ``'evspsbl'``, ``'smb'``, ``'ts'``
            (and ``'mrro'`` if ``mrro`` is also provided).  Values should be
            in the same units as the raw input arrays.
        mrro : np.ndarray, optional
            Raw runoff (86 values).  Only needed for ISEFlow v1.0.0.
        **kwargs
            All remaining keyword arguments are forwarded to
            ``ISEFlowAISInputs.__init__`` (e.g. ISM config fields such as
            ``numerics``, ``stress_balance``, ``model_configs``, etc.).

        Returns
        -------
        ISEFlowAISInputs
            Fully validated inputs object ready for ``model.predict()``.

        Examples
        --------
        Using a bundled ISMIP6 climatology::

            inputs = ISEFlowAISInputs.from_absolute_forcings(
                year=np.arange(2015, 2101),
                sector=10,
                pr=pr_array,
                evspsbl=evspsbl_array,
                smb=smb_array,
                ts=ts_array,
                ocean_thermal_forcing=otf_array,
                ocean_salinity=sal_array,
                ocean_temperature=temp_array,
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
                standard_melt_type="nonlocal",
            )

        Using a custom climatology for a new CMIP model::

            inputs = ISEFlowAISInputs.from_absolute_forcings(
                year=np.arange(2015, 2101),
                sector=10,
                pr=pr_array, evspsbl=evspsbl_array,
                smb=smb_array, ts=ts_array,
                ocean_thermal_forcing=otf_array,
                ocean_salinity=sal_array,
                ocean_temperature=temp_array,
                custom_climatology={
                    "pr": 1.3e-5, "evspsbl": 3.8e-6,
                    "smb": 9.0e-6, "ts": 253.7,
                },
                numerics="fd", ...
            )
        """
        from ise.data.anomaly import AnomalyConverter

        converter = AnomalyConverter("AIS")
        anomalies = converter.compute_ais(
            sector=sector,
            pr=pr,
            evspsbl=evspsbl,
            smb=smb,
            ts=ts,
            aogcm=aogcm,
            custom_climatology=custom_climatology,
            mrro=mrro,
        )

        return cls(
            year=year,
            sector=sector,
            pr_anomaly=anomalies["pr_anomaly"],
            evspsbl_anomaly=anomalies["evspsbl_anomaly"],
            smb_anomaly=anomalies["smb_anomaly"],
            ts_anomaly=anomalies["ts_anomaly"],
            ocean_thermal_forcing=ocean_thermal_forcing,
            ocean_salinity=ocean_salinity,
            ocean_temperature=ocean_temperature,
            mrro_anomaly=anomalies.get("mrro_anomaly"),
            **kwargs,
        )

    @classmethod
    def from_raw_values(cls, *args, **kwargs):
        """Deprecated — use ``from_absolute_forcings`` instead."""
        warnings.warn(
            "from_raw_values() is deprecated; use from_absolute_forcings() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_absolute_forcings(*args, **kwargs)

    # Validation logic runs after the object is created
    def __post_init__(self):

        if self.model_configs:
            self._load_all_ism_configs()

            if self.model_configs not in self.all_ism_configs:
                raise ValueError(
                    f"Model name {self.model_configs} in 'model_configs' not found, must be in {list(self.all_ism_configs.keys())}"
                )
            if (
                self.all_ism_configs[self.model_configs]["ocean_forcing_type"]
                != self.ocean_forcing_type
            ):
                raise ValueError(
                    f"Model {self.model_configs} has ocean_forcing_type {self.all_ism_configs[self.model_configs]['ocean_forcing_type']}, but received {self.ocean_forcing_type}"
                )

            self._assign_model_configs(self.model_configs)

        self._check_inputs()
        self._map_args()
        self._convert_arrays()
        self.df = None
        self.all_ism_configs = None if not self.model_configs else self.all_ism_configs

    def _check_inputs(
        self,
    ):
        """Validate all AIS input parameters and normalise array encodings.

        Converts ``year`` from calendar years (2015-2100) to model-internal
        encoding (1-86), broadcasts a scalar ``sector`` to an array, and raises
        ``ValueError`` for any out-of-range or mutually exclusive parameter
        combinations.
        """

        if self.year[0] == 2015:
            self.year = self.year - 2015 + 1  # convert 2015-2100 → 1-86 (model encoding)

        if isinstance(self.sector, int):
            self.sector = np.ones_like(self.year) * self.sector

        if not self.model_configs and (
            not self.numerics
            or not self.stress_balance
            or not self.resolution
            or not self.init_method
            or not self.initial_year
            or not self.melt_in_floating_cells
            or not self.icefront_migration
            or not self.ocean_forcing_type
            or not self.ocean_sensitivity
            or self.ice_shelf_fracture is None
        ):
            raise ValueError(
                "Either 'model_configs' must be provided or all individual configuration parameters must be specified."
            )

        if not isinstance(self.initial_year, int):
            raise ValueError("initial_year must be an integer")

        if str(self.numerics).lower() not in ("fe", "fd", "fe/fv"):
            raise ValueError("numerics must be one of 'fe', 'fd', or 'fe/fv'")

        if str(self.stress_balance) not in ("ho", "hybrid", "l1l2", "sia+ssa", "ssa", "stokes"):
            raise ValueError(
                "stress_balance must be one of 'ho', 'hybrid', 'l1l2', 'sia+ssa', 'ssa', or 'stokes'"
            )

        if str(self.resolution) not in ("16", "20", "32", "4", "8", "variable"):
            raise ValueError("resolution must be one of '16', '20', '32', '4', '8', or 'variable'")

        if str(self.init_method) not in ("da", "da*", "da+", "eq", "sp", "sp+"):
            raise ValueError("init_method must be one of 'da', 'da*', 'da+', 'eq', 'sp', or 'sp+'")

        if str(self.melt_in_floating_cells) not in (
            "floating condition",
            "sub-grid",
            "None",
            "False",
            "No",
        ):
            raise ValueError(
                "melt_in_floating_cells must be one of 'floating condition', 'sub-grid', 'No', 'None', or 'False'"
            )

        if str(self.icefront_migration) not in ("str", "fix", "mh", "ro", "div"):
            raise ValueError("icefront_migration must be one of 'str', 'fix', 'mh', 'ro', or 'div'")

        if str(self.ocean_forcing_type) not in ("standard", "open"):
            raise ValueError("ocean_forcing_type must be one of 'standard' or 'open'")

        if str(self.ocean_forcing_type) == "standard" and self.standard_melt_type is None:
            raise ValueError(
                "standard_melt_type must be provided if ocean_forcing_type is 'standard'"
            )
        elif str(self.ocean_forcing_type) == "standard" and self.standard_melt_type not in (
            "local",
            "nonlocal",
            "local anom",
            "nonlocal anom",
            "None",
        ):
            raise ValueError(
                "standard_melt_type must be one of 'local', 'nonlocal', 'local anom', 'nonlocal anom', or None"
            )

        if str(self.ocean_forcing_type) == "open" and self.open_melt_type is None:
            raise ValueError("open_melt_type must be provided if ocean_forcing_type is 'open'")
        elif str(self.ocean_forcing_type) == "open" and self.open_melt_type not in (
            "lin",
            "quad",
            "nonlocal+slope",
            "pico",
            "picop",
            "plume",
            "None",
        ):
            raise ValueError(
                "open_melt_type must be one of 'lin', 'quad', 'nonlocal+slope', 'pico', 'picop', 'plume', or None"
            )

        if str(self.ocean_sensitivity) not in ("low", "medium", "high", "pigl"):
            raise ValueError("ocean_sensitivity must be one of 'low', 'medium', 'high', or 'pigl'")

        if not isinstance(self.ice_shelf_fracture, bool):
            raise ValueError("ice_shelf_fracture must be a boolean")

    def _map_args(
        self,
    ):
        """Map user-facing string values to the internal encodings expected by the model.

        For example, ``numerics='fd'`` becomes ``'FD'``,
        ``init_method='da'`` becomes ``'DA'``, etc.  Also applies any
        overrides specified in ``self.override_params``.
        """

        # map from accepted input to how the model expects variable names
        arg_map = {
            "numerics": {
                "fe": "FE",
                "fd": "FD",
                "fe/fv": "FE/FV",
            },
            "stress_balance": {
                "ho": "HO",
                "hybrid": "Hybrid",
                "l1l2": "L1L2",
                "sia+ssa": "SIA_SSA",
                "ssa": "SSA",
                "stokes": "Stokes",
            },
            "init_method": {
                "da": "DA",
                "da*": "DA_geom",
                "da+": "DA_relax",
                "eq": "Eq",
                "sp": "SP",
                "sp+": "SP_icethickness",
            },
            "melt_in_floating_cells": {
                "floating condition": "Floating_condition",
                "sub-grid": "Sub-grid",
                "No": "No",
                "None": None,
            },
            "icefront_migration": {
                "str": "StR",
                "fix": "Fix",
                "mh": "MH",
                "ro": "RO",
                "div": "Div",
            },
            "ocean_forcing_type": {
                "standard": "Standard",
                "open": "Open",
            },
            "ocean_sensitivity": {
                "low": "Low",
                "medium": "Medium",
                "high": "High",
                "pigl": "PIGL",
            },
            "open_melt_type": {
                "lin": "Lin",
                "quad": "Quad",
                "nonlocal+slope": "Nonlocal_Slope",
                "pico": "PICO",
                "picop": "PICOP",
                "plume": "Plume",
                "None": None,
            },
            "standard_melt_type": {
                "local": "Local",
                "nonlocal": "Nonlocal",
                "local anom": "Local_anom",
                "nonlocal anom": "Nonlocal_anom",
                "None": None,
            },
        }

        for key, value in vars(self).items():
            current_value = getattr(self, key)

            if key in arg_map:
                # Normalise Python None to the string 'None' so the lookup succeeds
                lookup_key = "None" if current_value is None else current_value
                new_value = arg_map[key][lookup_key]
                setattr(self, key, new_value)

        if self.override_params:
            if not isinstance(self.override_params, dict):
                raise ValueError("override_params must be a dictionary")

            for key, value in self.override_params.items():
                if key not in arg_map and not hasattr(self, key):
                    raise ValueError(
                        f"Invalid configuration key '{key}' in 'override_params' mapping. Should be one of {list(arg_map.keys())}."
                    )

                if value not in arg_map.get(key, {}):
                    raise ValueError(
                        f"Invalid value '{value}' for key '{key}' in 'override_params'. Accepted values are: {list(arg_map.get(key, {}).keys())}"
                    )

                setattr(self, key, arg_map[key][value])

    def _convert_arrays(self):
        """Coerce all forcing arrays to ``numpy.ndarray``."""

        forcings = (
            "year",
            "pr_anomaly",
            "evspsbl_anomaly",
            "smb_anomaly",
            "ts_anomaly",
            "ocean_thermal_forcing",
            "ocean_salinity",
            "ocean_temperature",
        )
        forcings += ("mrro_anomaly",) if self.version == "v1.0.0" else ()

        for arr_name in forcings:
            forcing_array = getattr(self, arr_name)

            try:
                setattr(self, arr_name, np.array(forcing_array))
            except Exception as e:
                raise ValueError(
                    f"Variable {arr_name} must be a numpy array, received {type(forcing_array)}."
                ) from e

    def to_df(self):
        """Convert the dataclass fields to a pandas DataFrame.

        Returns:
            pandas.DataFrame: One row per timestep (86 rows) with all forcing
            and configuration columns needed by ``ISEFlow_AIS.process()``.
        """

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
            data["mrro_anomaly"] = self.mrro_anomaly

        self.df = pd.DataFrame(data)
        # self.df = self._order_columns(self.df)
        return self.df

    def __str__(self):
        def _arr_summary(arr):
            if arr is None:
                return "None"
            a = np.asarray(arr)
            return (
                f"array(shape={a.shape}, min={a.min():.4g}, max={a.max():.4g}, mean={a.mean():.4g})"
            )

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

    def _load_all_ism_configs(
        self,
    ):
        if not self.model_configs:
            raise ValueError("model_configs must be provided to get ISM characteristics.")

        with open(ismip6_model_configs_path) as file:
            self.all_ism_configs = json.load(file)

        return self.all_ism_configs

    def _assign_model_configs(self, model_name, characteristics_json=ismip6_model_configs_path):

        configs_provided = any(
            [
                self.numerics,
                self.stress_balance,
                self.resolution,
                self.init_method,
                self.initial_year,
                self.melt_in_floating_cells,
                self.icefront_migration,
            ]
        )
        if configs_provided:
            warnings.warn(
                "Both 'model_configs' and individual configuration parameters are provided. 'model_configs' will take precedence."
            )

        if not self.all_ism_configs:
            self._load_all_ism_configs()

        if model_name in self.all_ism_configs:
            model_config = self.all_ism_configs[model_name]
        else:
            raise ValueError(
                f"Model name {model_name} in 'model_configs' not found, must be in {list(self.all_ism_configs.keys())}"
            )

        for key, value in model_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key '{key}' in 'model_configs' mapping.")


@dataclass
class ISEFlowGrISInputs:
    """Inputs for an ISEFlow-GrIS prediction.

    Expects pre-computed anomaly arrays (``aSMB``, ``aST``).  If you have raw
    absolute forcing values instead, use the alternative constructor::

        inputs = ISEFlowGrISInputs.from_absolute_forcings(
            year=..., sector=..., smb=..., st=...,
            ocean_thermal_forcing=..., basin_runoff=...,
            aogcm="hadgem2-es_rcp85",  # or custom_climatology={...}
            **ism_config_kwargs,
        )

    ``from_absolute_forcings()`` subtracts the ISMIP6 1960-1989 MAR climatological
    baseline automatically.  Pass ``aogcm`` for a bundled ISMIP6 model or
    ``custom_climatology`` (dict with keys ``'smb'``, ``'st'``) for a CMIP
    model not in the bundled climatology.
    """

    # Forcing data
    year: np.ndarray
    sector: np.ndarray | int
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
    initial_year: int | None = None
    numerics: str | None = None
    ice_flow_model: str | None = None
    initialization: str | None = None
    initial_smb: str | None = None
    velocity: str | None = None
    bedrock_topography: str | None = None
    surface_thickness: str | None = None
    geothermal_heat_flux: str | None = None
    res_min: str | None = None
    res_max: str | None = None

    # ISMIP6 model to emulate
    model_configs: str | None = None

    # ISEFlow *model weights* version (distinct from the ise-py package version)
    version: str = "v1.1.0"

    # ------------------------------------------------------------------
    # Alternative constructor: raw (non-anomaly) forcing values
    # ------------------------------------------------------------------

    @classmethod
    def from_absolute_forcings(
        cls,
        year: np.ndarray,
        sector: int,
        smb: np.ndarray,
        st: np.ndarray,
        ocean_thermal_forcing: np.ndarray,
        basin_runoff: np.ndarray,
        aogcm: str | None = None,
        custom_climatology: dict | None = None,
        **kwargs,
    ) -> "ISEFlowGrISInputs":
        """Construct ISEFlowGrISInputs from raw (non-anomaly) atmospheric forcings.

        Subtracts the ISMIP6 1960-1989 MAR climatological baseline from each
        atmospheric variable to produce the anomaly arrays (``aSMB``, ``aST``)
        required by the model.  Ocean variables (``ocean_thermal_forcing``,
        ``basin_runoff``) are absolute values and are passed through unchanged.

        Exactly one of ``aogcm`` or ``custom_climatology`` must be provided.

        Parameters
        ----------
        year : np.ndarray
            Years (86 values, 2015-2100).
        sector : int
            GrIS drainage basin number (1-6).
        smb : np.ndarray
            Raw surface mass balance (86 values, in MAR units — mm w.e. yr⁻¹
            or equivalent, consistent with the reference file).
        st : np.ndarray
            Raw surface temperature (86 values, K or °C, consistent with
            the MAR reference).
        ocean_thermal_forcing : np.ndarray
            Ocean thermal forcing (86 values).  Passed through unchanged.
        basin_runoff : np.ndarray
            Basin-integrated runoff (86 values).  Passed through unchanged.
        aogcm : str, optional
            AOGCM name to look up in the bundled ISMIP6 climatology
            (e.g. ``'hadgem2-es_rcp85'``).  Common alternate spellings are
            normalised automatically.
        custom_climatology : dict, optional
            Baseline means for a CMIP model not in the bundled climatology.
            Must contain keys ``'smb'`` and ``'st'`` in MAR units.
        **kwargs
            All remaining keyword arguments are forwarded to
            ``ISEFlowGrISInputs.__init__`` (e.g. ISM config fields such as
            ``numerics``, ``ice_flow_model``, ``model_configs``, etc.).

        Returns
        -------
        ISEFlowGrISInputs
            Fully validated inputs object ready for ``model.predict()``.

        Examples
        --------
        Using a bundled ISMIP6 climatology::

            inputs = ISEFlowGrISInputs.from_absolute_forcings(
                year=np.arange(2015, 2101),
                sector=1,
                smb=smb_array,
                st=st_array,
                ocean_thermal_forcing=otf_array,
                basin_runoff=runoff_array,
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

        Using a custom climatology for a new CMIP model::

            inputs = ISEFlowGrISInputs.from_absolute_forcings(
                year=np.arange(2015, 2101),
                sector=1,
                smb=smb_array,
                st=st_array,
                ocean_thermal_forcing=otf_array,
                basin_runoff=runoff_array,
                custom_climatology={"smb": -241.2, "st": -22.8},
                initial_year=1990, ...
            )
        """
        from ise.data.anomaly import AnomalyConverter

        converter = AnomalyConverter("GrIS")
        anomalies = converter.compute_gris(
            sector=sector,
            smb=smb,
            st=st,
            aogcm=aogcm,
            custom_climatology=custom_climatology,
        )

        return cls(
            year=year,
            sector=sector,
            aSMB=anomalies["aSMB"],
            aST=anomalies["aST"],
            ocean_thermal_forcing=ocean_thermal_forcing,
            basin_runoff=basin_runoff,
            **kwargs,
        )

    @classmethod
    def from_raw_values(cls, *args, **kwargs):
        """Deprecated — use ``from_absolute_forcings`` instead."""
        warnings.warn(
            "from_raw_values() is deprecated; use from_absolute_forcings() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return cls.from_absolute_forcings(*args, **kwargs)

    # Validation logic runs after the object is created
    def __post_init__(self):
        self._assign_model_configs(self.model_configs) if self.model_configs else None
        self._check_inputs()
        self._map_args()
        self._convert_arrays()
        self.df = None

    def _check_inputs(
        self,
    ):
        """Validate all GrIS input parameters and normalise array encodings.

        Converts ``year`` from calendar years (2015-2100) to model-internal
        encoding (1-86), broadcasts a scalar ``sector`` to an array, and raises
        ``ValueError`` for any out-of-range or mutually exclusive parameter
        combinations.
        """

        if self.year[0] == 2015:
            self.year = self.year - 2015 + 1  # convert 2015-2100 → 1-86 (model encoding)

        if isinstance(self.sector, int):
            self.sector = np.ones_like(self.year) * self.sector

        if not isinstance(self.initial_year, int):
            raise ValueError("initial_year must be an integer")

        # velocity, surface_thickness, and geothermal_heat_flux are legitimately absent
        # for some models (stored as None/'None'). Only the core ISM config fields are required.
        if not self.model_configs and (
            not self.numerics
            or not self.ice_flow_model
            or not self.initialization
            or not self.initial_smb
            or not self.bedrock_topography
            or not self.res_min
            or not self.res_max
            or self.standard_ocean_forcing is None
            or not self.ocean_sensitivity
            or self.ice_shelf_fracture is None
        ):
            raise ValueError(
                "Either 'model_configs' must be provided or all individual configuration parameters must be specified."
            )

        elif self.model_configs and (
            self.numerics
            or self.ice_flow_model
            or self.initialization
            or self.initial_smb
            or self.velocity
            or self.bedrock_topography
            or self.surface_thickness
            or self.geothermal_heat_flux
            or self.res_min
            or self.res_max
            or self.standard_ocean_forcing
            or self.ocean_sensitivity
        ):
            warnings.warn(
                "Both 'model_configs' and individual configuration parameters are provided. 'model_configs' will take precedence."
            )

        if str(self.numerics).lower() not in ("fe", "fv", "fd", "fd/fv"):
            raise ValueError("numerics must be one of 'fe', 'fv', 'fd', or 'fd/fv'")

        if str(self.ice_flow_model) not in (
            "ho",
            "ssa",
            "sia",
            "hybrid",
        ):
            raise ValueError("ice_flow_model must be one of 'ho', 'ssa', 'sia', or 'hybrid'")

        if str(self.initialization) not in (
            "dav",
            "cyc/nds",
            "sp/ndm",
            "sp/dav",
            "sp/das",
            "cyc/ndm",
            "sp/dai",
            "cyc/dai",
            "sp/nds",
        ):
            raise ValueError(
                "initialization must be one of 'dav', 'cyc/nds', 'sp/ndm', 'sp/dav', 'sp/das', 'cyc/ndm', 'sp/dai', 'cyc/dai', or 'sp/nds'"
            )

        if str(self.initial_smb) not in ("ra3", "hir", "ismb", "box/mar", "box/ra3", "mar", "ra1"):
            raise ValueError(
                "initial_smb must be one of 'ra3', 'hir', 'ismb', 'box/mar', 'box/ra3', 'mar', or 'ra1'"
            )

        if str(self.bedrock_topography) not in ("morlighem", "bamber"):
            raise ValueError("bed must be one of 'morlighem' or 'bamber'")

        if str(self.surface_thickness) not in ("None", "morlighem"):
            raise ValueError("surface_thickness must be one of 'None' or 'morlighem'")

        if str(self.velocity) not in ("joughin", "rignot", "None"):
            raise ValueError("velocity must be one of 'joughin', 'rignot', or 'None'")

        if str(self.geothermal_heat_flux) not in ("g", "None", "sr", "mix"):
            raise ValueError("geothermal_heat_flux must be one of 'g', 'None', 'sr', or 'mix'")

        if float(self.res_min) not in [
            0.2,
            0.25,
            0.5,
            0.75,
            0.9,
            1.0,
            1.2,
            2.0,
            3.0,
            4.0,
            5.0,
            8.0,
            16.0,
        ]:
            raise ValueError(
                "res_min must be one of 0.2, 0.25, 0.5, 0.75, 0.9, 1., 1.2, 2., 3., 4., 5., 8., or 16."
            )

        if float(self.res_max) not in [
            0.9,
            2.0,
            4.0,
            4.8,
            5.0,
            7.5,
            8.0,
            14.0,
            15.0,
            16.0,
            20.0,
            25.0,
            30.0,
        ]:
            raise ValueError(
                "res_max must be one of 0.9, 2., 4., 4.8, 5., 7.5, 8., 14., 15., 16., 20., 25., or 30."
            )

        if not isinstance(self.ice_shelf_fracture, bool):
            raise ValueError("ice_shelf_fracture must be a boolean")

        if not isinstance(self.standard_ocean_forcing, bool):
            raise ValueError("standard_ocean_forcing must be a boolean")

    def _map_args(
        self,
    ):
        """Map user-facing string values to the internal encodings expected by the model.

        For example, ``numerics='fe'`` becomes ``'FE'``,
        ``ice_flow_model='ho'`` becomes ``'HO'``, etc.  Numeric resolution
        fields (``res_min``, ``res_max``) are converted to string representations
        of their float values (e.g. ``1.0`` → ``'1.0'``).
        """

        # map from accepted input to how the model expects variable names
        arg_map = {
            "numerics": {
                "fe": "FE",
                "fv": "FV",
                "fd": "FD",
                "fd/fv": "FD_FV5",
            },
            "ice_flow_model": {
                "ho": "HO",
                "ssa": "SSA",
                "sia": "SIA",
                "hybrid": "HYB",
            },
            "initialization": {
                "dav": "DAV",
                "cyc/nds": "CYC_NDS",
                "sp/ndm": "SP_NDM",
                "sp/dav": "SP_DAV",
                "sp/das": "SP_DAS",
                "cyc/ndm": "CYC_NDM",
                "sp/dai": "SP_DAI",
                "cyc/dai": "CYC_DAI",
                "sp/nds": "SP_NDS",
            },
            "initial_smb": {
                "ra3": "RA3",
                "hir": "HIR",
                "ismb": "ISMB",
                "box/mar": "BOX_MAR",
                "box/ra3": "BOX_RA3",
                "mar": "MAR",
                "ra1": "RA1",
            },
            "bedrock_topography": {
                "morlighem": "M",
                "bamber": "B",
            },
            "surface_thickness": {
                "None": None,
                "morlighem": "M",
            },
            "geothermal_heat_flux": {
                "g": "G",
                "None": None,
                "sr": "SR",
                "mix": "MIX",
            },
            "velocity": {
                "joughin": "J",
                "rignot": "RM",
                "None": None,
            },
            "ocean_sensitivity": {
                "low": "Low",
                "medium": "Medium",
                "high": "High",
            },
        }

        for key, value in vars(self).items():
            current_value = getattr(self, key)

            if key == "res_min" or key == "res_max":
                new_value = str(float(current_value))
                setattr(self, key, new_value)

            elif key in arg_map:
                # Normalise Python None to the string 'None' so the lookup succeeds
                lookup_key = "None" if current_value is None else current_value
                new_value = arg_map[key][lookup_key]
                setattr(self, key, new_value)

    def _convert_arrays(self):
        """Coerce all GrIS forcing arrays to ``numpy.ndarray``."""

        forcings = ("year", "aST", "aSMB", "ocean_thermal_forcing", "basin_runoff")

        for arr_name in forcings:
            forcing_array = getattr(self, arr_name)

            try:
                setattr(self, arr_name, np.array(forcing_array))
            except Exception as e:
                raise ValueError(
                    f"Variable {arr_name} must be a numpy array, received {type(forcing_array)}."
                ) from e

    def to_df(self):
        """Convert the dataclass fields to a pandas DataFrame.

        Returns:
            pandas.DataFrame: One row per timestep (86 rows) with all forcing
            and configuration columns needed by ``ISEFlow_GrIS.process()``.
        """

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
            "Ocean forcing": "Standard" if self.standard_ocean_forcing else "Open",
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
            return (
                f"array(shape={a.shape}, min={a.min():.4g}, max={a.max():.4g}, mean={a.mean():.4g})"
            )

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
        with open(characteristics_json) as file:
            characteristics = json.load(file)

        if model_name in characteristics:
            model_config = characteristics[model_name]
        else:
            raise ValueError(
                f"Model name {model_name} in 'model_configs' not found, must be in {list(characteristics.keys())}"
            )

        for key, value in model_config.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid configuration key '{key}' in 'model_configs' mapping.")
