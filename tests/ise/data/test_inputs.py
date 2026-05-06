"""Pure unit tests for ISEFlowAISInputs and ISEFlowGrISInputs.

These tests use only synthetic data and do not require external dataset files.
They exercise the dataclass constructors, validation logic, and to_df() output.
"""

import numpy as np
import pandas as pd
import pytest

from ise.data.inputs import ISEFlowAISInputs, ISEFlowGrISInputs

PROJ_LEN = 86
YEAR = np.arange(2015, 2101)  # 86 calendar years


# ---------------------------------------------------------------------------
# AIS helpers
# ---------------------------------------------------------------------------

def _ais_kwargs(**overrides):
    """Minimal valid kwargs for ISEFlowAISInputs."""
    rng = np.random.default_rng(0)
    base = dict(
        year=YEAR.copy(),
        sector=5,
        pr_anomaly=rng.random(PROJ_LEN) * 1e-5,
        evspsbl_anomaly=rng.random(PROJ_LEN) * 1e-6,
        smb_anomaly=rng.random(PROJ_LEN) * 1e-5,
        ts_anomaly=rng.random(PROJ_LEN) * 5.0,
        ocean_thermal_forcing=rng.random(PROJ_LEN) * 2.0,
        ocean_salinity=rng.random(PROJ_LEN) * 35.0,
        ocean_temperature=rng.random(PROJ_LEN) * 2.0,
        ice_shelf_fracture=False,
        ocean_sensitivity="medium",
        numerics="fd",
        stress_balance="hybrid",
        resolution="8",
        init_method="eq",
        initial_year=2005,
        melt_in_floating_cells="sub-grid",
        icefront_migration="str",
        ocean_forcing_type="open",
        open_melt_type="quad",
        standard_melt_type="nonlocal",
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ISEFlowAISInputs — construction and to_df
# ---------------------------------------------------------------------------

class TestISEFlowAISInputs:
    def test_construction_succeeds(self):
        inputs = ISEFlowAISInputs(**_ais_kwargs())
        assert isinstance(inputs, ISEFlowAISInputs)

    def test_to_df_returns_dataframe(self):
        inputs = ISEFlowAISInputs(**_ais_kwargs())
        df = inputs.to_df()
        assert isinstance(df, pd.DataFrame)

    def test_to_df_has_86_rows(self):
        inputs = ISEFlowAISInputs(**_ais_kwargs())
        df = inputs.to_df()
        assert len(df) == PROJ_LEN

    def test_to_df_contains_forcing_columns(self):
        inputs = ISEFlowAISInputs(**_ais_kwargs())
        df = inputs.to_df()
        for col in ("pr_anomaly", "evspsbl_anomaly", "smb_anomaly", "ts_anomaly"):
            assert col in df.columns, f"Missing column: {col}"

    def test_invalid_numerics_raises(self):
        kwargs = _ais_kwargs()
        kwargs["numerics"] = "invalid_numerics"
        with pytest.raises(ValueError):
            ISEFlowAISInputs(**kwargs)

    def test_sector_stored_as_array(self):
        inputs = ISEFlowAISInputs(**_ais_kwargs())
        df = inputs.to_df()
        assert "sector" in df.columns
        assert len(df["sector"]) == PROJ_LEN

    def test_year_column_present(self):
        inputs = ISEFlowAISInputs(**_ais_kwargs())
        df = inputs.to_df()
        assert "year" in df.columns

    def test_from_absolute_forcings_with_custom_climatology(self):
        """from_absolute_forcings should produce the same type as direct construction."""
        rng = np.random.default_rng(1)
        custom_clim = {"pr": 1e-5, "evspsbl": 4e-7, "smb": 8e-6, "ts": 248.0}
        inputs = ISEFlowAISInputs.from_absolute_forcings(
            year=YEAR.copy(),
            sector=5,
            pr=rng.random(PROJ_LEN) * 1e-4,
            evspsbl=rng.random(PROJ_LEN) * 1e-5,
            smb=rng.random(PROJ_LEN) * 1e-4,
            ts=rng.random(PROJ_LEN) * 10 + 245,
            ocean_thermal_forcing=rng.random(PROJ_LEN),
            ocean_salinity=rng.random(PROJ_LEN) * 35,
            ocean_temperature=rng.random(PROJ_LEN),
            custom_climatology=custom_clim,
            ice_shelf_fracture=False,
            ocean_sensitivity="medium",
            numerics="fd",
            stress_balance="hybrid",
            resolution="8",
            init_method="eq",
            initial_year=2005,
            melt_in_floating_cells="sub-grid",
            icefront_migration="str",
            ocean_forcing_type="open",
            open_melt_type="quad",
            standard_melt_type="nonlocal",
        )
        assert isinstance(inputs, ISEFlowAISInputs)
        df = inputs.to_df()
        assert len(df) == PROJ_LEN

    def test_from_absolute_forcings_anomalies_match_subtraction(self):
        """Anomalies from from_absolute_forcings == raw - climatology baseline."""
        rng = np.random.default_rng(2)
        pr_raw = rng.random(PROJ_LEN) * 1e-4
        custom_clim = {"pr": 1.3e-5, "evspsbl": 4e-7, "smb": 8e-6, "ts": 248.0}
        inputs = ISEFlowAISInputs.from_absolute_forcings(
            year=YEAR.copy(),
            sector=5,
            pr=pr_raw,
            evspsbl=rng.random(PROJ_LEN) * 1e-5,
            smb=rng.random(PROJ_LEN) * 1e-4,
            ts=rng.random(PROJ_LEN) * 10 + 245,
            ocean_thermal_forcing=rng.random(PROJ_LEN),
            ocean_salinity=rng.random(PROJ_LEN) * 35,
            ocean_temperature=rng.random(PROJ_LEN),
            custom_climatology=custom_clim,
            ice_shelf_fracture=False,
            ocean_sensitivity="medium",
            numerics="fd",
            stress_balance="hybrid",
            resolution="8",
            init_method="eq",
            initial_year=2005,
            melt_in_floating_cells="sub-grid",
            icefront_migration="str",
            ocean_forcing_type="open",
            open_melt_type="quad",
            standard_melt_type="nonlocal",
        )
        np.testing.assert_allclose(
            inputs.pr_anomaly, pr_raw - custom_clim["pr"], rtol=1e-9
        )

    def test_from_raw_values_deprecated(self):
        """from_raw_values should emit DeprecationWarning and still work."""
        rng = np.random.default_rng(3)
        custom_clim = {"pr": 1e-5, "evspsbl": 4e-7, "smb": 8e-6, "ts": 248.0}
        with pytest.warns(DeprecationWarning, match="from_raw_values"):
            inputs = ISEFlowAISInputs.from_raw_values(
                year=YEAR.copy(),
                sector=5,
                pr=rng.random(PROJ_LEN) * 1e-4,
                evspsbl=rng.random(PROJ_LEN) * 1e-5,
                smb=rng.random(PROJ_LEN) * 1e-4,
                ts=rng.random(PROJ_LEN) * 10 + 245,
                ocean_thermal_forcing=rng.random(PROJ_LEN),
                ocean_salinity=rng.random(PROJ_LEN) * 35,
                ocean_temperature=rng.random(PROJ_LEN),
                custom_climatology=custom_clim,
                ice_shelf_fracture=False,
                ocean_sensitivity="medium",
                numerics="fd",
                stress_balance="hybrid",
                resolution="8",
                init_method="eq",
                initial_year=2005,
                melt_in_floating_cells="sub-grid",
                icefront_migration="str",
                ocean_forcing_type="open",
                open_melt_type="quad",
                standard_melt_type="nonlocal",
            )
        assert isinstance(inputs, ISEFlowAISInputs)


# ---------------------------------------------------------------------------
# GrIS helpers
# ---------------------------------------------------------------------------

def _gris_kwargs(**overrides):
    """Minimal valid kwargs for ISEFlowGrISInputs."""
    rng = np.random.default_rng(10)
    base = dict(
        year=YEAR.copy(),
        sector=3,
        aSMB=rng.random(PROJ_LEN) * 200 - 100,
        aST=rng.random(PROJ_LEN) * 5.0 - 2.0,
        ocean_thermal_forcing=rng.random(PROJ_LEN) * 1.5,
        basin_runoff=rng.random(PROJ_LEN) * 500,
        ice_shelf_fracture=False,
        ocean_sensitivity="medium",
        standard_ocean_forcing=True,
        initial_year=2005,
        numerics="fd",
        ice_flow_model="ho",
        initialization="cyc/dai",
        initial_smb="mar",
        velocity="joughin",
        bedrock_topography="bamber",
        surface_thickness="morlighem",
        geothermal_heat_flux="g",
        res_min=1.0,
        res_max=5.0,
    )
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# ISEFlowGrISInputs — construction and to_df
# ---------------------------------------------------------------------------

class TestISEFlowGrISInputs:
    def test_construction_succeeds(self):
        inputs = ISEFlowGrISInputs(**_gris_kwargs())
        assert isinstance(inputs, ISEFlowGrISInputs)

    def test_to_df_returns_dataframe(self):
        inputs = ISEFlowGrISInputs(**_gris_kwargs())
        df = inputs.to_df()
        assert isinstance(df, pd.DataFrame)

    def test_to_df_has_86_rows(self):
        inputs = ISEFlowGrISInputs(**_gris_kwargs())
        df = inputs.to_df()
        assert len(df) == PROJ_LEN

    def test_to_df_contains_forcing_columns(self):
        inputs = ISEFlowGrISInputs(**_gris_kwargs())
        df = inputs.to_df()
        for col in ("aSMB", "aST", "thermal_forcing", "basin_runoff"):
            assert col in df.columns, f"Missing column: {col}"

    def test_invalid_numerics_raises(self):
        kwargs = _gris_kwargs()
        kwargs["numerics"] = "invalid_numerics"
        with pytest.raises(ValueError):
            ISEFlowGrISInputs(**kwargs)

    def test_sector_column_present(self):
        inputs = ISEFlowGrISInputs(**_gris_kwargs())
        df = inputs.to_df()
        assert "sector" in df.columns

    def test_year_column_present(self):
        inputs = ISEFlowGrISInputs(**_gris_kwargs())
        df = inputs.to_df()
        assert "year" in df.columns
