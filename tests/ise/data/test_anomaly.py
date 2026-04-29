"""Tests for ise/data/anomaly.py — AnomalyConverter.

Covers:
  - CSV integrity (shape, columns, NaN rules)
  - Constructor validation
  - list_aogcms() / get_climatology() round-trips
  - compute_ais(): correct subtraction against bundled CSV values
  - compute_ais(): mrro handling (present, absent clim, not provided)
  - compute_ais(): custom_climatology path and missing-key errors
  - compute_ais(): argument mutex (neither/both aogcm+custom raises)
  - compute_ais(): array-length / dimensionality validation
  - compute_gris(): correct subtraction against bundled CSV values
  - compute_gris(): custom_climatology path and error paths
  - AOGCM alias normalisation for both ice sheets
"""

import warnings

import numpy as np
import pandas as pd
import pytest

from ise.data.anomaly import AnomalyConverter, _AIS_ALIASES, _GRIS_ALIASES

# ---------------------------------------------------------------------------
# Constants pulled from the bundled CSV for regression tests.
# These are the exact values stored in the CSV; any discrepancy means either
# the CSV or the subtraction logic is broken.
# ---------------------------------------------------------------------------
_AIS_CSV = "ise/data/data_files/AIS_atmos_climatologies.csv"
_GRIS_CSV = "ise/data/data_files/GrIS_atmos_climatologies.csv"

# AIS — noresm1-m_rcp85, sector 5 (has mrro_clim)
_AIS_AOGCM = "noresm1-m_rcp85"
_AIS_SECTOR = 5
_AIS_PR_CLIM        = 9.090845196624286e-06
_AIS_EVSPSBL_CLIM   = 4.421370647378353e-07
_AIS_SMB_CLIM       = 8.355646968993824e-06
_AIS_TS_CLIM        = 245.5599822998047
_AIS_MRRO_CLIM      = 2.9306085025382345e-07

# AIS — csiro-mk3.6_rcp85, sector 3 (mrro_clim is NaN)
_AIS_NO_MRRO_AOGCM  = "csiro-mk3.6_rcp85"
_AIS_NO_MRRO_SECTOR = 3

# GrIS — noresm1-m_rcp85, sector 2
_GRIS_AOGCM   = "noresm1-m_rcp85"
_GRIS_SECTOR  = 2
_GRIS_SMB_CLIM = 17.879426956176758
_GRIS_ST_CLIM  = -21.672592163085938

PROJ_LEN = 86


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng_array(seed=0, length=PROJ_LEN, base=0.0, scale=1.0):
    rng = np.random.default_rng(seed)
    return rng.random(length) * scale + base


# ---------------------------------------------------------------------------
# 1. CSV integrity
# ---------------------------------------------------------------------------

class TestCSVIntegrity:
    def test_ais_csv_shape(self):
        df = pd.read_csv(_AIS_CSV)
        assert len(df) == 270, f"Expected 270 rows (15 AOGCMs × 18 sectors), got {len(df)}"

    def test_ais_csv_columns(self):
        df = pd.read_csv(_AIS_CSV)
        required = {"aogcm", "sector", "pr_clim", "evspsbl_clim", "smb_clim", "ts_clim", "mrro_clim"}
        assert required.issubset(set(df.columns))

    def test_ais_csv_no_nan_in_core_vars(self):
        df = pd.read_csv(_AIS_CSV)
        for col in ("pr_clim", "evspsbl_clim", "smb_clim", "ts_clim"):
            assert df[col].notna().all(), f"Unexpected NaN in AIS column '{col}'"

    def test_ais_csv_mrro_nan_only_for_known_aogcms(self):
        df = pd.read_csv(_AIS_CSV)
        nan_aogcms = set(df[df["mrro_clim"].isna()]["aogcm"].unique())
        expected_nan = {"csiro-mk3.6_rcp85", "ipsl-cm5-mr_rcp85", "ipsl-cm5-mr_rcp26"}
        assert nan_aogcms == expected_nan, (
            f"mrro_clim NaN AOGCMs changed.\n  expected: {expected_nan}\n  got: {nan_aogcms}"
        )

    def test_gris_csv_shape(self):
        df = pd.read_csv(_GRIS_CSV)
        assert len(df) == 72, f"Expected 72 rows (12 AOGCMs × 6 sectors), got {len(df)}"

    def test_gris_csv_columns(self):
        df = pd.read_csv(_GRIS_CSV)
        assert {"aogcm", "sector", "smb_clim", "st_clim"}.issubset(set(df.columns))

    def test_gris_csv_no_nan(self):
        df = pd.read_csv(_GRIS_CSV)
        for col in ("smb_clim", "st_clim"):
            assert df[col].notna().all(), f"Unexpected NaN in GrIS column '{col}'"

    def test_ais_all_sectors_present(self):
        df = pd.read_csv(_AIS_CSV)
        assert set(df["sector"].unique()) == set(range(1, 19))

    def test_gris_all_sectors_present(self):
        df = pd.read_csv(_GRIS_CSV)
        assert set(df["sector"].unique()) == set(range(1, 7))


# ---------------------------------------------------------------------------
# 2. Constructor validation
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_ais_upper(self):
        conv = AnomalyConverter("AIS")
        assert conv.ice_sheet == "AIS"

    def test_gris_mixed_case(self):
        conv = AnomalyConverter("gris")
        assert conv.ice_sheet == "GrIS"

    def test_gris_canonical_case(self):
        conv = AnomalyConverter("GrIS")
        assert conv.ice_sheet == "GrIS"

    def test_invalid_ice_sheet(self):
        with pytest.raises(ValueError, match="ice_sheet"):
            AnomalyConverter("WAIS")

    def test_climatology_lazy_loaded(self):
        conv = AnomalyConverter("AIS")
        assert conv._clim is None
        _ = conv.climatology
        assert conv._clim is not None

    def test_climatology_cached(self):
        conv = AnomalyConverter("AIS")
        df1 = conv.climatology
        df2 = conv.climatology
        assert df1 is df2


# ---------------------------------------------------------------------------
# 3. list_aogcms / get_climatology
# ---------------------------------------------------------------------------

class TestListAndGetClimatology:
    def test_ais_list_aogcms_sorted(self):
        names = AnomalyConverter("AIS").list_aogcms()
        assert names == sorted(names)
        assert "noresm1-m_rcp85" in names

    def test_gris_list_aogcms_sorted(self):
        names = AnomalyConverter("GrIS").list_aogcms()
        assert names == sorted(names)
        assert "noresm1-m_rcp85" in names

    def test_ais_get_climatology_values(self):
        row = AnomalyConverter("AIS").get_climatology(_AIS_AOGCM, _AIS_SECTOR)
        assert pytest.approx(row["pr_clim"],      rel=1e-9) == _AIS_PR_CLIM
        assert pytest.approx(row["evspsbl_clim"], rel=1e-9) == _AIS_EVSPSBL_CLIM
        assert pytest.approx(row["smb_clim"],     rel=1e-9) == _AIS_SMB_CLIM
        assert pytest.approx(row["ts_clim"],      rel=1e-9) == _AIS_TS_CLIM
        assert pytest.approx(row["mrro_clim"],    rel=1e-9) == _AIS_MRRO_CLIM

    def test_gris_get_climatology_values(self):
        row = AnomalyConverter("GrIS").get_climatology(_GRIS_AOGCM, _GRIS_SECTOR)
        assert pytest.approx(row["smb_clim"], rel=1e-9) == _GRIS_SMB_CLIM
        assert pytest.approx(row["st_clim"],  rel=1e-9) == _GRIS_ST_CLIM

    def test_ais_get_climatology_unknown_aogcm_raises(self):
        with pytest.raises(KeyError):
            AnomalyConverter("AIS").get_climatology("not-a-real-model_rcp99", 1)

    def test_gris_get_climatology_unknown_aogcm_raises(self):
        with pytest.raises(KeyError):
            AnomalyConverter("GrIS").get_climatology("not-a-real-model_rcp99", 1)

    def test_ais_get_climatology_unknown_sector_raises(self):
        with pytest.raises(KeyError):
            AnomalyConverter("AIS").get_climatology(_AIS_AOGCM, sector=99)


# ---------------------------------------------------------------------------
# 4. compute_ais — core correctness
# ---------------------------------------------------------------------------

class TestComputeAIS:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("AIS")

    @pytest.fixture(scope="class")
    def arrays(self):
        rng = np.random.default_rng(42)
        return {
            "pr":      rng.random(PROJ_LEN) * 1e-4,
            "evspsbl": rng.random(PROJ_LEN) * 1e-4,
            "smb":     rng.random(PROJ_LEN) * 1e-4,
            "ts":      rng.random(PROJ_LEN) * 30 + 240,
            "mrro":    rng.random(PROJ_LEN) * 1e-6,
        }

    def test_output_keys(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        assert set(result.keys()) == {"pr_anomaly", "evspsbl_anomaly", "smb_anomaly", "ts_anomaly"}

    def test_pr_anomaly_values(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        expected = arrays["pr"] - _AIS_PR_CLIM
        np.testing.assert_allclose(result["pr_anomaly"], expected, rtol=1e-9)

    def test_evspsbl_anomaly_values(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        expected = arrays["evspsbl"] - _AIS_EVSPSBL_CLIM
        np.testing.assert_allclose(result["evspsbl_anomaly"], expected, rtol=1e-9)

    def test_smb_anomaly_values(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        expected = arrays["smb"] - _AIS_SMB_CLIM
        np.testing.assert_allclose(result["smb_anomaly"], expected, rtol=1e-9)

    def test_ts_anomaly_values(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        expected = arrays["ts"] - _AIS_TS_CLIM
        np.testing.assert_allclose(result["ts_anomaly"], expected, rtol=1e-9)

    def test_output_lengths(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        for key, arr in result.items():
            assert len(arr) == PROJ_LEN, f"{key} has wrong length"

    def test_output_is_numpy(self, converter, arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        for key, arr in result.items():
            assert isinstance(arr, np.ndarray), f"{key} is not ndarray"


# ---------------------------------------------------------------------------
# 5. compute_ais — mrro handling
# ---------------------------------------------------------------------------

class TestComputeAISMrro:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("AIS")

    @pytest.fixture(scope="class")
    def base_arrays(self):
        rng = np.random.default_rng(7)
        return {k: rng.random(PROJ_LEN) for k in ("pr", "evspsbl", "smb", "ts", "mrro")}

    def test_mrro_anomaly_included_when_clim_available(self, converter, base_arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=base_arrays["pr"], evspsbl=base_arrays["evspsbl"],
            smb=base_arrays["smb"], ts=base_arrays["ts"],
            mrro=base_arrays["mrro"],
        )
        assert "mrro_anomaly" in result
        expected = base_arrays["mrro"] - _AIS_MRRO_CLIM
        np.testing.assert_allclose(result["mrro_anomaly"], expected, rtol=1e-9)

    def test_mrro_anomaly_absent_when_not_provided(self, converter, base_arrays):
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=base_arrays["pr"], evspsbl=base_arrays["evspsbl"],
            smb=base_arrays["smb"], ts=base_arrays["ts"],
        )
        assert "mrro_anomaly" not in result

    def test_mrro_warns_when_clim_unavailable(self, converter, base_arrays):
        with pytest.warns(UserWarning, match="mrro"):
            result = converter.compute_ais(
                sector=_AIS_NO_MRRO_SECTOR, aogcm=_AIS_NO_MRRO_AOGCM,
                pr=base_arrays["pr"], evspsbl=base_arrays["evspsbl"],
                smb=base_arrays["smb"], ts=base_arrays["ts"],
                mrro=base_arrays["mrro"],
            )
        assert "mrro_anomaly" not in result

    def test_mrro_not_warned_when_not_provided_no_clim(self, converter, base_arrays):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            converter.compute_ais(
                sector=_AIS_NO_MRRO_SECTOR, aogcm=_AIS_NO_MRRO_AOGCM,
                pr=base_arrays["pr"], evspsbl=base_arrays["evspsbl"],
                smb=base_arrays["smb"], ts=base_arrays["ts"],
            )


# ---------------------------------------------------------------------------
# 6. compute_ais — custom_climatology
# ---------------------------------------------------------------------------

class TestComputeAISCustomClimatology:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("AIS")

    @pytest.fixture(scope="class")
    def arrays(self):
        rng = np.random.default_rng(99)
        return {k: rng.random(PROJ_LEN) for k in ("pr", "evspsbl", "smb", "ts")}

    _custom = {"pr": 1.3e-5, "evspsbl": 4e-6, "smb": 9e-6, "ts": 253.7}

    def test_custom_subtraction_correct(self, converter, arrays):
        result = converter.compute_ais(
            sector=1, custom_climatology=self._custom,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        np.testing.assert_allclose(result["pr_anomaly"],      arrays["pr"]      - 1.3e-5, rtol=1e-9)
        np.testing.assert_allclose(result["evspsbl_anomaly"], arrays["evspsbl"] - 4e-6,   rtol=1e-9)
        np.testing.assert_allclose(result["smb_anomaly"],     arrays["smb"]     - 9e-6,   rtol=1e-9)
        np.testing.assert_allclose(result["ts_anomaly"],      arrays["ts"]      - 253.7,  rtol=1e-9)

    def test_custom_missing_key_raises(self, converter, arrays):
        bad_custom = {"pr": 1e-5, "evspsbl": 4e-6, "ts": 253.7}  # missing smb
        with pytest.raises(ValueError, match="smb"):
            converter.compute_ais(
                sector=1, custom_climatology=bad_custom,
                pr=arrays["pr"], evspsbl=arrays["evspsbl"],
                smb=arrays["smb"], ts=arrays["ts"],
            )

    def test_neither_arg_raises(self, converter, arrays):
        with pytest.raises(ValueError):
            converter.compute_ais(
                sector=1,
                pr=arrays["pr"], evspsbl=arrays["evspsbl"],
                smb=arrays["smb"], ts=arrays["ts"],
            )

    def test_both_args_raise(self, converter, arrays):
        with pytest.raises(ValueError):
            converter.compute_ais(
                sector=1, aogcm=_AIS_AOGCM, custom_climatology=self._custom,
                pr=arrays["pr"], evspsbl=arrays["evspsbl"],
                smb=arrays["smb"], ts=arrays["ts"],
            )


# ---------------------------------------------------------------------------
# 7. compute_ais — array validation
# ---------------------------------------------------------------------------

class TestComputeAISArrayValidation:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("AIS")

    def _good_arrays(self):
        rng = np.random.default_rng(0)
        return {k: rng.random(PROJ_LEN) for k in ("pr", "evspsbl", "smb", "ts")}

    def test_wrong_length_raises(self, converter):
        a = self._good_arrays()
        with pytest.raises(ValueError, match="86"):
            converter.compute_ais(
                sector=1, aogcm=_AIS_AOGCM,
                pr=a["pr"][:50], evspsbl=a["evspsbl"],
                smb=a["smb"], ts=a["ts"],
            )

    def test_2d_array_raises(self, converter):
        a = self._good_arrays()
        with pytest.raises(ValueError, match="1-D"):
            converter.compute_ais(
                sector=1, aogcm=_AIS_AOGCM,
                pr=a["pr"].reshape(2, 43), evspsbl=a["evspsbl"],
                smb=a["smb"], ts=a["ts"],
            )

    def test_list_input_accepted(self, converter):
        a = self._good_arrays()
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=a["pr"].tolist(), evspsbl=a["evspsbl"].tolist(),
            smb=a["smb"].tolist(), ts=a["ts"].tolist(),
        )
        assert "pr_anomaly" in result

    def test_mrro_wrong_length_raises(self, converter):
        a = self._good_arrays()
        with pytest.raises(ValueError, match="86"):
            converter.compute_ais(
                sector=1, aogcm=_AIS_AOGCM,
                pr=a["pr"], evspsbl=a["evspsbl"],
                smb=a["smb"], ts=a["ts"],
                mrro=np.zeros(50),
            )


# ---------------------------------------------------------------------------
# 8. compute_gris — core correctness
# ---------------------------------------------------------------------------

class TestComputeGrIS:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("GrIS")

    @pytest.fixture(scope="class")
    def arrays(self):
        rng = np.random.default_rng(21)
        return {
            "smb": rng.random(PROJ_LEN) * 500 - 300,
            "st":  rng.random(PROJ_LEN) * 20  - 30,
        }

    def test_output_keys(self, converter, arrays):
        result = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=arrays["smb"], st=arrays["st"],
        )
        assert set(result.keys()) == {"aSMB", "aST"}

    def test_asmb_values(self, converter, arrays):
        result = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=arrays["smb"], st=arrays["st"],
        )
        expected = arrays["smb"] - _GRIS_SMB_CLIM
        np.testing.assert_allclose(result["aSMB"], expected, rtol=1e-9)

    def test_ast_values(self, converter, arrays):
        result = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=arrays["smb"], st=arrays["st"],
        )
        expected = arrays["st"] - _GRIS_ST_CLIM
        np.testing.assert_allclose(result["aST"], expected, rtol=1e-9)

    def test_output_lengths(self, converter, arrays):
        result = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=arrays["smb"], st=arrays["st"],
        )
        assert len(result["aSMB"]) == PROJ_LEN
        assert len(result["aST"])  == PROJ_LEN

    def test_output_is_numpy(self, converter, arrays):
        result = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=arrays["smb"], st=arrays["st"],
        )
        assert isinstance(result["aSMB"], np.ndarray)
        assert isinstance(result["aST"],  np.ndarray)

    def test_wrong_length_raises(self, converter):
        with pytest.raises(ValueError, match="86"):
            converter.compute_gris(
                sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
                smb=np.zeros(50), st=np.zeros(PROJ_LEN),
            )

    def test_2d_array_raises(self, converter):
        with pytest.raises(ValueError, match="1-D"):
            converter.compute_gris(
                sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
                smb=np.zeros((2, 43)), st=np.zeros(PROJ_LEN),
            )


# ---------------------------------------------------------------------------
# 9. compute_gris — custom_climatology
# ---------------------------------------------------------------------------

class TestComputeGrISCustomClimatology:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("GrIS")

    @pytest.fixture(scope="class")
    def arrays(self):
        rng = np.random.default_rng(55)
        return {"smb": rng.random(PROJ_LEN) * 200 - 100, "st": rng.random(PROJ_LEN) * 15 - 25}

    _custom = {"smb": -250.0, "st": -18.5}

    def test_custom_subtraction_correct(self, converter, arrays):
        result = converter.compute_gris(
            sector=1, custom_climatology=self._custom,
            smb=arrays["smb"], st=arrays["st"],
        )
        np.testing.assert_allclose(result["aSMB"], arrays["smb"] - (-250.0), rtol=1e-9)
        np.testing.assert_allclose(result["aST"],  arrays["st"]  - (-18.5),  rtol=1e-9)

    def test_custom_missing_key_raises(self, converter, arrays):
        with pytest.raises(ValueError, match="st"):
            converter.compute_gris(
                sector=1, custom_climatology={"smb": -200.0},
                smb=arrays["smb"], st=arrays["st"],
            )

    def test_neither_arg_raises(self, converter, arrays):
        with pytest.raises(ValueError):
            converter.compute_gris(sector=1, smb=arrays["smb"], st=arrays["st"])

    def test_both_args_raise(self, converter, arrays):
        with pytest.raises(ValueError):
            converter.compute_gris(
                sector=1, aogcm=_GRIS_AOGCM, custom_climatology=self._custom,
                smb=arrays["smb"], st=arrays["st"],
            )


# ---------------------------------------------------------------------------
# 10. AOGCM alias normalisation — AIS
# ---------------------------------------------------------------------------

class TestAISAliasNormalisation:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("AIS")

    @pytest.fixture(scope="class")
    def arrays(self):
        rng = np.random.default_rng(11)
        return {k: rng.random(PROJ_LEN) for k in ("pr", "evspsbl", "smb", "ts")}

    @pytest.mark.parametrize("alias, canonical", [
        ("NorESM1-M_rcp8.5",      "noresm1-m_rcp85"),
        ("noresm1-m_rcp8.5",      "noresm1-m_rcp85"),
        ("CCSM4_rcp8.5",          "ccsm4_rcp85"),
        ("HadGEM2-ES_rcp8.5",     "hadgem2-es_rcp85"),
        ("IPSL-CM5A-MR_rcp8.5",   "ipsl-cm5-mr_rcp85"),
        ("CNRM-CM6-1_ssp585",     "cnrm-cm6_ssp585"),
        ("CNRM-ESM2-1_ssp585",    "cnrm-esm2_ssp585"),
    ])
    def test_alias_resolves_to_same_clim_as_canonical(self, converter, arrays, alias, canonical):
        result_alias = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=alias,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        result_canon = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=canonical,
            pr=arrays["pr"], evspsbl=arrays["evspsbl"],
            smb=arrays["smb"], ts=arrays["ts"],
        )
        for key in ("pr_anomaly", "evspsbl_anomaly", "smb_anomaly", "ts_anomaly"):
            np.testing.assert_array_equal(
                result_alias[key], result_canon[key],
                err_msg=f"alias='{alias}' produced different '{key}' from canonical='{canonical}'"
            )

    def test_alias_dict_keys_are_lowercase(self):
        for key in _AIS_ALIASES:
            assert key == key.lower(), f"AIS alias key not lowercase: '{key}'"

    def test_alias_dict_values_are_lowercase(self):
        for val in _AIS_ALIASES.values():
            assert val == val.lower(), f"AIS alias value not lowercase: '{val}'"


# ---------------------------------------------------------------------------
# 11. AOGCM alias normalisation — GrIS
# ---------------------------------------------------------------------------

class TestGrISAliasNormalisation:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("GrIS")

    @pytest.fixture(scope="class")
    def arrays(self):
        rng = np.random.default_rng(13)
        return {"smb": rng.random(PROJ_LEN) * 300, "st": rng.random(PROJ_LEN) * 10 - 20}

    @pytest.mark.parametrize("alias, canonical", [
        ("access1-3_rcp85",      "access1.3_rcp85"),
        ("access1.3_rcp8.5",     "access1.3_rcp85"),
        ("MIROC5_rcp8.5",        "miroc5_rcp85"),
        ("NorESM1-M_rcp8.5",     "noresm1-m_rcp85"),
        ("CNRM-CM6-1_ssp585",    "cnrm-cm6_ssp585"),
        ("CNRM-ESM2-1_ssp585",   "cnrm-esm2_ssp585"),
        ("ukesm1-cm6_ssp585",    "ukesm1-0-ll_ssp585"),
    ])
    def test_alias_resolves_to_same_clim_as_canonical(self, converter, arrays, alias, canonical):
        result_alias = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=alias,
            smb=arrays["smb"], st=arrays["st"],
        )
        result_canon = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=canonical,
            smb=arrays["smb"], st=arrays["st"],
        )
        np.testing.assert_array_equal(result_alias["aSMB"], result_canon["aSMB"])
        np.testing.assert_array_equal(result_alias["aST"],  result_canon["aST"])

    def test_alias_dict_keys_are_lowercase(self):
        for key in _GRIS_ALIASES:
            assert key == key.lower(), f"GrIS alias key not lowercase: '{key}'"


# ---------------------------------------------------------------------------
# 12. Cross-sector consistency — same AOGCM, different sectors differ
# ---------------------------------------------------------------------------

class TestCrossSectorConsistency:

    @pytest.fixture(scope="class")
    def converter(self):
        return AnomalyConverter("AIS")

    def test_different_sectors_produce_different_anomalies(self, converter):
        rng = np.random.default_rng(77)
        arrays = {k: rng.random(PROJ_LEN) for k in ("pr", "evspsbl", "smb", "ts")}

        res1 = converter.compute_ais(sector=1, aogcm=_AIS_AOGCM, **arrays)
        res5 = converter.compute_ais(sector=5, aogcm=_AIS_AOGCM, **arrays)

        # Sectors 1 and 5 have different climatologies — results must differ
        assert not np.allclose(res1["pr_anomaly"], res5["pr_anomaly"]), (
            "Sectors 1 and 5 produced identical pr_anomaly — climatology lookup may be broken"
        )

    def test_different_aogcms_same_sector_produce_different_anomalies(self, converter):
        rng = np.random.default_rng(88)
        arrays = {k: rng.random(PROJ_LEN) for k in ("pr", "evspsbl", "smb", "ts")}

        res_nor = converter.compute_ais(sector=1, aogcm="noresm1-m_rcp85",   **arrays)
        res_csm = converter.compute_ais(sector=1, aogcm="cesm2_ssp585",       **arrays)

        assert not np.allclose(res_nor["ts_anomaly"], res_csm["ts_anomaly"]), (
            "Two distinct AOGCMs produced identical ts_anomaly — climatology lookup may be broken"
        )


# ---------------------------------------------------------------------------
# 13. End-to-end: bundled CSV values exactly reproduce ISMIP6 anomaly
#
# The reference anomaly is computed directly from the CSV without going
# through AnomalyConverter — i.e. reference = raw_array - csv_clim_value.
# The converter must produce an identical result (regression guard).
# ---------------------------------------------------------------------------

class TestEndToEndRegression:

    def test_ais_anomaly_matches_direct_csv_subtraction(self):
        rng = np.random.default_rng(1234)
        pr      = rng.random(PROJ_LEN) * 1e-4
        evspsbl = rng.random(PROJ_LEN) * 1e-4
        smb     = rng.random(PROJ_LEN) * 1e-4
        ts      = rng.random(PROJ_LEN) * 30 + 240
        mrro    = rng.random(PROJ_LEN) * 1e-6

        df = pd.read_csv(_AIS_CSV)
        row = df[(df["aogcm"] == _AIS_AOGCM) & (df["sector"] == _AIS_SECTOR)].iloc[0]

        expected = {
            "pr_anomaly":      pr      - row["pr_clim"],
            "evspsbl_anomaly": evspsbl - row["evspsbl_clim"],
            "smb_anomaly":     smb     - row["smb_clim"],
            "ts_anomaly":      ts      - row["ts_clim"],
            "mrro_anomaly":    mrro    - row["mrro_clim"],
        }

        converter = AnomalyConverter("AIS")
        result = converter.compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=pr, evspsbl=evspsbl, smb=smb, ts=ts, mrro=mrro,
        )

        for key in expected:
            np.testing.assert_allclose(
                result[key], expected[key], rtol=1e-12,
                err_msg=f"End-to-end mismatch for '{key}'"
            )

    def test_gris_anomaly_matches_direct_csv_subtraction(self):
        rng = np.random.default_rng(5678)
        smb = rng.random(PROJ_LEN) * 500 - 300
        st  = rng.random(PROJ_LEN) * 20  - 30

        df = pd.read_csv(_GRIS_CSV)
        row = df[(df["aogcm"] == _GRIS_AOGCM) & (df["sector"] == _GRIS_SECTOR)].iloc[0]

        expected = {
            "aSMB": smb - row["smb_clim"],
            "aST":  st  - row["st_clim"],
        }

        converter = AnomalyConverter("GrIS")
        result = converter.compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=smb, st=st,
        )

        for key in expected:
            np.testing.assert_allclose(
                result[key], expected[key], rtol=1e-12,
                err_msg=f"End-to-end mismatch for '{key}'"
            )

    @pytest.mark.parametrize("aogcm,sector", [
        ("noresm1-m_rcp85",    1),
        ("ccsm4_rcp85",        9),
        ("cesm2_ssp585",      18),
        ("hadgem2-es_rcp85",   4),
        ("cnrm-cm6_ssp126",   12),
    ])
    def test_ais_multiple_aogcm_sector_pairs(self, aogcm, sector):
        """Each (AOGCM, sector) combo must produce anomaly == raw - csv_clim."""
        rng = np.random.default_rng(aogcm.__hash__() % (2**31) + sector)
        pr      = rng.random(PROJ_LEN) * 1e-4
        evspsbl = rng.random(PROJ_LEN) * 1e-4
        smb     = rng.random(PROJ_LEN) * 1e-4
        ts      = rng.random(PROJ_LEN) * 30 + 240

        df = pd.read_csv(_AIS_CSV)
        row = df[(df["aogcm"] == aogcm) & (df["sector"] == sector)].iloc[0]

        converter = AnomalyConverter("AIS")
        result = converter.compute_ais(
            sector=sector, aogcm=aogcm,
            pr=pr, evspsbl=evspsbl, smb=smb, ts=ts,
        )

        np.testing.assert_allclose(result["pr_anomaly"],      pr      - row["pr_clim"],      rtol=1e-12)
        np.testing.assert_allclose(result["evspsbl_anomaly"], evspsbl - row["evspsbl_clim"], rtol=1e-12)
        np.testing.assert_allclose(result["smb_anomaly"],     smb     - row["smb_clim"],     rtol=1e-12)
        np.testing.assert_allclose(result["ts_anomaly"],      ts      - row["ts_clim"],      rtol=1e-12)

    @pytest.mark.parametrize("aogcm,sector", [
        ("access1.3_rcp85",   3),
        ("miroc5_rcp85",      6),
        ("ukesm1-0-ll_ssp585", 1),
        ("cnrm-esm2_ssp585",  4),
    ])
    def test_gris_multiple_aogcm_sector_pairs(self, aogcm, sector):
        """Each GrIS (AOGCM, sector) combo must produce anomaly == raw - csv_clim."""
        rng = np.random.default_rng(aogcm.__hash__() % (2**31) + sector)
        smb = rng.random(PROJ_LEN) * 500 - 300
        st  = rng.random(PROJ_LEN) * 20  - 30

        df = pd.read_csv(_GRIS_CSV)
        row = df[(df["aogcm"] == aogcm) & (df["sector"] == sector)].iloc[0]

        converter = AnomalyConverter("GrIS")
        result = converter.compute_gris(sector=sector, aogcm=aogcm, smb=smb, st=st)

        np.testing.assert_allclose(result["aSMB"], smb - row["smb_clim"], rtol=1e-12)
        np.testing.assert_allclose(result["aST"],  st  - row["st_clim"],  rtol=1e-12)
