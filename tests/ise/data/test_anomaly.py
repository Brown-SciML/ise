"""Tests for ise/data/anomaly.py — AnomalyConverter."""

import warnings

import numpy as np
import pytest

from ise.data.anomaly import AnomalyConverter

PROJ_LEN = 86

_AIS_AOGCM      = "noresm1-m_rcp85"
_AIS_SECTOR     = 5
_AIS_PR_CLIM    = 9.090845196624286e-06
_AIS_EVSPSBL_CLIM = 4.421370647378353e-07
_AIS_SMB_CLIM   = 8.355646968993824e-06
_AIS_TS_CLIM    = 245.5599822998047
_AIS_MRRO_CLIM  = 2.9306085025382345e-07

_AIS_NO_MRRO_AOGCM  = "csiro-mk3.6_rcp85"
_AIS_NO_MRRO_SECTOR = 3

_GRIS_AOGCM    = "noresm1-m_rcp85"
_GRIS_SECTOR   = 2
_GRIS_SMB_CLIM = 17.879426956176758
_GRIS_ST_CLIM  = -21.672592163085938


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def ais_arrays():
    rng = np.random.default_rng(42)
    return {
        "pr":      rng.random(PROJ_LEN) * 1e-4,
        "evspsbl": rng.random(PROJ_LEN) * 1e-4,
        "smb":     rng.random(PROJ_LEN) * 1e-4,
        "ts":      rng.random(PROJ_LEN) * 30 + 240,
        "mrro":    rng.random(PROJ_LEN) * 1e-6,
    }

@pytest.fixture(scope="module")
def gris_arrays():
    rng = np.random.default_rng(21)
    return {
        "smb": rng.random(PROJ_LEN) * 500 - 300,
        "st":  rng.random(PROJ_LEN) * 20  - 30,
    }


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestConstructor:
    def test_ais_initialises(self):
        conv = AnomalyConverter("AIS")
        assert conv.ice_sheet == "AIS"

    def test_gris_case_insensitive(self):
        assert AnomalyConverter("gris").ice_sheet == "GrIS"

    def test_invalid_ice_sheet_raises(self):
        with pytest.raises(ValueError, match="ice_sheet"):
            AnomalyConverter("WAIS")

    def test_climatology_lazy_loaded(self):
        conv = AnomalyConverter("AIS")
        assert conv._clim is None
        _ = conv.climatology
        assert conv._clim is not None

    def test_climatology_cached(self):
        conv = AnomalyConverter("AIS")
        assert conv.climatology is conv.climatology


# ---------------------------------------------------------------------------
# compute_ais — core correctness
# ---------------------------------------------------------------------------

class TestComputeAIS:
    @pytest.fixture(scope="class")
    def result(self, ais_arrays):
        return AnomalyConverter("AIS").compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
            smb=ais_arrays["smb"], ts=ais_arrays["ts"],
        )

    def test_output_keys(self, result):
        assert set(result.keys()) == {"pr_anomaly", "evspsbl_anomaly", "smb_anomaly", "ts_anomaly"}

    def test_anomaly_values(self, result, ais_arrays):
        np.testing.assert_allclose(result["pr_anomaly"],      ais_arrays["pr"]      - _AIS_PR_CLIM,      rtol=1e-9)
        np.testing.assert_allclose(result["evspsbl_anomaly"], ais_arrays["evspsbl"] - _AIS_EVSPSBL_CLIM, rtol=1e-9)
        np.testing.assert_allclose(result["smb_anomaly"],     ais_arrays["smb"]     - _AIS_SMB_CLIM,     rtol=1e-9)
        np.testing.assert_allclose(result["ts_anomaly"],      ais_arrays["ts"]      - _AIS_TS_CLIM,      rtol=1e-9)

    def test_output_length_and_type(self, result):
        for arr in result.values():
            assert isinstance(arr, np.ndarray)
            assert len(arr) == PROJ_LEN

    def test_wrong_length_raises(self, ais_arrays):
        with pytest.raises(ValueError, match="86"):
            AnomalyConverter("AIS").compute_ais(
                sector=1, aogcm=_AIS_AOGCM,
                pr=ais_arrays["pr"][:50], evspsbl=ais_arrays["evspsbl"],
                smb=ais_arrays["smb"], ts=ais_arrays["ts"],
            )

    def test_2d_array_raises(self, ais_arrays):
        with pytest.raises(ValueError, match="1-D"):
            AnomalyConverter("AIS").compute_ais(
                sector=1, aogcm=_AIS_AOGCM,
                pr=ais_arrays["pr"].reshape(2, 43), evspsbl=ais_arrays["evspsbl"],
                smb=ais_arrays["smb"], ts=ais_arrays["ts"],
            )

    def test_neither_aogcm_nor_custom_raises(self, ais_arrays):
        with pytest.raises(ValueError):
            AnomalyConverter("AIS").compute_ais(
                sector=1,
                pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
                smb=ais_arrays["smb"], ts=ais_arrays["ts"],
            )

    def test_both_aogcm_and_custom_raises(self, ais_arrays):
        with pytest.raises(ValueError):
            AnomalyConverter("AIS").compute_ais(
                sector=1, aogcm=_AIS_AOGCM,
                custom_climatology={"pr": 1e-5, "evspsbl": 4e-6, "smb": 9e-6, "ts": 253.7},
                pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
                smb=ais_arrays["smb"], ts=ais_arrays["ts"],
            )


# ---------------------------------------------------------------------------
# compute_ais — mrro handling
# ---------------------------------------------------------------------------

class TestComputeAISMrro:
    def test_mrro_anomaly_included_when_clim_available(self, ais_arrays):
        result = AnomalyConverter("AIS").compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
            smb=ais_arrays["smb"], ts=ais_arrays["ts"],
            mrro=ais_arrays["mrro"],
        )
        assert "mrro_anomaly" in result
        np.testing.assert_allclose(result["mrro_anomaly"], ais_arrays["mrro"] - _AIS_MRRO_CLIM, rtol=1e-9)

    def test_mrro_absent_when_not_provided(self, ais_arrays):
        result = AnomalyConverter("AIS").compute_ais(
            sector=_AIS_SECTOR, aogcm=_AIS_AOGCM,
            pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
            smb=ais_arrays["smb"], ts=ais_arrays["ts"],
        )
        assert "mrro_anomaly" not in result

    def test_mrro_warns_when_clim_unavailable(self, ais_arrays):
        with pytest.warns(UserWarning, match="mrro"):
            AnomalyConverter("AIS").compute_ais(
                sector=_AIS_NO_MRRO_SECTOR, aogcm=_AIS_NO_MRRO_AOGCM,
                pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
                smb=ais_arrays["smb"], ts=ais_arrays["ts"],
                mrro=ais_arrays["mrro"],
            )


# ---------------------------------------------------------------------------
# compute_ais — custom_climatology
# ---------------------------------------------------------------------------

class TestComputeAISCustomClimatology:
    _custom = {"pr": 1.3e-5, "evspsbl": 4e-6, "smb": 9e-6, "ts": 253.7}

    def test_custom_subtraction_correct(self, ais_arrays):
        result = AnomalyConverter("AIS").compute_ais(
            sector=1, custom_climatology=self._custom,
            pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
            smb=ais_arrays["smb"], ts=ais_arrays["ts"],
        )
        np.testing.assert_allclose(result["pr_anomaly"],      ais_arrays["pr"]      - 1.3e-5, rtol=1e-9)
        np.testing.assert_allclose(result["evspsbl_anomaly"], ais_arrays["evspsbl"] - 4e-6,   rtol=1e-9)
        np.testing.assert_allclose(result["smb_anomaly"],     ais_arrays["smb"]     - 9e-6,   rtol=1e-9)
        np.testing.assert_allclose(result["ts_anomaly"],      ais_arrays["ts"]      - 253.7,  rtol=1e-9)

    def test_custom_missing_key_raises(self, ais_arrays):
        with pytest.raises(ValueError, match="smb"):
            AnomalyConverter("AIS").compute_ais(
                sector=1, custom_climatology={"pr": 1e-5, "evspsbl": 4e-6, "ts": 253.7},
                pr=ais_arrays["pr"], evspsbl=ais_arrays["evspsbl"],
                smb=ais_arrays["smb"], ts=ais_arrays["ts"],
            )


# ---------------------------------------------------------------------------
# compute_gris — core correctness
# ---------------------------------------------------------------------------

class TestComputeGrIS:
    @pytest.fixture(scope="class")
    def result(self, gris_arrays):
        return AnomalyConverter("GrIS").compute_gris(
            sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
            smb=gris_arrays["smb"], st=gris_arrays["st"],
        )

    def test_output_keys(self, result):
        assert set(result.keys()) == {"aSMB", "aST"}

    def test_anomaly_values(self, result, gris_arrays):
        np.testing.assert_allclose(result["aSMB"], gris_arrays["smb"] - _GRIS_SMB_CLIM, rtol=1e-9)
        np.testing.assert_allclose(result["aST"],  gris_arrays["st"]  - _GRIS_ST_CLIM,  rtol=1e-9)

    def test_output_length_and_type(self, result):
        for arr in result.values():
            assert isinstance(arr, np.ndarray)
            assert len(arr) == PROJ_LEN

    def test_wrong_length_raises(self, gris_arrays):
        with pytest.raises(ValueError, match="86"):
            AnomalyConverter("GrIS").compute_gris(
                sector=_GRIS_SECTOR, aogcm=_GRIS_AOGCM,
                smb=np.zeros(50), st=gris_arrays["st"],
            )

    def test_neither_aogcm_nor_custom_raises(self, gris_arrays):
        with pytest.raises(ValueError):
            AnomalyConverter("GrIS").compute_gris(
                sector=1, smb=gris_arrays["smb"], st=gris_arrays["st"],
            )


# ---------------------------------------------------------------------------
# compute_gris — custom_climatology
# ---------------------------------------------------------------------------

class TestComputeGrISCustomClimatology:
    _custom = {"smb": -250.0, "st": -18.5}

    def test_custom_subtraction_correct(self, gris_arrays):
        result = AnomalyConverter("GrIS").compute_gris(
            sector=1, custom_climatology=self._custom,
            smb=gris_arrays["smb"], st=gris_arrays["st"],
        )
        np.testing.assert_allclose(result["aSMB"], gris_arrays["smb"] - (-250.0), rtol=1e-9)
        np.testing.assert_allclose(result["aST"],  gris_arrays["st"]  - (-18.5),  rtol=1e-9)

    def test_custom_missing_key_raises(self, gris_arrays):
        with pytest.raises(ValueError, match="st"):
            AnomalyConverter("GrIS").compute_gris(
                sector=1, custom_climatology={"smb": -200.0},
                smb=gris_arrays["smb"], st=gris_arrays["st"],
            )
