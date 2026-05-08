"""End-to-end tests for ISEFlow_AIS / ISEFlow_GrIS — the user-facing API.

These exercise the full pipeline:

    inputs (synthetic, realistic-magnitude) → model.predict()
        → process() → scale → lag → dummies → forward → unscale → smoothing

Marked ``slow`` because each AIS/GrIS predict takes ~50s on CPU (the NF
aleatoric MC-sampling step is the bottleneck — we cannot reduce it without
changing the source). All tests share module-scoped predict fixtures so
each ice sheet's predict only runs once per session.

Realistic-magnitude inputs
--------------------------
We use *small, plausible* anomaly arrays rather than ``rng.random`` because the
NF was trained on realistic forcings and produces infinite samples (then NaNs)
on out-of-distribution inputs. This is expected behaviour, not a bug.
"""

import os
import warnings

import numpy as np
import pytest

from ise.data.inputs import ISEFlowAISInputs, ISEFlowGrISInputs
from ise.models.iseflow import ISEFlow_AIS, ISEFlow_GrIS

pytestmark = pytest.mark.slow

PROJ_LEN = 86

# Skip if the pretrained weight directories aren't present locally and HF can't
# resolve them. Mirrors the pattern used in test_input_classes.py.
try:
    from ise.models.pretrained import get_model_dir

    AIS_DIR = get_model_dir("v1.1.0", "AIS")
    GRIS_DIR = get_model_dir("v1.1.0", "GrIS")
    _weights_available = os.path.isfile(
        os.path.join(AIS_DIR, "deep_ensemble.pth")
    ) and os.path.isfile(os.path.join(GRIS_DIR, "deep_ensemble.pth"))
except Exception:
    _weights_available = False

skip_no_weights = pytest.mark.skipif(
    not _weights_available, reason="Pretrained ISEFlow weights not available locally or via HF"
)


# ---------------------------------------------------------------------------
# Realistic-magnitude synthetic inputs
# ---------------------------------------------------------------------------


def _ais_inputs(version="v1.1.0", **overrides):
    """Realistic-magnitude AIS inputs that the pretrained NF can sample from cleanly."""
    base = dict(
        year=np.arange(2015, 2101),
        sector=5,
        # Anomalies near zero — model trained on similar magnitudes
        pr_anomaly=np.zeros(PROJ_LEN),
        evspsbl_anomaly=np.zeros(PROJ_LEN),
        smb_anomaly=np.zeros(PROJ_LEN),
        ts_anomaly=np.linspace(0, 3.0, PROJ_LEN),
        ocean_thermal_forcing=np.linspace(0.5, 2.0, PROJ_LEN),
        ocean_salinity=np.full(PROJ_LEN, 34.5),
        ocean_temperature=np.linspace(-1.0, 1.0, PROJ_LEN),
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
        version=version,
    )
    base.update(overrides)
    return ISEFlowAISInputs(**base)


def _gris_inputs(version="v1.1.0", **overrides):
    """Realistic-magnitude GrIS inputs."""
    base = dict(
        year=np.arange(2015, 2101),
        sector=3,
        aSMB=np.zeros(PROJ_LEN),
        aST=np.linspace(0, 3.0, PROJ_LEN),
        ocean_thermal_forcing=np.linspace(0.5, 2.0, PROJ_LEN),
        basin_runoff=np.linspace(100, 200, PROJ_LEN),
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
        version=version,
    )
    base.update(overrides)
    return ISEFlowGrISInputs(**base)


# ---------------------------------------------------------------------------
# Module-scoped fixtures — load each pretrained model and predict once
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ais_model():
    return ISEFlow_AIS(version="v1.1.0")


@pytest.fixture(scope="module")
def gris_model():
    return ISEFlow_GrIS(version="v1.1.0")


@pytest.fixture(scope="module")
def ais_predict_result(ais_model):
    """Run AIS predict ONCE and reuse for all assertions (~50s on CPU)."""
    inputs = _ais_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, unc = ais_model.predict(inputs)
    return preds, unc


@pytest.fixture(scope="module")
def gris_predict_result(gris_model):
    """Run GrIS predict ONCE and reuse for all assertions (~50s on CPU)."""
    inputs = _gris_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        preds, unc = gris_model.predict(inputs)
    return preds, unc


# ---------------------------------------------------------------------------
# Pretrained loading
# ---------------------------------------------------------------------------


@skip_no_weights
class TestPretrainedLoading:
    def test_iseflow_ais_v1_1_0_loads_trained(self, ais_model):
        assert ais_model.trained is True
        assert ais_model.version == "v1.1.0"
        assert ais_model.ice_sheet == "AIS"

    def test_iseflow_gris_v1_1_0_loads_trained(self, gris_model):
        assert gris_model.trained is True
        assert gris_model.version == "v1.1.0"
        assert gris_model.ice_sheet == "GrIS"


# ---------------------------------------------------------------------------
# Predict — full pipeline integration test (the canonical user path)
# ---------------------------------------------------------------------------


@skip_no_weights
class TestPretrainedPredictAIS:
    """Real pretrained AIS predict from synthetic ISEFlowAISInputs.

    If anything in the chain breaks (process/scale/lag/dummies/forward/unscale),
    these assertions fail.
    """

    def test_predictions_have_expected_shape(self, ais_predict_result):
        preds, _ = ais_predict_result
        assert preds.shape == (PROJ_LEN, 1)

    def test_predictions_finite(self, ais_predict_result):
        preds, _ = ais_predict_result
        assert np.all(np.isfinite(preds))

    def test_uncertainty_keys(self, ais_predict_result):
        _, unc = ais_predict_result
        assert set(unc.keys()) == {"total", "epistemic", "aleatoric"}

    def test_uncertainty_total_equals_epi_plus_ale(self, ais_predict_result):
        """The contract documented in iseflow.py: total = epistemic + aleatoric."""
        _, unc = ais_predict_result
        np.testing.assert_allclose(unc["total"], unc["epistemic"] + unc["aleatoric"], rtol=1e-5)

    def test_uncertainty_non_negative(self, ais_predict_result):
        _, unc = ais_predict_result
        for key in ("total", "epistemic", "aleatoric"):
            assert (unc[key] >= 0).all(), f"{key} has negative values"


@skip_no_weights
class TestPretrainedPredictGrIS:
    def test_predictions_have_expected_shape(self, gris_predict_result):
        preds, _ = gris_predict_result
        assert preds.shape == (PROJ_LEN, 1)

    def test_predictions_finite(self, gris_predict_result):
        preds, _ = gris_predict_result
        assert np.all(np.isfinite(preds))

    def test_uncertainty_total_equals_epi_plus_ale(self, gris_predict_result):
        _, unc = gris_predict_result
        np.testing.assert_allclose(unc["total"], unc["epistemic"] + unc["aleatoric"], rtol=1e-5)


# ---------------------------------------------------------------------------
# Version-specific behaviour
# ---------------------------------------------------------------------------


@skip_no_weights
class TestVersionContract:
    def test_v1_0_0_process_raises_when_mrro_anomaly_missing(self):
        """ISEFlow_AIS v1.0.0's process() must reject inputs without mrro_anomaly
        with the documented ValueError.

        Guards the v1.0.0 vs v1.1.0 codepath split in iseflow.py — the defining
        behavioural difference between the two versions.
        """
        model = ISEFlow_AIS(version="v1.0.0")
        inputs = _ais_inputs(version="v1.0.0", mrro_anomaly=None)
        with pytest.raises(ValueError, match="mrro_anomaly"):
            model.process(inputs)

    def test_invalid_version_raises(self):
        with pytest.raises(NotImplementedError, match="not implemented"):
            ISEFlow_AIS(version="v9.9.9")

    def test_v1_0_0_process_handles_nan_mrro_via_year_mean_fallback(self):
        """v1.0.0 path must fill NaN values in mrro_anomaly via the per-year
        climatological mean lookup, including the last year (model encoding 86).

        Regression test for the year_mean_map off-by-one: previously the map was
        keyed 0..85 while ``data['year']`` is 1..86 (model encoding), so any NaN
        in the last year crashed with ``KeyError: 86``.
        """
        model = ISEFlow_AIS(version="v1.0.0")
        # Provide mrro_anomaly with NaN at every year — exercises every key in
        # year_mean_map including the previously-broken last-year entry.
        mrro_with_nans = np.full(PROJ_LEN, np.nan)
        inputs = _ais_inputs(version="v1.0.0", mrro_anomaly=mrro_with_nans)
        df = model.process(inputs)  # must not raise
        assert df.shape[0] == PROJ_LEN
        # The mrro_anomaly column has been filled (no NaNs) — the lookup
        # filled in real climatological values.
        assert not df["mrro_anomaly"].isna().any()

    def test_v1_0_0_inputs_accept_none_standard_melt_type(self):
        """ISEFlowAISInputs must not raise when standard_melt_type=None and
        ocean_forcing_type='open' — guards against None-valued optional config
        fields being rejected during validation.
        """
        inputs = _ais_inputs(
            version="v1.0.0",
            ocean_forcing_type="open",
            standard_melt_type=None,
            mrro_anomaly=np.zeros(PROJ_LEN),
        )
        # Validation passed — to_df() must work without error too
        df = inputs.to_df()
        assert df.shape[0] == PROJ_LEN
