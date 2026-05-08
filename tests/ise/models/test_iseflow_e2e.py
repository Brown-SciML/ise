"""End-to-end ISEFlow tests — exercise real training, save/load, and prediction.

Marked ``slow`` because each test runs a tiny but real training loop. CI should
skip with ``-m "not slow"``. These tests catch integration breakage that the
shape-only mocked tests in ``test_iseflow.py`` cannot:

- NF latent → DE training pipeline (concat, dimensions, device transfers)
- The ``trained`` flag composition across submodules
- Save/load round-trip preserving NF + DE weights and DE forward output
- ISEFlow construction without a previously-loaded NormalizingFlow
- ``DeepEnsemble.save()`` is repeatable on the same trained model

Note that ``NormalizingFlow.get_latent()`` and ``NormalizingFlow.aleatoric()``
are stochastic by design (they sample from a learned distribution), so
end-to-end ``ISEFlow.predict()`` outputs cannot be compared bitwise across
runs. We compare weight equality and DE-only forward (with a fixed latent)
instead.

Test sizing
-----------
Tiny architecture (NF: 2 transforms, 8 hidden; DE: 2 LSTMs with 8 hidden), 2
projections × 86 timesteps, 1 epoch each. The ``fitted_iseflow`` fixture is
**module-scoped** so the real fit runs once and is reused across tests.
"""

import json
import os

import numpy as np
import pytest
import torch

from ise.models.deep_ensemble import DeepEnsemble
from ise.models.iseflow import ISEFlow
from ise.models.lstm import LSTM
from ise.models.normalizing_flow import NormalizingFlow

pytestmark = pytest.mark.slow

PROJ_LEN = 86
N_PROJ = 2
N = PROJ_LEN * N_PROJ
N_FEATURES = 8
LATENT_DIM = 1


def _build_untrained_iseflow():
    """Tiny ISEFlow architecture for fast training tests."""
    torch.manual_seed(0)
    nf = NormalizingFlow(
        input_size=N_FEATURES,
        output_size=1,
        num_flow_transforms=2,
        flow_hidden_features=8,
    )
    members = [
        LSTM(
            lstm_num_layers=1,
            lstm_hidden_size=8,
            input_size=N_FEATURES + LATENT_DIM,
            output_size=1,
            criterion=torch.nn.MSELoss(),
        )
        for _ in range(2)
    ]
    de = DeepEnsemble(ensemble_members=members)
    return ISEFlow(de, nf)


@pytest.fixture(scope="module")
def synthetic_data():
    rng = np.random.default_rng(42)
    X = torch.from_numpy(rng.random((N, N_FEATURES)).astype(np.float32))
    y = torch.from_numpy(rng.random((N, 1)).astype(np.float32))
    return X, y


@pytest.fixture(scope="module")
def fitted_iseflow(synthetic_data, tmp_path_factory):
    """A fully-trained ISEFlow on tiny data — module-scoped to share across tests."""
    X, y = synthetic_data
    model = _build_untrained_iseflow()
    ckpt_dir = tmp_path_factory.mktemp("ckpt")
    model.fit(
        X,
        y,
        nf_epochs=1,
        de_epochs=1,
        batch_size=32,
        save_checkpoints=True,
        checkpoint_path=str(ckpt_dir / "ckpt"),
        early_stopping=False,
        verbose=False,
    )
    return model


@pytest.fixture(scope="module")
def saved_iseflow_dir(fitted_iseflow, tmp_path_factory):
    """Single shared save directory used by save-inspection and load tests."""
    save_dir = str(tmp_path_factory.mktemp("saved_iseflow"))
    feat_names = [f"f{i}" for i in range(N_FEATURES)]
    fitted_iseflow.save(save_dir, input_features=feat_names)
    return save_dir, feat_names


# ---------------------------------------------------------------------------
# Construction (does not require fit) — covers Issue 2 fix
# ---------------------------------------------------------------------------


class TestISEFlowConstructionWithoutLoad:
    def test_fresh_nf_de_construction_does_not_require_model_dir(self):
        """ISEFlow(de, nf) must work for freshly-constructed (un-loaded) submodules.

        Regression test: previously raised AttributeError because __init__ accessed
        normalizing_flow.model_dir, which is only set by NormalizingFlow.load().
        """
        model = _build_untrained_iseflow()
        assert model.model_dir is None
        assert model.trained is False


# ---------------------------------------------------------------------------
# Full training loop — Issue 1 regression + integration coverage
# ---------------------------------------------------------------------------


class TestISEFlowFit:
    def test_fit_marks_model_trained(self, fitted_iseflow):
        assert fitted_iseflow.trained is True
        assert fitted_iseflow.normalizing_flow.trained is True
        assert fitted_iseflow.deep_ensemble.trained is True

    def test_fit_produces_finite_predictions(self, fitted_iseflow, synthetic_data):
        """After real fit, predict must produce finite outputs and well-formed
        uncertainties (regression test for the Issue 1 positional-arg bug, which
        previously crashed before any training could happen)."""
        X, _ = synthetic_data
        with pytest.warns(UserWarning, match="scaler"):
            preds, unc = fitted_iseflow.predict(X[:PROJ_LEN], output_scaler=False)
        assert preds.shape[0] == PROJ_LEN
        assert np.all(np.isfinite(preds))
        assert set(unc.keys()) == {"total", "epistemic", "aleatoric"}
        for key in ("total", "epistemic", "aleatoric"):
            assert (unc[key] >= 0).all(), f"{key} uncertainty has negative values"


# ---------------------------------------------------------------------------
# Save / load round-trip — the most important integration coverage
# ---------------------------------------------------------------------------


class TestISEFlowSaveLoad:
    def test_save_creates_expected_files(self, saved_iseflow_dir):
        save_dir, _ = saved_iseflow_dir
        assert os.path.isfile(os.path.join(save_dir, "deep_ensemble.pth"))
        assert os.path.isfile(os.path.join(save_dir, "normalizing_flow.pth"))
        # NF writes a metadata.json next to the .pth
        assert os.path.isfile(os.path.join(save_dir, "normalizing_flow.pth_metadata.json"))

    def test_save_writes_input_features_json(self, saved_iseflow_dir):
        save_dir, feat_names = saved_iseflow_dir
        feat_path = os.path.join(save_dir, "input_features.json")
        assert os.path.isfile(feat_path)
        with open(feat_path) as f:
            assert json.load(f) == feat_names

    def test_load_marks_trained_and_restores_submodules(self, saved_iseflow_dir):
        """ISEFlow.load() must produce a trained model with the same submodule types."""
        save_dir, _ = saved_iseflow_dir
        loaded = ISEFlow.load(model_dir=save_dir)
        assert loaded.trained is True
        assert isinstance(loaded.normalizing_flow, NormalizingFlow)
        assert isinstance(loaded.deep_ensemble, DeepEnsemble)
        assert loaded.normalizing_flow.trained is True
        assert loaded.deep_ensemble.trained is True

    def test_load_preserves_nf_state_dict(self, fitted_iseflow, saved_iseflow_dir):
        """NormalizingFlow weights must round-trip exactly through save/load.

        We compare state_dict tensors directly because get_latent() and aleatoric()
        are stochastic by design (they sample from a learned distribution). Only
        weight equality is a deterministic check.
        """
        save_dir, _ = saved_iseflow_dir
        loaded = ISEFlow.load(model_dir=save_dir)
        for name, p_orig in fitted_iseflow.normalizing_flow.state_dict().items():
            p_load = loaded.normalizing_flow.state_dict()[name]
            assert torch.allclose(p_orig, p_load, atol=1e-7), f"NF param {name} drifted"

    def test_load_preserves_deep_ensemble_member_state_dicts(
        self, fitted_iseflow, saved_iseflow_dir, synthetic_data
    ):
        """Each LSTM member's weights must round-trip, AND DE.forward with a fixed
        latent must produce the same output (covers both weight load and the
        members ordering / sequence_length restore).
        """
        save_dir, _ = saved_iseflow_dir
        X, _ = synthetic_data
        loaded = ISEFlow.load(model_dir=save_dir)

        # Per-member weight equality
        for i, (orig_m, load_m) in enumerate(
            zip(
                fitted_iseflow.deep_ensemble.ensemble_members, loaded.deep_ensemble.ensemble_members
            )
        ):
            for name, p_orig in orig_m.state_dict().items():
                p_load = load_m.state_dict()[name]
                assert torch.allclose(p_orig, p_load, atol=1e-7), (
                    f"DE member {i} param {name} drifted"
                )

        # End-to-end DE forward with a *fixed* latent vector — this catches issues
        # with sequence_length restoration that pure weight equality wouldn't.
        z_fixed = torch.zeros(PROJ_LEN, LATENT_DIM)
        x_in = X[:PROJ_LEN]
        with torch.no_grad():
            mean_orig, _ = fitted_iseflow.deep_ensemble(torch.cat((x_in, z_fixed), dim=1))
            mean_load, _ = loaded.deep_ensemble(torch.cat((x_in, z_fixed), dim=1))
        np.testing.assert_allclose(mean_orig.cpu().numpy(), mean_load.cpu().numpy(), atol=1e-5)

    def test_save_is_repeatable(self, fitted_iseflow, tmp_path):
        """Saving a trained model twice (e.g. to two locations) must succeed.

        Regression test: ``DeepEnsemble.save()`` previously deleted member
        checkpoint files as a side effect, so the second save() crashed with
        FileNotFoundError. Now save() does best-effort cleanup that tolerates
        already-deleted (or never-created) files.
        """
        first = str(tmp_path / "save1")
        second = str(tmp_path / "save2")
        fitted_iseflow.save(first)
        fitted_iseflow.save(second)  # must not raise
        for d in (first, second):
            assert os.path.isfile(os.path.join(d, "deep_ensemble.pth"))
            assert os.path.isfile(os.path.join(d, "normalizing_flow.pth"))
