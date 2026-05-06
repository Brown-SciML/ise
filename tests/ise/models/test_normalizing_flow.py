import os
import pytest
import torch
import numpy as np

from ise.models.normalizing_flow import NormalizingFlow


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_nf():
    """Tiny NF (2 transforms, 8 hidden) for fast CPU tests."""
    return NormalizingFlow(input_size=8, output_size=1, num_flow_transforms=2, flow_hidden_features=8)


@pytest.fixture
def small_features():
    """(N=10, features=8) conditioning feature matrix."""
    return torch.rand(10, 8)


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------

class TestNormalizingFlowConstructor:
    def test_num_transforms_stored(self, small_nf):
        assert small_nf.num_flow_transforms == 2

    def test_input_size_stored(self, small_nf):
        assert small_nf.num_input_features == 8

    def test_output_size_stored(self, small_nf):
        assert small_nf.num_predicted_sle == 1

    def test_trained_false_on_init(self, small_nf):
        assert small_nf.trained is False

    def test_flow_attribute_exists(self, small_nf):
        from nflows.flows.base import Flow
        assert isinstance(small_nf.flow, Flow)


# ---------------------------------------------------------------------------
# get_latent — does not require training
# ---------------------------------------------------------------------------

class TestNormalizingFlowGetLatent:
    def test_output_shape(self, small_nf, small_features):
        z = small_nf.get_latent(small_features)
        assert z.shape == (10, 1)

    def test_output_is_tensor(self, small_nf, small_features):
        z = small_nf.get_latent(small_features)
        assert isinstance(z, torch.Tensor)

    def test_different_n(self, small_nf):
        for n in [1, 5, 50]:
            x = torch.rand(n, 8)
            z = small_nf.get_latent(x)
            assert z.shape[0] == n


# ---------------------------------------------------------------------------
# sample — does not require training
# ---------------------------------------------------------------------------

class TestNormalizingFlowSample:
    def test_numpy_return_shape(self, small_nf, small_features):
        samples = small_nf.sample(small_features, num_samples=20)
        assert samples.shape == (10, 20)

    def test_tensor_return_shape(self, small_nf, small_features):
        samples = small_nf.sample(small_features, num_samples=5, return_type="tensor")
        assert samples.shape == (10, 5)

    def test_no_nans(self, small_nf, small_features):
        samples = small_nf.sample(small_features, num_samples=10)
        assert not np.isnan(samples).any()


# ---------------------------------------------------------------------------
# aleatoric — does not require training
# ---------------------------------------------------------------------------

class TestNormalizingFlowAleatoric:
    def test_output_shape(self, small_nf, small_features):
        unc = small_nf.aleatoric(small_features, num_samples=20)
        assert unc.shape == (10,)

    def test_output_is_numpy(self, small_nf, small_features):
        unc = small_nf.aleatoric(small_features, num_samples=20)
        assert isinstance(unc, np.ndarray)

    def test_non_negative(self, small_nf, small_features):
        unc = small_nf.aleatoric(small_features, num_samples=50)
        assert (unc >= 0).all()

    def test_batch_size_respected(self, small_nf):
        x = torch.rand(25, 8)
        unc = small_nf.aleatoric(x, num_samples=10, batch_size=10)
        assert unc.shape == (25,)


# ---------------------------------------------------------------------------
# Save — requires trained=True flag + metadata attrs set manually
# ---------------------------------------------------------------------------

class TestNormalizingFlowSave:
    def test_save_raises_if_not_trained(self, small_nf, tmp_path):
        with pytest.raises(ValueError, match="Train"):
            small_nf.save(str(tmp_path / "nf.pth"))

    def test_save_creates_files(self, small_nf, tmp_path):
        small_nf.trained = True
        small_nf.best_loss = 0.5
        small_nf.epochs_trained = 10
        path = str(tmp_path / "nf.pth")
        small_nf.save(path)
        assert os.path.isfile(path)
        assert os.path.isfile(path + "_metadata.json")

    def test_load_restores_architecture(self, small_nf, tmp_path):
        small_nf.trained = True
        small_nf.best_loss = 0.5
        small_nf.epochs_trained = 10
        path = str(tmp_path / "nf.pth")
        small_nf.save(path)

        loaded = NormalizingFlow.load(path)
        assert loaded.num_input_features == small_nf.num_input_features
        assert loaded.num_predicted_sle == small_nf.num_predicted_sle
        assert loaded.num_flow_transforms == small_nf.num_flow_transforms
        assert loaded.trained is True

    def test_load_latent_shape_preserved(self, small_nf, tmp_path, small_features):
        small_nf.trained = True
        small_nf.best_loss = 0.5
        small_nf.epochs_trained = 10
        path = str(tmp_path / "nf.pth")
        small_nf.save(path)

        loaded = NormalizingFlow.load(path)
        z = loaded.get_latent(small_features)
        assert z.shape == (10, 1)
