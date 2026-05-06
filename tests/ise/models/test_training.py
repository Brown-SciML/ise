import pytest
import torch
from torch import nn, optim

from ise.models.training import CheckpointSaver, EarlyStoppingCheckpointer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tiny_model():
    return nn.Linear(2, 1)


@pytest.fixture
def tiny_optimizer(tiny_model):
    return optim.SGD(tiny_model.parameters(), lr=0.01)


@pytest.fixture
def saver(tiny_model, tiny_optimizer, tmp_path):
    return CheckpointSaver(tiny_model, tiny_optimizer, str(tmp_path / "ckpt.pt"), verbose=False)


@pytest.fixture
def stopper(tiny_model, tiny_optimizer, tmp_path):
    return EarlyStoppingCheckpointer(
        tiny_model, tiny_optimizer, str(tmp_path / "ckpt.pt"), patience=3, verbose=False
    )


# ---------------------------------------------------------------------------
# CheckpointSaver
# ---------------------------------------------------------------------------

class TestCheckpointSaver:
    def test_saves_on_first_call(self, saver, tmp_path):
        saved = saver(loss=1.0, epoch=1)
        assert saved is True
        assert (tmp_path / "ckpt.pt").exists()

    def test_saves_when_loss_improves(self, saver):
        saver(1.0, epoch=1)
        saved = saver(0.5, epoch=2)
        assert saved is True
        assert saver.best_loss == pytest.approx(0.5)

    def test_does_not_save_when_loss_does_not_improve(self, saver):
        saver(0.5, epoch=1)
        saved = saver(0.9, epoch=2)
        assert saved is False
        assert saver.best_loss == pytest.approx(0.5)

    def test_always_saves_when_save_best_only_false(self, saver):
        saver(0.5, epoch=1)
        saved = saver(0.9, epoch=2, save_best_only=False)
        assert saved is True

    def test_best_loss_initialised_to_inf(self, saver):
        assert saver.best_loss == float("inf")

    def test_load_checkpoint_restores_best_loss(self, saver):
        saver(loss=0.42, epoch=7)
        epoch = saver.load_checkpoint()
        assert saver.best_loss == pytest.approx(0.42)
        assert epoch == 8  # checkpoint["epoch"] + 1

    def test_checkpoint_dict_keys(self, saver, tmp_path):
        saver(loss=0.3, epoch=5)
        ckpt = torch.load(str(tmp_path / "ckpt.pt"), weights_only=False)
        assert {"epoch", "model_state_dict", "optimizer_state_dict", "best_loss"}.issubset(ckpt.keys())


# ---------------------------------------------------------------------------
# EarlyStoppingCheckpointer
# ---------------------------------------------------------------------------

class TestEarlyStoppingCheckpointer:
    def test_counter_increments_on_no_improvement(self, stopper):
        stopper(0.5, epoch=1)   # sets best
        stopper(0.9, epoch=2)   # no improvement → counter = 1
        assert stopper.counter == 1

    def test_counter_resets_on_improvement(self, stopper):
        stopper(0.5, epoch=1)
        stopper(0.9, epoch=2)   # counter = 1
        stopper(0.3, epoch=3)   # improvement → counter resets to 0
        assert stopper.counter == 0

    def test_early_stop_false_before_patience_reached(self, stopper):
        stopper(0.5, epoch=1)
        stopper(0.9, epoch=2)
        stopper(0.9, epoch=3)
        assert stopper.early_stop is False

    def test_early_stop_true_after_patience_exceeded(self, stopper):
        stopper(0.5, epoch=1)
        # patience=3, so three non-improving calls trigger early_stop
        stopper(0.9, epoch=2)
        stopper(0.9, epoch=3)
        stopper(0.9, epoch=4)
        assert stopper.early_stop is True

    def test_inherits_checkpoint_saving(self, stopper, tmp_path):
        stopper(0.3, epoch=1)
        assert (tmp_path / "ckpt.pt").exists()

    def test_best_loss_updated_on_improvement(self, stopper):
        stopper(1.0, epoch=1)
        stopper(0.2, epoch=2)
        assert stopper.best_loss == pytest.approx(0.2)
