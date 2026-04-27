"""Experimental and legacy models from prior ISE manuscripts.

These models (GP, PCA, ScenarioPredictor, VariationalLSTMEmulator) were used in
earlier versions of the package and prior publications. They are no longer actively
maintained. Use ISEFlow (ise.models.ISEFlow) for current ice sheet emulation.

All classes in this subpackage emit a DeprecationWarning on import.
"""
import warnings

warnings.warn(
    "ise.models.experimental contains legacy models that are no longer maintained. "
    "Use ise.models.ISEFlow for current ice sheet emulation.",
    DeprecationWarning,
    stacklevel=2,
)

from .gp import GP, PowerExponentialKernel, NuggetKernel, EmulandiceGP
from .pca import PCA, DimensionProcessor
from .scenario import ScenarioPredictor
from .variational_lstm_emulator import VariationalLSTMEmulator

__all__ = [
    "GP", "PowerExponentialKernel", "NuggetKernel", "EmulandiceGP",
    "PCA", "DimensionProcessor",
    "ScenarioPredictor",
    "VariationalLSTMEmulator",
]
