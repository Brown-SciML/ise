"""Experimental and legacy models from prior ISE manuscripts.

These models (GP, PCA, ScenarioPredictor, VariationalLSTMEmulator) were used in
earlier versions of the package and prior publications. They are no longer actively
maintained. Use ``ise.models.iseflow.ISEFlow`` (or the pretrained convenience
classes ``ISEFlow_AIS`` / ``ISEFlow_GrIS``) for current ice sheet emulation.

All classes in this subpackage emit a DeprecationWarning on import.
"""

import warnings

warnings.warn(
    "ise.models._experimental contains legacy models that are no longer maintained. "
    "Use ise.models.iseflow.ISEFlow for current ice sheet emulation.",
    DeprecationWarning,
    stacklevel=2,
)

from .gp import GP, EmulandiceGP, NuggetKernel, PowerExponentialKernel
from .pca import PCA, DimensionProcessor
from .scenario import ScenarioPredictor
from .variational_lstm_emulator import VariationalLSTMEmulator

__all__ = [
    "GP",
    "PowerExponentialKernel",
    "NuggetKernel",
    "EmulandiceGP",
    "PCA",
    "DimensionProcessor",
    "ScenarioPredictor",
    "VariationalLSTMEmulator",
]
