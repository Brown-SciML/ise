"""Shared pytest fixtures for the ise test suite.

Markers
-------
@pytest.mark.slow
    Tests that take more than a few seconds (e.g. full training loops).
    Skipped in CI with ``-m "not slow"``.

@pytest.mark.gpu
    Tests that require a CUDA-capable GPU.
    Skipped in CI with ``-m "not gpu"``.
"""

import os

import numpy as np
import pytest
import torch

TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Skip gpu-marked tests automatically when no CUDA device is available.
_no_cuda = not torch.cuda.is_available()
skip_if_no_gpu = pytest.mark.skipif(_no_cuda, reason="CUDA GPU not available")

PROJ_LEN = 86


@pytest.fixture
def ais_year():
    """Calendar years for a standard AIS projection (2015-2100)."""
    return np.arange(2015, 2101)


@pytest.fixture
def zeros86():
    """Array of 86 zeros — useful for default/null forcing arrays."""
    return np.zeros(PROJ_LEN)


@pytest.fixture
def rng():
    """Seeded numpy RNG for reproducible random arrays in tests."""
    return np.random.default_rng(42)
