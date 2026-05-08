"""Tests for ise/models/pretrained — model directory resolution and weight loading.

The HuggingFace fallback chain is silently load-bearing: every user-facing
``ISEFlow_AIS()``/``ISEFlow_GrIS()`` instantiation depends on
``get_model_dir()`` returning a directory that contains the four expected
artefacts (``deep_ensemble.pth``, ``normalizing_flow.pth``, ``scaler_X.pkl``,
``scaler_y.pkl``).  The variable lists tested here are the canonical column
ordering applied by ``ISEFlow_AIS.process()`` / ``ISEFlow_GrIS.process()`` —
silent edits to them shift every column in the feature tensor.
"""

import os

import pytest

from ise.models.pretrained import (
    ISEFLOW_LATEST_MODEL_VERSION,
    ISEFlow_AIS_v1_0_0_variables,
    ISEFlow_AIS_v1_1_0_variables,
    ISEFlow_GrIS_v1_0_0_variables,
    ISEFlow_GrIS_v1_1_0_variables,
    _subfolder,
    get_model_dir,
)

EXPECTED_ARTEFACTS = (
    "deep_ensemble.pth",
    "normalizing_flow.pth",
    "scaler_X.pkl",
    "scaler_y.pkl",
)


# ---------------------------------------------------------------------------
# _subfolder — pure formatting (no I/O)
# ---------------------------------------------------------------------------


class TestSubfolder:
    def test_format_v1_1_0_AIS(self):
        assert _subfolder("v1.1.0", "AIS") == "v1.1.0/ISEFlow_AIS_v1-1-0"

    def test_format_v1_0_0_GrIS(self):
        assert _subfolder("v1.0.0", "GrIS") == "v1.0.0/ISEFlow_GrIS_v1-0-0"


# ---------------------------------------------------------------------------
# get_model_dir — resolves to a directory containing the 4 required artefacts
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestGetModelDir:
    """Marked slow because the first call may trigger a HuggingFace download."""

    @pytest.mark.parametrize(
        "version,ice_sheet",
        [
            ("v1.1.0", "AIS"),
            ("v1.1.0", "GrIS"),
            ("v1.0.0", "AIS"),
            ("v1.0.0", "GrIS"),
        ],
    )
    def test_returns_dir_with_required_artefacts(self, version, ice_sheet):
        path = get_model_dir(version, ice_sheet)
        assert os.path.isdir(path), f"Returned path is not a directory: {path}"
        contents = set(os.listdir(path))
        missing = set(EXPECTED_ARTEFACTS) - contents
        assert not missing, (
            f"{version}/{ice_sheet} model dir missing required artefacts: {missing}\n"
            f"  path: {path}\n"
            f"  contents: {sorted(contents)}"
        )


# ---------------------------------------------------------------------------
# Fallback behaviour — mock HF failure, ensure local fallback is used.
# This test does NOT trigger HF downloads.
# ---------------------------------------------------------------------------


class TestGetModelDirFallback:
    def test_raises_when_hf_unavailable_and_no_local(self, monkeypatch, tmp_path):
        """When snapshot_download raises and no local fallback exists, RuntimeError."""
        from ise.models import pretrained as pretrained_mod

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated HF failure")

        monkeypatch.setattr(pretrained_mod, "snapshot_download", _raise)
        # Point the local fallback at a directory that doesn't exist
        monkeypatch.setattr(pretrained_mod, "_LOCAL_PRETRAINED_DIR", str(tmp_path))

        with pytest.raises(RuntimeError, match="Could not download"):
            get_model_dir("v1.1.0", "AIS")


# ---------------------------------------------------------------------------
# Variable list integrity — guards the column-order source of truth
# ---------------------------------------------------------------------------


class TestVariableLists:
    """If these lists drift silently, every prediction silently misaligns columns."""

    def test_ISEFLOW_LATEST_MODEL_VERSION_is_v1_1_0(self):
        assert ISEFLOW_LATEST_MODEL_VERSION == "v1.1.0"

    @pytest.mark.parametrize(
        "name,variables,expected_len",
        [
            ("AIS_v1_0_0", ISEFlow_AIS_v1_0_0_variables, 98),
            ("AIS_v1_1_0", ISEFlow_AIS_v1_1_0_variables, 92),
            ("GrIS_v1_0_0", ISEFlow_GrIS_v1_0_0_variables, 90),
            ("GrIS_v1_1_0", ISEFlow_GrIS_v1_1_0_variables, 90),
        ],
    )
    def test_variable_list_length(self, name, variables, expected_len):
        assert len(variables) == expected_len, (
            f"{name} variable list length changed: expected {expected_len}, got {len(variables)}. "
            "If intentional (e.g. retraining with new features), update this test AND the "
            "corresponding pretrained model weights."
        )

    @pytest.mark.parametrize(
        "name,variables",
        [
            ("AIS_v1_0_0", ISEFlow_AIS_v1_0_0_variables),
            ("AIS_v1_1_0", ISEFlow_AIS_v1_1_0_variables),
            ("GrIS_v1_0_0", ISEFlow_GrIS_v1_0_0_variables),
            ("GrIS_v1_1_0", ISEFlow_GrIS_v1_1_0_variables),
        ],
    )
    def test_variable_list_no_duplicates(self, name, variables):
        # Duplicates here would cause `data.loc[:, ~data.columns.duplicated()]` in
        # process() to silently drop the second occurrence, shifting downstream features.
        assert len(set(variables)) == len(variables), (
            f"{name} contains duplicate columns: {[v for v in variables if variables.count(v) > 1]}"
        )

    def test_AIS_v1_0_0_includes_mrro(self):
        """v1.0.0 AIS must include mrro_anomaly (the defining v1.0.0 feature)."""
        assert "mrro_anomaly" in ISEFlow_AIS_v1_0_0_variables

    def test_AIS_v1_1_0_excludes_mrro(self):
        """v1.1.0 AIS must NOT include mrro_anomaly (the defining v1.1.0 change)."""
        assert "mrro_anomaly" not in ISEFlow_AIS_v1_1_0_variables
