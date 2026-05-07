import pickle

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from ise.utils.functions import (
    check_input,
    get_all_filepaths,
    get_X_y,
    to_tensor,
    undummify,
    unscale_output,
)

# ---------------------------------------------------------------------------
# to_tensor
# ---------------------------------------------------------------------------


class TestToTensor:
    def test_from_dataframe(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        t = to_tensor(df)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert t.shape == (2, 2)

    def test_from_series(self):
        s = pd.Series([1.0, 2.0, 3.0])
        t = to_tensor(s)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32

    def test_from_numpy(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        t = to_tensor(arr)
        assert isinstance(t, torch.Tensor)
        assert t.dtype == torch.float32
        assert t.shape == (2, 2)

    def test_from_tensor_passthrough(self):
        original = torch.rand(3, 5)
        t = to_tensor(original)
        assert t is original or torch.equal(t, original)
        assert t.dtype == torch.float32

    def test_from_none_returns_none(self):
        assert to_tensor(None) is None

    def test_invalid_type_raises(self):
        with pytest.raises(ValueError):
            to_tensor([1, 2, 3])

    def test_values_preserved_from_numpy(self):
        arr = np.array([[1.5, 2.5]])
        t = to_tensor(arr)
        assert t[0, 0].item() == pytest.approx(1.5)
        assert t[0, 1].item() == pytest.approx(2.5)

    def test_integer_numpy_cast_to_float32(self):
        arr = np.array([[1, 2, 3]], dtype=np.int32)
        t = to_tensor(arr)
        assert t.dtype == torch.float32


# ---------------------------------------------------------------------------
# get_all_filepaths
# ---------------------------------------------------------------------------


class TestGetAllFilepaths:
    @pytest.fixture
    def file_tree(self, tmp_path):
        """Create a small directory tree with mixed file types."""
        (tmp_path / "a.csv").write_text("1")
        (tmp_path / "b.csv").write_text("2")
        (tmp_path / "c.txt").write_text("3")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "d.csv").write_text("4")
        (subdir / "e_exclude.csv").write_text("5")
        return tmp_path

    def test_finds_all_files_no_filter(self, file_tree):
        files = get_all_filepaths(str(file_tree))
        assert len(files) == 5

    def test_filetype_filter(self, file_tree):
        files = get_all_filepaths(str(file_tree), filetype="csv")
        assert all(f.endswith(".csv") for f in files)
        assert len(files) == 4

    def test_txt_filter(self, file_tree):
        files = get_all_filepaths(str(file_tree), filetype="txt")
        assert len(files) == 1

    def test_contains_filter(self, file_tree):
        files = get_all_filepaths(str(file_tree), contains="exclude")
        assert len(files) == 1
        assert "exclude" in files[0]

    def test_not_contains_filter(self, file_tree):
        files = get_all_filepaths(str(file_tree), not_contains="exclude", filetype="csv")
        assert all("exclude" not in f for f in files)
        assert len(files) == 3

    def test_empty_directory_returns_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert get_all_filepaths(str(empty)) == []


# ---------------------------------------------------------------------------
# check_input
# ---------------------------------------------------------------------------


class TestCheckInput:
    def test_valid_input_passes(self):
        check_input("numpy", ["numpy", "tensor", "pandas"])  # should not raise

    def test_case_insensitive(self):
        check_input("NUMPY", ["numpy", "tensor", "pandas"])  # should not raise

    def test_invalid_input_raises(self):
        with pytest.raises(ValueError):
            check_input("zarr", ["numpy", "tensor", "pandas"])

    def test_argname_appears_in_error_message(self):
        with pytest.raises(ValueError, match="return_format"):
            check_input("bad", ["numpy", "tensor"], argname="return_format")

    def test_valid_option_at_boundary(self):
        check_input("pandas", ["numpy", "tensor", "pandas"])  # should not raise


# ---------------------------------------------------------------------------
# undummify — round-trip from get_dummies
# ---------------------------------------------------------------------------


class TestUndummify:
    def test_round_trips_categorical_column(self):
        """undummify(get_dummies(df)) recovers the original categorical column."""
        df = pd.DataFrame({"numerics": ["fd", "fe", "fd", "fe"], "value": [1.0, 2.0, 3.0, 4.0]})
        dummies = pd.get_dummies(df, columns=["numerics"], prefix_sep="-")
        recovered = undummify(dummies, prefix_sep="-")
        assert "numerics" in recovered.columns
        assert list(recovered["numerics"]) == list(df["numerics"])
        # Non-categorical column passes through unchanged
        np.testing.assert_array_equal(recovered["value"].values, df["value"].values)


# ---------------------------------------------------------------------------
# unscale_output — round-trip via pickled sklearn scaler
# ---------------------------------------------------------------------------


class TestUnscaleOutput:
    def test_round_trips_via_pickled_scaler(self, tmp_path):
        """unscale_output must invert scaler.transform exactly (within numerical tolerance)."""
        rng = np.random.default_rng(0)
        y = rng.random((20, 1)) * 100 - 50
        scaler = SkStandardScaler().fit(y)
        scaled = scaler.transform(y)

        scaler_path = tmp_path / "scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        recovered = unscale_output(scaled, str(scaler_path))
        np.testing.assert_allclose(recovered, y, rtol=1e-9)


# ---------------------------------------------------------------------------
# get_X_y — sectors path drops id/model and returns sle as target
# ---------------------------------------------------------------------------


class TestGetXy:
    def test_sectors_drops_id_and_model_returns_sle_as_y(self):
        """X must not contain id/model/exp; y must be the sle column.

        These columns are how get_X_y separates metadata from features — if the
        drop-list ever changes silently, training data leaks identity into the
        model.
        """
        df = pd.DataFrame(
            {
                "id": ["a", "b", "c"],
                "model": ["m1", "m1", "m2"],
                "exp": ["e1", "e1", "e2"],
                "sle": [0.1, 0.2, 0.3],
                "feature1": [1.0, 2.0, 3.0],
                "feature2": [4.0, 5.0, 6.0],
            }
        )
        X, y = get_X_y(df, dataset_type="sectors", return_format="pandas")
        assert "id" not in X.columns
        assert "model" not in X.columns
        assert "exp" not in X.columns
        assert "sle" not in X.columns
        # Features are preserved
        assert "feature1" in X.columns
        assert "feature2" in X.columns
        # y is the sle column
        np.testing.assert_array_equal(y["sle"].values, df["sle"].values)
