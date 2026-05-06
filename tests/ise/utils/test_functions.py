import os
import pytest
import numpy as np
import pandas as pd
import torch

from ise.utils.functions import to_tensor, get_all_filepaths, check_input


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
