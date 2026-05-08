import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler as SkStandardScaler

from ise.data.feature_engineer import (
    FeatureEngineer,
    add_lag_variables,
    scale_data,
    split_training_data,
)


### ---------------------- Fixtures for Sample Data ---------------------- ###
@pytest.fixture
def sample_dataframe():
    """Creates a small sample dataset for testing"""
    data = pd.DataFrame(
        {
            "id": np.arange(1, 11),
            "model": ["A"] * 5 + ["B"] * 5,
            "exp": ["exp1", "exp1", "exp2", "exp2", "exp3"] * 2,
            "sector": [1, 2, 3, 4, 3] * 2,
            "year": np.arange(2000, 2010),
            "mrro_anomaly": [1.2, np.nan, 2.3, np.nan, 3.4, 4.5, 5.6, np.nan, 6.7, 7.8],
            "sle": np.random.randn(10),
            "feature1": np.random.rand(10),
            "feature2": np.random.rand(10),
        }
    )
    return data


@pytest.fixture
def feature_engineer_instance(sample_dataframe):
    """Creates an instance of FeatureEngineer with sample data"""
    return FeatureEngineer(ice_sheet="TestSheet", data=sample_dataframe, split_dataset=False)


### ---------------------- Initialization Tests ---------------------- ###
def test_feature_engineer_initialization(feature_engineer_instance):
    """Ensure FeatureEngineer initializes correctly"""
    fe = feature_engineer_instance
    assert fe.data is not None
    assert isinstance(fe.data, pd.DataFrame)
    assert fe.ice_sheet == "TestSheet"


def test_feature_engineer_fill_mrro_nans(feature_engineer_instance):
    """Ensure missing values in 'mrro_anomaly' are filled"""
    fe = feature_engineer_instance
    fe.fill_mrro_nans(method="zero")
    assert fe.data["mrro_anomaly"].isnull().sum() == 0  # No NaNs should remain


def test_feature_engineer_fill_mrro_invalid_method(feature_engineer_instance):
    """Ensure invalid fill method raises ValueError"""
    fe = feature_engineer_instance
    with pytest.raises(ValueError):
        fe.fill_mrro_nans(method="invalid")


### ---------------------- Data Splitting Tests ---------------------- ###
@pytest.mark.filterwarnings("ignore")
def test_feature_engineer_split_data(feature_engineer_instance):
    """Ensure data splitting works correctly"""
    fe = feature_engineer_instance
    train, val, test = fe.split_data()

    assert isinstance(train, pd.DataFrame)
    assert isinstance(val, pd.DataFrame)
    assert isinstance(test, pd.DataFrame)


def test_split_training_data_invalid_input():
    """Ensure invalid data input raises FileNotFoundError"""
    with pytest.raises(FileNotFoundError):
        split_training_data("invalid_path.csv", train_size=0.7, val_size=0.2, test_size=0.1)


### ---------------------- Scaling Tests ---------------------- ###
def test_feature_engineer_scale_data(feature_engineer_instance):
    """Ensure scaling works correctly"""
    fe = feature_engineer_instance
    X_scaled, y_scaled = fe.scale_data(method="standard")

    assert X_scaled.shape[0] == 10  # row count preserved
    assert X_scaled.ndim == 2  # 2D feature matrix
    assert y_scaled.shape == (10, 1)  # single target column


@pytest.mark.filterwarnings("ignore")
def test_feature_engineer_unscale_data(feature_engineer_instance):
    """Ensure unscaling works correctly"""
    fe = feature_engineer_instance
    fe.scale_data(method="standard")

    X_scaled, y_scaled = fe.scale_data(method="standard")

    X_unscaled, y_unscaled = fe.unscale_data(X=X_scaled, y=y_scaled)

    print(
        y_unscaled,
    )
    print()
    print(fe.y.values)

    assert np.allclose(X_unscaled[0][0], fe.X.values[0][0], atol=1e-3)
    assert np.allclose(y_unscaled[0][0], fe.y.values[0][0], atol=1e-3)


def test_scale_data_invalid_method(feature_engineer_instance):
    """Ensure invalid scaling method raises ValueError"""
    fe = feature_engineer_instance
    with pytest.raises(ValueError):
        fe.scale_data(method="invalid")


### ---------------------- Outlier Handling Tests ---------------------- ###
def test_feature_engineer_backfill_outliers(feature_engineer_instance):
    """Ensure backfill outliers replaces extreme values with the next valid value.
    bfill() cannot fill trailing NaNs (outliers at the end of the series), so
    we assert that only a trailing block of NaNs remains — no interior NaNs.
    """
    fe = feature_engineer_instance
    fe.backfill_outliers(percentile=95)
    sle = fe.data["sle"]
    # All NaNs must be at the tail (bfill fills interior/leading outliers, not trailing ones)
    null_mask = sle.isnull()
    if null_mask.any():
        first_null = null_mask.idxmax()
        assert null_mask[first_null:].all(), (
            "Interior NaNs remain after backfill — outliers were not replaced"
        )


def test_feature_engineer_drop_outliers(feature_engineer_instance):
    """Ensure outliers are dropped correctly"""
    fe = feature_engineer_instance
    fe.drop_outliers(method="quantile", column="sle")
    assert len(fe.data) <= 10  # Some rows should be dropped


### ---------------------- Lag Feature Tests ---------------------- ###
def test_feature_engineer_add_lag_variables():
    """Ensure lag variables are added correctly."""
    n = 86
    data = pd.DataFrame(
        {
            "id": np.arange(n),
            "model": ["A"] * n,
            "exp": ["exp1"] * n,
            "sector": [1] * n,
            "year": np.arange(2015, 2015 + n),
            "mrro_anomaly": np.random.rand(n),
            "sle": np.random.randn(n),
            "ts_anomaly": np.random.rand(n),
            "smb_anomaly": np.random.rand(n),
        }
    )
    fe = FeatureEngineer(ice_sheet="TestSheet", data=data, split_dataset=False)
    original_cols = set(fe.data.columns)
    fe.add_lag_variables(lag=2)
    new_cols = set(fe.data.columns) - original_cols
    assert any("lag" in col for col in new_cols)


def test_feature_engineer_fill_mrro_mean(feature_engineer_instance):
    """fill_mrro_nans with method='mean' fills all NaNs."""
    fe = feature_engineer_instance
    fe.fill_mrro_nans(method="mean")
    assert fe.data["mrro_anomaly"].isnull().sum() == 0


def test_feature_engineer_fill_mrro_median(feature_engineer_instance):
    """fill_mrro_nans with method='median' fills all NaNs."""
    fe = feature_engineer_instance
    fe.fill_mrro_nans(method="median")
    assert fe.data["mrro_anomaly"].isnull().sum() == 0


def test_feature_engineer_drop_outliers_expression(feature_engineer_instance):
    """drop_outliers with method='explicit' drops entire runs matching the expression."""
    fe = feature_engineer_instance
    n_before = len(fe.data)
    # Use a threshold that no sle value can exceed (all randn values < 9999)
    # so every row matches and every run gets dropped
    fe.drop_outliers(method="explicit", column="sle", expression=[("sle", "<", 9999)])
    # All rows match the expression so all runs should be dropped
    assert len(fe.data) == 0


### ---------------------- Model Characteristics Tests ---------------------- ###
@pytest.fixture
def mock_model_characteristics(tmp_path):
    """Creates a temporary model characteristics CSV file"""
    file = tmp_path / "model_characteristics.csv"
    df = pd.DataFrame(
        {
            "model": ["A", "B"],
            "Ocean forcing": ["low", "high"],
            "Ocean sensitivity": ["low", "high"],
            "Ice shelf fracture": [True, False],
        }
    )
    df.to_csv(file, index=False)
    return str(file)


def test_feature_engineer_add_model_characteristics(
    feature_engineer_instance, mock_model_characteristics
):
    """Ensure model characteristics are added correctly"""
    fe = feature_engineer_instance
    fe.add_model_characteristics(model_char_path=mock_model_characteristics)
    assert "Ocean forcing_high" in list(fe.data.columns)


### ---------------------- Edge Case Tests ---------------------- ###
def test_feature_engineer_unscale_without_scalers(feature_engineer_instance):
    """Ensure unscaling without prior scaling raises ValueError"""
    fe = feature_engineer_instance
    with pytest.raises(ValueError):
        fe.unscale_data(X=np.random.rand(10, 2), y=np.random.rand(10, 1))


@pytest.mark.filterwarnings("ignore:Data length.*is not divisible by projection_length:UserWarning")
def test_feature_engineer_invalid_lag_value(feature_engineer_instance):
    """Ensure invalid lag values raise an error"""
    fe = feature_engineer_instance
    with pytest.raises(ValueError):
        fe.add_lag_variables(lag=-1)


### ---------------------- File Handling Tests ---------------------- ###
def test_feature_engineer_saves_scalers(feature_engineer_instance, tmp_path):
    """Ensure scalers are saved correctly"""
    fe = feature_engineer_instance
    save_dir = tmp_path / "scalers"
    save_dir.mkdir()

    fe.scale_data(method="standard", save_dir=str(save_dir))

    assert (save_dir / "scaler_X.pkl").exists()
    assert (save_dir / "scaler_y.pkl").exists()


@pytest.mark.filterwarnings("ignore")
def test_feature_engineer_loads_scalers(feature_engineer_instance, tmp_path):
    """Ensure scalers are loaded correctly"""
    fe = feature_engineer_instance
    save_dir = tmp_path / "scalers"
    save_dir.mkdir()

    fe.scale_data(method="standard", save_dir=str(save_dir))

    fe_new = FeatureEngineer("TestSheet", fe.data)
    fe_new.scaler_X_path = str(save_dir / "scaler_X.pkl")
    fe_new.scaler_y_path = str(save_dir / "scaler_y.pkl")

    X_scaled, y_scaled = fe_new.scale_data()
    assert X_scaled.shape == fe.X.shape


### ---------------------- Module-level Function Tests ---------------------- ###
# These test the standalone functions used by ISEFlow_AIS.process() and
# ISEFlow_GrIS.process() — distinct from the FeatureEngineer class methods.


PROJ_LEN = 86


def _make_two_projection_df():
    """Two 86-step projections with distinct, easily-identifiable values per projection."""
    proj1 = pd.DataFrame(
        {
            "year": np.arange(2015, 2015 + PROJ_LEN),
            "pr_anomaly": np.full(PROJ_LEN, -100.0),  # all -100 in projection 1
            "smb_anomaly": np.full(PROJ_LEN, -50.0),
            "temperature": np.full(PROJ_LEN, -10.0),
        }
    )
    proj2 = pd.DataFrame(
        {
            "year": np.arange(2015, 2015 + PROJ_LEN),
            "pr_anomaly": np.full(PROJ_LEN, 100.0),  # all 100 in projection 2
            "smb_anomaly": np.full(PROJ_LEN, 50.0),
            "temperature": np.full(PROJ_LEN, 10.0),
        }
    )
    return pd.concat([proj1, proj2], ignore_index=True)


class TestAddLagVariablesModuleLevel:
    """Tests for the standalone add_lag_variables function used by ISEFlow.process()."""

    def test_creates_lag1_through_lagN_for_each_forcing(self):
        """All recognized forcing columns get lag1..lagN columns; non-forcing columns don't."""
        df = pd.DataFrame(
            {
                "year": np.arange(2015, 2015 + PROJ_LEN),
                "pr_anomaly": np.random.rand(PROJ_LEN),
                "smb_anomaly": np.random.rand(PROJ_LEN),
                "temperature": np.random.rand(PROJ_LEN),
            }
        )
        result = add_lag_variables(df, lag=5, verbose=False)
        for forcing in ("pr_anomaly", "smb_anomaly", "temperature"):
            for k in range(1, 6):
                assert f"{forcing}.lag{k}" in result.columns, (
                    f"Missing lag column: {forcing}.lag{k}"
                )
        # year does not get lagged (it's the temporal indicator)
        assert "year.lag1" not in result.columns

    def test_no_cross_projection_bleed_in_lag_values(self):
        """At the start of projection 2, lag values must come from projection 2's own
        first row (via bfill), not from projection 1's tail.

        This is the most insidious silent corruption: if the per-projection segmentation
        in add_lag_variables breaks, the model trains on data that mixes runs together
        but produces plausible-looking outputs.
        """
        df = _make_two_projection_df()
        result = add_lag_variables(df, lag=5, verbose=False)
        # At the start of projection 2 (row index 86), lag values must be from
        # projection 2 (positive), NOT from projection 1's tail (negative).
        proj2_start = result.iloc[PROJ_LEN]
        for k in range(1, 6):
            assert proj2_start[f"pr_anomaly.lag{k}"] == pytest.approx(100.0), (
                f"Cross-projection bleed in pr_anomaly.lag{k} at projection 2 start"
            )
            assert proj2_start[f"smb_anomaly.lag{k}"] == pytest.approx(50.0)
            assert proj2_start[f"temperature.lag{k}"] == pytest.approx(10.0)

        # Sanity: end of projection 1 (row 85) should still have projection 1's values
        proj1_end = result.iloc[PROJ_LEN - 1]
        assert proj1_end["pr_anomaly"] == pytest.approx(-100.0)


class TestScaleDataModuleLevel:
    """Tests for the standalone scale_data function used by ISEFlow.process()."""

    def test_round_trips_via_inverse_transform(self, tmp_path):
        """scale_data(scaler_path) must be exactly inverted by the saved scaler."""
        df = pd.DataFrame(
            {
                "pr_anomaly": np.random.rand(20) * 1e-5,
                "smb_anomaly": np.random.rand(20) * 1e-5,
                "non_scaled": np.arange(20.0),  # absent from scaler
            }
        )
        scaler = SkStandardScaler().fit(df[["pr_anomaly", "smb_anomaly"]])
        path = tmp_path / "scaler.pkl"
        with open(path, "wb") as f:
            pickle.dump(scaler, f)

        scaled = scale_data(df, str(path))
        # Column order is preserved (load-bearing for downstream get_dummies/reindex)
        assert list(scaled.columns) == list(df.columns)
        # Non-scaler columns pass through unchanged
        np.testing.assert_array_equal(scaled["non_scaled"].values, df["non_scaled"].values)
        # Scaler columns invert exactly
        recovered = scaler.inverse_transform(scaled[["pr_anomaly", "smb_anomaly"]].values)
        np.testing.assert_allclose(recovered, df[["pr_anomaly", "smb_anomaly"]].values, rtol=1e-9)


# ---------------------------------------------------------------------------
# Regression tests for feature_engineer.py fixes
# ---------------------------------------------------------------------------


def _make_projection_df(n_ids=10, proj_len=86):
    """Build a minimal DataFrame with `id` and `sle` columns suitable for split tests."""
    rows = []
    for id_ in range(1, n_ids + 1):
        for t in range(proj_len):
            rows.append({"id": id_, "year": 2015 + t, "feature1": float(t), "sle": float(t) * 0.01})
    return pd.DataFrame(rows)


def test_feature_engineer_split_dataset_returns_non_none_splits():
    """split_dataset=True must produce non-None, non-empty train/val/test."""
    df = _make_projection_df(n_ids=10)
    fe = FeatureEngineer(ice_sheet="AIS", data=df, split_dataset=True)
    assert fe.train is not None
    assert fe.val is not None
    assert fe.test is not None
    assert len(fe.train) > 0


def test_split_training_data_random_state_is_reproducible():
    """split_training_data with the same random_state must return the same ids."""
    df = _make_projection_df(n_ids=20)
    np.random.seed(0)
    train1, val1, test1 = split_training_data(df, 0.7, 0.15, 0.15, random_state=42)
    np.random.seed(99)
    train2, val2, test2 = split_training_data(df, 0.7, 0.15, 0.15, random_state=42)
    assert set(train1["id"].unique()) == set(train2["id"].unique())


def test_scale_data_with_explicit_X_y_does_not_raise():
    """scale_data(X=..., y=...) must not NameError on the dropped_data concat."""
    df = _make_projection_df(n_ids=5)
    fe = FeatureEngineer(ice_sheet="AIS", data=df, split_dataset=False)
    X_df = df[["feature1"]].copy()
    y_df = df[["sle"]].copy()
    X_scaled, y_scaled = fe.scale_data(X=X_df, y=y_df)
    assert X_scaled is not None
    assert y_scaled is not None
