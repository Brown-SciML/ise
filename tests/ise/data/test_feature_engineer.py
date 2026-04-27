import pytest
import pandas as pd
import numpy as np
import pickle
import os
import json
import torch
from sklearn.preprocessing import StandardScaler
from ise.data.feature_engineer import FeatureEngineer, scale_data, split_training_data, drop_outliers, fill_mrro_nans

### ---------------------- Fixtures for Sample Data ---------------------- ###
@pytest.fixture
def sample_dataframe():
    """Creates a small sample dataset for testing"""
    data = pd.DataFrame({
        "id": np.arange(1, 11),
        "model": ["A"] * 5 + ["B"] * 5,
        "exp": ["exp1", "exp1", "exp2", "exp2", "exp3"] * 2,
        "sector": [1, 2, 3, 4, 3] * 2,
        "year": np.arange(2000, 2010),
        "mrro_anomaly": [1.2, np.nan, 2.3, np.nan, 3.4, 4.5, 5.6, np.nan, 6.7, 7.8],
        "sle": np.random.randn(10),
        "feature1": np.random.rand(10),
        "feature2": np.random.rand(10),
    })
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

    assert X_scaled.shape == (10, 5)  # Only feature1 & feature2 should be scaled
    assert y_scaled.shape == (10, 1)  # Only sle_value should be scaled

@pytest.mark.filterwarnings("ignore")
def test_feature_engineer_unscale_data(feature_engineer_instance):
    """Ensure unscaling works correctly"""
    fe = feature_engineer_instance
    fe.scale_data(method="standard")

    X_scaled, y_scaled = fe.scale_data(method="standard")
    
    X_unscaled, y_unscaled = fe.unscale_data(X=X_scaled, y=y_scaled)
    
    print(y_unscaled, )
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
    """Ensure backfill outliers replaces extreme values"""
    fe = feature_engineer_instance
    fe.backfill_outliers(percentile=95)
    assert fe.data["sle"].isnull().sum() <= 1  # No NaNs should be left after backfilling

def test_feature_engineer_drop_outliers(feature_engineer_instance):
    """Ensure outliers are dropped correctly"""
    fe = feature_engineer_instance
    fe.drop_outliers(method="quantile", column="sle")
    assert len(fe.data) <= 10  # Some rows should be dropped

### ---------------------- Lag Feature Tests ---------------------- ###
# def test_feature_engineer_add_lag_variables(feature_engineer_instance):
#     """Ensure lag variables are added correctly"""
#     fe = feature_engineer_instance
#     fe.add_lag_variables(lag=2)
#     assert any("lag" in col for col in fe.data.columns)

### ---------------------- Model Characteristics Tests ---------------------- ###
@pytest.fixture
def mock_model_characteristics(tmp_path):
    """Creates a temporary model characteristics CSV file"""
    file = tmp_path / "model_characteristics.csv"
    df = pd.DataFrame({"model": ["A", "B"], "Ocean forcing": ["low", "high"], "Ocean sensitivity": ["low", "high"], "Ice shelf fracture": [True, False]})
    df.to_csv(file, index=False)
    return str(file)

def test_feature_engineer_add_model_characteristics(feature_engineer_instance, mock_model_characteristics):
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
