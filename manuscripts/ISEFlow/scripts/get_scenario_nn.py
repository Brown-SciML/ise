import shap
from sklearn.ensemble import RandomForestClassifier
import torch
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# Try importing ISE-specific modules, add a fallback for non-standard paths
try:
    from ise.data.dataclasses import ScenarioDataset
    from ise.models.scenario import ScenarioPredictor
    from ise.utils import functions as f
    from ise.evaluation import metrics as m
except ModuleNotFoundError:
    import sys
    sys.path.append('/users/pvankatw/research/ise/')
    from ise.data.dataclasses import ScenarioDataset
    from ise.models.scenario import ScenarioPredictor
    from ise.utils import functions as f
    from ise.evaluation import metrics as m


def train_scenario_model(ice_sheet='AIS'):
    """
    Trains a scenario classification model using data from a given ice sheet and generates SHAP values for feature importance.

    Args:
        ice_sheet (str): The ice sheet to use for training ('AIS' for Antarctic Ice Sheet, 'GrIS' for Greenland Ice Sheet). Default is 'AIS'.
    
    Outputs:
        - Trains the `ScenarioPredictor` model using the dataset.
        - Generates SHAP values for the model predictions.
        - Saves SHAP summary plots and feature importance values to CSV.
    """
    
    start_time = time.time()
    data_dir = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ice_sheet}/"

    # Load training and validation data
    data = pd.read_csv(f"{data_dir}/train.csv")
    val_data = pd.read_csv(f'{data_dir}/val.csv')
    
    # Prepare features and labels for training and validation datasets
    X_train, y_train = f.get_X_y(data, dataset_type='scenario', return_format='pandas')
    X_val, y_val = f.get_X_y(val_data, dataset_type='scenario', return_format='pandas')
    X_train['id'], X_val['id'] = X_train.index // 86 + 1, X_val.index // 86 + 1
    X_train['set'] = 'train'
    X_val['set'] = 'val'

    train = pd.concat([X_train, y_train], axis=1)
    val = pd.concat([X_val, y_val], axis=1)
    all_data = pd.concat([train, val], axis=0)

    # Define the relevant columns for each ice sheet
    if ice_sheet == 'AIS':
        cols = ['pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly', 'ts_anomaly', 'thermal_forcing', 'salinity', 'temperature', 'Scenario']
    else:
        cols = ['aST', 'aSMB', 'basin_runoff', 'thermal_forcing', 'Scenario']
    print(cols)

    # Subset the data to the relevant columns
    subset_data = all_data[cols].drop_duplicates()

    # Balance the dataset by sampling equal numbers of both scenarios (rcp2.6 and rcp8.5)
    data_26 = subset_data[subset_data['Scenario'] == 'rcp2.6']
    subset_data = pd.concat([data_26, subset_data[subset_data['Scenario'] == 'rcp8.5'].sample(n=len(data_26))], axis=0)

    # Separate the features (X) and labels (y)
    X = subset_data.drop(columns=['Scenario'])
    y = subset_data['Scenario']

    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert to numpy arrays and prepare for PyTorch
    X_train = X_train.to_numpy()
    X_val = X_val.to_numpy()
    if X_train.shape[1] == 1:
        X_train = X_train.reshape(-1, 1)
        X_val = X_val.reshape(-1, 1)
    y_train, y_val = y_train.to_numpy(), y_val.to_numpy()

    # Convert labels to binary (1 for 'rcp8.5', 0 for 'rcp2.6')
    y_train, y_val = (y_train == 'rcp8.5').astype(int), (y_val == 'rcp8.5').astype(int)

    # Convert data to PyTorch tensors
    X_train, y_train = torch.tensor(X_train).float(), torch.tensor(y_train).float()
    X_val, y_val = torch.tensor(X_val).float(), torch.tensor(y_val).float()

    # Create PyTorch datasets and data loaders
    train_dataset = ScenarioDataset(X_train, y_train)
    val_dataset = ScenarioDataset(X_val, y_val)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True)

    # Initialize and train the model
    model = ScenarioPredictor(input_size=X_train.shape[1], hidden_layers=[512, 128, 32])
    model.fit(train_loader, val_loader, epochs=100)
    model.load_state_dict(torch.load('checkpoint.pth'))

    # Function to use model for SHAP predictions
    def predict_fn(x):
        return model.predict(x).detach().cpu().numpy()

    # Generate SHAP values
    sample_rows = X_val.numpy()
    explainer = shap.Explainer(predict_fn, sample_rows)
    shap_values = explainer(sample_rows)

    # Convert SHAP values to DataFrame
    shap_df = pd.DataFrame(shap_values.values, columns=[x for x in cols if x != 'Scenario'])

    # Save SHAP values to CSV
    timestamp = time.time()
    shap_df.to_csv(f"./{ice_sheet}_shap_values_{timestamp}.csv", index=False)

    # Plot SHAP summary plots
    shap.summary_plot(shap_values, sample_rows, feature_names=cols)
    plt.savefig(f'{ice_sheet}_shap_values_plot_{timestamp}.png')
    plt.close('all')

    shap.summary_plot(shap_values, sample_rows, feature_names=cols, plot_type='bar')
    plt.savefig(f'{ice_sheet}_shap_values_plot_{timestamp}_bar.png')

    # Create a bar plot of SHAP values and save it
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values)
    plt.gcf().set_size_inches(12, 8)
    plt.savefig(f'{ice_sheet}_shap_values_plot_{timestamp}_bar.svg')

    print(f"Completed SHAP analysis and saved plots for {ice_sheet}.")

if __name__ == '__main__':
    # Train model and run SHAP analysis for the Antarctic Ice Sheet (AIS) by default
    train_scenario_model('AIS')
    # Uncomment for Greenland Ice Sheet (GrIS) scenario
    # train_scenario_model('GrIS')
