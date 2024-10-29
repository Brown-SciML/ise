import os
import torch
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
import time
import json

try:
    from ise.data.feature_engineer import FeatureEngineer
    from ise.models.ISEFlow import ISEFlow, WeakPredictor
    from ise.utils.functions import get_X_y, unscale
except ModuleNotFoundError:
    import sys
    sys.path.append('/users/pvankatw/research/ise/')
    from ise.data.feature_engineer import FeatureEngineer
    from ise.models.ISEFlow import ISEFlow, WeakPredictor
    from ise.utils.functions import get_X_y, unscale

def add_lagged_shaps(shap_df):
    """
    Aggregate lagged SHAP values by summing values with a shared base variable.

    Args:
        shap_df (pd.DataFrame): SHAP values dataframe with lagged variables.
    
    Returns:
        pd.DataFrame: Aggregated SHAP values by base variable.
    """
    base_variables = set(name.split('.')[0] for name in list(shap_df.columns) if 'lag' in name)
    aggregated_shap_values = pd.DataFrame(index=shap_df.index)

    # Copy non-lagged columns as-is
    for name in list(shap_df.columns):
        if name.split('.')[0] not in base_variables and not 'lag' in name:
            aggregated_shap_values[name] = shap_df[name]

    # Aggregate lagged SHAP values
    for base_var in base_variables:
        related_columns = [col for col in list(shap_df.columns) if col.startswith(base_var) and ('lag' in col or col.endswith(tuple('.12345')))]
        aggregated_shap_values[base_var] = shap_df[related_columns].sum(axis=1)

    return aggregated_shap_values

def aggregate_lagged_columns(dataframe):
    """
    Aggregate lagged columns by summing them into their base variables.

    Args:
        dataframe (pd.DataFrame): DataFrame with lagged columns.

    Returns:
        pd.DataFrame: DataFrame with aggregated base variables.
    """
    base_variables = set(name.split('.')[0] for name in dataframe.columns if 'lag' in name)
    aggregated_data = pd.DataFrame()

    # Copy non-lagged columns as-is
    for name in dataframe.columns:
        if name.split('.')[0] not in base_variables and not 'lag' in name and not name.endswith(tuple('.12345')):
            aggregated_data[name] = dataframe[name]

    # Aggregate lagged columns
    for base_var in base_variables:
        related_columns = [col for col in dataframe.columns if col.startswith(base_var) and ('lag' in col or col.endswith(tuple('.12345')))]
        if related_columns:
            aggregated_data[base_var] = dataframe[related_columns].sum(axis=1)

    return aggregated_data

def aggregate_categorical_columns(dataframe):
    """
    Aggregate categorical columns that have lagged versions.

    Args:
        dataframe (pd.DataFrame): DataFrame with categorical columns.

    Returns:
        pd.DataFrame: Aggregated DataFrame by base variables.
    """
    base_variables = [
        'numerics', 'initial-year', 'stress-balance', 'resolution', 'init-method',
        'melt', 'ice-front', 'open-melt-param', 'standard-melt-param', 'Ocean forcing',
        'Ocean sensitivity', 'Ice shelf fracture', 'initialization', 'res-max', 'res-min',
        'bed', 'ghf', 'velocity', 'surface-thickness', 'ice-flow', 'initial_smb'
    ]
    aggregated_data = pd.DataFrame()

    # Copy non-lagged columns as-is
    for name in dataframe.columns:
        if name.split('.')[0] not in base_variables:
            aggregated_data[name] = dataframe[name]

    # Aggregate categorical columns
    for base_var in base_variables:
        related_columns = [col for col in dataframe.columns if col.startswith(base_var)]
        if related_columns:
            aggregated_data[base_var] = dataframe[related_columns].sum(axis=1)

    return aggregated_data

def run_shap(model, sample_rows, X_val_cols, out_dir):
    """
    Generate SHAP values for a model and save results as plots and CSV.

    Args:
        model (torch.nn.Module): The trained model for generating predictions.
        sample_rows (np.ndarray): Sample data used for SHAP analysis.
        X_val_cols (list of str): Feature names for validation data.
        out_dir (str): Output directory for saving results.
    """
    # Define prediction function for SHAP explainer
    def f(x):
        return model.predict(x)

    # Initialize SHAP explainer and compute SHAP values
    explainer = shap.Explainer(f, sample_rows)
    shap_values = explainer.shap_values(sample_rows)

    # Create DataFrame for SHAP values and add lagged SHAP values
    df = pd.DataFrame(shap_values, columns=X_val_cols)
    df = add_lagged_shaps(df)
    print(df.shape)

    # Save sample rows and SHAP values to CSV
    curr = time.time()
    pd.DataFrame(sample_rows, columns=X_val_cols).to_csv(f"{out_dir}/sample_rows_{n_projections}_{curr}.csv", index=False)
    df['run_time'] = curr
    df.to_csv(f"{out_dir}/shap_values_{n_projections}_{curr}.csv", index=False)

    # Generate SHAP summary plot
    shap.summary_plot(shap_values, sample_rows, feature_names=X_val_cols)
    plt.savefig(f'{out_dir}/shap_values_plot_{n_projections}_{curr}.png')
    plt.close('all')

    # Aggregate lagged columns and plot the summarized SHAP values
    sample_rows_df = pd.DataFrame(sample_rows, columns=X_val_cols)
    sample_rows_df = aggregate_lagged_columns(sample_rows_df)
    shap_values_df = pd.DataFrame(shap_values, columns=X_val_cols)
    shap_values_df = aggregate_lagged_columns(shap_values_df)
    plt.figure()
    shap.summary_plot(shap_values_df.values, sample_rows_df.values, feature_names=sample_rows_df.columns)
    plt.savefig(f'{out_dir}/shap_values_plot_{n_projections}_{curr}_summed.png')

    # Drop lagged columns and generate a new SHAP plot
    df = df.drop(columns=[x for x in df.columns if 'lag' in x])
    sample_rows_df = sample_rows_df.drop(columns=[x for x in sample_rows_df.columns if 'lag' in x])
    plt.figure()
    shap.summary_plot(df.values, sample_rows_df.values, feature_names=sample_rows_df.columns)
    plt.savefig(f'{out_dir}/shap_values_plot_{n_projections}_{curr}_dropped.png')

    return df

if __name__ == '__main__':
    ITERATIONS = 100
    n_projections = 50
    ICE_SHEET = 'AIS'
    print('Ice Sheet:', ICE_SHEET)

    # Define directories and model path
    dir_ = f"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/data/ml/{ICE_SHEET}/"
    best_models = json.load(open(f'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/models/bests_model_paths.json'))
    model_dir = best_models['WeakPredictor'][ICE_SHEET]
    modelname = model_dir.split('/')[-2]

    # Load validation data
    val = pd.read_csv(f'{dir_}/val.csv')
    X_val, y_val = get_X_y(val, 'sectors', return_format='numpy')
    X_val_df, y_val_df = get_X_y(val, 'sectors', return_format='pandas')

    # Initialize model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if ICE_SHEET == 'AIS':
        model = WeakPredictor(lstm_num_layers=1, lstm_hidden_size=512, input_size=X_val.shape[1], ice_sheet='AIS', criterion=torch.nn.MSELoss())
    else:
        model = WeakPredictor(lstm_num_layers=1, lstm_hidden_size=512, input_size=X_val.shape[1], ice_sheet='GrIS', criterion=torch.nn.MSELoss())

    model.load_state_dict(torch.load(f"{model_dir}/wp_model.pth", map_location=device))
    model.to(device)
    model.train()

    # Create output directory
    out_dir = f'/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/ISEFlow/explainability/SHAP/SLE_target/{ICE_SHEET}/nprojs{n_projections}_{modelname}'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Generate SHAP values for multiple iterations
    all_results = []
    for _ in range(ITERATIONS):
        if len(X_val) // 86 < n_projections:
            n_projections = len(X_val) // 86

        # Randomly sample projections for SHAP analysis
        sample_proj_starts = np.random.choice(np.arange(0, len(X_val) // 86) * 86, n_projections, replace=False)
        print('n_projections', n_projections)
        print('sample_proj_starts', sample_proj_starts)

        sample_rows = np.vstack([X_val[start:start + 86, :] for start in sample_proj_starts])

        # Run SHAP analysis and collect results
        df = run_shap(model, sample_rows, X_val_df.columns, out_dir)
        all_results.append(df)

    # Save all results to a CSV file
    all_results = pd.concat(all_results, axis=0)
    all_results.to_csv(f"{out_dir}/all_results.csv", index=False)
