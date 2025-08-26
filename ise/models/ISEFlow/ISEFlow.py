import json
import os
import pickle
import warnings
import shutil

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from nflows import distributions, flows, transforms

from ise.data.ISMIP6.dataclasses import EmulatorDataset
from ise.utils.functions import to_tensor
from ise.utils.training import EarlyStoppingCheckpointer, CheckpointSaver
from ise.data.ISMIP6 import feature_engineer as fe
from ise.models.predictors.deep_ensemble import DeepEnsemble
from ise.models.density_estimators.normalizing_flow import NormalizingFlow
from .de import ISEFlow_AIS_DE, ISEFlow_GrIS_DE
from .nf import ISEFlow_AIS_NF, ISEFlow_GrIS_NF
from ise.models.pretrained import ISEFlow_AIS_v1_0_0_path, ISEFlow_GrIS_v1_0_0_path

class ISEFlow(torch.nn.Module):
    """
    ISEFlow is a hybrid ice sheet emulator that combines a deep ensemble model and a normalizing flow model.

    This class provides methods to train, predict, save, and load hybrid models for ice sheet emulation.
    It integrates a deep ensemble to capture epistemic uncertainties and a normalizing flow to model aleatoric uncertainties.

    Attributes:
        device (str): The computing device ('cuda' if available, else 'cpu').
        deep_ensemble (DeepEnsemble): The deep ensemble model for epistemic uncertainty.
        normalizing_flow (NormalizingFlow): The normalizing flow model for aleatoric uncertainty.
        trained (bool): Flag indicating whether the model has been trained.
        scaler_path (str or None): Path to the scaler used for output transformation.
    """


    def __init__(self, deep_ensemble, normalizing_flow):
        """
        Initializes the ISEFlow model with a deep ensemble and a normalizing flow.

        Args:
            deep_ensemble (DeepEnsemble): A deep ensemble model for epistemic uncertainty estimation.
            normalizing_flow (NormalizingFlow): A normalizing flow model for aleatoric uncertainty estimation.

        Raises:
            ValueError: If `deep_ensemble` is not an instance of DeepEnsemble.
            ValueError: If `normalizing_flow` is not an instance of NormalizingFlow.
        """

        super(ISEFlow, self).__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

        if not isinstance(deep_ensemble, DeepEnsemble):
            raise ValueError("deep_ensemble must be a DeepEnsemble instance")
        if not isinstance(normalizing_flow, NormalizingFlow):
            raise ValueError("normalizing_flow must be a NormalizingFlow instance")

        self.deep_ensemble = deep_ensemble.to(self.device)
        self.normalizing_flow = normalizing_flow.to(self.device)
        self.trained = self.deep_ensemble.trained and self.normalizing_flow.trained
        self.scaler_path = None

    def fit(self, X, y, nf_epochs, de_epochs, batch_size=64, X_val=None, y_val=None, save_checkpoints=True, checkpoint_path='checkpoint_ensemble', early_stopping=True,  
             sequence_length=5, patience=10, verbose=True):
        """
        Trains the hybrid emulator using the provided data.

        This method trains the normalizing flow model first, then uses its latent representations 
        to train the deep ensemble model.

        Args:
            X (array-like): Input feature matrix.
            y (array-like): Target values.
            nf_epochs (int): Number of training epochs for the normalizing flow.
            de_epochs (int): Number of training epochs for the deep ensemble.
            batch_size (int, optional): Batch size for training. Defaults to 64.
            X_val (array-like, optional): Validation feature matrix. Defaults to None.
            y_val (array-like, optional): Validation target values. Defaults to None.
            save_checkpoints (bool, optional): Whether to save training checkpoints. Defaults to True.
            checkpoint_path (str, optional): Path prefix for saving model checkpoints. Defaults to 'checkpoint_ensemble'.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to True.
            sequence_length (int, optional): Sequence length for recurrent architectures. Defaults to 5.
            patience (int, optional): Number of epochs with no improvement before stopping. Defaults to 10.
            verbose (bool, optional): Whether to print training progress. Defaults to True.

        Raises:
            Warning: If the model has already been trained.
        """

        
        if early_stopping is None:
            early_stopping = X_val is not None and y_val is not None

        torch.manual_seed(np.random.randint(0, 100000))

        X, y = to_tensor(X).to(self.device), to_tensor(y).to(self.device)

        if self.trained:
            warnings.warn("This model has already been trained. Training again.")

        # Train Normalizing Flow
        if not self.normalizing_flow.trained:
            print(f"\nTraining Normalizing Flow ({'Maximum ' if early_stopping else ''}{nf_epochs} epochs):")
            self.normalizing_flow.fit(X, y, nf_epochs, batch_size, save_checkpoints, f"{checkpoint_path}_nf.pth", early_stopping, patience, verbose)      

        # Latent representation
        z = self.normalizing_flow.get_latent(X).detach()
        X_latent = torch.cat((X, z), axis=1)
        
        X_val_latent = None
        if X_val is not None and y_val is not None:
            X_val, y_val = to_tensor(X_val).to(self.device), to_tensor(y_val).to(self.device)
            z_val = self.normalizing_flow.get_latent(X_val).detach()
            X_val_latent = torch.cat((X_val, z_val), axis=1)
        
        # Train Deep Ensemble
        if not self.deep_ensemble.trained:
            print(f"\nTraining Deep Ensemble ({'Maximum ' if early_stopping else ''}{de_epochs} epochs):")
            
            self.deep_ensemble.fit(X_latent, y, X_val_latent, y_val, save_checkpoints, f"{checkpoint_path}_de", early_stopping, de_epochs, batch_size, sequence_length, patience, verbose,)

        self.trained = True

    def forward(self, x, smooth_projection=False):
        """
        Performs a forward pass through the hybrid emulator.

        Args:
            x (array-like): Input data.
            smooth_projection (bool, optional): Whether to apply smoothing to projections. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - prediction (numpy.ndarray): Model predictions.
                - uncertainties (dict): Dictionary with keys:
                    - 'total' (numpy.ndarray): Total uncertainty.
                    - 'epistemic' (numpy.ndarray): Epistemic uncertainty.
                    - 'aleatoric' (numpy.ndarray): Aleatoric uncertainty.

        Raises:
            Warning: If the model has not been trained.
        """

        self.eval()
        x = to_tensor(x).to(self.device)
        if not self.trained:
            warnings.warn("This model has not been trained. Predictions may not be accurate.")
        z = self.normalizing_flow.get_latent(x).detach()
        X_latent = torch.cat((x, z), axis=1)
        prediction, epistemic = self.deep_ensemble(X_latent)
        aleatoric = self.normalizing_flow.aleatoric(x, 100)
        prediction = prediction.detach().cpu().numpy()
        epistemic = epistemic.detach().cpu().numpy()
        uncertainties = dict(
            total=aleatoric + epistemic,
            epistemic=epistemic,
            aleatoric=aleatoric,
        )
        return prediction, uncertainties

    def predict(self, x, output_scaler=True, smooth_projection=False):
        """
        Makes predictions using the trained hybrid emulator.

        Args:
            x (array-like): Input data.
            output_scaler (bool or str, optional): Path to the output scaler or whether to apply scaling. Defaults to True.
            smooth_projection (bool, optional): Whether to apply smoothing. Defaults to False.

        Returns:
            tuple: A tuple containing:
                - unscaled_predictions (numpy.ndarray): Model predictions in the original scale.
                - uncertainties (dict): Dictionary with keys:
                    - 'total' (numpy.ndarray): Total uncertainty.
                    - 'epistemic' (numpy.ndarray): Epistemic uncertainty.
                    - 'aleatoric' (numpy.ndarray): Aleatoric uncertainty.

        Raises:
            Warning: If no scaler path is provided.
        """

        self.eval()
        if output_scaler is True:
            output_scaler = os.path.join(self.model_dir, "scaler_y.pkl")
            with open(output_scaler, "rb") as f:
                output_scaler = pickle.load(f)
        elif output_scaler is False and self.scaler_path is None:
            warnings.warn("No scaler path provided, uncertainties are not in units of SLE.")
            return self.forward(x, smooth_projection=smooth_projection)
        elif isinstance(output_scaler, str):
            self.scaler_path = output_scaler
            with open(self.scaler_path, "rb") as f:
                output_scaler = pickle.load(f)

        predictions, uncertainties = self.forward(x, smooth_projection=smooth_projection)
        unscaled_predictions = output_scaler.inverse_transform(predictions.reshape(-1, 1))
        bound_epistemic = predictions + uncertainties["epistemic"]
        bound_aleatoric = predictions + uncertainties["aleatoric"]
        unscaled_bound_epistemic = output_scaler.inverse_transform(bound_epistemic.reshape(-1, 1))
        unscaled_bound_aleatoric = output_scaler.inverse_transform(bound_aleatoric.reshape(-1, 1))
        epistemic = unscaled_bound_epistemic - unscaled_predictions
        aleatoric = unscaled_bound_aleatoric - unscaled_predictions

        uncertainties = dict(
            total=epistemic + aleatoric,
            epistemic=epistemic,
            aleatoric=aleatoric,
        )
        return unscaled_predictions, uncertainties
    

    def save(self, save_dir, input_features=None, output_scaler_path=None):
        """
        Saves the trained model and related components to a specified directory.

        Args:
            save_dir (str): Directory where the model should be saved.
            input_features (list, optional): List of input feature names. Defaults to None.
            output_scaler_path (str, optional): Path to the output scaler. Defaults to None.

        Raises:
            ValueError: If the model has not been trained.
            ValueError: If `save_dir` is a file instead of a directory.
            ValueError: If `input_features` is not a list.
        """

        if not self.trained:
            raise ValueError("This model has not been trained yet. Train the model before saving.")
        if save_dir.endswith(".pth"):
            raise ValueError("save_dir must be a directory, not a file")
        os.makedirs(save_dir, exist_ok=True)

        self.deep_ensemble.save(os.path.join(save_dir, "deep_ensemble.pth"))
        self.normalizing_flow.save(os.path.join(save_dir, "normalizing_flow.pth"))
        
        if input_features is not None:
            if not isinstance(input_features, list):
                raise ValueError("input_features must be a list of feature names")
            with open(os.path.join(save_dir, "input_features.json"), "w") as f:
                json.dump(input_features, f, indent=4)
        
        if output_scaler_path is not None and output_scaler_path.endswith(".pkl"):
            self.scaler_path = output_scaler_path
            
        if self.scaler_path is not None:
            shutil.copy(self.scaler_path, os.path.join(save_dir, "scaler_y.pkl"))

    @staticmethod
    def load(model_dir=None, deep_ensemble_path=None, normalizing_flow_path=None,):
        """
        Loads a trained ISEFlow model from specified paths.

        Args:
            model_dir (str, optional): Directory containing the saved model. Defaults to None.
            deep_ensemble_path (str, optional): Path to the saved deep ensemble model. Defaults to None.
            normalizing_flow_path (str, optional): Path to the saved normalizing flow model. Defaults to None.

        Returns:
            ISEFlow: The loaded ISEFlow model.

        Raises:
            NotImplementedError: If an unsupported version is specified.
        """

            
        if model_dir:
            deep_ensemble_path = os.path.join(model_dir, "deep_ensemble.pth")
            normalizing_flow_path = os.path.join(model_dir, "normalizing_flow.pth")
        
        deep_ensemble = DeepEnsemble.load(deep_ensemble_path)
        normalizing_flow = NormalizingFlow.load(normalizing_flow_path)
        model = ISEFlow(deep_ensemble, normalizing_flow)
        model.trained = True
        model.model_dir = model_dir
        
        return model



class ISEFlow_AIS(ISEFlow):
    """
    ISEFlow_AIS is a specialized version of ISEFlow for the Antarctic Ice Sheet (AIS).

    This subclass initializes the deep ensemble and normalizing flow models specifically
    for AIS and provides a loading method with pre-trained model paths.
    """

    def __init__(self,):
        """
        Initializes the ISEFlow_AIS model.

        Sets the ice sheet type to 'AIS' and initializes pre-configured deep ensemble 
        and normalizing flow models specific to AIS.
        """

        self.ice_sheet = "AIS"
        deep_ensemble = ISEFlow_AIS_DE()
        normalizing_flow = ISEFlow_AIS_NF()
        super(ISEFlow_AIS, self).__init__(deep_ensemble, normalizing_flow)
    
    @staticmethod
    def load(version="v1.0.0", model_dir=None, deep_ensemble_path=None, normalizing_flow_path=None):
        """
        Loads a trained ISEFlow_AIS model.

        Args:
            version (str, optional): Model version. Defaults to "v1.0.0".
            model_dir (str, optional): Directory of the saved model. Defaults to None.
            deep_ensemble_path (str, optional): Path to deep ensemble model. Defaults to None.
            normalizing_flow_path (str, optional): Path to normalizing flow model. Defaults to None.

        Returns:
            ISEFlow_AIS: The loaded model.

        Raises:
            NotImplementedError: If an unsupported version is specified.
        """
        
        # TODO: Add support for deep ensemble and normalizing flow paths

        if model_dir is None:
            if version == "v1.0.0":
                model_dir = ISEFlow_AIS_v1_0_0_path
            else:
                raise NotImplementedError("Only version v1.0.0 is supported")

        # Load components using the parent class logic
        deep_ensemble = DeepEnsemble.load(os.path.join(model_dir, "deep_ensemble.pth"))
        normalizing_flow = NormalizingFlow.load(os.path.join(model_dir, "normalizing_flow.pth"))

        # Return an instance of ISEFlow_AIS instead of ISEFlow
        model = ISEFlow_AIS()
        model.deep_ensemble = deep_ensemble
        model.normalizing_flow = normalizing_flow
        model.trained = True
        model.model_dir = model_dir

        return model
    
    def process(
        self, 
        year: np.array,
        pr_anomaly: np.array, 
        evspsbl_anomaly: np.array,
        mrro_anomaly: np.array,
        smb_anomaly: np.array,
        ts_anomaly: np.array,
        ocean_thermal_forcing: np.array,
        ocean_salinity: np.array,
        ocean_temperature: np.array,
        initial_year: int,
        numerics: str,
        stress_balance: str,
        resolution: int,
        init_method: str,
        melt_in_floating_cells: str,
        icefront_migration: str,
        ocean_forcing_type: str,
        ocean_sensitivity: str,
        ice_shelf_fracture: bool,
        open_melt_type: str=None,
        standard_melt_type: str=None,
        
    ):
        """
        Processes input data for prediction by applying necessary transformations and encoding.

        Args:
            year (np.array): Years of the input data.
            pr_anomaly (np.array): Precipitation anomaly data.
            evspsbl_anomaly (np.array): Evaporation anomaly data.
            mrro_anomaly (np.array): Runoff anomaly data.
            smb_anomaly (np.array): Surface mass balance anomaly.
            ts_anomaly (np.array): Surface temperature anomaly.
            ocean_thermal_forcing (np.array): Ocean thermal forcing.
            ocean_salinity (np.array): Ocean salinity.
            ocean_temperature (np.array): Ocean temperature.
            initial_year (int): Initial year for modeling.
            numerics (str): Numerical scheme used.
            stress_balance (str): Stress balance model.
            resolution (int): Resolution of the model.
            init_method (str): Initialization method.
            melt_in_floating_cells (str): Melt treatment method.
            icefront_migration (str): Ice front migration scheme.
            ocean_forcing_type (str): Type of ocean forcing applied.
            ocean_sensitivity (str): Ocean sensitivity setting.
            ice_shelf_fracture (bool): Whether ice shelf fracture is considered.
            open_melt_type (str, optional): Type of open melt model. Defaults to None.
            standard_melt_type (str, optional): Type of standard melt model. Defaults to None.

        Returns:
            pd.DataFrame: Processed input data ready for prediction.

        Raises:
            ValueError: If any input arguments are invalid.
        """

        
        if year[0] == 2015:
            year = year - 2015
            
        data = {    
            "year": year,
            "pr_anomaly": pr_anomaly,
            "evspsbl_anomaly": evspsbl_anomaly,
            "mrro_anomaly": mrro_anomaly,
            "smb_anomaly": smb_anomaly,
            "ts_anomaly": ts_anomaly,
            "thermal_forcing": ocean_thermal_forcing,
            "salinity": ocean_salinity,
            "temperature": ocean_temperature,
            "initial_year": initial_year,
            "numerics": numerics,
            "stress_balance": stress_balance,
            "resolution": resolution,
            "init_method": init_method,
            "melt": melt_in_floating_cells,
            "ice_front": icefront_migration,
            "Ocean sensitivity": ocean_sensitivity,
            "Ice shelf fracture": ice_shelf_fracture,
            "Ocean forcing": ocean_forcing_type,
            "open_melt_param": open_melt_type,
            "standard_melt_param": standard_melt_type,
        }

        
        # map from accepted input to how the model expects variable names
        arg_map = {
            'numerics': {
                'fe': 'FE',
                'fd': 'FD',
                'fe/fv': 'FE/FV',
            },
            'stress_balance': {
                'ho': 'HO',
                'hybrid': 'Hybrid',
                'l1l2': 'L1L2',
                'sia+ssa': 'SIA_SSA',
                'ssa': 'SSA',
                'stokes': 'Stokes',
            },
            "init_method": {
                'da': 'DA',
                'da*': 'DA_geom',
                'da+': 'DA_relax',
                'eq': 'Eq',
                'sp': 'SP',
                'sp+': 'SP_icethickness',
            },
            "melt": {
                'floating condition': 'Floating condition',
                'sub-grid': 'Sub-grid',
            },
            'ice_front': {
                'str': 'StR',
                'fix': 'Fix',
                'mh': 'MH',
                'ro': 'RO',
                'div': 'Div',
            },
            'Ocean forcing': {
                'standard': 'Standard',
                'open': 'Open',
            },
            'Ocean sensitivity': {
                'low': 'Low',
                'medium': 'Medium',
                'high': 'High',
                'pigl': 'PIGL',
            },
            "open_melt_param": {
                'lin': 'Lin',
                'quad': 'Quad',
                'nonlocal+slope': 'Nonlocal_Slope',
                'pico': 'PICO',
                'picop': 'PICOP',
                'plume': 'Plume',
            },
            "standard_melt_param": {
                'local': 'Local',
                'nonlocal': 'Nonlocal',
                'local anom': 'Local anom',
                'nonlocal anom': 'Nonlocal anom',
            }
        }
        
        mrro_means = np.array([3.61493220e-08, 2.77753815e-08, 5.50841177e-08, 4.17617754e-08, 5.58558082e-08, 5.74870861e-08, 1.07017988e-07, 7.72183085e-08, 6.44275121e-08, 2.10466987e-08, 5.36071770e-08, 8.32501757e-08, 9.31873131e-08, 7.84747761e-08, 8.41751157e-08, 8.56960829e-08, 7.81743956e-08, 9.74934761e-08, 6.04155892e-08, 8.31572351e-08, 1.16800344e-07, 9.96168899e-08, 1.41262144e-07, 8.76467771e-08, 1.03335698e-07, 1.23414214e-07, 9.29483909e-08, 1.95530928e-07, 1.18321950e-07, 1.68664275e-07, 1.56460562e-07, 1.40309916e-07, 1.08267844e-07, 1.85627395e-07, 1.29400203e-07, 1.98725020e-07, 1.39994753e-07, 1.86775688e-07, 1.68388442e-07, 2.04534154e-07, 1.49715175e-07, 1.50418319e-07, 1.44444531e-07, 1.67211070e-07, 1.83698063e-07, 2.05489898e-07, 2.42246565e-07, 1.98110423e-07, 2.40505470e-07, 2.37863389e-07, 2.55668987e-07, 2.93048624e-07, 2.57849749e-07, 2.72915753e-07, 2.82135517e-07, 2.27647208e-07, 2.21859448e-07, 2.07266200e-07, 2.42241281e-07, 2.55693726e-07, 2.52039399e-07, 2.82802604e-07, 2.94193847e-07, 3.00380753e-07, 3.60152406e-07, 3.47886784e-07, 3.58344925e-07, 3.84398045e-07, 4.41053179e-07, 3.84072892e-07, 4.42520286e-07, 4.30170222e-07, 4.34444387e-07, 4.77483307e-07, 3.52802246e-07, 4.96503280e-07, 5.22078462e-07, 4.78644041e-07, 4.86755806e-07, 5.04600526e-07, 4.80814514e-07, 5.38276914e-07, 5.91539053e-07, 5.84794672e-07, 5.33792907e-07, 5.37435986e-07])
        

        # check inputs
        if not isinstance(initial_year, int):
            raise ValueError("initial_year must be an integer")

        if str(numerics).lower() not in ('fe', 'fd', 'fe/fv'):
            raise ValueError("numerics must be one of 'fe', 'fd', or 'fe/fv'")
        
        if str(stress_balance) not in ('ho', 'hybrid', "l1l2", 'sia+ssa', 'ssa', 'stokes'):
            raise ValueError("stress_balance must be one of 'ho', 'hybrid', 'l1l2', 'sia+ssa', 'ssa', or 'stokes'")
        
        if str(resolution) not in ('16', '20', '32', '4', '8', 'variable'):
            raise ValueError("resolution must be one of '16', '20', '32', '4', '8', or 'variable'")
        
        if str(init_method) not in ('da', 'da*', 'da+', 'eq', 'sp', 'sp+'):
            raise ValueError("init_method must be one of 'da', 'da*', 'da+', 'eq', 'sp', or 'sp+'")
        
        if str(melt_in_floating_cells) not in ('floating condition', 'sub-grid', 'None', 'False'):
            raise ValueError("melt_in_floating_cells must be one of 'floating condition', 'sub-grid', 'None', or 'False'")

        if str(icefront_migration) not in ('str', 'fix', 'mh', 'ro', 'div'):
            raise ValueError("icefront_migration must be one of 'str', 'fix', 'mh', 'ro', or 'div'")
        
        if str(ocean_forcing_type) not in ('standard', 'open'):
            raise ValueError("ocean_forcing_type must be one of 'standard' or 'open'")
        
        if str(ocean_forcing_type) == 'standard' and standard_melt_type is None:
            raise ValueError("standard_melt_type must be provided if ocean_forcing_type is 'standard'")
        elif str(ocean_forcing_type) == 'standard' and standard_melt_type not in ("local", "nonlocal", "local anom", "nonlocal anom", "None"):
            raise ValueError("standard_melt_type must be one of 'local', 'nonlocal', 'local anom', 'nonlocal anom', or None")
        
        if str(ocean_forcing_type) == 'open' and open_melt_type is None:
            raise ValueError("open_melt_type must be provided if ocean_forcing_type is 'open'")
        elif str(ocean_forcing_type) == 'open' and open_melt_type not in ("lin", "quad", "nonlocal+slope", "pico", "picop", "plume", "None"):
            raise ValueError("open_melt_type must be one of 'lin', 'quad', 'nonlocal+slope', 'pico', 'picop', 'plume', or None")
        
        if str(ocean_sensitivity) not in ('low', 'medium', 'high', 'pigl'):
            raise ValueError("ocean_sensitivity must be one of 'low', 'medium', 'high', or 'pigl'")

        if not isinstance(ice_shelf_fracture, bool):
            raise ValueError("ice_shelf_fracture must be a boolean")
        
        
        for key, value in data.items():
            # make sure forcings are numpy arrays           
            if key in ("year", "pr_anomaly", "evspsbl_anomaly", "mrro_anomaly", "smb_anomaly", "ts_anomaly", "thermal_forcing", "salinity", "temperature"):
                try:
                    data[key] = np.array(value)
                except Exception as e:
                    raise ValueError(f"Variable {key} must be a numpy array.") from e
            # remap args
            elif key in arg_map:
                if value in arg_map[key]:
                    data[key] = arg_map[key][value]
                else:
                    raise ValueError(f"Invalid value for {key}: {value}. Must be one of {list(arg_map[key].keys())}")


            
        data = pd.DataFrame(data)
        year_mean_map = {year: mean for year, mean in enumerate(mrro_means)}
        data["mrro_anomaly"] = data.apply(
            lambda row: year_mean_map[row["year"]] if pd.isna(row["mrro_anomaly"]) else row["mrro_anomaly"],
            axis=1
        )
        data = fe.add_lag_variables(data, lag=5, verbose=False)
        data = pd.get_dummies(data, columns=['numerics', 'stress_balance', 'resolution', 'init_method', 'melt', 'ice_front', 'Ocean forcing', 'Ocean sensitivity', 'open_melt_param', 'standard_melt_param'])
        # need to add other columns as zeros from get_dummies (all true)
        columns = ['year', 'sector', 'pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly',
       'smb_anomaly', 'ts_anomaly', 'thermal_forcing', 'salinity',
       'temperature', 'pr_anomaly.lag1', 'evspsbl_anomaly.lag1',
       'mrro_anomaly.lag1', 'smb_anomaly.lag1', 'ts_anomaly.lag1',
       'thermal_forcing.lag1', 'salinity.lag1', 'temperature.lag1',
       'pr_anomaly.lag2', 'evspsbl_anomaly.lag2', 'mrro_anomaly.lag2',
       'smb_anomaly.lag2', 'ts_anomaly.lag2', 'thermal_forcing.lag2',
       'salinity.lag2', 'temperature.lag2', 'pr_anomaly.lag3',
       'evspsbl_anomaly.lag3', 'mrro_anomaly.lag3', 'smb_anomaly.lag3',
       'ts_anomaly.lag3', 'thermal_forcing.lag3', 'salinity.lag3',
       'temperature.lag3', 'pr_anomaly.lag4', 'evspsbl_anomaly.lag4',
       'mrro_anomaly.lag4', 'smb_anomaly.lag4', 'ts_anomaly.lag4',
       'thermal_forcing.lag4', 'salinity.lag4', 'temperature.lag4',
       'pr_anomaly.lag5', 'evspsbl_anomaly.lag5', 'mrro_anomaly.lag5',
       'smb_anomaly.lag5', 'ts_anomaly.lag5', 'thermal_forcing.lag5',
       'salinity.lag5', 'temperature.lag5', 'initial_year', 'numerics_FD',
       'numerics_FE', 'numerics_FE/FV', 'stress_balance_HO',
       'stress_balance_Hybrid', 'stress_balance_L1L2',
       'stress_balance_SIA_SSA', 'stress_balance_SSA', 'stress_balance_Stokes',
       'resolution_16', 'resolution_20', 'resolution_32', 'resolution_4',
       'resolution_8', 'resolution_variable', 'init_method_DA',
       'init_method_DA_geom', 'init_method_DA_relax', 'init_method_Eq',
       'init_method_SP', 'init_method_SP_icethickness',
       'melt_Floating_condition', 'melt_No', 'melt_Sub-grid', 'ice_front_Div',
       'ice_front_Fix', 'ice_front_MH', 'ice_front_RO', 'ice_front_StR',
       'open_melt_param_Lin', 'open_melt_param_Nonlocal_Slope',
       'open_melt_param_PICO', 'open_melt_param_PICOP',
       'open_melt_param_Plume', 'open_melt_param_Quad',
       'standard_melt_param_Local', 'standard_melt_param_Local_anom',
       'standard_melt_param_Nonlocal', 'standard_melt_param_Nonlocal_anom',
       'Ocean forcing_Open', 'Ocean forcing_Standard',
       'Ocean sensitivity_High', 'Ocean sensitivity_Low',
       'Ocean sensitivity_Medium', 'Ocean sensitivity_PIGL',
       'Ice shelf fracture_False', 'Ice shelf fracture_True']
        
        for col in columns:
            if col not in data.columns:
                data[col] = 0
        
        data = data[columns]
        data['outlier'] = False
        data = fe.scale_data(data, scaler_path=f"{ISEFlow_AIS_v1_0_0_path}/scaler_X.pkl")
        
        return data
    
    def predict(        
        self, 
        year: np.array,
        pr_anomaly: np.array, 
        evspsbl_anomaly: np.array,
        mrro_anomaly: np.array,
        smb_anomaly: np.array,
        ts_anomaly: np.array,
        ocean_thermal_forcing: np.array,
        ocean_salinity: np.array,
        ocean_temperature: np.array,
        initial_year: int,
        numerics: str,
        stress_balance: str,
        resolution: int,
        init_method: str,
        melt_in_floating_cells: str,
        icefront_migration: str,
        ocean_forcing_type: str,
        ocean_sensitivity: str,
        ice_shelf_fracture: bool,
        open_melt_type: str=None,
        standard_melt_type: str=None,   
    ):
        """
        Predicts ice sheet evolution using the trained ISEFlow_AIS model.

        Args:
            (Same as process method)

        Returns:
            tuple: A tuple containing:
                - unscaled_predictions (numpy.ndarray): Model predictions in the original scale.
                - uncertainties (dict): Dictionary containing different uncertainty components.
        """

        
        data = self.process(
            year, pr_anomaly, evspsbl_anomaly, mrro_anomaly, smb_anomaly, ts_anomaly, ocean_thermal_forcing, ocean_salinity, ocean_temperature, initial_year, numerics, stress_balance, resolution, init_method, melt_in_floating_cells, icefront_migration, ocean_forcing_type, ocean_sensitivity, ice_shelf_fracture, open_melt_type, standard_melt_type
        )
        X = data.values
        X = to_tensor(X).to(self.device)
        return super().predict(X, output_scaler=f"{ISEFlow_AIS_v1_0_0_path}/scaler_y.pkl")
    
    def test(self, X_test,):
        """
        Tests the model on a test dataset.

        Args:
            X_test (array-like): Test feature matrix.
            y_test (array-like): Test target values.

        Returns:
            tuple: A tuple containing:
                - unscaled_predictions (numpy.ndarray): Model predictions in the original scale.
                - uncertainties (dict): Dictionary with keys:
                    - 'total' (numpy.ndarray): Total uncertainty.
                    - 'epistemic' (numpy.ndarray): Epistemic uncertainty.
                    - 'aleatoric' (numpy.ndarray): Aleatoric uncertainty.
        """
        
        return super().predict(X_test, output_scaler=f"{ISEFlow_AIS_v1_0_0_path}/scaler_y.pkl")

class ISEFlow_GrIS(ISEFlow):
    """
    ISEFlow_GrIS is a specialized version of ISEFlow for the Greenland Ice Sheet (GrIS).

    This subclass initializes the deep ensemble and normalizing flow models specifically
    for GrIS and provides a loading method with pre-trained model paths.
    """

    def __init__(self,):
        """
        Initializes the ISEFlow_GrIS model.

        Sets the ice sheet type to 'GrIS' and initializes pre-configured deep ensemble 
        and normalizing flow models specific to GrIS.
        """

        self.ice_sheet = "GrIS"
        deep_ensemble = ISEFlow_GrIS_DE()
        normalizing_flow = ISEFlow_GrIS_NF()
        super(ISEFlow_GrIS, self).__init__(deep_ensemble, normalizing_flow)
    
    @staticmethod
    def load(version="v1.0.0", model_dir=None, deep_ensemble_path=None, normalizing_flow_path=None,):
        """
        Loads a trained ISEFlow_GrIS model.

        (Same arguments and return type as `ISEFlow_AIS.load`.)
        """

        if model_dir is None:
            if version == "v1.0.0":
                model_dir = ISEFlow_GrIS_v1_0_0_path
            else:
                raise NotImplementedError("Only version v1.0.0 is supported")
        return super(ISEFlow_GrIS, ISEFlow_GrIS).load(model_dir, deep_ensemble_path, normalizing_flow_path)

    # TODO: ISEFlow GrIS process, predict