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
from ise.models.pretrained import ISEFlow_AIS_v1_0_0_path, ISEFlow_GrIS_v1_0_0_path, ISEFlow_AIS_v1_1_0_path, ISEFlow_GrIS_v1_1_0_path, ISEFlow_AIS_v1_1_0_variables, ISEFlow_AIS_v1_0_0_variables
from ise.models.predictors.lstm import LSTM
from ise.models.predictors.deep_ensemble import DeepEnsemble

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

    def forward(self, x,):
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

    def predict(self, x, output_scaler=True, smoothing_window=0):
        """
        Makes predictions using the trained hybrid emulator with optional smoothing.
        
        IMPORTANT: Smoothing is applied to the final unscaled predictions and uncertainties
        to ensure smooth output curves.

        Args:
            x (array-like): Input data.
            output_scaler (bool or str, optional): Path to the output scaler or whether to apply scaling. 
                Defaults to True.
            smoothing_window (int, optional): Size of the smoothing window. 0 means no smoothing. 
                Defaults to 0.

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
        
        # Handle scaler loading
        if output_scaler is True:
            output_scaler = os.path.join(self.model_dir, "scaler_y.pkl")
            with open(output_scaler, "rb") as f:
                output_scaler = pickle.load(f)
        elif output_scaler is False and self.scaler_path is None:
            warnings.warn("No scaler path provided, uncertainties are not in units of SLE.")
            predictions, uncertainties = self.forward(x)
            
            # Apply smoothing to raw predictions and uncertainties if requested
            if smoothing_window > 0:
                predictions = smooth_projections(predictions, smoothing_window)
                for key in uncertainties:
                    uncertainties[key] = smooth_projections(uncertainties[key], smoothing_window)
            
            return predictions, uncertainties
        elif isinstance(output_scaler, str):
            self.scaler_path = output_scaler
            with open(self.scaler_path, "rb") as f:
                output_scaler = pickle.load(f)
        
        # Get raw predictions and uncertainties (no smoothing yet)
        predictions, uncertainties = self.forward(x)
        
        # Inverse transform predictions first
        unscaled_predictions = output_scaler.inverse_transform(predictions.reshape(-1, 1))
        
        # Calculate uncertainty bounds in scaled space
        bound_epistemic_upper = predictions + uncertainties["epistemic"]
        bound_epistemic_lower = predictions - uncertainties["epistemic"]
        bound_aleatoric_upper = predictions + uncertainties["aleatoric"]
        bound_aleatoric_lower = predictions - uncertainties["aleatoric"]
        
        # Inverse transform all bounds
        unscaled_bound_epistemic_upper = output_scaler.inverse_transform(bound_epistemic_upper.reshape(-1, 1))
        unscaled_bound_epistemic_lower = output_scaler.inverse_transform(bound_epistemic_lower.reshape(-1, 1))
        unscaled_bound_aleatoric_upper = output_scaler.inverse_transform(bound_aleatoric_upper.reshape(-1, 1))
        unscaled_bound_aleatoric_lower = output_scaler.inverse_transform(bound_aleatoric_lower.reshape(-1, 1))
        
        # Calculate unscaled uncertainties (symmetric around predictions)
        epistemic_upper = unscaled_bound_epistemic_upper - unscaled_predictions
        epistemic_lower = unscaled_predictions - unscaled_bound_epistemic_lower
        aleatoric_upper = unscaled_bound_aleatoric_upper - unscaled_predictions
        aleatoric_lower = unscaled_predictions - unscaled_bound_aleatoric_lower
        
        # Use average of upper and lower uncertainties for symmetry
        epistemic = (epistemic_upper + epistemic_lower) / 2
        aleatoric = (aleatoric_upper + aleatoric_lower) / 2
        
        # NOW apply smoothing to the final unscaled values
        if smoothing_window > 0:
            unscaled_predictions = smooth_projections(unscaled_predictions, smoothing_window)
            epistemic = smooth_projections(epistemic, smoothing_window)
            aleatoric = smooth_projections(aleatoric, smoothing_window)
        
        # Ensure uncertainties are non-negative
        epistemic = np.abs(epistemic)
        aleatoric = np.abs(aleatoric)
        
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
            try:
                scaler_x_path = self.scaler_path.replace('scaler_y', 'scaler_X')
                shutil.copy(scaler_x_path, os.path.join(save_dir, "scaler_X.pkl"))
            except:
                pass

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
    def __init__(self, version="v1.1.0"):
        self.ice_sheet = "AIS"
        self.version = version

        if version == "v1.0.0":
            model_dir = ISEFlow_AIS_v1_0_0_path
        elif version == "v1.1.0":
            model_dir = ISEFlow_AIS_v1_1_0_path
        else:
            raise NotImplementedError(f"Version {version} not implemented. Try v1.0.0 or v1.1.0")

        # Load the actual submodules
        deep_ensemble = DeepEnsemble.load(os.path.join(model_dir, "deep_ensemble.pth"))
        normalizing_flow = NormalizingFlow.load(os.path.join(model_dir, "normalizing_flow.pth"))

        # ✅ This calls nn.Module.__init__ properly and registers submodules
        super(ISEFlow_AIS, self).__init__(deep_ensemble, normalizing_flow)

        self.model_dir = model_dir
        self.trained = True
    


    def process(
        self, 
        year: np.array,
        sector: np.array,
        pr_anomaly: np.array, 
        evspsbl_anomaly: np.array,
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
        mrro_anomaly: np.array=None,
        
        
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
        if mrro_anomaly is None and self.version == "v1.0.0":
            raise ValueError("mrro_anomaly is required for version v1.0.0")
        
        if year[0] == 2015:
            year = year - 2015
        
        if isinstance(sector, int):
            sector = np.ones_like(year) * sector
            
        data = {    
            "year": year,
            "sector": sector,
            "pr_anomaly": pr_anomaly,
            "evspsbl_anomaly": evspsbl_anomaly,
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
        
        if self.version == "v1.0.0":
            data['mrro_anomaly'] = mrro_anomaly

        
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
            cols = ("year", "pr_anomaly", "evspsbl_anomaly", "smb_anomaly", "ts_anomaly", "thermal_forcing", "salinity", "temperature")
            cols += ("mrro_anomaly",) if self.version == "v1.0.0" else ()
            if key in cols:
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
    
        if self.version == "v1.0.0":
            year_mean_map = {year: mean for year, mean in enumerate(mrro_means)}
            data["mrro_anomaly"] = data.apply(
                lambda row: year_mean_map[row["year"]] if pd.isna(row["mrro_anomaly"]) else row["mrro_anomaly"],
                axis=1
            )
        data = fe.scale_data(data, scaler_path=f"{self.model_dir}/scaler_X.pkl")
        data = fe.add_lag_variables(data, lag=5, verbose=False)
        data = pd.get_dummies(data, columns=['numerics', 'stress_balance', 'resolution', 'init_method', 'melt', 'ice_front', 'Ocean forcing', 'Ocean sensitivity', 'open_melt_param', 'standard_melt_param', 'Ice shelf fracture'], dtype=bool)
        
        # need to add other columns as zeros from get_dummies (all true)
        if self.version == "v1.1.0":
            columns = ISEFlow_AIS_v1_1_0_variables
        elif self.version == "v1.0.0":
            columns = ISEFlow_AIS_v1_0_0_variables
        else:
            raise NotImplementedError(f"Version {self.version} not implemented. Use v1.0.0 or v1.1.0")
        
        
        for col in columns:
            if col not in data.columns:
                data[col] = False
            
        data = data[columns]
        data = data.loc[:, ~data.columns.duplicated()]
        
        
        return data
    
    def predict(        
        self, 
        year: np.array,
        sector: np.array,
        pr_anomaly: np.array, 
        evspsbl_anomaly: np.array,
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
        mrro_anomaly: np.array=None,
        smoothing_window: int=0
        
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
            year=year, sector=sector, pr_anomaly=pr_anomaly,
            evspsbl_anomaly=evspsbl_anomaly, smb_anomaly=smb_anomaly,
            ts_anomaly=ts_anomaly, ocean_thermal_forcing=ocean_thermal_forcing, 
            ocean_salinity=ocean_salinity, ocean_temperature=ocean_temperature, 
            initial_year=initial_year, numerics=numerics, stress_balance=stress_balance, 
            resolution=resolution, init_method=init_method, 
            melt_in_floating_cells=melt_in_floating_cells, icefront_migration=icefront_migration, 
            ocean_forcing_type=ocean_forcing_type, ocean_sensitivity=ocean_sensitivity, 
            ice_shelf_fracture=ice_shelf_fracture, open_melt_type=open_melt_type, 
            standard_melt_type=standard_melt_type, mrro_anomaly=mrro_anomaly
        )
        X = data.to_numpy(dtype=float)
        X = to_tensor(X).to(self.device)
        return super().predict(X, output_scaler=f"{self.model_dir}/scaler_y.pkl", smoothing_window=smoothing_window)
    
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
        
        warnings.warn("ISEFlow_GrIS is deprecated and will be removed in future versions. Please use ISEFlow with custom models instead.", DeprecationWarning)

        self.ice_sheet = "GrIS"
        deep_ensemble = ISEFlow_GrIS_DE_v1_0_0()
        normalizing_flow = ISEFlow_GrIS_NF_v1_0_0()
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
    
    
    
    
class ISEFlow_AIS_DE_v1_0_0(DeepEnsemble):
    """
    ISEFlow Deep ensemble model for Antarctic Ice Sheet (AIS) emulation.

    This class implements an ensemble of Long Short-Term Memory (LSTM) networks 
    to predict ice sheet dynamics using deep learning. It extends the `DeepEnsemble` 
    class and combines multiple LSTM models to enhance predictive performance.

    Attributes:
        input_size (int): The number of input features, Defaults to 99.
        output_size (int): The number of output features, Defaults to 1.
        iseflow_ais_ensemble (list): A list of LSTM models with different architectures and loss functions.
    
    Inherits from:
        DeepEnsemble: A base class for deep ensemble models.

    """

    def __init__(self, ):
        warnings.warn("ISEFlow_AIS_DE_v1_0_0 is deprecated and will be removed in future versions. Please use ISEFlow_AIS instead.", DeprecationWarning)
        
        self.input_size = 99
        self.output_size = 1
        iseflow_ais_ensemble = [
            LSTM(1, 128, 99, 1, nn.HuberLoss()),
            LSTM(1, 512, 99, 1, nn.HuberLoss()),
            LSTM(1, 512, 99, 1, nn.HuberLoss()),
            LSTM(2, 128, 99, 1, nn.HuberLoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
            LSTM(1, 512, 99, 1, nn.MSELoss()),
            LSTM(2, 128, 99, 1, nn.MSELoss()),
            LSTM(2, 512, 99, 1, nn.MSELoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
            LSTM(1, 64, 99, 1, nn.HuberLoss()),
        ]
        super().__init__(ensemble_members=iseflow_ais_ensemble, input_size=self.input_size, output_size=self.output_size, output_sequence_length=86,)
    
    

class ISEFlow_GrIS_DE_v1_0_0(DeepEnsemble):
    """
    ISEFlow Deep ensemble model for Greenland Ice Sheet (GrIS) emulation.

    This class constructs an ensemble of LSTM models to predict ice sheet behavior 
    for the Greenland Ice Sheet (GrIS). It extends the `DeepEnsemble` framework 
    and integrates multiple LSTM-based predictors to improve accuracy.

    Attributes:
        input_size (int): The number of input features (90).
        output_size (int): The number of output features (1).
        iseflow_gris_ensemble (list): A list of LSTM models with varying architectures and loss functions.

    Inherits from:
        DeepEnsemble: A base class for deep ensemble models.
    """

    def __init__(self,):
        self.input_size = 90
        self.output_size = 1
        iseflow_gris_ensemble = [
            LSTM(2, 128, 99, 1, nn.HuberLoss()),
            LSTM(2, 256, 99, 1, nn.MSELoss()),
            LSTM(2, 128, 99, 1, nn.HuberLoss()),
            LSTM(2, 128, 99, 1, nn.MSELoss()),
            LSTM(2, 256, 99, 1, nn.HuberLoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
            LSTM(1, 128, 99, 1, nn.HuberLoss()),
            LSTM(2, 64, 99, 1, nn.MSELoss()),
            LSTM(2, 256, 99, 1, nn.HuberLoss()),
            LSTM(1, 256, 99, 1, nn.L1Loss()),
        ]
        super().__init__(ensemble_members=iseflow_gris_ensemble, input_size=self.input_size, output_size=self.output_size, output_sequence_length=86,)
        
        
from ise.models.density_estimators.normalizing_flow import NormalizingFlow

class ISEFlow_AIS_NF_v1_0_0(NormalizingFlow):
    """
    ISEFlow_AIS_NF is a specialized normalizing flow model for the Antarctic Ice Sheet (AIS).

    This class extends the `NormalizingFlow` class, configuring it with AIS-specific input sizes and transformations.
    """

    def __init__(self, version="1.0.0"):
        """
        Initializes the ISEFlow_AIS_NF model.

        This model is pre-configured with:
        - `input_size` of 99, representing the number of input features.
        - `output_size` of 1, representing the target variable.
        - `num_flow_transforms` of 5, specifying the number of flow transformations.

        Calls the `NormalizingFlow` constructor with these preset parameters.
        """
        warnings.warn("ISEFlow_AIS_NF_v1_0_0 is deprecated and will be removed in future versions. Please use ISEFlow_AIS instead.", DeprecationWarning)

        self.input_size = 99 if version == "v1.0.0" else 93
        self.output_size = 1
        self.num_flow_transforms = 5
        super().__init__(input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)
        
class ISEFlow_GrIS_NF_v1_0_0(NormalizingFlow):
    """
    ISEFlow_GrIS_NF is a specialized normalizing flow model for the Greenland Ice Sheet (GrIS).

    This class extends the `NormalizingFlow` class, configuring it with GrIS-specific input sizes and transformations.
    """

    def __init__(self,):
        """
        Initializes the ISEFlow_GrIS_NF model.

        This model is pre-configured with:
        - `input_size` of 90, representing the number of input features.
        - `output_size` of 1, representing the target variable.
        - `num_flow_transforms` of 5, specifying the number of flow transformations.

        Calls the `NormalizingFlow` constructor with these preset parameters.
        """
        warnings.warn("ISEFlow_GrIS_NF_v1_0_0 is deprecated and will be removed in future versions. Please use ISEFlow with custom models instead.", DeprecationWarning)
        self.input_size = 90
        self.output_size = 1
        self.num_flow_transforms = 5
        super().__init__(input_size=self.input_size, output_size=self.output_size, num_flow_transforms=self.num_flow_transforms)
        
        
        
from scipy.ndimage import uniform_filter1d

def smooth_projections(data, window_size, projection_length=86):
    """
    Apply smoothing to projections while respecting projection boundaries.
    Uses scipy's uniform_filter1d for more effective smoothing.
    
    Args:
        data (np.ndarray): Array of shape (n_samples,) or (n_samples, 1) containing values to smooth
        window_size (int): Size of the smoothing window
        projection_length (int): Length of each projection segment (default: 86 years)
    
    Returns:
        np.ndarray: Smoothed data with same shape as input
    """
    if window_size <= 0 or window_size == 1:
        return data
    
    # Ensure window_size is odd for symmetric smoothing
    if window_size % 2 == 0:
        window_size += 1
    
    # Handle both 1D and 2D arrays
    original_shape = data.shape
    data_was_1d = False
    if data.ndim == 1:
        data = data.reshape(-1, 1)
        data_was_1d = True
    
    smoothed_data = np.zeros_like(data)
    n_samples = data.shape[0]
    
    # Calculate number of complete projections
    n_complete_projections = n_samples // projection_length
    
    # Process each projection separately to avoid boundary mixing
    for proj_idx in range(n_complete_projections):
        start_idx = proj_idx * projection_length
        end_idx = (proj_idx + 1) * projection_length
        
        # Extract current projection segment
        projection_segment = data[start_idx:end_idx, :]
        
        # Apply uniform filter for each column
        for col in range(data.shape[1]):
            # Use mode='nearest' at boundaries to avoid edge effects
            smoothed_segment = uniform_filter1d(
                projection_segment[:, col], 
                size=window_size, 
                mode='nearest'
            )
            smoothed_data[start_idx:end_idx, col] = smoothed_segment
    
    # Handle any remaining samples that don't form a complete projection
    if n_samples % projection_length != 0:
        start_idx = n_complete_projections * projection_length
        remaining_segment = data[start_idx:, :]
        
        for col in range(data.shape[1]):
            smoothed_segment = uniform_filter1d(
                remaining_segment[:, col],
                size=min(window_size, len(remaining_segment)),
                mode='nearest'
            )
            smoothed_data[start_idx:, col] = smoothed_segment
    
    # Restore original shape
    if data_was_1d:
        smoothed_data = smoothed_data.flatten()
    
    return smoothed_data