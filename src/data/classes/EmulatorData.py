from multiprocessing.sharedctypes import Value
import pandas as pd
from sklearn import preprocessing as sp
from sklearn.model_selection import train_test_split
import numpy as np

class EmulatorData:
    def __init__(self, directory):
        
        self.directory = directory
        
        
        try:
            self.data = pd.read_csv(f"{self.directory}/master.csv", low_memory=False)
        except FileNotFoundError:
            try:
                self.inputs = pd.read_csv(f"{self.directory}/inputs.csv")
                self.outputs = pd.read_csv(f"{self.directory}/outputs.csv")
            except FileNotFoundError:
                raise FileNotFoundError('Files not found, make sure to run all processing functions.')
        
        self.output_columns = ['icearea','iareafl','iareagr','ivol','ivaf','smb','smbgr', 'bmbfl']
        self.X = None
        self.y = None
        self.scaler_X = None
        self.scaler_y = None
        
    def process(self, target_column, drop_missing=True, drop_columns=True, boolean_indices=True, scale=True, ):
        if drop_missing:
            self = self.drop_missing()
        if drop_columns:
            if drop_columns is True:
                self = self.drop_columns(columns=['experiment', 'exp_id', 'groupname', 'regions'])
            elif isinstance(drop_columns, list):
                self = self.drop_columns(columns=drop_columns)
            else:
                raise ValueError(f'drop_columns argument must be of type boolean|list, received {type(drop_columns)}')
        
        if boolean_indices:
            self = self.create_boolean_indices()
        
        self = self.split_data(target_column=target_column)
        
        if scale:
            self.X = self.scale(self.X, 'inputs', scaler='MinMaxScaler')
            self.y = self.scale(self.y, 'outputs', scaler='MinMaxScaler')
        
        self = self.train_test_split()
        
        return self, self.train_features, self.test_features, self.train_labels, self.test_labels
            
    
    def split_data(self, target_column: str, ):
        self.target_column = target_column
        self.X = self.data.drop(columns=self.output_columns)
        self.y = self.data[target_column]
        self.input_columns = self.X.columns
        return self
    
    def train_test_split(self, train_size=0.7, shuffle=True):
        # if isinstance(self.X, pd.DataFrame):
        #     train_dataset = dataset.sample(frac=0.8, random_state=0)
        #     test_dataset = dataset.drop(train_dataset.index)
        # self.train_features, self.test_features, self.train_labels, self.test_labels = train_test_split(self.X, self.y, train_size=train_size, random_state=0, shuffle=False)
        
        if not isinstance(self.X, pd.DataFrame):
            self.X = pd.DataFrame(self.X, columns=self.input_columns)
    
        self.train_features = self.X.sample(frac=train_size, random_state=0)
        training_indices = self.train_features.index
        self.train_labels = self.y[training_indices].squeeze()
        
        self.test_features = self.X.drop(training_indices)
        self.test_labels = pd.Series(self.y.squeeze()).drop(training_indices)
        
        return self
        
    def drop_missing(self):
        self.data = self.data.dropna()
        return self
    
    def create_boolean_indices(self, columns='all'):
        if columns == 'all':
            self.data = pd.get_dummies(self.data)
        else:
            if not isinstance(columns, list):
                raise ValueError(f'Columns argument must be of type: list, received {type(columns)}.')
            
            self.data = pd.get_dummies(self.data, columns=columns)
            
            for col in self.data.columns:
                self.data[col] = self.data[col].astype(float)
        return self
    
    def drop_columns(self, columns):
        if not isinstance(columns, list) and not isinstance(columns, str):
            raise ValueError(f'Columns argument must be of type: str|list, received {type(columns)}.')
        columns = list(columns)
        
        self.data = self.data.drop(columns=columns)
            
        return self
    
    def scale(self, values, values_type, scaler="MinMaxScaler"):
        if self.X is None and self.y is None:
            raise AttributeError('Data must be split before scaling using model.split_data method.')
        
        if "minmax" in scaler.lower():
            self.scaler_X = sp.MinMaxScaler()
            self.scaler_y = sp.MinMaxScaler()
        elif "standard" in scaler.lower():
            self.scaler_X = sp.StandardScaler()
            self.scaler_y = sp.StandardScaler()
        else:
            raise ValueError(f'scaler argument must be in [\'MinMaxScaler\', \'StandardScaler\'], received {scaler}')
        
        
        if 'input' in values_type.lower():
            self.input_columns = self.X.columns
            self.scaler_X.fit(self.X)
            return pd.DataFrame(self.scaler_X.transform(values), columns=self.X.columns)
        
        elif 'output' in values_type.lower():
            self.scaler_y.fit(np.array(self.y).reshape(-1,1))
            return self.scaler_y.transform(np.array(values).reshape(-1,1))
        
        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")

    
    def unscale(self, values, values_type):
        if not self.scaler_X and not self.scaler_y:
            raise AttributeError('Data has not been scaled.')
        
        if 'input' in values_type.lower():
            return pd.DataFrame(self.scaler_X.inverse_transform(values), columns=self.input_columns)
        
        elif 'output' in values_type.lower():
            return self.scaler_y.inverse_transform(values)
        
        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")
        
    
        
            