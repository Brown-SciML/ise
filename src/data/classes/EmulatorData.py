from multiprocessing.sharedctypes import Value
import pandas as pd
from sklearn import preprocessing as sp

class EmulatorData:
    def __init__(self, directory):
        
        self.directory = directory
        
        self.output_columns = ['icearea', 'iareafl', 'iareagr', 'ivol', 'ivaf', 'smb', 'smbgr', 'bmbfl']
        
        try:
            self.data = pd.read_csv(f"{self.directory}/master.csv")
        except FileNotFoundError:
            try:
                self.inputs = pd.read_csv(f"{self.directory}/inputs.csv")
                self.outputs = pd.read_csv(f"{self.directory}/outputs.csv")
            except FileNotFoundError:
                raise FileNotFoundError('Files not found, make sure to run all processing functions.')
            
        self.X = None
        self.y = None
        self.scaler_X = None
        self.scaler_y = None
    
    def split_data(self, target_column: str):
        self.X = self.data.drop(columns=self.output_columns)
        self.y = self.data[target_column]
        return self.X, self.y
        
    def drop_missing(self):
        self.data = self.data.dropna()
        return self
    
    def create_boolean_indices(self, columns='all'):
        if columns == 'all':
            self.data = pd.get_dummies(self.data)
        else:
            if not columns.isinstance(list):
                raise ValueError(f'Columns argument must be of type: list, received {type(columns)}.')
            
            self.data = pd.get_dummies(self.data, columns=columns)
        return self
    
    def drop_columns(self, columns):
        if not columns.isinstance(list) or not columns.isinstance(str):
            raise ValueError(f'Columns argument must be of type: str|list, received {type(columns)}.')
        columns = list(columns)
        
        self.data = self.data.drop(columns=columns)
    
    def scale_dataset(self, scaler="MinMaxScaler"):
        if not self.X and not self.y:
            raise AttributeError('Data must be split before scaling using model.split_data method.')
        
        if "minmax" in scaler.lower():
            self.scaler = sp.MinMaxScaler()
        elif "standard" in scaler.lower():
            self.scaler = sp.StandardScaler()
        else:
            raise ValueError(f'scaler argument must be in [\'MinMaxScaler\', \'StandardScaler\'], received {scaler}')
        
        
        self.scaler_X = self.scaler.fit(self.X)
        self.scaler_y = self.scaler.fit(self.y)
        
        self.X = self.scaler_X.transform(self.X)
        self.y = self.scaler_X.transform(self.y)
        
        return self
    
    def unscale_dataset(self,):
        if not self.scaler_X and not self.scaler_y:
            raise AttributeError('Data has not been scaled.')
        
        self.X = self.scaler_X.inverse_transform(self.X)
        self.y = self.scaler_y.inverse_transform(self.y)
        
        return self
        
    
        
            