"""Module containing EmulatorData class with all associated methods and attributes. Primarily carries out data loading, feature engineering & processing of formatted data."""
import random
import pandas as pd
import numpy as np
from sklearn import preprocessing as sp
np.random.seed(10)


class EmulatorData:
    """Class containing attributes and methods for storing and handling ISMIP6 ice sheet data."""
    def __init__(self, directory: str):
        """Initializes class and opens/stores data. Includes loading processed files from 
        ise.data.processors functions, converting IVAF to SLE, and saving initial condition data
        for later use.

        Args:
            directory (str): Directory containing processed files from ise.data.processors functions. Should contain master.csv
        """

        self.directory = directory

        try:
            self.data = pd.read_csv(f"{self.directory}/master.csv", low_memory=False)
        except FileNotFoundError:
            try:
                self.inputs = pd.read_csv(f"{self.directory}/inputs.csv")
                self.outputs = pd.read_csv(f"{self.directory}/outputs.csv")
            except FileNotFoundError:
                raise FileNotFoundError('Files not found, make sure to run all processing functions.')

        # convert to SLE
        self.data['sle'] = self.data.ivaf / 1e9 / 362.5  # m^3 / 1e9 / 362.5 = Gt / 1 mm --> mm SLE
        self.data['modelname'] = self.data.groupname + '_' + self.data.modelname


        # Save data on initial conditions for use in data splitting
        unique_batches = self.data.groupby(['modelname', 'sectors', 'exp_id']).size().reset_index().rename(
            columns={0: 'count'}).drop(columns='count')
        self.batches = unique_batches.values.tolist()
        self.output_columns = ['icearea', 'iareafl', 'iareagr', 'ivol', 'ivaf', 'smb', 'smbgr', 'bmbfl', 'sle']

        for col in ['icearea', 'iareafl', 'iareagr', 'ivol', 'smb', 'smbgr', 'bmbfl', 'sle']:
            self.data[col] = self.data[col].fillna(self.data[col].mean())

        self.X = None
        self.y = None
        self.scaler_X = None
        self.scaler_y = None

    def process(self, target_column: str='sle', drop_missing: bool=True, 
                drop_columns: list[str]=True, boolean_indices: bool=True, scale: bool=True,
                split_type: str='batch', drop_outliers: str=False, drop_expression: list[tuple]=None,
                time_series: bool=False, lag: int=None):
        """Carries out feature engineering & processing of formatted data. Includes dropping missing
        values, eliminating columns, creating boolean indices of categorical variables, scaling, 
        dataset splitting, and more. Refer to the individual functions contained in the source
        code for more information on each process.

        Args:
            target_column (str, optional): Column to be predicted. Defaults to 'sle'.
            drop_missing (bool, optional): Flag denoting whether to drop missing values. Defaults to True.
            drop_columns (list[str], optional): List containing which columns (variables) to be dropped. Should be List[str] or boolean. If True is chosen, columns are dropped that will result in optimal performance. Defaults to True.
            boolean_indices (bool, optional): Flag denoting whether to create boolean indices for all categorical variables left after dropping columns. Defaults to True.
            scale (bool, optional): Flag denoting whether to scale data between zero and 1.  Sklearn's MinMaxScaler is used. Defaults to True.
            split_type (str, optional): Method to split data into training and testing set, must be in [random, batch]. Random is not recommended but is included for completeness. Defaults to 'batch'.
            drop_outliers (str, optional): Method by which outliers will be dropped, must be in [quantile, explicit]. Defaults to False.
            drop_expression (list[tuple], optional): Expressions by which to drop outliers, see EmulatorData.drop_outliers. If drop_outliers==quantile, drop_expression must be a list[float] containing quantile bounds. Defaults to None. 
            time_series (bool, optional): Flag denoting whether to process the data as a time-series dataset or traditional non-time dataset. Defaults to False.
            lag (int, optional): Lag variable for time-series processing. Defaults to None.
            
        Returns:
            tuple: Multi-output returning [EmulatorData, train_features, test_features, train_labels, test_labels]
        """


        if time_series:
            if lag is None:
                raise ValueError('If time_series == True, lag cannot be None')
            time_dependent_columns = ['salinity', 'temperature', 'thermal_forcing',
                                      'pr_anomaly', 'evspsbl_anomaly', 'mrro_anomaly', 'smb_anomaly',
                                      'ts_anomaly',]
            separated_dfs = [y for x, y in self.data.groupby(['sectors', 'exp_id', 'modelname'])]
            for df in separated_dfs:
                for shift in range(1, lag + 1):
                    for column in time_dependent_columns:
                        df[f"{column}.lag{shift}"] = df[column].shift(shift, fill_value=0)
            self.data = pd.concat(separated_dfs)


        if drop_columns:
            if drop_columns is True:
                self.drop_columns(columns=['experiment', 'exp_id', 'groupname', 'regions'])
            elif isinstance(drop_columns, list):
                self.drop_columns(columns=drop_columns)
            else:
                raise ValueError(f'drop_columns argument must be of type boolean|list, received {type(drop_columns)}')

        if drop_missing:
            self = self.drop_missing()

        if boolean_indices:
            self = self.create_boolean_indices()

        if drop_outliers:
            if drop_outliers.lower() == 'quantile':
                self = self.drop_outliers(method=drop_outliers, quantiles=drop_expression)
            elif drop_outliers.lower() == 'explicit':
                self = self.drop_outliers(method=drop_outliers, expression=drop_expression)
            else:
                raise ValueError('drop_outliers argument must be in [quantile, explicit]')
            
            

        self = self.split_data(target_column=target_column)

        if scale:
            self.X = self.scale(self.X, 'inputs', scaler='MinMaxScaler')
            if target_column == 'sle':
                self.y = np.array(self.y)
            else:
                self.y = self.scale(self.y, 'outputs', scaler='MinMaxScaler')

        self = self.train_test_split(split_type=split_type)

        return self, self.train_features, self.test_features, self.train_labels, self.test_labels

    def drop_outliers(self, method: str, expression: list[tuple]=None, quantiles: list[float]=[0.01, 0.99]):
        """Drops simulations that are outliers based on the provided method and expression. 
        Extra complexity is handled due to the necessity of removing the entire 85 row series from 
        the dataset rather than simply removing the rows with given conditions. Note that the 
        condition indicates rows to be DROPPED, not kept (e.g. 'sle', '>', '20' would drop all
        simulations containing sle values over 20). If quantile method is used, outliers are dropped
        from the SLE column based on the provided quantile in the quantiles argument. If explicit is
        chosen, expression must contain a list of tuples such that the tuple contains 
        [(column, operator, expression)] of the subset, e.g. [("sle", ">", 20), ("sle", "<", -20)].

        Args:
            method (str): Method of outlier deletion, must be in [quantile, explicit]
            expression (list[tuple]): List of subset expressions in the form [column, operator, value], defaults to None.
            quantiles (list[float]): , defaults to [0.01, 0.99].

        Returns:
            EmulatorData: self, with self.data having outliers dropped.
        """
        
        if method.lower() == 'quantile':
            if quantiles is None:
                raise AttributeError('If method == quantile, quantiles argument cannot be None')
            lower_sle, upper_sle = np.quantile(np.array(self.data.sle), quantiles)
            outlier_data = self.data[(self.data['sle'] <= lower_sle) | (self.data['sle'] >= upper_sle)]
        elif method.lower() == 'explicit':
            
            if expression is None:
                raise AttributeError('If method == explicit, expression argument cannot be None')
            elif not isinstance(expression, list) or not isinstance(expression[0], tuple):
                raise AttributeError('Expression argument must be a list of tuples, e.g. [("sle", ">", 20), ("sle", "<", -20)]')
            
            outlier_data = self.data
            for subset_expression in expression:
                column, operator, value = subset_expression            

                if operator.lower() in ('equal', 'equals', '=', '=='):
                    outlier_data = outlier_data[outlier_data[column] == value]
                elif operator.lower() in ('not equal', 'not equals', '!=', '~='):
                    outlier_data = outlier_data[outlier_data[column] != value]
                elif operator.lower() in ('greather than', 'greater', '>=', '>'):
                    outlier_data = outlier_data[outlier_data[column] > value]
                elif operator.lower() in ('less than', 'less', '<=', '<'):
                    outlier_data = outlier_data[outlier_data[column] < value]
                else:
                    raise ValueError(f'Operator must be in [\"==\", \"!=\", \">\", \"<\"], received {operator}')

        cols = outlier_data.columns
        nonzero_columns = outlier_data.apply(lambda x: x > 0).apply(lambda x: list(cols[x.values]), axis=1)

        # Create dataframe of experiments with outliers (want to delete the entire 85 rows)
        outlier_runs = pd.DataFrame()
        outlier_runs['modelname'] = nonzero_columns.apply(lambda x: x[-6])
        outlier_runs['exp_id'] = nonzero_columns.apply(lambda x: x[-5])
        outlier_runs['sectors'] = outlier_data.sectors
        outlier_runs_list = outlier_runs.values.tolist()
        unique_outliers = [list(x) for x in set(tuple(x) for x in outlier_runs_list)]

        # Drop those runs
        for i in unique_outliers:
            modelname = i[0]
            exp_id = i[1]
            sector = i[2]
            self.data = self.data.drop(
                self.data[(self.data[modelname] == 1) & (self.data[exp_id] == 1) & (self.data.sectors == sector)].index
            )

        return self

    def split_data(self, target_column: str, ):
        """Splits data into features and labels based on target column.

        Args:
            target_column (str): Output column to be predicted.

        Returns:
            EmulatorData: self, with self.X and self.y as attributes.
        """
        self.target_column = target_column
        self.X = self.data.drop(columns=self.output_columns)
        self.y = self.data[target_column]
        self.input_columns = self.X.columns
        return self

    def train_test_split(self, train_size: float=0.7, split_type: str='batch'):
        """Splits dataset into training set and testing set. Can be split using two different
        methods: random and batch. The random method splits by randomly sampling rows, whereas 
        batch method randomly samples entire simulation series (85 rows) in order to keep simulations
        together during testing. Random method is included for completeness but is not recommended
        for use in emulator creation.

        Args:
            train_size (float, optional): Proportion of data in training set, between 0 and 1. Defaults to 0.7.
            split_type (str, optional): Splitting method, must be in [random, batch]. Defaults to 'batch'.

        Returns:
            EmulatorData: self, with self.train_features, self.test_features, self.train_labels, self.test_labels as attributes.
        """

        if not isinstance(self.X, pd.DataFrame):
            self.X = pd.DataFrame(self.X, columns=self.input_columns)

        if 'random' in split_type.lower():
            self.train_features = self.X.sample(frac=train_size, random_state=0)
            training_indices = self.train_features.index
            self.train_labels = self.y[training_indices].squeeze()

            self.test_features = self.X.drop(training_indices)
            self.test_labels = pd.Series(self.y.squeeze()).drop(training_indices)


        elif split_type.lower() == "batch":
            # batch -- grouping of 85 years of a particular model, experiment, and sector
            # Calculate how many batches you'll need (roughly) for train/test proportion
            test_num_rows = len(self.X) * (1 - train_size)
            num_years = len(set(self.data.year))
            num_test_batches = test_num_rows // num_years

            # Get all possible values for sector, experiment, and model
            all_sectors = list(set(self.X.sectors))
            all_experiments = [col for col in self.X.columns if "exp_id" in col]
            all_modelnames = [col for col in self.X.columns if "modelname" in col]

            # Set up concatenation of test data scenarios...
            test_scenarios = []
            test_dataset = pd.DataFrame()

            # Keep this running until you have enough samples
            np.random.seed(10)
            while len(test_scenarios) < num_test_batches:
                # Get a random
                random_model = np.random.choice(all_modelnames)
                random_sector = np.random.choice(all_sectors)
                random_experiment = np.random.choice(all_experiments)
                test_scenario = [random_model, random_sector, random_experiment]
                if test_scenario not in test_scenarios:
                    scenario_df = self.X[(self.X[random_model] == 1) & (self.X['sectors'] == random_sector) & (
                            self.X[random_experiment] == 1)]
                    if not scenario_df.empty:
                        test_scenarios.append(test_scenario)
                        test_dataset = pd.concat([test_dataset, scenario_df])
            self.test_features = test_dataset
            testing_indices = self.test_features.index
            self.test_labels = self.y[testing_indices].squeeze()

            self.train_features = self.X.drop(testing_indices)
            self.train_labels = pd.Series(self.y.squeeze()).drop(testing_indices)

            self.test_scenarios = test_scenarios
        
        else:
            raise(f'split_type must be in [random, batch], received {split_type}')

        return self

    def drop_missing(self):
        """Drops rows with missing values (wrapper for pandas.DataFrame.dropna()).

        Returns:
            EmulatorData: self, with NA values dropped from self.data.
        """
        self.data = self.data.dropna()
        return self

    def create_boolean_indices(self, columns: str='all'):
        """Creates boolean indices (one hot encoding) for categoritcal variables in columns 
        argument. Wrapper for pandas.get_dummies() with added functionality for prefix separation.

        Args:
            columns (str, optional): Categorical variables to be encoded. Defaults to 'all'.

        Returns:
            EmulatorData: self, with boolean indices in self.data.
        """
        if columns == 'all':
            self.data = pd.get_dummies(self.data, prefix_sep="-")
        else:
            if not isinstance(columns, list):
                raise ValueError(f'Columns argument must be of type: list, received {type(columns)}.')

            self.data = pd.get_dummies(self.data, columns=columns, prefix_sep="-")

            for col in self.data.columns:
                self.data[col] = self.data[col].astype(float)
        return self

    def drop_columns(self, columns: list[str]):
        """Drops columns in columns argument from the dataset. Wrapper for pandas.DataFrame.drop()
        with error checking.

        Args:
            columns (list[str]): List of columns (or singular string column) to be dropped from the dataset.

        Returns:
            EmulatorData: self, with desired columns dropped from self.data.
        """
        if not isinstance(columns, list) and not isinstance(columns, str):
            raise ValueError(f'Columns argument must be of type: str|list, received {type(columns)}.')
        columns = list(columns)

        self.data = self.data.drop(columns=columns)

        return self

    def scale(self, values: pd.DataFrame, values_type: str, scaler: str="MinMaxScaler"):
        """Scales dataframe and saves scaler for future use in unscaling. Sklearn's scaling API is
        used. MinMaxScaler is recommended but StandardScaler is also supported.

        Args:
            values (pd.DataFrame): Dataframe to be scaled.
            values_type (str): Whether the dataframe to be scaled is a feature or labels dataframe, must be in [inputs, outputs]
            scaler (str, optional): Type of scaler to be used, must be in [MinMaxScaler, StandardScaler]. Defaults to "MinMaxScaler".

        Returns:
            pd.DataFrame: scaled dataset with self.scaler_X and self.scaler_y as attributes in the EmulatorData class.
        """
        if self.X is None and self.y is None:
            raise AttributeError('Data must be split before scaling using model.split_data method.')

        if "minmax" in scaler.lower():
            if 'input' in values_type.lower():
                self.scaler_X = sp.MinMaxScaler()
            else:
                self.scaler_y = sp.MinMaxScaler()
        elif "standard" in scaler.lower():
            if 'input' in values_type.lower():
                self.scaler_X = sp.StandardScaler()
            else:
                self.scaler_y = sp.StandardScaler()
        else:
            raise ValueError(f'scaler argument must be in [\'MinMaxScaler\', \'StandardScaler\'], received {scaler}')

        if 'input' in values_type.lower():
            self.input_columns = self.X.columns
            self.scaler_X.fit(self.X)
            return pd.DataFrame(self.scaler_X.transform(values), columns=self.X.columns)

        # TODO: Don't need this anymore with SLE as the prediction
        elif 'output' in values_type.lower():
            self.scaler_y.fit(np.array(self.y).reshape(-1, 1))
            return self.scaler_y.transform(np.array(values).reshape(-1, 1))

        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")

    def unscale(self, values: pd.DataFrame, values_type: str):
        """Unscales data based on scalers trained in EmulatorData.scale().

        Args:
            values (pd.DataFrame): Dataframe to be unscaled.
            values_type (str): Whether the dataframe to be unscaled is a feature or labels dataframe, must be in [inputs, outputs]

        Returns:
           pd.DataFrame: unscaled dataset.
        """

        if 'input' in values_type.lower():
            return pd.DataFrame(self.scaler_X.inverse_transform(values), columns=self.input_columns)

        elif 'output' in values_type.lower():
            return self.scaler_y.inverse_transform(values.reshape(-1, 1))

        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")
