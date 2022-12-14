import pandas as pd
from sklearn import preprocessing as sp
import numpy as np
np.random.seed(10)
import random


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

        self.data['sle'] = self.data.ivaf / 1e9 / 362.5  # m^3 / 1e9 / 362.5 = Gt / 1 mm --> mm SLE
        self.data['modelname'] = self.data.groupname + '_' + self.data.modelname
        # self.data.smb = self.data.smb.fillna(method="ffill")

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

    def process(self, target_column='sle', drop_missing=True, drop_columns=True, boolean_indices=True, scale=True,
                split_type='batch_test', drop_outliers=False, time_series=False, lag=None):


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
                self = self.drop_columns(columns=['experiment', 'exp_id', 'groupname', 'regions'])
            elif isinstance(drop_columns, list):
                self = self.drop_columns(columns=drop_columns)
            else:
                raise ValueError(f'drop_columns argument must be of type boolean|list, received {type(drop_columns)}')

        if drop_missing:
            self = self.drop_missing()

        if boolean_indices:
            self = self.create_boolean_indices()

        if drop_outliers:
            column = drop_outliers['column']
            operator = drop_outliers['operator']
            value = drop_outliers['value']
            self = self.drop_outliers(column, operator, value, )

        self = self.split_data(target_column=target_column)

        if scale:
            self.X = self.scale(self.X, 'inputs', scaler='MinMaxScaler')
            if target_column == 'sle':
                self.y = np.array(self.y)
            else:
                self.y = self.scale(self.y, 'outputs', scaler='MinMaxScaler')

        self = self.train_test_split(split_type=split_type)

        return self, self.train_features, self.test_features, self.train_labels, self.test_labels

    def drop_outliers(self, column, operator, value, ):
        if operator.lower() in ('equal', 'equals', '=', '=='):
            outlier_data = self.data[self.data[column] == value]
        elif operator.lower() in ('not equal', 'not equals', '!=', '~='):
            outlier_data = self.data[self.data[column] != value]
        elif operator.lower() in ('greather than', 'greater', '>=', '>'):
            outlier_data = self.data[self.data[column] > value]
        elif operator.lower() in ('less than', 'less', '<=', '<'):
            outlier_data = self.data[self.data[column] < value]
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
                self.data[(self.data[modelname] == 1) & (self.data[exp_id] == 1) & (self.data.sectors == sector)].index)

        return self

    def split_data(self, target_column: str, ):
        self.target_column = target_column
        self.X = self.data.drop(columns=self.output_columns)
        self.y = self.data[target_column]
        self.input_columns = self.X.columns
        return self

    def train_test_split(self, train_size=0.7, split_type='random'):

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

            sectors_scaler = sp.MinMaxScaler().fit(np.array(self.data.sectors).reshape(-1, 1))

            test_index = 0
            test_scenarios = []
            test_indices = []
            test_features = []
            test_labels = []
            # Keep this running until you have enough samples
            while len(test_scenarios) < num_test_batches:
                try:
                    random_index = random.sample(range(len(self.batches)), 1)[0]
                    batch = self.batches[random_index]
                    if batch not in test_scenarios:
                        data = self.X[(self.X[f"modelname_{batch[0]}"] == 1) & (
                                self.X['sectors'] == sectors_scaler.transform(
                            np.array(batch[1]).reshape(1, -1)).squeeze()) & (self.X[f"exp_id_{batch[2]}"] == 1)]
                        labels = self.y[(self.X[f"modelname_{batch[0]}"] == 1) & (
                                self.X['sectors'] == sectors_scaler.transform(
                            np.array(batch[1]).reshape(1, -1)).squeeze()) & (self.X[f"exp_id_{batch[2]}"] == 1)]
                        if not data.empty:
                            test_scenarios.append(batch)
                            test_indices.append(random_index)
                            test_features.append(np.array(data))
                            test_labels.append(labels.squeeze())
                            test_index += 1
                except KeyError:
                    pass

            self.test_features = np.array(test_features)
            self.test_labels = np.array(test_labels)
            train_scenarios = np.delete(np.array(self.batches), test_indices, axis=0).tolist()

            train_features = []
            train_labels = []
            for batch in train_scenarios:
                try:
                    data = self.X[(self.X[f"modelname_{batch[0]}"] == 1) & (
                            self.X['sectors'] == sectors_scaler.transform(
                        np.array(batch[1]).reshape(1, -1)).squeeze()) & (self.X[f"exp_id_{batch[2]}"] == 1)]
                    labels = self.y[(self.X[f"modelname_{batch[0]}"] == 1) & (
                            self.X['sectors'] == sectors_scaler.transform(
                        np.array(batch[1]).reshape(1, -1)).squeeze()) & (self.X[f"exp_id_{batch[2]}"] == 1)]
                    if not data.empty:
                        train_features.append(np.array(data))
                        train_labels.append(labels.squeeze())
                except KeyError:
                    pass
            self.train_features = np.array(train_features)
            self.train_labels = np.array(train_labels)
            self.test_scenarios = test_scenarios



        elif split_type.lower() == "batch_test":
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

        return self

    def drop_missing(self):
        self.data = self.data.dropna()
        return self

    def create_boolean_indices(self, columns='all'):
        if columns == 'all':
            self.data = pd.get_dummies(self.data, prefix_sep="-")
        else:
            if not isinstance(columns, list):
                raise ValueError(f'Columns argument must be of type: list, received {type(columns)}.')

            self.data = pd.get_dummies(self.data, columns=columns, prefix_sep="-")

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
            # dump(self.scaler_X, open('./src/data/output_files/scaler_X.pkl', 'wb'))
            return pd.DataFrame(self.scaler_X.transform(values), columns=self.X.columns)

        # TODO: Don't need this anymore with SLE as the prediction
        elif 'output' in values_type.lower():
            self.scaler_y.fit(np.array(self.y).reshape(-1, 1))
            # dump(self.scaler_y, open('./src/data/output_files/scaler_y.pkl', 'wb'))
            return self.scaler_y.transform(np.array(values).reshape(-1, 1))

        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")

    def unscale(self, values, values_type):
        # self.scaler_X = load(open('./src/data/output_files/scaler_X.pkl', 'rb'))
        # self.scaler_y = load(open('./src/data/output_files/scaler_y.pkl', 'rb'))

        if 'input' in values_type.lower():
            return pd.DataFrame(self.scaler_X.inverse_transform(values), columns=self.input_columns)

        elif 'output' in values_type.lower():
            return self.scaler_y.inverse_transform(values.reshape(-1, 1))

        else:
            raise ValueError(f"values_type must be in ['inputs', 'outputs'], received {values_type}")
