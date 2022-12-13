from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import r2_score
import numpy as np
np.random.seed(10)

class GP(GaussianProcessRegressor):
    def __init__(self, kernel, verbose=True):
        super().__init__(n_restarts_optimizer=9)
        self.kernel = kernel
        self.verbose = verbose
        
    def train(self, train_features, train_labels,):
        self.train_features, self.train_labels = train_features, train_labels
        self.fit(train_features, train_labels,)
        return self
    
    def test(self, test_features, test_labels):
        self.test_features, self.test_labels = test_features, test_labels
        preds, std_prediction = self.predict(test_features, return_std=True)
        test_labels = np.array(test_labels.squeeze())
        mse = sum((preds - test_labels)**2) / len(preds)
        mae = sum(abs((preds - test_labels))) / len(preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(test_labels, preds)
        
        metrics = {'MSE': mse, 'MAE': mae, 'RMSE': rmse, 'R2': r2}

        if self.verbose:
            print(f"""Test Metrics
MSE: {mse:0.6f}
MAE: {mae:0.6f}
RMSE: {rmse:0.6f}
R2: {r2:0.6f}""")
        return preds, std_prediction, metrics
    


