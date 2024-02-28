import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ise.grids.models.PCA import PCA
import os

ice_sheet = 'AIS'
pca_dir = f"/oscar/scratch/pvankatw/pca/{ice_sheet}/pca_models/"
out_dir = r"/users/pvankatw/research/current/supplemental/pca_results/"
num_components = 10
pca_models = os.listdir(pca_dir)

for model_path in pca_models:
    pca = PCA.load(f"{pca_dir}/{model_path}")
    pca.components = pca.components.T
    
    exp_var_pca = pca.explained_variance_ratio
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    var = model_path.split('_')[-1].split('.')[0]
    
    if not os.path.exists(f"{out_dir}/{var}"):
        os.makedirs(f"{out_dir}/{var}")
    
    if var == 'sle':
        stop = ''
    
    for i in range(1, num_components+1):
        pca_1 = pca.components[i-1, :].reshape(761, 761)
        plt.figure()
        plt.imshow(pca_1)
        plt.title(f"PC{i}, {exp_var_pca[i-1]*100:0.2f}% Var Explained")
        plt.savefig(f'{out_dir}/{var}/pca{i}_{var}_pytorch.png')
        plt.show()

stop = ''