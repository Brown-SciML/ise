import pickle as pkl
import numpy as np
from ise.models.PCA import PCA
import os

import matplotlib.pyplot as plt

ice_sheet = 'AIS'
pca_dir = f"/oscar/scratch/pvankatw/pca/{ice_sheet}/pca_models/"
out_dir = f"/users/pvankatw/research/current/supplemental/pca_results/{ice_sheet}/"
num_components = 8
pca_models = os.listdir(pca_dir)

ice_sheet_shape = (761, 761) if ice_sheet == 'AIS' else (337, 577)

for model_path in pca_models:
    pca = PCA.load(f"{pca_dir}/{model_path}")
    pca.components = pca.components
    print(model_path, pca.components.shape[1])
    
    exp_var_pca = pca.explained_variance_ratio
    cum_sum_eigenvalues = np.cumsum(exp_var_pca)
    var = model_path.split('_')[-1].split('.')[0]
    if var == 'anomaly':
        var = '_'.join(model_path.split('_')[-2:]).replace('.pth', '')
        
    
    if not os.path.exists(f"{out_dir}/{var}"):
        os.makedirs(f"{out_dir}/{var}")
    
    if var == 'sle':
        stop = ''
    
    for i in range(1, num_components+1):
        # pca_1 = pca.components[:, i-1].reshape(ice_sheet_shape).numpy()
        pca_1 = np.flipud(pca.components[:, i-1].reshape(ice_sheet_shape).T,)
        
        # Mask out zero values and make them white
        pca_1_masked = np.ma.masked_where(pca_1 == 0, pca_1)
        
        plt.figure()
        plt.imshow(pca_1_masked, cmap='viridis', vmin=np.min(pca_1), vmax=np.max(pca_1))
        plt.title(f"PC{i}, {exp_var_pca[i-1]*100:0.2f}% Var Explained")
        plt.savefig(f'{out_dir}/{var}/pca{i}_{var}_pytorch.png')
        plt.show()

stop = ''