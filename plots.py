import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from ise.grids.models.PCA import PCA

var = 'salinity'
# pca_path = "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/pca/pca_models/pca_smb_anomaly_94pcs.pkl"
pca_path = f"/oscar/scratch/pvankatw/pca_pytorch/AIS/pca_models/AIS_pca_{var}.pth"
# pca_path = "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/pca/pca_models/pca_smb_anomaly_68pcs.pkl"
# pca = pkl.load(open(pca_path, "rb"))
pca = PCA.load(pca_path)
pca.components = pca.components.T
first_component_number = 3
second_component_number = 4
pca_1 = pca.components[first_component_number-1, :].reshape(761, 761)
pca_2 = pca.components[second_component_number-1, :].reshape(761, 761)

exp_var_pca = pca.explained_variance_ratio
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure()
plt.imshow(pca_1)
plt.title(f"PC{first_component_number}, {cum_sum_eigenvalues[0]*100:0.2f}% Var Explained")
plt.savefig(f'pca{first_component_number}_{var}_pytorch.png')
plt.show()

plt.figure()
plt.imshow(pca_2)
plt.title(f"PC{second_component_number}, {(cum_sum_eigenvalues[second_component_number-1]*100) - (cum_sum_eigenvalues[second_component_number-2]*100):0.2f}% Var Explained")
plt.savefig(f'pca{second_component_number}_{var}_pytorch.png')
plt.show()

stop = ''