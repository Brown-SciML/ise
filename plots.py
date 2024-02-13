import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

dir_ = "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/pca/pca_models"
# pca_path = "/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/pca/pca_models/pca_smb_anomaly_68pcs.pkl"
pca_path = f"{dir_}/pca_sle_22pcs.pkl"
pca = pkl.load(open(pca_path, "rb"))
first_component_number = 1
second_component_number = 3
pca_1 = pca.components_[first_component_number-1, :].reshape(761, 761)
pca_2 = pca.components_[second_component_number-1, :].reshape(761, 761)

exp_var_pca = pca.explained_variance_ratio_
cum_sum_eigenvalues = np.cumsum(exp_var_pca)

plt.figure()
plt.imshow(pca_1)
plt.title(f"PC{first_component_number}, {cum_sum_eigenvalues[0]*100:0.2f}% Var Explained")
plt.savefig('pca1.png')
plt.show()

plt.figure()
plt.imshow(pca_2)
plt.title(f"PC{second_component_number}, {(cum_sum_eigenvalues[second_component_number-1]*100) - (cum_sum_eigenvalues[second_component_number-2]*100):0.2f}% Var Explained")
plt.savefig('pca2.png')
plt.show()

stop = ''