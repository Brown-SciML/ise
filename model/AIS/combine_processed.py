from ise.utils import functions as f
import pandas as pd
import os

data_dir = r"/oscar/home/pvankatw/data/pvankatw/pvankatw-bfoxkemp/CMIP/dataset"

aogcms = os.listdir(data_dir)


for aogcm in aogcms:
    aogcm_path = os.path.join(data_dir, aogcm)
    if not os.path.isdir(aogcm_path):
        continue
    
    ssps = [x for x in os.listdir(aogcm_path) if x.startswith("ssp")]
    for ssp in ssps:
        
        data = []
        for realm in ["atmosphere", "ocean"]:
            realm_dir = os.path.join(aogcm_path, ssp, realm)
            if not os.path.isdir(realm_dir):
                continue
            
            csv_file = os.path.join(realm_dir, f"{aogcm}_{ssp}_atmospheric.csv") if realm == "atmosphere" else os.path.join(realm_dir, "1995-2100", f"{aogcm}_{ssp}_oceanic.csv")
            if os.path.exists(csv_file):
                data.append(csv_file)
                
        if len(data) == 2:
            ssp_data = pd.concat([pd.read_csv(x) for x in data], axis=1)
            ssp_data = ssp_data.loc[:, ~ssp_data.columns.duplicated()]
            ssp_data.to_csv(os.path.join(aogcm_path, f"{aogcm}_{ssp}_combined.csv"), index=False)


data_paths = f.get_all_filepaths(
    path=data_dir,
    filetype=".csv",
    contains="combined"
)
data = []
for path in data_paths:
    df = pd.read_csv(path)
    print(f"Loaded {path} with shape {df.shape}")
    data.append(df)

if data:
    combined = pd.concat(data, ignore_index=True)
    print(f"Combined DataFrame shape: {combined.shape}")
else:
    combined = pd.DataFrame()
    print("No data files loaded; combined DataFrame is empty.")

combined.to_csv("all_aogcm_ssp_combined.csv", index=False)

stop = ''