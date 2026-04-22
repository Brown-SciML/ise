import os
import json
import shutil

# Base directory
base_dir = "/users/pvankatw/research/ise/model/AIS/nf"

# Variables to track the best model
best_loss = float("inf")
best_model_dir = None

# Traverse subdirectories
for subdir in os.listdir(base_dir):
    full_path = os.path.join(base_dir, subdir)
    if os.path.isdir(full_path) and subdir.startswith("nf_"):
        metadata_path = os.path.join(full_path, "best_model.pt_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                try:
                    metadata = json.load(f)
                    loss = metadata.get("best_loss", None)
                    if loss is not None and loss < best_loss:
                        best_loss = loss
                        best_model_dir = full_path
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse {metadata_path}")

# If we found the best model, copy files
if best_model_dir:
    src_model = os.path.join(best_model_dir, "best_model.pt")
    src_metadata = os.path.join(best_model_dir, "best_model.pt_metadata.json")
    dst_model = os.path.join(base_dir, "nf.pt")
    dst_metadata = os.path.join(base_dir, "nf.pt_metadata.json")
    
    shutil.copy(src_model, dst_model)
    shutil.copy(src_metadata, dst_metadata)
    
    print(f"Copied best model from {best_model_dir}")
    print(f"Lowest best_loss: {best_loss}")
else:
    print("No valid best_model.pt_metadata.json files found.")
