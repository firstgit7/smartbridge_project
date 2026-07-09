import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("rikdifos/credit-card-approval-prediction")
print("Path to dataset files:", path)

# Copy to our local dataset directory
dest_dir = "dataset"
os.makedirs(dest_dir, exist_ok=True)
for file_name in os.listdir(path):
    full_file_name = os.path.join(path, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, dest_dir)
        print(f"Copied {file_name} to {dest_dir}")
