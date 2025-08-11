import pandas as pd
import glob
import os
import re

# Path to dataset
dataset_path = "../datasets/final"

def patient_num(path):
    match = re.search(r"patient_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else float('inf')

rows = []

for patient_dir in sorted(glob.glob(os.path.join(dataset_path, "patient_*")), key=patient_num):
    patient_name = os.path.basename(patient_dir)
    
    # Find PCA file
    pca_file = glob.glob(os.path.join(patient_dir, "unloaded_pc_scores_*.csv"))
    if not pca_file:
        continue
    
    pca_df = pd.read_csv(pca_file[0], header=None)
    pca_flat = pca_df.values.flatten()[1:11].tolist()  # first 10 elements regardless of shape
    if len(pca_flat) < 10:
        continue  # skip if not enough PCA values
    
    # Loop through result files
    results_path = os.path.join(patient_dir, "results-full/mode_-1/unloaded_ED")
    for result_file in glob.glob(os.path.join(results_path, "PLV*.bp")):
        filename = os.path.basename(result_file).replace(".bp", "")
        parts = filename.split("__")
        
        try:
            lv_ed = float(parts[0].split("_")[1])
            rv_ed = float(parts[1].split("_")[1])
            lv_es = float(parts[2].split("_")[1])
            rv_es = float(parts[3].split("_")[1])
            a_val = float(parts[5].split("_")[1])
            af_val = float(parts[6].split("_")[1])
        except (IndexError, ValueError):
            continue  # skip malformed filename
        
        row = [patient_name] + pca_flat + [lv_ed, lv_es, rv_ed, rv_es, a_val, af_val]
        rows.append(row)

columns = ["Patient"] + [f"PCA{i+1}" for i in range(10)] + ["LV_ED", "LV_ES", "RV_ED", "RV_ES", "a", "a_f"]

df = pd.DataFrame(rows, columns=columns)
df.to_csv("aggregated_results.csv", index=False)

print("Saved aggregated_results.csv")
