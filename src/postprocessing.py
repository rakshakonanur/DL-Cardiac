import pandas as pd
import glob
import os
import re
from pathlib import Path
from extract_pca import resample, deform, main

# Path to dataset
dataset_path = "../datasets/updated_final/5"

def patient_num(path):
    match = re.search(r"patient_(\d+)", os.path.basename(path))
    return int(match.group(1)) if match else float('inf')

rows = []

for patient_dir in sorted(glob.glob(os.path.join(dataset_path, "patient_*")), key=patient_num):
    patient_name = os.path.basename(patient_dir)
    patient_id = int(patient_name.split("_")[1])
    print(f"Processing {patient_name}...")

    rows = []
    # Find PCA file
    pca_file = glob.glob(os.path.join(patient_dir, "unloaded_pc_scores_*.csv"))
    if not pca_file:
        continue
    
    pca_df = pd.read_csv(pca_file[0], header=None)
    pca_flat = pca_df.values.flatten()[1:26].tolist()  # first 10 elements regardless of shape
    if len(pca_flat) < 10:
        continue  # skip if not enough PCA values
    
    # Loop through result files
    results_path = os.path.join(patient_dir, "results-full/mode_-1/unloaded_ED")
    for result_file in glob.glob(os.path.join(results_path, "PLV*.bp")):
        if result_file.endswith("_checkpoint.bp"):
            continue  # skip checkpoint files

        filename = os.path.basename(result_file).replace(".bp", "")
        parts = filename.split("__")
        
        try:
            lv_ed = float(parts[0].split("_")[1])
            rv_ed = float(parts[1].split("_")[1])
            lv_es = float(parts[2].split("_")[1])
            rv_es = float(parts[3].split("_")[1])
            a_val = float(parts[6].split("_")[1])
            af_val = float(parts[7].split("_")[1])
        except (IndexError, ValueError):
            continue  # skip malformed filename
        
        ES_file = result_file
        ED_file = os.path.join(results_path, f"unloaded_to_ED_PLVED_{lv_ed:.2f}__PRVED_{rv_ed:.2f}__TA_0.0__eta_0.2__a_{a_val:.2f}__af_{af_val:.2f}.bp")
        u_ED, coords, geodir = resample(bpl=ED_file, mode=-1, datadir=Path(f"../datasets/updated_final/5/patient_{patient_id}/data-full"), resultsdir=Path(f"../datasets/updated_final/5/patient_{patient_id}/results-full"), case="ED")
        u_ES, coords, geodir = resample(bpl=ES_file, mode=-1, datadir=Path(f"../datasets/updated_final/5/patient_{patient_id}/data-full"), resultsdir=Path(f"../datasets/updated_final/5/patient_{patient_id}/results-full"), case="ES")
        outdir = Path(f"../datasets/updated_final/5/patient_{patient_id}/results-full/mode_-1/unloaded_ED")
        csv_dir = f"../datasets/updated_final/5/patient_{patient_id}/unloaded_pc_scores_patient_{patient_id}.csv"
        points_ED, undeformed = deform(Path(f"../datasets/updated_final/5/patient_{patient_id}/results-full/mode_-1/unloaded_ED"),  u_ED, geodir, csv_dir, coords, case="ED", patient_id=patient_id)
        points_ES, _ = deform(Path(f"../datasets/updated_final/5/patient_{patient_id}/results-full/mode_-1/unloaded_ED"), u_ES, geodir, csv_dir, coords, case="ES", patient_id=patient_id)
        deformed_pca, volume = main(points_ED, points_ES, undeformed,outdir)
        volume_items = [volume[k] for k in sorted(volume.keys())]  # sorted keeps it consistent

        row = [patient_name] + pca_flat + [lv_ed, lv_es, rv_ed, rv_es, a_val, af_val] + deformed_pca.tolist() + volume_items
        rows.append(row)

    columns = (["Patient"] + [f"PCA{i+1}" for i in range(25)] + 
               ["LV_ED", "LV_ES", "RV_ED", "RV_ES", "a", "a_f"] + 
               [f"defPCA{i+1}" for i in range(200)] + sorted(volume.keys()))  # column names match the order of values

    df = pd.DataFrame(rows, columns=columns)
    output_file = "aggregated_results_0827_2.csv"

    # Append mode, but write header only if file doesn't exist or is empty
    df.to_csv(
        output_file,
        mode='a',
        header=not os.path.exists(output_file) or os.path.getsize(output_file) == 0,
        index=False
    )

print("Saved aggregated_results.csv")
