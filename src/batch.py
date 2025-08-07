import itertools
import shutil
import subprocess
import os
from pathlib import Path

# Define the parameter values
# PLV_ED_vals = [5, 10, 20]
# PLV_ES_vals = [5.5, 16, 30]
# PRV_ED_vals = [1, 1.5, 4]
# PRV_ES_vals = [1.5, 2.67, 8]
# a_vals = [0.1, 0.25, 2.0]
# a_f_vals = [1, 20, 30]

PLV_ED_vals = [10]
PLV_ES_vals = [16, 30]
PRV_ED_vals = [1.5]
PRV_ES_vals = [2.67]
a_vals = [0.25]
a_f_vals = [20, 30]

# Number of patients to simulate
num_patients = 1  # or any number you want

# Paths
base_dir = Path.cwd()
output_dir = base_dir / "output"
results_dir = base_dir / "output" / "results-full"
data_root = base_dir / "dataset"

data_root.mkdir(exist_ok=True)

# Generate all combinations of parameters
param_combinations = list(itertools.product(PLV_ED_vals, PLV_ES_vals,
                                            PRV_ED_vals, PRV_ES_vals,
                                            a_vals, a_f_vals))

for patient_id in range(num_patients):
    print(f"\n=== Generating mesh for patient {patient_id} ===")
    
    # Step 1: Run mesh.py
    subprocess.run(["python", "mesh.py", "--patient_id", str(patient_id)], check=True)

    # Step 2: Copy output folder
    patient_folder = data_root / f"patient_{patient_id}"
    # patient_folder.mkdir(parents=True, exist_ok=True)
    # # mesh_copy = patient_folder / "mesh"
    # # if mesh_copy.exists():
    # #     shutil.rmtree(mesh_copy)
    shutil.copytree(output_dir, patient_folder)

    # Step 3–5: Iterate through all parameter combinations
    for (PLV_ED, PLV_ES, PRV_ED, PRV_ES, a, a_f) in param_combinations:
        print(f" -> Running simulation for parameters: PLV=({PLV_ED}, {PLV_ES}), PRV=({PRV_ED}, {PRV_ES}), a={a}, a_f={a_f}")
        
        # Format parameters
        plv = f"[{PLV_ED},{PLV_ES}]"
        prv = f"[{PRV_ED},{PRV_ES}]"

        # Run simulation
        subprocess.run([
            "mpirun", "-n", "8",  # Adjust the number of processes as needed
            "python", "simulation.py",
            "--PLV", plv,
            "--PRV", prv,
            "--a", str(a),
            "--a_f", str(a_f)
        ], check=True)

        # Step 4: Copy results-full to patient folder with descriptive name
        sim_name = f"PLV_{PLV_ED}_{PLV_ES}__PRV_{PRV_ED}_{PRV_ES}__a_{a}__af_{a_f}"
        sim_folder = patient_folder / sim_name
        if sim_folder.exists():
            shutil.rmtree(sim_folder)
        shutil.copytree(results_dir, sim_folder)

print("\n✅ All simulations complete.")
