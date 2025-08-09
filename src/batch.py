import itertools
import shutil
import subprocess
import os
from pathlib import Path
import numpy as np
import time

# Define the parameter values
PLV_ED_vals = [5, 10, 20]
# PLV_ES_vals = [5.5, 16, 30]
PRV_ED_vals = [1, 1.5, 4]
# PRV_ES_vals = [1.5, 2.67, 8]
# a_vals = [0.1, 0.25, 2.0]
# a_f_vals = [1, 20, 30]

PLV_ED = 20
PLV_ES = 30
PRV_ED = 4
PRV_ES = 8

a = 2.280
a_f = 1.685
x = 1
n_max = 1  # number of steps in each direction (positive and negative)

def generate_sorted_array(center, x, n_max):
    result = [center]
    for n in range(1, n_max + 1):
        result.append(center + n * x)
        result.append(center - n * x)
    return result

a_array = generate_sorted_array(a, x, n_max)
a_f_array = [1.685, 20, 30]
# a_f_array = a_f_array[1:]

print("a_array:", a_array)
print("a_f_array:", a_f_array)

# Number of patients to simulate
num_patients = 10  # or any number you want

# Generate all combinations of parameters
param_combinations = list(itertools.product(a_array, a_f_array, PLV_ED_vals, PRV_ED_vals))

count = 0
start_patient = 0
end_patient = 10

# Stores the number of retries for meshing
max_retries = 2

# Storage for results
success_cases = []
failure_cases = []

for patient_id in range(start_patient, end_patient):

    print(f"\n=== Generating mesh for patient {patient_id} ===")

    # Paths
    base_dir = Path.cwd()
    output_dir = base_dir / f"patient_{patient_id}"
    results_dir = output_dir / "results-full"
    data_root = base_dir / "dataset"
    data_root.mkdir(exist_ok=True)

    # Mesh generation with retries
    attempt = 0
    while attempt <= max_retries:
        try:
            subprocess.run(["python", "mesh.py", "--patient_id", str(patient_id)], check=True)
            break
        except subprocess.CalledProcessError:
            print(f"Attempt {attempt + 1} failed for patient {patient_id}. Retrying...")
            attempt += 1
            time.sleep(1)
            if attempt > max_retries:
                with open("error_log.txt", "a") as log_file:
                    log_file.write(f"Failed to mesh patient {patient_id} after {max_retries + 1} attempts.\n")
                # Skip to next patient
                patient_id += 1
                continue

    # Parameter sweep
    for (a, a_f, PLV_ED, PRV_ED) in param_combinations:
        print(f" -> Running simulation for parameters: PLV=({PLV_ED}, {PLV_ES}), PRV=({PRV_ED}, {PRV_ES}), a={a}, a_f={a_f}")
        
        plv = f"[{PLV_ED},{PLV_ES}]"
        prv = f"[{PRV_ED},{PRV_ES}]"

        try:
            # Command to run simulation.py with parameters
            cmd = [
                "mpirun", "-n", "30",
                "python", "simulation.py",
                "--patient_id", str(patient_id),
                "--PLV", str(plv),
                "--PRV", str(prv),
                "--a", str(a),
                "--a_f", str(a_f)
            ]

            # Start subprocess
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1  # Line-buffered
            )

            # Capture output while printing live
            output_lines = []
            for line in process.stdout:
                print(line, end='')         # Live print
                output_lines.append(line)   # Store for later

            process.wait()

            # After completion, analyze the output
            if output_lines:
                last_line = output_lines[-1].strip()
                success = last_line.lower() == "True"
                if success:
                    success_cases.append((a, a_f, PLV_ED, PRV_ED))
                else:
                    failure_cases.append((a, a_f, PLV_ED, PRV_ED))
            else:
                print("No output captured from simulation.py")
                failure_cases.append(a, a_f, PLV_ED, PRV_ED)

        except subprocess.CalledProcessError as e:
            print(f"Simulation failed to run for patient {patient_id}, parameters a={a}, a_f={a_f}")
            failure_cases.append((a, a_f, PLV_ED, PRV_ED))
        

# Save results for later analysis (optional)
import csv

with open("success_cases.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["a", "a_f", "PLV_ED", "PRV_ED"])
    writer.writerows(success_cases)

with open("failure_cases.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["a", "a_f", "PLV_ED", "PRV_ED"])
    writer.writerows(failure_cases)

print("\n✅ All simulations complete.")
