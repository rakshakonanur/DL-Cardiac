import sys
from pathlib import Path
import numpy as np

def parse_array(s):
    # Accepts strings like "[1.0, 2.0]" or "1.0,2.0"
    s = s.strip("[]")
    return np.array([float(x) for x in s.split(",")])

class Simulation:
    def __init__(self, mode: int = -1, case = "unloaded_ED",
                 results_dir: str = f"patient_0/results-full", 
                 data_dir: str = f"patient_0/data-full/", 
                 solver_path: str = "../clones/rk-sscp25-deep-learning-cardiac-mechanics",
                 PLV: np.array = [30.0, 40.0],
                 PRV: np.array = [6.0, 8.0],
                 Ta: np.array = [0.0, 120.0],
                 N: np.array = [500, 200],
                 eta: float = 0.3,
                 a: float = 2.280,
                 a_f: float = 1.685):

        self.mode = mode
        if case == "both":
            cases = ["ED", "ES"]
        elif case == "all":
            cases = ["ED","ES","unloaded_ED"]
        else:
            cases = [case]

        self.cases = cases
        self.PLV = PLV
        self.PRV = PRV
        self.Ta = Ta
        self.N = N
        self.eta = eta
        self.a = a
        self.a_f = a_f
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.solver_path = solver_path

    def run(self, max_retries: int = 3):
        for case in self.cases:
            sys.path.append(self.solver_path)
            import run_simulation_full

            retries = 0
            current_N = self.N

            while retries <= max_retries:
                try:
                    print(f"Running with N = {current_N}")
                    run_simulation_full.main(
                        mode=self.mode,
                        case=case,
                        datadir=self.data_dir,
                        resultsdir=self.results_dir,
                        PLV=self.PLV,
                        PRV=self.PRV,
                        TA=self.Ta,
                        N=current_N,
                        eta=self.eta,
                        a=self.a,
                        a_f=self.a_f
                    )
                    print("✅ Simulation converged successfully.")
                    break  # Exit retry loop on success

                except RuntimeError as e:
                    # Check if error is convergence-related
                    if "convergence" in str(e).lower() or "did not converge" in str(e).lower():
                        print(f"⚠️ Solver failed to converge with N = {current_N}. Retrying with N = {2 * current_N}...")
                        current_N =[current_N[0] * 2, current_N[1]]
                        retries += 1
                    else:
                        # Some other error; re-raise
                        print(f"Solver cannot converge with N = {current_N}- skipping this case.")
                        break

            else:
                print(f"❌ Failed to converge after {max_retries} retries with max N = {current_N}.")
                with open("error_log.txt", "a") as log_file:
                        patient = self.data_dir.split("/data-full/")[0]
                        log_file.write(f"Failed to converge {patient} with PLV: {self.PLV},PRV: {self.PRV}, a: {self.a}, a_f: {self.a_f} after {max_retries + 1} attempts.\n")
                print("False")
        print("True")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cardiac mechanics simulation.")
    parser.add_argument("--case", type=str, default="unloaded_ED", help="Simulation case")
    parser.add_argument("--PLV", type=parse_array, default=[20.0, 30.0], help="Left ventricular pressure")
    parser.add_argument("--PRV", type=parse_array, default=[4.0, 8.0], help="Right ventricular pressure")
    parser.add_argument("--Ta", type=parse_array, default=[0.0, 120.0], help="Active stress time constant")
    parser.add_argument("--eta", type=float, default=0.3, help="Active stress scaling factor")
    parser.add_argument("--N", type=np.array, default=[100, 150], help="Number of time steps for simulation")
    parser.add_argument("--a", type=float, default=2.280, help="Material parameter a")
    parser.add_argument("--a_f", type=float, default=1.685, help="Material parameter a_f")
    parser.add_argument("--mode", type=int, default=-1, help="Simulation mode")
    parser.add_argument("--patient_id", type=int, default=0, help="Patient ID")
    parser.add_argument("--solver_path", type=str, default="../clones/rk-sscp25-deep-learning-cardiac-mechanics", help="Path to solver")

    args = parser.parse_args()
    patient_id = args.patient_id

    sim = Simulation(mode=args.mode, results_dir=Path(f"patient_{patient_id}/results-full"), 
                     data_dir=Path(f"patient_{patient_id}/data-full"), solver_path=args.solver_path,
                     PLV=args.PLV, PRV=args.PRV, Ta=args.Ta, N=args.N,
                     a=args.a, a_f=args.a_f)
    sim.run()