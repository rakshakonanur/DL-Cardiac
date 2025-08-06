import sys
from pathlib import Path
import numpy as np

class Simulation:
    def __init__(self, mode: int = -1, case = "unloaded_ED",
                 results_dir: str = "output/results-full", 
                 data_dir: str = "output/data-full/", 
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

    def run(self):
        for case in self.cases:
            sys.path.append(self.solver_path)
            import run_simulation_full
            print("Material parameters:", self.a, self.a_f)
            run_simulation_full.main(mode = self.mode, case = case, datadir=self.data_dir, resultsdir = self.results_dir,
                                     PLV = self.PLV, PRV = self.PRV, TA = self.Ta, N = self.N, eta = self.eta,
                                     a = self.a, a_f = self.a_f)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cardiac mechanics simulation.")
    parser.add_argument("--case", type=str, default="unloaded_ED", help="Simulation case")
    parser.add_argument("--PLV", type=np.array, default=[20.0, 30.0], help="Left ventricular pressure")
    parser.add_argument("--PRV", type=np.array, default=[4.0, 6.0], help="Right ventricular pressure")
    parser.add_argument("--Ta", type=np.array, default=[0.0, 120.0], help="Active stress time constant")
    parser.add_argument("--eta", type=float, default=0.3, help="Active stress scaling factor")
    parser.add_argument("--N", type=np.array, default=[400, 200], help="Number of time steps for simulation")
    parser.add_argument("--a", type=float, default=2.280, help="Material parameter a")
    parser.add_argument("--a_f", type=float, default=1.685, help="Material parameter a_f")
    parser.add_argument("--mode", type=int, default=-1, help="Simulation mode")
    parser.add_argument("--results_dir", type=str, default="output/results-full", help="Directory for results")
    parser.add_argument("--data_dir", type=str, default="output/data-full/", help="Directory for data")
    parser.add_argument("--solver_path", type=str, default="../clones/rk-sscp25-deep-learning-cardiac-mechanics", help="Path to solver")

    args = parser.parse_args()

    sim = Simulation(mode=args.mode, results_dir=Path(args.results_dir), 
                     data_dir=Path(args.data_dir), solver_path=args.solver_path,
                     PLV=args.PLV, PRV=args.PRV, Ta=args.Ta, N=args.N,
                     a=args.a, a_f=args.a_f)
    sim.run()