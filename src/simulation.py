import sys
from pathlib import Path

class Simulation:
    def __init__(self, mode: int = -1, case = "unloaded_ED",
                 results_dir: str = "output/results-full", 
                 data_dir: str = "output/data-full/", 
                 solver_path: str = "../clones/rk-sscp25-deep-learning-cardiac-mechanics"):

        self.mode = mode
        if case == "both":
            cases = ["ED", "ES"]
        elif case == "all":
            cases = ["ED","ES","unloaded_ED"]
        else:
            cases = [case]
        self.cases = cases
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.solver_path = solver_path

    def run(self):
        for case in self.cases:
            sys.path.append(self.solver_path)
            import run_simulation_full
            run_simulation_full.main(mode = self.mode, case = case, datadir=self.data_dir, resultsdir = self.results_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run cardiac mechanics simulation.")
    parser.add_argument("--mode", type=int, default=-1, help="Simulation mode")
    parser.add_argument("--results_dir", type=str, default="output/results-full", help="Directory for results")
    parser.add_argument("--data_dir", type=str, default="output/data-full/", help="Directory for data")
    parser.add_argument("--solver_path", type=str, default="../clones/rk-sscp25-deep-learning-cardiac-mechanics", help="Path to solver")

    args = parser.parse_args()

    sim = Simulation(mode=args.mode, results_dir=Path(args.results_dir), 
                     data_dir=Path(args.data_dir), solver_path=args.solver_path)
    sim.run()