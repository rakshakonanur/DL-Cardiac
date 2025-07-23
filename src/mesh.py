import sys, os
from pathlib import Path
from unittest import case
import numpy as np
import pandas as pd
from rich import inspect
import scipy.io
import time
from argparse import ArgumentParser
import shape, volume
from typing import NamedTuple
from mpi4py import MPI
import json
from typing import Literal
from importlib.metadata import metadata
import dolfinx
from structlog import get_logger
import datetime
import cardiac_geometries as cgx

logger = get_logger()
meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]
comm = MPI.COMM_WORLD

def set_path(ukb_path: str):
    # Path to the ukb-atlas/src folder : important to download my fork of the UKB atlas
    if ukb_path is None:
        ukb_path =  "../clones/rk-ukb-atlas/src"

    sys.path.insert(0, ukb_path)
    import ukb, cardiac_geometries as cgx
    from ukb import atlas, surface, mesh, clip
    return ukb, atlas, surface, mesh, clip
class Mesh():
    def __init__(self, ukb_atlas_path, pca_path, connectivity_path, virtual_cohort_path = None, patient_id=None):

        self.ukb, self.atlas, self.surface, self.mesh, self.clip = set_path(ukb_atlas_path)
        pca = shape.load_pca(pca_path)
        connectivity_indices = shape.load_connectivity(connectivity_path)
        virtual_cohort = shape.load_virtual_cohort(virtual_cohort_path)
        pc_columns = [col for col in virtual_cohort.columns if col.startswith('PC')]
        pc_scores = virtual_cohort[pc_columns]
        patient = list(pc_scores.loc[patient_id,:])
        patient = pc_scores.loc[patient_id].to_numpy()  # Example: get the specified patient's PC scores
        patient_shape = shape.reconstruct_shape(score = patient, atlas = pca)
        self.patient_ed = shape.get_ED_mesh_from_shape(patient_shape)
        self.patient_es = shape.get_ES_mesh_from_shape(patient_shape)
        # test calculating the volumes
        vol = volume.find_volume(self.patient_ed)

        target_volume = virtual_cohort["Unloaded Volume"].iloc[patient_id]
        target_mass = vol["lv_mass"]

        print(f"Target LV Volume: {target_volume:.2f} mL")
        print(f"Original LV Mass:    {target_mass:.2f} g\n")

        # store test which is a dictionary as a dataframe
        df = pd.DataFrame([vol])
        print(df)

        # prompt user for optimization choice
        choice = input("Do you want to optimize the PC scores for volume (v) or mass + volume (m)? ").strip().lower()

        if choice == "m":
            # Optimize for mass + volume
            unloaded_pc_scores, volume_history, mass_history = volume.find_unloaded_pcs_by_gradient_descent_mass_volume(
                initial_pc_scores=patient,
                target_volume=target_volume,
                original_lv_mass=target_mass,
                reconstruct_shape=shape.reconstruct_shape,
                get_ED_mesh_from_shape=shape.get_ED_mesh_from_shape,
                find_volume=volume.find_volume,
                pca=pca,
                learning_rate=0.05,
                max_iterations=200,
                tolerance=1,
                epsilon=0.5,
                mass_constraint_weight=0.1  # tune this parameter, higher values put more emphasis on mass constraint (higher values will converge slower)
            )
            pass
        elif choice == "v":
            # Optimize for volume only
            unloaded_pc_scores = volume.find_unloaded_pcs_by_gradient_descent_volume(
                initial_pc_scores=patient,
                target_volume=target_volume,
                reconstruct_shape=shape.reconstruct_shape,
                get_ED_mesh_from_shape=shape.get_ED_mesh_from_shape,
                find_volume=volume.find_volume,
                pca = pca,
                learning_rate=0.05,
                max_iterations=200,
                tolerance= 0.1,
                epsilon=0.5)
        else:
            print("Invalid choice. Please enter 'v' or 'm'.")

        # Reconstruct the unloaded shape from the optimized PC scores
        unloaded_shape = shape.reconstruct_shape(unloaded_pc_scores, pca)
        self.unloaded_ed = shape.get_ED_mesh_from_shape(unloaded_shape)


    def set_output_directory(self, mode):
        self.mode = mode
        self.outdir = Path("output/data-full") / f"mode_{mode}"
        self.outdir.mkdir(parents=True, exist_ok=True)

    def generate_points(self):
        unwanted_nodes = (5630, 5655, 5696, 5729)
        self.points = shape.Points(
            ED=np.delete(self.patient_ed, unwanted_nodes, axis=0),
            ES=np.delete(self.patient_es, unwanted_nodes, axis=0),
            unloaded_ED=np.delete(self.unloaded_ed, unwanted_nodes, axis=0),
        )

    def generate_surfaces(self, case="all"):
        self.ukb.surface.main(case=case, folder=self.outdir, custom_points=self.points)

    def generate_surface_mesh(self, case="all",char_length_max=5.0, char_length_min=5.0, clipped=False):
        self.ukb.mesh.main(
            folder=self.outdir,
            case=case,
            char_length_max=char_length_max,
            char_length_min=char_length_min,
            clipped=clipped,
        )

    def generate_volume_mesh(self, std = 1.5, case="all", char_length_max =5.0, 
                             char_length_min = 5.0, fiber_angle_endo=60.0, fiber_angle_epi=-60.0, 
                             fiber_space="DG_0", clipped=False, create_fibers=True):
        if case == "both":
            cases = ["ED", "ES"]
        elif case == "all":
            cases = ["ED","ES","unloaded_ED"]
        else:
            cases = [case]

        for case in cases:
            geometry = cgx.utils.gmsh2dolfin(comm=comm, msh_file=self.outdir / case / "mesh.msh")
            if comm.rank == 0:
                (self.outdir / case /"markers.json").write_text(
                    json.dumps(geometry.markers, default=cgx.utils.json_serial)
                )
                (self.outdir / case /"info.json").write_text(
                    json.dumps(
                        {
                            "mode": self.mode,
                            "std": std,
                            "case": case,
                            "char_length_max": char_length_max,
                            "char_length_min": char_length_min,
                            "fiber_angle_endo": fiber_angle_endo,
                            "fiber_angle_epi": fiber_angle_epi,
                            "fiber_space": fiber_space,
                            "cardiac_geometry_version": __version__,
                            "mesh_type": "ukb",
                            "clipped": clipped,
                            "timestamp": datetime.datetime.now().isoformat(),
                        }
                    )
                )

            if create_fibers:
                try:
                    import ldrb
                except ImportError as ex:
                    msg = (
                        "To create fibers you need to install the ldrb package "
                        "which you can install with pip install fenicsx-ldrb"
                    )
                    raise ImportError(msg) from ex

                markers = cgx.mesh.transform_markers(geometry.markers, clipped=clipped)
                system = ldrb.dolfinx_ldrb(
                    mesh=geometry.mesh,
                    ffun=geometry.ffun,
                    markers=markers,
                    alpha_endo_lv=fiber_angle_endo,
                    alpha_epi_lv=fiber_angle_epi,
                    beta_endo_lv=0,
                    beta_epi_lv=-0,
                    fiber_space=fiber_space,
                )

                cgx.fibers.utils.save_microstructure(
                    mesh=geometry.mesh,
                    functions=(system.f0, system.s0, system.n0),
                    outdir=self.outdir/case,
                )

                for k, v in system._asdict().items():
                    if v is None:
                        continue
                    if fiber_space.startswith("Q"):
                        # Cannot visualize Quadrature spaces yet
                        continue

                    logger.debug(f"Write {k}: {v}")
                    with dolfinx.io.VTXWriter(comm, self.outdir / case / f"{k}-viz.bp", [v], engine="BP4") as vtx:
                        vtx.write(0.0)

            geo = cgx.geometry.Geometry.from_folder(comm=comm, folder=self.outdir / case)


if __name__ == "__main__":
    parser = ArgumentParser(description="Load PCA, connectivity indices, and virtual cohort data.")
    parser.add_argument("--pca_path", type=str, default=None, help="Path to the PCA .mat file.")
    parser.add_argument("--connectivity_path", type=str, default=None, help="Path to the connectivity indices file.")
    parser.add_argument("--virtual_cohort_path", type=str, default=None, help="Path to the virtual cohort data file.")

    # prompt for a specific patient
    patient_id = int(input("Enter patient ID (0-based index): "))
    
    args = parser.parse_args()
    mesh = Mesh(
        ukb_atlas_path="../clones/rk-ukb-atlas/src",
        pca_path=args.pca_path,
        connectivity_path=args.connectivity_path,
        virtual_cohort_path=args.virtual_cohort_path,
        patient_id=patient_id
    )
    mesh.set_output_directory(mode="-1")
    mesh.generate_points()
    mesh.generate_surfaces()
    mesh.generate_surface_mesh()
    mesh.generate_volume_mesh()


