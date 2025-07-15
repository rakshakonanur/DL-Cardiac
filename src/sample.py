import pandas as pd
import pyvista as pv
import scipy.io
import numpy as np
from pathlib import Path
import os
import json
from argparse import ArgumentParser
import logging
from typing import NamedTuple, Literal
import meshio
from mpi4py import MPI
import scipy.io
import datetime
import dolfinx
from structlog import get_logger
from importlib.metadata import metadata
meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]

logger = get_logger()

import sys
import os

# Path to the ukb-atlas/src folder
ukb_atlas_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../clones/rk-ukb-atlas/src"))
sys.path.insert(0, ukb_atlas_path)

comm = MPI.COMM_WORLD
mode = 0 # corresponds to a custom mode in the UKB EDES atlas
# Load UKB EDES atlas from .mat file
mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat") #Replace with the actual path to your library folder
pca = mat_data['pca200'][0, 0]

# functions to reconstruct shape given PC scores
def reconstruct_shape(score, atlas, num_scores = 25):
    d = score * np.sqrt(atlas["LATENT"][0:num_scores]).T
    shape = atlas["MU"] + np.matmul(d, atlas["COEFF"][:, :num_scores].T)
    return shape.T

# Extract ED phase as (N, 3) mesh
def get_ED_mesh_from_shape(shape):
    N = len(shape)
    return shape[:N // 2].reshape(-1, 3)

def get_ES_mesh_from_shape(shape):
    N = len(shape)
    return shape[N // 2:].reshape(-1, 3)

class Points(NamedTuple):
    ED: np.ndarray
    ES: np.ndarray

# import necessary ET Indices File- this is the same as the connectivity.txt file in the UKB atlas
et_indices = np.loadtxt("../clones/biv-me/src/model/ETIndicesSorted.txt").astype(int) - 1 #matlab indexing to python
print(et_indices)
# Import virtual cohort sheet
virtual_cohort = pd.read_excel("../refs/virtual_cohort_data.xlsx")
pc_columns = [col for col in virtual_cohort.columns if col.startswith('PC')]
pc_scores = virtual_cohort[pc_columns]

patient = pc_scores.loc[1].to_numpy()  # Example: get the first patient's PC scores
patient_shape = reconstruct_shape(patient, pca)
patient_ed = get_ED_mesh_from_shape(patient_shape)
patient_es = get_ES_mesh_from_shape(patient_shape)

import sys
sys.path.append("../clones/rk-ukb-atlas/src/ukb")

import ukb
from ukb import atlas, surface, mesh, clip
out_dir = f"output/data-full/mode_{mode}"
outdir = Path("output/data-full")  / f"mode_{mode}"
outdir.mkdir(parents=True, exist_ok=True)
# custom_surface = ukb.surface.Surface()
unwanted_nodes = (5630, 5655, 5696, 5729)
points = Points(
    ED=np.delete(patient_ed, unwanted_nodes, axis=0),
    ES=np.delete(patient_es, unwanted_nodes, axis=0),
)

ukb.surface.main(folder=outdir, custom_points=points) # generates the various surfaces

ukb.mesh.main(
    folder=out_dir,
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    clipped=False,
)

sys.path.append("../clones/cardiac-geometriesx/src")
import cardiac_geometries as cgx

geometry = cgx.utils.gmsh2dolfin(comm=comm, msh_file = out_dir + "/ED.msh")
std = 1.5
case = "ED"
char_length_max = 5.0
char_length_min = 5.0
fiber_angle_endo = 60.0
fiber_angle_epi = -60.0
fiber_space = "DG_0"
clipped = False
create_fibers = True


if comm.rank == 0:
    (outdir / "markers.json").write_text(
        json.dumps(geometry.markers, default=cgx.utils.json_serial)
    )
    (outdir / "info.json").write_text(
        json.dumps(
            {
                "mode": mode,
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
        outdir=outdir,
    )

    for k, v in system._asdict().items():
        if v is None:
            continue
        if fiber_space.startswith("Q"):
            # Cannot visualize Quadrature spaces yet
            continue

        logger.debug(f"Write {k}: {v}")
        with dolfinx.io.VTXWriter(comm, outdir / f"{k}-viz.bp", [v], engine="BP4") as vtx:
            vtx.write(0.0)

geo = cgx.geometry.Geometry.from_folder(comm=comm, folder=outdir)

sys.path.append("/Users/rakshakonanur/Documents/SSCP25/Cardiac_Mechanics_DL/clones/sscp25-deep-learning-cardiac-mechanics")

results_dir = Path("output/results-full")
results_dir.mkdir(parents=True, exist_ok=True)
import run_simulation_full
results_dir = "output/results-full/"
run_simulation_full.main(mode = mode, datadir="output/data-full/", resultsdir =results_dir)