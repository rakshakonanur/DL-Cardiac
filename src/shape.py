import pandas as pd
import scipy.io
import numpy as np
import os, sys
from argparse import ArgumentParser
from typing import NamedTuple
# import dolfinx as dfx
from structlog import get_logger
from importlib.metadata import metadata

meta = metadata("cardiac-geometriesx")
__version__ = meta["Version"]
logger = get_logger()


class Points(NamedTuple):
    ED: np.ndarray
    ES: np.ndarray
    unloaded_ED: np.ndarray

def load_pca(path: str):
    """
    Load PCA data from a .mat file.
    Args:
        path (str): Path to the .mat file containing PCA data. If none, it will use the default path.
    Returns:
        pca (dict): Dictionary containing PCA data including 'MU', 'COEFF', 'and 'LATENT'.
    """

    if path is None:
        path = "../refs/BioBank_EDES_200.mat"

    # Load UKB EDES atlas from .mat file
    mat_data = scipy.io.loadmat(path) #Replace with the actual path to your library folder
    pca = mat_data['pca200'][0, 0]
    return pca

# Functions to reconstruct shape given PC scores
def reconstruct_shape(score, atlas, num_scores = 25):
    d = score * np.sqrt(atlas["LATENT"][0:num_scores]).T
    shape = atlas["MU"] + np.matmul(d, atlas["COEFF"][:, :num_scores].T)
    print(f"Reconstructed shape with {num_scores} scores has size {shape.shape}.", flush=True)
    return shape.T

# Extract ED phase as (N, 3) mesh
def get_ED_mesh_from_shape(shape):
    N = len(shape)
    return shape[:N // 2].reshape(-1, 3)

def get_ES_mesh_from_shape(shape):
    N = len(shape)
    return shape[N // 2:].reshape(-1, 3)

def load_connectivity(path: str):
    """
    Load connectivity indices from a text file.
    Args:
        path (str): Path to the connectivity indices file.
    Returns:
        et_indices (np.ndarray): Array of connectivity indices.
    """
    if path is None:
        path = "../clones/biv-me/src/model/ETIndicesSorted.txt"

    et_indices = np.loadtxt(path).astype(int) - 1  # Convert MATLAB indexing to Python
    return et_indices



def load_virtual_cohort(path: str):
    """
    Load virtual cohort data from an Excel file.
    Args:
        path (str): Path to the virtual cohort data file.
    Returns:
        virtual_cohort (pd.DataFrame): DataFrame containing the virtual cohort data.
    """
    if path is None:
        path = "../refs/virtual_cohort_data.xlsx"

    virtual_cohort = pd.read_excel(path)
    # linear equation found from applying the Garg et al. equation to UKB data
    virtual_cohort["Unloaded Volume"] = 0.5025 * virtual_cohort["LV_EDV"] + 5.7574
    virtual_cohort.to_excel(path, index=False)
    return virtual_cohort


if __name__ == "__main__":
    parser = ArgumentParser(description="Load PCA, connectivity indices, and virtual cohort data.")
    parser.add_argument("--pca_path", type=str, default=None, help="Path to the PCA .mat file.")
    parser.add_argument("--connectivity_path", type=str, default=None, help="Path to the connectivity indices file.")
    parser.add_argument("--virtual_cohort_path", type=str, default=None, help="Path to the virtual cohort data file.")
    
    args = parser.parse_args()

    pca = load_pca(args.pca_path)
    et_indices = load_connectivity(args.connectivity_path)
    virtual_cohort = load_virtual_cohort(args.virtual_cohort_path)

    # prompt for a specific patient
    patient_id = int(input("Enter patient ID (0-based index): "))
    pc_columns = [col for col in virtual_cohort.columns if col.startswith('PC')]
    pc_scores = virtual_cohort[pc_columns]

    patient = pc_scores.loc[patient_id].to_numpy()  # Get the specified patient's PC scores
    patient_shape = reconstruct_shape(patient, pca)
    patient_ed = get_ED_mesh_from_shape(patient_shape)
    patient_es = get_ES_mesh_from_shape(patient_shape)

    
