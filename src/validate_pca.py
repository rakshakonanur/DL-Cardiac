import numpy as np
import h5py as h5
from pathlib import Path
import pandas as pd
import meshio
import dolfinx
import adios4dolfinx
from mpi4py import MPI
import cardiac_geometries
import scipy
from mesh import Mesh
import shape
import volume
import sys
from adios2 import Stream, FileReader
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
import pyvista as pv
import logging
import shape, mesh
from mesh import Mesh

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

"""
sample_counts = {
        "LV": 1500 - 0,
        "RV": 3224 - 1500,
        "EPI": 5582 - 3224,
        "MV": 5631 - 5582,
        "AV": 5656 - 5631,
        "TV": 5697 - 5656,
        "PV": 5730 - 5697,
        "EPI_2": 5810 - 5730 ---> RVFW
    }

surfaces = {
    "LV": Surface("LV", [(0, 1500)], [(0, 3072)]),
    "RV": Surface(
        "RV",
        [(1500, 2165), (2165, 3224)],
        [(3072, 4480)],
    ),
    "RVFW": Surface(
        "RVFW",
        [(5729, 5808)],
        [(4480, 6752)],
    ),
    "EPI": Surface("Epi", [(3224, 5582)], [(6752, 11616)]),
    "MV": Surface("MV", [(5582, 5629)], [(6752, 11616)]),
    "AV": Surface("AV", [(5630, 5653)], [(6752, 11616)]),
    "TV": Surface("TV", [(5654, 5693)], [(6752, 11616)]),
    "PV": Surface("PV", [(5694, 5729)], [(6752, 11616)]),
}
"""
def project_patient_to_atlas(patient_shape_flat, atlas, numModes = 10):

    MU = np.transpose(atlas["MU"]) # mean shape
    COEFF = np.transpose(atlas["COEFF"]) # PCA eigenvectors (basis)
    LATENT = np.transpose(atlas["LATENT"]) # PCA eigenvalues (variances)
    patient3D = patient_shape_flat.reshape(-1, 3)
    patient1Daligned = patient3D.flatten()
    patient1Dnormalized = patient1Daligned - np.transpose(MU) # center the patient shape for PCA projection
    projectedScores = np.dot(patient1Dnormalized, np.transpose(COEFF[0:numModes,:])) / np.sqrt(np.array(np.transpose(LATENT[0,0:numModes]))) 
    # project the patient shape onto the PCA basis by taking the dot product with the first numModes eigenvectors and normalizing by the square root of the eigenvalues
    return projectedScores

def resample(
    mode: int = -1,
    datadir: Path = Path("/mnt/c/Users/aleja/Desktop/Simula 2025/DL-Cardiac/src/test/patient_0/data-full"),
    resultsdir: Path = Path("/mnt/c/Users/aleja/Desktop/Simula 2025/DL-Cardiac/src/test/patient_0/results-full"),
    bpl: str = "/mnt/c/Users/aleja/Desktop/Simula 2025/DL-Cardiac/src/test/patient_0/results-full/mode_-1/unloaded_ED/unloaded_to_ED_PLVED_20.00__PRVED_4.00__TA_0.0__a_2.28__af_1.69.bp",
    case: str = "ED"
):
    print("Starting resample...", flush=True)
    geodir = Path(datadir) / f"mode_{mode}" / "unloaded_ED"
    outdir = Path(resultsdir) / f"mode_{mode}" / "unloaded_ED"
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {outdir}", flush=True)

    # Load geometry and displacement function (geo, u)
    comm = MPI.COMM_WORLD
    print("Loading geometry...", flush=True)
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)

    V = dolfinx.fem.functionspace(geo.mesh, ("CG", 2, (3,)))
    u = dolfinx.fem.Function(V)

    from adios2 import FileReader
    print(f"Reading BPL file: {bpl}", flush=True)
    with FileReader(bpl) as ibpFile:
        blocks_info = ibpFile.all_blocks_info("f")[0]
        blocks_info_sorted = sorted(blocks_info, key=lambda b: int(b["BlockID"]), reverse=True)

        f_list = []
        geometry_list = []

        for block in blocks_info_sorted:
            block_id = int(block["BlockID"])
            f_list.append(ibpFile.read("f", block_id=block_id))
            geometry_list.append(ibpFile.read("geometry", block_id=block_id))

        f = np.vstack(f_list)
        geometry = np.vstack(geometry_list)

        print("f.shape:", f.shape, flush=True)
        print("geometry.shape:", geometry.shape, flush=True)

    dof_coords = V.tabulate_dof_coordinates().reshape((-1, 3))
    block_size = V.dofmap.index_map_bs

    print("Building KDTree for geometry matching...", flush=True)
    tree = cKDTree(geometry)
    dist, vert_index = tree.query(dof_coords, distance_upper_bound=1e-12)
    if np.any(dist > 1e-12):
        raise RuntimeError("Some DOFs could not be matched to geometry points")
    print("Geometry matching done.", flush=True)

    values_flat = np.zeros_like(u.x.array)
    for dof_id, vtx_id in enumerate(vert_index):
        comp = dof_id % block_size
        values_flat[dof_id] = f[vtx_id, comp]

    u.x.array[:] = f[vert_index, :].reshape(-1)
    u.x.scatter_forward()
    print("Updated function u with displacement values.", flush=True)
    return u.x.array[:].reshape(-1,3),dof_coords, geodir

def deform(patient_id):

    mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
    pca = mat_data['pca200'][0, 0]
    patient = pd.read_csv(f"test/patient_{patient_id}/unloaded_pc_scores_patient_{patient_id}.csv").to_numpy()
    patient_shape = shape.reconstruct_shape(score = patient.ravel()[:25], atlas = pca, num_scores=25)
    print("PC scores for patient", patient_id, ":", patient.ravel(), flush=True)
    # print(f"Patient {patient_id} shape has {len(patient_shape)} points.", flush=True)
    unloaded = shape.get_ED_mesh_from_shape(patient_shape)
    ES = shape.get_ES_mesh_from_shape(patient_shape)
    
    return unloaded, ES


def main(patient_ED, patient_ES, unloaded):
    print("ED shape:", patient_ED.shape)
    print("ES shape:", patient_ES.shape)
    print("Unloaded shape:", unloaded.shape)

    def set_path(ukb_path: str):
    # Path to the ukb-atlas/src folder : important to download my fork of the UKB atlas
        if ukb_path is None:
            ukb_path =  "../clones/rk-ukb-atlas/src"

        sys.path.insert(0, ukb_path)
        import ukb, cardiac_geometries as cgx
        from ukb import atlas, surface, mesh, clip
        return ukb, atlas, surface, mesh, clip

    def generate_points(patient_ED, patient_ES, unloaded):
        unwanted_nodes = (5630, 5655, 5696, 5729)
        points = shape.Points(
            ED=np.delete(patient_ED, unwanted_nodes, axis=0),
            ES=np.delete(patient_ES, unwanted_nodes, axis=0),
            unloaded_ED=np.delete(unloaded, unwanted_nodes, axis=0),
        )
        return points

    def generate_surfaces(points):
        ukb.surface.main(case="both", folder=outdir, custom_points=points)

    ukb, atlas, surface, mesh, clip = set_path("../clones/rk-ukb-atlas/src")
    points = generate_points(patient_ED, patient_ES, unloaded)
    print("Points generated for all labels.", flush=True)
    generate_surfaces(points)
    print("Surfaces generated for all labels.", flush=True)

if __name__ == "__main__":

    patient_id = 0
    ED_file = f"../datasets/updated_final/1/patient_{patient_id}/results-full/mode_-1/unloaded_ED/unloaded_to_ED_PLVED_1.20__PRVED_0.53__TA_0.0__eta_0.2__a_1.28__af_1.69.bp"
    u_ED, coords, geodir = resample(bpl=ED_file, mode=-1, datadir=Path(f"../datasets/updated_final/1/patient_{patient_id}/data-full"), resultsdir=Path(f".../datasets/updated_final/1/patient_{patient_id}/results-full"), case="ED")
    points_ED, ES = deform(patient_id=patient_id)

    outdir = Path(f"../datasets/updated_final/1/patient_{patient_id}/results-full/mode_-1/unloaded_ED")
    mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
    pca = mat_data['pca200'][0, 0]

    example_1d_ed = points_ED.flatten()
    example_flattened = np.concatenate((example_1d_ed, ES.flatten()))

    projectedScores = project_patient_to_atlas(example_flattened, pca, numModes=25)
    
    print("Projected scores:", projectedScores[0])

    patient_shape = shape.reconstruct_shape(score = projectedScores, atlas = pca, num_scores=25)
    patient_ed = shape.get_ED_mesh_from_shape(patient_shape)
    patient_es = shape.get_ES_mesh_from_shape(patient_shape)
    vol_ed = volume.find_volume(patient_ed)
    print("ED Volume:", vol_ed)
    vol_es = volume.find_volume(patient_es)
    print("ES Volume:", vol_es)

     # Create PolyData
    point_cloud = pv.PolyData(patient_ed)

    # Save as VTP
    point_cloud.save(f"unloaded_original_ED.vtp")

    print("Saved unloaded.vtp successfully!")

     # Create PolyData
    point_cloud = pv.PolyData(patient_es)

    # Save as VTP
    point_cloud.save(f"unloaded_original_ES.vtp")

    print("Saved unloaded.vtp successfully!")


    # Flatten into one dict
    flattened = {f"ED_{k}": float(v) for k, v in vol_ed.items()}
    flattened.update({f"ES_{k}": float(v) for k, v in vol_es.items()})

    # print(flattened)

    # def set_path(ukb_path: str):
    # # Path to the ukb-atlas/src folder : important to download my fork of the UKB atlas
    #     if ukb_path is None:
    #         ukb_path =  "../clones/rk-ukb-atlas/src"

    #     sys.path.insert(0, ukb_path)
    #     import ukb, cardiac_geometries as cgx
    #     from ukb import atlas, surface, mesh, clip
    #     return ukb, atlas, surface, mesh, clip

    # ukb, atlas, surface, mesh, clip = set_path("../clones/rk-ukb-atlas/src")
    # unwanted_nodes = (5630, 5655, 5696, 5729)
    # points = shape.Points(
    #     ED=np.delete(patient_ed, unwanted_nodes, axis=0),
    #     ES=np.delete(patient_es, unwanted_nodes, axis=0),
    #     unloaded_ED=np.delete(patient_ed, unwanted_nodes, axis=0),
    # )

    ### LOOK AT ED/ FOR UNDEFORMED MESHES TO VALIDATE