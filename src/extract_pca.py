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

def procrustes(X,Y,scaling=True,reflection='best'):
    
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
    d, Z, [tform] = procrustes(X, Y)

    Parameters
    ----------
    X : MEAN shape (n, m)
    Y : PATIENT shape (n, m)

    """
    n,m = X.shape # n = number of points, m = dimension
    ny,my = Y.shape # ny = number of points, my = dimension
    muX = X.mean(0) # compute the mean of X along each dimension
    muY = Y.mean(0) # compute the mean of Y along each dimension
    X0 = X - muX # center the data around the origin by subtracting the mean
    Y0 = Y - muY # center the data around the origin by subtracting the mean
    ssX = (X0**2.).sum() # finds the total variance (sum of squares) in X
    ssY = (Y0**2.).sum() # finds the total variance (sum of squares) in Y
    normX = np.sqrt(ssX) # Euclidean norm of X
    normY = np.sqrt(ssY) # Euclidean norm of Y
    X0 /= normX # normalize X0 by dividing by its norm (makes it scale invariant)
    Y0 /= normY # normalize Y0 by dividing by its norm (makes it scale invariant)
    if my < m: # if Y has fewer dimensions than X, pad Y0 with zeros
        Y0 = np.concatenate((Y0,np.zeros(n, m-my)),0)
    A = np.dot(X0.T,Y0) # compute the covariance matrix between centered, normalized shapes
    U,s,Vt = np.linalg.svd(A,full_matrices=True)
    V = Vt.T
    T = np.dot(V,U.T) # compute the optimal rotation using singular value decomposition
    if reflection != 'best': # if reflection is false, ensures that the transformation is not a rotation
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V,U.T)
    traceTA = s.sum() # sum of singular values = how well the aligned shapes are
    if scaling:
        b = traceTA*normX/normY # compute the scaling factor
        d = 1-traceTA**2 # computes dissimilarity (how different the shapes are)
        Z = normX*traceTA*np.dot(Y0,T)+muX # reconstructs the aligned shape Z
    else:
        b = 1
        d = 1+ssY/ssX-2*traceTA*normY/normX
        Z = normY*np.dot(Y0,T)+muX
    if my<m:
        T = T[:my,:] # truncates if it was padded earlier
    c = muX-b*np.dot(muY,T) # compute the translation vector
    # Transformation values 
    tform = {'rotation':T,'scale':b,'translation':c}
    print(f"Procrustes analysis: d={d:.4f}, scale={b:.4f}, translation={c}")
    #return d, Z, tform
    return T, c

def project_patient_to_atlas(patient_shape_flat, atlas, numModes = 10):

    MU = np.transpose(atlas["MU"]) # mean shape
    COEFF = np.transpose(atlas["COEFF"]) # PCA eigenvectors (basis)
    LATENT = np.transpose(atlas["LATENT"]) # PCA eigenvalues (variances)
    patient3D = patient_shape_flat.reshape(-1, 3)
    mean3D = np.array(np.transpose(MU)).reshape(-1, 3)

    # Procrustes alignment
    # T, c = procrustes(mean3D, patient3D, scaling=True, reflection='best')
    # patient3Daligned = np.dot(patient3D, T) + c # rotates and translates the patient shape to align with the mean shape
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

def deform(outdir, u, geodir, coords, case, patient_id):

    mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
    pca = mat_data['pca200'][0, 0]
    patient = pd.read_csv(f"test/patient_{patient_id}/unloaded_pc_scores_patient_{patient_id}.csv").to_numpy()
    patient_shape = shape.reconstruct_shape(score = patient.ravel()[:25], atlas = pca, num_scores=25)
    print("PC scores for patient", patient_id, ":", patient.ravel(), flush=True)
    # print(f"Patient {patient_id} shape has {len(patient_shape)} points.", flush=True)
    unloaded = shape.get_ED_mesh_from_shape(patient_shape)
    ES = shape.get_ES_mesh_from_shape(patient_shape)
    # print(f"Unloaded shape for patient {patient_id} has {len(unloaded)} points.", flush=True)
    connectivity = shape.load_connectivity("../clones/biv-me/src/model/ETIndicesSorted.txt")
    points, cells = {}, {}

    # write unloaded coordinates to txt file
    np.savetxt(f"unloaded_{case}.txt", unloaded, fmt='%.6f', comments='')

    # connectivity: array of shape (num_triangles, 3)
    connectivity = np.array(connectivity, dtype=int)

    # Build KDTree once on source coordinates
    tree = cKDTree(coords)

    # Query nearest neighbors for all points in unloaded mesh
    distances, indices = tree.query(unloaded, k=4)

    # Compute inverse-distance weights
    weights = 1.0 / (distances + 1e-12)
    weights /= weights.sum(axis=1, keepdims=True)

    # Interpolate displacement u at unloaded points
    interpolated_u = np.sum(u[indices] * weights[..., np.newaxis], axis=1)

    # Apply displacement to unloaded coordinates
    unloaded_deformed = unloaded + interpolated_u
    # Create PolyData
    point_cloud = pv.PolyData(unloaded_deformed)

    # Save as VTP
    point_cloud.save(f"unloaded_deformed_{case}.vtp")

    print("Saved unloaded.vtp successfully!")

    # deform both meshes- code to test stl creation from point cloud
    for label in ["LV", "RV", "RVFW", "EPI", "MV", "AV", "TV", "PV"]:
        mesh = meshio.read(geodir / f"{label}_unloaded_ED.stl")
        tree = cKDTree(coords)
        distances, indices = tree.query(mesh.points, k=4)
        # Inverse distance weights
        weights = 1.0 / (distances + 1e-12)
        weights /= weights.sum(axis=1, keepdims=True)
        interpolated_u = np.sum(u[indices] * weights[..., np.newaxis], axis=1)
        tri_cells = None
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                tri_cells = cell_block.data
                break
        if tri_cells is None:
            raise RuntimeError(f"No triangles found in {label}_unloaded_ED.stl")
        cells[label] = tri_cells

        print(f"Loaded {label} mesh with {len(mesh.points)} points and {len(tri_cells)} triangles.", flush=True)
        points[label] = mesh.points + interpolated_u
        


    def get_mesh(faces, points) -> meshio.Mesh:
        triangle_data_local = faces

        node_indices_that_we_need = np.unique(triangle_data_local)
        node_data_local = points[node_indices_that_we_need, :]

        node_id_map_original_to_local = {
            original: local for local, original in enumerate(node_indices_that_we_need)
        }

        # now apply the mapping to the triangle_data
        for i in range(triangle_data_local.shape[0]):
            triangle_data_local[i, 0] = node_id_map_original_to_local[triangle_data_local[i, 0]]
            triangle_data_local[i, 1] = node_id_map_original_to_local[triangle_data_local[i, 1]]
            triangle_data_local[i, 2] = node_id_map_original_to_local[triangle_data_local[i, 2]]

        # node_indices_that_we_need = np.unique(triangle_data_local)
        # node_data_local = points[node_indices_that_we_need, :]

        return meshio.Mesh(points=node_data_local, cells=[("triangle", triangle_data_local)])
        
    for label in points:
        get_mesh(cells[label], points[label]).write(outdir / f"{label}_{case}_deformed.stl")
    
    return unloaded_deformed, unloaded

    # combine all points
    # all_points = np.vstack([points[label] for label in points])
    # all_cells = np.vstack([cells[label] + sum(len(cells[l]) for l in points if l < label) for label in points])
    # Example input: points and cells dictionaries
    # points = {'label1': np.array(...), 'label2': np.array(...)}
    # cells  = {'label1': np.array(...), 'label2': np.array(...))}

    # all_points = []
    # all_cells = []
    # label_ranges = {}  # store start/end index for each label

    # print("\n--- Mesh summary per label ---")
    # offset = 0
    # for label in ["LV", "RV", "EPI", "MV", "AV", "TV", "PV", "RVFW"]:
    #     verts = points[label]
    #     tris  = cells[label]

    #     start = offset
    #     end   = offset + len(verts)
    #     label_ranges[label] = (start, end)

    #     print(f"{label}: {len(verts)} points, {len(tris)} triangles, offset={offset}")

    #     all_points.append(verts)
    #     all_cells.append(tris + offset)
    #     offset += len(verts)

    # all_points = np.vstack(all_points)
    # all_cells  = np.vstack(all_cells)

    # print("\n--- Combined mesh summary (before deduplication) ---")
    # print(f"Total points: {len(all_points)}")
    # print(f"Total triangles: {len(all_cells)}")
    # print(f"Max cell index: {all_cells.max()}")

    # # Deduplicate while ensuring total points >= 5810
    # target_min = 5810
    # seen = {}
    # unique_points_list = []
    # inverse = np.zeros(len(all_points), dtype=int)
    # removed_count = 0

    # for i, pt in enumerate(all_points):
    #     key = tuple(pt)
    #     if key in seen:
    #         if (len(all_points) - removed_count) > target_min:
    #             inverse[i] = seen[key]
    #             removed_count += 1
    #         else:
    #             inverse[i] = len(unique_points_list)
    #             seen[key] = inverse[i]
    #             unique_points_list.append(pt)
    #     else:
    #         inverse[i] = len(unique_points_list)
    #         seen[key] = inverse[i]
    #         unique_points_list.append(pt)

    # final_points = np.vstack(unique_points_list)
    # final_cells  = inverse[all_cells]

    # print("\n--- After deduplication (ensuring >= 5810 points) ---")
    # print(f"Total points: {len(final_points)}")
    # print(f"Removed duplicates (global): {removed_count}")
    # print(f"Max cell index (remapped): {final_cells.max()}")

    # print("\n--- Per-label point reduction (accurate) ---")
    # for label, (start, end) in label_ranges.items():
    #     orig_count = end - start
    #     # count how many points in this label were mapped to an existing point (i.e., removed)
    #     removed_in_label = sum(inverse[start:end] != np.arange(start, end))
    #     print(f"{label}: {orig_count} → {orig_count - removed_in_label} (reduced {removed_in_label})")


    # return final_points, final_cells


def main(points_ED, points_ES, outdir):
    mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
    pca = mat_data['pca200'][0, 0]

    example_1d_ed = points_ED.flatten()
    example_1d_es = points_ES.flatten()
    example_flattened = np.concatenate((example_1d_ed, example_1d_es))

    projectedScores = project_patient_to_atlas(example_flattened, pca, numModes=200)
    
    print("Projected scores:", projectedScores[0])

    patient_shape = shape.reconstruct_shape(score = projectedScores, atlas = pca, num_scores=200)
    patient_ed = shape.get_ED_mesh_from_shape(patient_shape)
    patient_es = shape.get_ES_mesh_from_shape(patient_shape)
    vol_ed = volume.find_volume(patient_ed)
    print("ED Volume:", vol_ed)
    vol_es = volume.find_volume(patient_es)
    print("ES Volume:", vol_es)

    # Flatten into one dict
    flattened = {f"ED_{k}": float(v) for k, v in vol_ed.items()}
    flattened.update({f"ES_{k}": float(v) for k, v in vol_es.items()})

    return projectedScores[0], flattened, patient_ed, patient_es

if __name__ == "__main__":

    # patient_id = 0
    # ED_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/unloaded_to_ED_PLVED_20.00__PRVED_4.00__TA_0.0__a_2.28__af_1.69.bp"
    # ES_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/PLVED_20.00__PRVED_4.00__PLVES_30.0000__PRVES_8.0000__TA_120.0__a_2.28__af_1.69.bp"
    patient_id = 71
    ED_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/unloaded_to_ED_PLVED_10.00__PRVED_4.00__TA_0.0__a_3.28__af_30.00.bp"
    ES_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/PLVED_10.00__PRVED_4.00__PLVES_16.0000__PRVES_8.0000__TA_120.0__a_3.28__af_30.00.bp"
    u_ED, coords, geodir = resample(bpl=ED_file, mode=-1, datadir=Path(f"test/patient_{patient_id}/data-full"), resultsdir=Path(f"test/patient_{patient_id}/results-full"), case="ED")
    u_ES, coords, geodir = resample(bpl=ES_file, mode=-1, datadir=Path(f"test/patient_{patient_id}/data-full"), resultsdir=Path(f"test/patient_{patient_id}/results-full"), case="ES")

    points_ED, undeformed = deform(Path(f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED"), u_ED, geodir, coords, case="ED", patient_id=patient_id)
    points_ES, _ = deform(Path(f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED"), u_ES, geodir, coords, case="ES", patient_id=patient_id)

    outdir = Path(f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED")
    projectedScores, flattened, patient_ed, patient_es = main(points_ED, points_ES, outdir)

    def set_path(ukb_path: str):
    # Path to the ukb-atlas/src folder : important to download my fork of the UKB atlas
        if ukb_path is None:
            ukb_path =  "../clones/rk-ukb-atlas/src"

        sys.path.insert(0, ukb_path)
        import ukb, cardiac_geometries as cgx
        from ukb import atlas, surface, mesh, clip
        return ukb, atlas, surface, mesh, clip

    ukb, atlas, surface, mesh, clip = set_path("../clones/rk-ukb-atlas/src")
    unwanted_nodes = (5630, 5655, 5696, 5729)
    points = shape.Points(
        ED=np.delete(patient_ed, unwanted_nodes, axis=0),
        ES=np.delete(patient_es, unwanted_nodes, axis=0),
        unloaded_ED=np.delete(undeformed, unwanted_nodes, axis=0),
    )

    ukb.surface.main(case="all", folder=outdir, custom_points=points)

