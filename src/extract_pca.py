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

def procrustes(X,Y,scaling=True,reflection='best'):
    
    """
    Procrustes analysis determines a linear transformation (translation,
    reflection, orthogonal rotation and scaling) of the points in Y to best
    conform them to the points in matrix X, using the sum of squared errors
    as the goodness of fit criterion.
    d, Z, [tform] = procrustes(X, Y)
    """
    n,m = X.shape
    ny,my = Y.shape    
    muX = X.mean(0)
    muY = Y.mean(0)    
    X0 = X - muX
    Y0 = Y - muY    
    ssX = (X0**2.).sum()
    ssY = (Y0**2.).sum()
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)
    X0 /= normX
    Y0 /= normY
    if my < m:
        Y0 = np.concatenate((Y0,np.zeros(n, m-my)),0)
    A = np.dot(X0.T,Y0)
    U,s,Vt = np.linalg.svd(A,full_matrices=True)
    V = Vt.T
    T = np.dot(V,U.T)
    if reflection != 'best':
        have_reflection = np.linalg.det(T) < 0
        if reflection != have_reflection:
            V[:,-1] *= -1
            s[-1] *= -1
            T = np.dot(V,U.T)
    traceTA = s.sum()
    if scaling:
        b = traceTA*normX/normY
        d = 1-traceTA**2
        Z = normX*traceTA*np.dot(Y0,T)+muX
    else:
        b = 1
        d = 1+ssY/ssX-2*traceTA*normY/normX
        Z = normY*np.dot(Y0,T)+muX
    if my<m:
        T = T[:my,:]
    c = muX-b*np.dot(muY,T)
    # Transformation values 
    tform = {'rotation':T,'scale':b,'translation':c}
    #return d, Z, tform
    return T, c

def project_patient_to_atlas(patient_shape_flat, atlas, numModes = 10):

    MU = np.transpose(atlas["MU"])
    COEFF = np.transpose(atlas["COEFF"])
    LATENT = np.transpose(atlas["LATENT"])
    patient3D = patient_shape_flat.reshape(-1, 3)
    mean3D = np.array(np.transpose(MU)).reshape(-1, 3)

    # Procrustes alignment
    T, c = procrustes(mean3D, patient3D, scaling=True, reflection='best')
    patient3Daligned = np.dot(patient3D, T) + c
    patient1Daligned = patient3Daligned.flatten()

    patient1Dnormalized = patient1Daligned - np.transpose(MU)
    projectedScores = np.dot(patient1Dnormalized, np.transpose(COEFF[0:numModes,:])) / np.sqrt(np.array(np.transpose(LATENT[0,0:numModes])))
    return projectedScores

def sample_points_on_triangles_uniform(points, triangles, n_samples):
    tri_pts = points[triangles]
    vec0 = tri_pts[:, 1] - tri_pts[:, 0]
    vec1 = tri_pts[:, 2] - tri_pts[:, 0]
    cross_prod = np.cross(vec0, vec1)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    tri_probs = tri_areas / np.sum(tri_areas)

    # Allocate floor values first
    samples_per_triangle = np.floor(tri_probs * n_samples).astype(int)

    # Distribute remaining points to largest fractional parts
    fractional_parts = tri_probs * n_samples - np.floor(tri_probs * n_samples)
    leftover = n_samples - np.sum(samples_per_triangle)
    if leftover > 0:
        top_indices = np.argsort(fractional_parts)[::-1][:leftover]
        samples_per_triangle[top_indices] += 1

    sampled_points = []
    for tri_idx, n_pts in enumerate(samples_per_triangle):
        if n_pts == 0:
            continue
        m = int(np.ceil((np.sqrt(8 * n_pts + 1) - 1) / 2))
        bary_coords = []
        count = 0
        for i in range(m + 1):
            for j in range(m + 1 - i):
                if count >= n_pts:
                    break
                w0 = i / m
                w1 = j / m
                w2 = 1 - w0 - w1
                bary_coords.append([w0, w1, w2])
                count += 1
            if count >= n_pts:
                break
        bary_coords = np.array(bary_coords)
        pts = (
            bary_coords[:, 0:1] * tri_pts[tri_idx, 0]
            + bary_coords[:, 1:2] * tri_pts[tri_idx, 1]
            + bary_coords[:, 2:3] * tri_pts[tri_idx, 2]
        )
        sampled_points.append(pts)
    sampled_points = np.vstack(sampled_points)
    return sampled_points

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

    # Now load the STL meshes and combine RV and RVFW
    points = {}
    triangles = {}

    for label in ["LV", "RV", "EPI", "MV", "AV", "TV", "PV"]:
        mesh = meshio.read(geodir / f"{label}_unloaded_ED.stl")
        if label == "RV":
            mesh2 = meshio.read(geodir / f"RVFW_unloaded_ED.stl")

            combined_points = np.vstack([mesh.points, mesh2.points])
            offset = len(mesh.points)

            cells1 = mesh.cells_dict.get("triangle", np.empty((0, 3), dtype=int))
            cells2 = mesh2.cells_dict.get("triangle", np.empty((0, 3), dtype=int)) + offset
            combined_cells = np.vstack([cells1, cells2])

            mesh = meshio.Mesh(points=combined_points, cells=[("triangle", combined_cells)])

            points[label] = combined_points
            triangles[label] = combined_cells
        else:
            points[label] = mesh.points
            tri_cells = None
            for cell_block in mesh.cells:
                if cell_block.type == "triangle":
                    tri_cells = cell_block.data
                    break
            if tri_cells is None:
                raise RuntimeError(f"No triangles found in {label}_unloaded_ED.stl")
            triangles[label] = tri_cells

    print("Loaded and combined meshes:", flush=True)
    for label in points:
        print(f" - {label}: {len(points[label])} points, {len(triangles[label])} triangles", flush=True)

    # Create connectivity and bb_tree once
    fdim = geo.mesh.topology.dim - 1
    geo.mesh.topology.create_connectivity(fdim, 0)
    ftree = dolfinx.geometry.bb_tree(geo.mesh, fdim, padding=0.1)
    bb_tree = dolfinx.geometry.bb_tree(geo.mesh, geo.mesh.topology.dim)

    all_deformed_points = []
    for label, coords in points.items():
        print(f"Processing deformation for {label}...", flush=True)
        entities = geo.ffun.find(geo.markers[label][0])
        mid_tree = dolfinx.geometry.create_midpoint_tree(geo.mesh, fdim, entities)
        entity = dolfinx.geometry.compute_closest_entity(ftree, mid_tree, geo.mesh, coords)
        midpoint_coords = dolfinx.mesh.compute_midpoints(geo.mesh, 2, entity)
        potential_colliding_cells = dolfinx.geometry.compute_collisions_points(bb_tree, midpoint_coords)
        colliding_cells = dolfinx.geometry.compute_colliding_cells(geo.mesh, potential_colliding_cells, midpoint_coords)
        cells = []
        for i, coord in enumerate(midpoint_coords):
            if len(colliding_cells.links(i)) > 0:
                cells.append(colliding_cells.links(i)[0])
            else:
                raise ValueError(f"Point {i} at coordinate {coord} not found in mesh")
        cells = np.array(cells, dtype=np.int32)

        print(f"Evaluating displacement for {len(midpoint_coords)} points in {label}...", flush=True)
        u_values = u.eval(midpoint_coords, cells)
        print(f"Displacement evaluation done for {label}.", flush=True)

        deformed_vertices = midpoint_coords + u_values
        all_deformed_points.append((label, deformed_vertices, triangles[label]))
        print(f"Deformed points for {label}: min {deformed_vertices.min(axis=0)}, max {deformed_vertices.max(axis=0)}", flush=True)

    sample_counts = {
        "LV": 1500 - 0,
        "RV": 3224 - 1500,
        "EPI": 5582 - 3224,
        "MV": 5631 - 5582,
        "AV": 5656 - 5631,
        "TV": 5697 - 5656,
        "PV": 5730 - 5697,
        "EPI_2": 5810 - 5730
    }

    deformed_dict = {label: (pts, tris) for label, pts, tris in all_deformed_points}

    sampled_points_list = []

    print("Starting sampling points per label...", flush=True)
    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["LV"][0], deformed_dict["LV"][1], sample_counts["LV"]))
    print("Sampled LV points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["RV"][0], deformed_dict["RV"][1], sample_counts["RV"]))
    print("Sampled RV points.", flush=True)

    # You do not have 'WALL' label in deformed_dict, remove or fix this if needed
    # sampled_points_list.append(sample_points_on_triangles(deformed_dict["WALL"][0], deformed_dict["WALL"][1], sample_counts["WALL"]))
    # print("Sampled WALL points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["EPI"][0], deformed_dict["EPI"][1], sample_counts["EPI"]))
    print("Sampled EPI points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["MV"][0], deformed_dict["MV"][1], sample_counts["MV"]))
    print("Sampled MV points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["AV"][0], deformed_dict["AV"][1], sample_counts["AV"]))
    print("Sampled AV points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["TV"][0], deformed_dict["TV"][1], sample_counts["TV"]))
    print("Sampled TV points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["PV"][0], deformed_dict["PV"][1], sample_counts["PV"]))
    print("Sampled PV points.", flush=True)

    sampled_points_list.append(sample_points_on_triangles_uniform(deformed_dict["EPI"][0], deformed_dict["EPI"][1], sample_counts["EPI_2"]))
    print("Sampled second batch of EPI points.", flush=True)

    all_sampled_points = np.vstack(sampled_points_list)
    print("All sampling done, now saving file...", flush=True)

    np.savetxt(outdir / f"resample_ordered_coordinates_{case}.txt", all_sampled_points, fmt="%.6f")
    print(f"Saved {all_sampled_points.shape[0]} points in total at {outdir}.", flush=True)


if __name__ == "__main__":

    patient_id = 71
    ED_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/unloaded_to_ED_PLVED_10.00__PRVED_4.00__TA_0.0__a_3.28__af_30.00.bp"
    ES_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/PLVED_10.00__PRVED_4.00__PLVES_16.0000__PRVES_8.0000__TA_120.0__a_3.28__af_30.00.bp"
    resample(bpl=ED_file, mode=-1, datadir=Path(f"test/patient_{patient_id}/data-full"), resultsdir=Path(f"test/patient_{patient_id}/results-full"), case="ED")
    resample(bpl=ES_file, mode=-1, datadir=Path(f"test/patient_{patient_id}/data-full"), resultsdir=Path(f"test/patient_{patient_id}/results-full"),case="ES")

    outdir = Path(f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED")

    mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
    pca = mat_data['pca200'][0, 0]
    # Load files
    my_coords_ed = np.loadtxt(outdir / "resample_ordered_coordinates_ED.txt")
    my_coords_es = np.loadtxt(outdir / "resample_ordered_coordinates_ES.txt")

    mean3D = np.array(np.transpose(pca["MU"])).reshape(-1, 3)
    n = len(mean3D)
    mean3D_ed = mean3D[:n // 2]
    mean3D_es = mean3D[n // 2:]

    # Build KDTree from your coordinates
    tree_ed = cKDTree(my_coords_ed)
    tree_es = cKDTree(my_coords_es)

    # Query nearest for each UKB point
    distances_es, indices_es = tree_es.query(mean3D_es, k=1)
    distances_ed, indices_ed = tree_ed.query(mean3D_ed, k=1)

    # Reorder your coordinates to match UKB order uniquely
    example_3d_es = my_coords_es[indices_es]
    np.savetxt(outdir / "example_3d_es.txt", example_3d_es, fmt="%.6f")
    example_3d_ed = my_coords_ed[indices_ed]
    np.savetxt(outdir / "example_3d_ed.txt", example_3d_ed, fmt="%.6f")

    example_1d_ed = example_3d_ed.flatten()
    example_1d_es = example_3d_es.flatten()
    example_flattened = np.concatenate((example_1d_ed, example_1d_es))

    # Load shape atlas
    file_path_in = '../clones/SSA_tutorial/' #replace with your file path

    # Update paths here
    # pc = h5.File(file_path_in + 'UKBRVLV_All.h5', 'r')
    projectedScores = project_patient_to_atlas(example_flattened, pca, numModes=25)
    
    print("Projected scores:", projectedScores[0])
    
    patient_shape = shape.reconstruct_shape(score = projectedScores, atlas = pca, num_scores=25)
    patient_ed = shape.get_ED_mesh_from_shape(patient_shape)
    patient_es = shape.get_ES_mesh_from_shape(patient_shape)
    vol_ed = volume.find_volume(patient_ed)
    print("ED Volume:", vol_ed)
    vol_es = volume.find_volume(patient_es)
    print("ES Volume:", vol_es)

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
        unloaded_ED=np.delete(patient_ed, unwanted_nodes, axis=0),
    )

    ukb.surface.main(case="both", folder=Path(f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED"), custom_points=points)


