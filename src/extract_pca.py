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
    patient3D = patient_shape_flat.reshape(-1, 3)
    mean3D = np.array(np.transpose(atlas["MU"])).reshape(-1, 3)
    # Save mean3D to a file for debugging
    np.savetxt("mean3D.txt", mean3D, fmt="%.6f")
    # Procrustes alignment
    T, c = procrustes(mean3D, patient3D, scaling=True, reflection='best')
    patient3Daligned = np.dot(patient3D, T) + c
    patient1Daligned = patient3Daligned.flatten()

    patient1Dnormalized = patient1Daligned - np.transpose(atlas["MU"])
    projectedScores = np.dot(patient1Dnormalized, np.transpose(atlas["COEFF"][0:numModes,:])) / np.sqrt(np.array(np.transpose(atlas['LATENT'][0,0:numModes])))
    return projectedScores

def sample_points_on_triangles(points, triangles, n_samples):
    """
    Uniformly sample n_samples points on the mesh surface defined by points and triangles.
    """

    # Compute areas of each triangle
    tri_pts = points[triangles]  # shape (num_triangles, 3, 3)
    vec0 = tri_pts[:, 1] - tri_pts[:, 0]
    vec1 = tri_pts[:, 2] - tri_pts[:, 0]
    cross_prod = np.cross(vec0, vec1)
    tri_areas = 0.5 * np.linalg.norm(cross_prod, axis=1)

    # Normalize to get probability distribution over triangles
    tri_probs = tri_areas / np.sum(tri_areas)

    # Sample triangles according to their area
    sampled_tri_indices = np.random.choice(len(triangles), size=n_samples, p=tri_probs)

    # Sample barycentric coordinates within triangles
    r1 = np.sqrt(np.random.rand(n_samples))
    r2 = np.random.rand(n_samples)
    w0 = 1 - r1
    w1 = r1 * (1 - r2)
    w2 = r1 * r2

    sampled_points = (
        w0[:, None] * points[triangles[sampled_tri_indices, 0]]
        + w1[:, None] * points[triangles[sampled_tri_indices, 1]]
        + w2[:, None] * points[triangles[sampled_tri_indices, 2]]
    )
    return sampled_points


def resample(
    mode: int = -1,
    datadir: Path = Path("test/patient_0/data-full"),
    resultsdir: Path = Path("test/patient_0/results-full"),
    bpl: str = "test/patient_0/results-full/mode_-1/unloaded_to_ED_PLVED_20.00__PRVED_4.00__TA_0.0__a_2.28__af_1.69.bp",
    n_total_points: int = 5352,
    case: str ="ED"
):
    geodir = Path(datadir) / f"mode_{mode}" / "unloaded_ED"
    outdir = Path(resultsdir) / f"mode_{mode}" / "unloaded_ED"

    comm = MPI.COMM_WORLD
    geo = cardiac_geometries.geometry.Geometry.from_folder(comm=comm, folder=geodir)
    V = dolfinx.fem.functionspace(geo.mesh, ("CG", 2, (3,)))
    u = dolfinx.fem.Function(V)

    from adios2 import FileReader
    with FileReader(bpl) as ibpFile:
        # Get list of blocks (first [0] because all_blocks_info returns a list per step)
        blocks_info = ibpFile.all_blocks_info("f")[0]

        # Sort by BlockID (as integer), largest first
        blocks_info_sorted = sorted(blocks_info, key=lambda b: int(b["BlockID"]), reverse=True)

        f_list = []
        geometry_list = []

        for block in blocks_info_sorted:
            block_id = int(block["BlockID"])
            
            f_list.append(ibpFile.read("f", block_id=block_id))
            geometry_list.append(ibpFile.read("geometry", block_id=block_id))

        # Vertically concatenate all blocks
        f = np.vstack(f_list)
        geometry = np.vstack(geometry_list)

        print("f.shape:", f.shape)
        print("geometry.shape:", geometry.shape)
    
    # Get DOF coordinates and mapping
    dof_coords = V.tabulate_dof_coordinates().reshape((-1, 3))  # one row per DOF

    # For vector space, DOFs come in blocks of 3 for each point
    block_size = V.dofmap.index_map_bs  # should be 3 for vector CG2

    # Match geometry points to DOF coords
    # We'll use nearest neighbor matching (tolerant to floating-point issues)
    from scipy.spatial import cKDTree
    tree = cKDTree(geometry)

    dist, vert_index = tree.query(dof_coords, distance_upper_bound=1e-12)

    # dist > 1e-12 means no match — handle errors
    if np.any(dist > 1e-12):
        raise RuntimeError("Some DOFs could not be matched to geometry points")

    # Create flat array for u
    values_flat = np.zeros_like(u.x.array)

    # Fill array
    for dof_id, vtx_id in enumerate(vert_index):
        comp = dof_id % block_size  # component 0,1,2
        values_flat[dof_id] = f[vtx_id, comp]

    # Assign to u
    u.x.array[:] = values_flat
    # u.x.array[:] = 0
    u.x.scatter_forward()

    print("u.x.array shape:", u.x.array.shape)
    print("u.x.array min/max:", u.x.array.min(), u.x.array.max())

    points = {}
    triangles = {}
    for label in ["LV", "RV","EPI","MV", "AV", "TV", "PV"]:
        mesh = meshio.read(geodir / f"{label}_unloaded_ED.stl")
        if label == "RV":
            # Read second STL
            mesh2 = meshio.read(geodir / f"RVFW_unloaded_ED.stl")
            # Combine points (concatenate)
            combined_points = np.vstack([mesh.points, mesh2.points])

            # Adjust cell indices of mesh2 triangles by offsetting by len(mesh1.points)
            offset = len(mesh.points)
            cells1 = mesh.cells_dict.get("triangle", np.empty((0, 3), dtype=int))
            cells2 = mesh2.cells_dict.get("triangle", np.empty((0, 3), dtype=int)) + offset

            # Combine cells (triangles)
            cells = np.vstack([cells1, cells2])

            # Create new mesh object with combined data
            merged_mesh = meshio.Mesh(points=points, cells=[("triangle", cells)])
            mesh = merged_mesh
            points[label] = combined_points
        else:
            points[label] = mesh.points
        # Get triangles (assume mesh.cells contains triangles under 'triangle' key)
        tri_cells = None
        for cell_block in mesh.cells:
            if cell_block.type == "triangle":
                tri_cells = cell_block.data
                break
        if tri_cells is None:
            raise RuntimeError(f"No triangles found in {label}_unloaded_ED.stl")
        triangles[label] = tri_cells

    fdim = geo.mesh.topology.dim - 1
    geo.mesh.topology.create_connectivity(fdim, 0)
     
    print(geo.mesh.geometry.x.max(axis=0), geo.mesh.geometry.x.min(axis=0))

    ftree = dolfinx.geometry.bb_tree(geo.mesh, fdim, padding=0.1)
    bb_tree = dolfinx.geometry.bb_tree(geo.mesh, geo.mesh.topology.dim)

    all_deformed_points = []

    for label, coords in points.items():
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
        u_values = u.eval(midpoint_coords, cells)

        # Compute deformed vertex positions
        deformed_vertices = midpoint_coords + u_values

        # Number of points to sample proportionally to the number of vertices on this surface
        # We'll assign sampling count proportional to number of vertices on this surface
        all_deformed_points.append((label, deformed_vertices, triangles[label]))
        # Save deformed vertices to a text file
        print(f"Deformed points for {label}: {deformed_vertices.min(axis=0)} to {deformed_vertices.max(axis=0)}")


    # Now combine all points and sample to get exactly n_total_points
    total_vertices = sum(points.shape[0] for _, points, _ in all_deformed_points)

    # save the total number of points to sample
    print(f"Total vertices across all surfaces: {total_vertices}")

    sampled_points_list = []

    points_label = []
    for label, deformed_pts, tris in all_deformed_points:
        # Number of points to sample for this surface (proportional allocation)
        n_points_surface = int(n_total_points * deformed_pts.shape[0] / total_vertices)
        # Sample points on triangles with deformed vertex positions
        sampled_pts = sample_points_on_triangles(deformed_pts, tris, n_points_surface)
        sampled_points_list.append(sampled_pts)

    # Concatenate all sampled points
    all_sampled_points = np.vstack(sampled_points_list)
    
    # save the sampled points to a text file
    np.savetxt(outdir / f"deformed_coordinates_{case}.txt", all_sampled_points, fmt="%.6f")

    # If due to rounding we have fewer than n_total_points, sample extra randomly from combined surfaces
    if all_sampled_points.shape[0] < n_total_points:
        n_extra = n_total_points - all_sampled_points.shape[0]
        # Just pick one surface arbitrarily (here first one)
        label, deformed_pts, tris = all_deformed_points[0]
        extra_pts = sample_points_on_triangles(deformed_pts, tris, n_extra)
        all_sampled_points = np.vstack([all_sampled_points, extra_pts])

    # Save all points to text file
    print(f"Total sampled points shape: {all_sampled_points.min(axis=0)} to {all_sampled_points.max(axis=0)}")
    np.savetxt(outdir / f"resample_deformed_coordinates_{case}.txt", all_sampled_points, fmt="%.6f")

if __name__ == "__main__":

    patient_id = 71
    ED_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/unloaded_to_ED_PLVED_10.00__PRVED_4.00__TA_0.0__a_3.28__af_30.00.bp"
    ES_file = f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED/PLVED_10.00__PRVED_4.00__PLVES_16.0000__PRVES_8.0000__TA_120.0__a_3.28__af_30.00.bp"
    resample(bpl=ED_file, mode=-1, datadir=Path(f"test/patient_{patient_id}/data-full"), resultsdir=Path(f"test/patient_{patient_id}/results-full"), n_total_points=5352, case="ED")
    resample(bpl=ES_file, mode=-1, datadir=Path(f"test/patient_{patient_id}/data-full"), resultsdir=Path(f"test/patient_{patient_id}/results-full"), n_total_points=5352, case="ES")

    outdir = Path(f"test/patient_{patient_id}/results-full/mode_-1/unloaded_ED")
    # Load files
    my_coords_ed = np.loadtxt(outdir / "resample_deformed_coordinates_ED.txt")
    ukb_coords_ed = np.loadtxt("test/anna/dummy_ed.txt")[:, :3]  # first 3 columns

    my_coords_es = np.loadtxt(outdir / "resample_deformed_coordinates_ES.txt")
    ukb_coords_es = np.loadtxt("test/anna/dummy_es.txt")[:, :3]  # first 3 columns

    # Build KDTree from your coordinates
    tree_ed = cKDTree(my_coords_ed)
    tree_es = cKDTree(my_coords_es)

    # Query nearest for each UKB point
    distances_es, indices_es = tree_es.query(ukb_coords_es, k=1)
    distances_ed, indices_ed = tree_ed.query(ukb_coords_ed, k=1)

    # Reorder your coordinates to match UKB order uniquely
    example_3d_es = my_coords_es[indices_es]
    example_3d_ed = my_coords_ed[indices_ed]

    example_1d_ed = example_3d_ed.flatten()
    example_1d_es = example_3d_es.flatten()
    example_flattened = np.concatenate((example_1d_ed, example_1d_es))

    # Load shape atlas
    file_path_in = '../clones/SSA_tutorial/' #replace with your file path

    # Update paths here
    pc = h5.File(file_path_in + 'UKBRVLV_All.h5', 'r')
    projectedScores = project_patient_to_atlas(example_flattened, pc, numModes=25)
    
    print("Projected scores:", projectedScores[0])
    mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
    pca = mat_data['pca200'][0, 0]

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


