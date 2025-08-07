import sys, os
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io
import time
from argparse import ArgumentParser
import shape

def find_volume(bivvertices, precision = 1) -> None:
    """
        # Authors: ldt, cm
        # Date: 09/22, revised 08/24 by cm
S
        This function measures the mass and volume of LV and RV.
        #--------------------------------------------------------------
        Inputs: model_file = fitted model (.txt), containing only data relative to one frame
                output_file = path to the output csv file
                biv_model_folder = path to the model folder - default: MODEL_RESOURCE_DIR
                precision - output precision for the volumes
        Output: None
    """
    if Path("../clones/biv-me/src/").exists():
        biv_model_dir = Path("../clones/biv-me/src/")
    else:
        print("biv-me model folder not found. Please provide the path to the biv-me model folder.")

    sys.path.append(str(biv_model_dir))
    from bivme import MODEL_RESOURCE_DIR
    from bivme.meshing.mesh import Mesh

    biv_model_folder = MODEL_RESOURCE_DIR

    # assign values to dict
    results_dict = {
        k: '' for k in ['lv_vol', 'rv_vol', 'lv_epivol', 'rv_epivol', 'lv_mass', 'rv_mass']
    }

    subdivision_matrix_file = biv_model_folder / "subdivision_matrix_sparse.mat"
    assert subdivision_matrix_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {subdivision_matrix_file} file!"

    elements_file = biv_model_folder / 'ETIndicesSorted.txt'
    assert elements_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {elements_file} file!"

    material_file = biv_model_folder / 'ETIndicesMaterials.txt'
    assert material_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {material_file} file!"

    thru_wall_file = biv_model_folder / 'thru_wall_et_indices.txt'
    assert thru_wall_file.exists(), \
        f"biv_model_folder does not exist. Cannot find {thru_wall_file} file!"
    
    control_points = bivvertices

    if control_points.shape[0] > 0:
        subdivision_matrix = scipy.io.loadmat(subdivision_matrix_file)['S'].toarray()
        faces = np.loadtxt(elements_file).astype(int)-1
        mat = np.loadtxt(material_file, dtype='str')

        # A.M. :there is a gap between septum surface and the epicardial
        #   Which needs to be closed if the RV/LV epicardial volume is needed
        #   this gap can be closed by using the et_thru_wall facets
        et_thru_wall = np.loadtxt(thru_wall_file, delimiter='\t').astype(int)-1

        ## convert labels to integer corresponding to the sorted list
        # of unique labels types
        unique_material = np.unique(mat[:,1])

        materials = np.zeros(mat.shape)
        for index, m in enumerate(unique_material):
            face_index = mat[:, 1] == m
            materials[face_index, 0] = mat[face_index, 0].astype(int)
            materials[face_index, 1] = [index] * np.sum(face_index)

        # add material for the new facets
        new_elem_mat = [list(range(materials.shape[0], materials.shape[0] + et_thru_wall.shape[0])),
                        [len(unique_material)] * len(et_thru_wall)]

        vertices = control_points
        faces = np.concatenate((faces.astype(int), et_thru_wall))
        materials = np.concatenate((materials.T, new_elem_mat), axis=1).T.astype(int)

        model = Mesh('mesh')
        model.set_nodes(vertices)
        model.set_elements(faces)
        model.set_materials(materials[:, 0], materials[:, 1])

        # components list, used to get the correct mesh components:
        # ['0 AORTA_VALVE' '1 AORTA_VALVE_CUT' '2 LV_ENDOCARDIAL' '3 LV_EPICARDIAL'
        # ' 4 MITRAL_VALVE' '5 MITRAL_VALVE_CUT' '6 PULMONARY_VALVE' '7 PULMONARY_VALVE_CUT'
        # '8 RV_EPICARDIAL' '9 RV_FREEWALL' '10 RV_SEPTUM' '11 TRICUSPID_VALVE'
        # '12 TRICUSPID_VALVE_CUT', '13' THRU WALL]

        lv_endo = model.get_mesh_component([0, 2, 4], reindex_nodes=False)

        # Select RV endocardial
        rv_endo = model.get_mesh_component([6, 9, 10, 11], reindex_nodes=False)

        # switching the normal direction for the septum
        rv_endo.elements[rv_endo.materials == 10, :] = \
            np.array([rv_endo.elements[rv_endo.materials == 10, 0],
                      rv_endo.elements[rv_endo.materials == 10, 2],
                      rv_endo.elements[rv_endo.materials == 10, 1]]).T

        lv_epi = model.get_mesh_component([0, 1, 3, 4, 5, 10, 13], reindex_nodes=False)
        # switching the normal direction for the thru wall
        lv_epi.elements[lv_epi.materials == 13, :] = \
            np.array([lv_epi.elements[lv_epi.materials == 13, 0],
                      lv_epi.elements[lv_epi.materials == 13, 2],
                      lv_epi.elements[lv_epi.materials == 13, 1]]).T

        # switching the normal direction for the septum
        rv_epi = model.get_mesh_component([6, 7, 8, 10, 11, 12, 13], reindex_nodes=False)
        rv_epi.elements[rv_epi.materials == 10, :] = \
            np.array([rv_epi.elements[rv_epi.materials == 10, 0],
                      rv_epi.elements[rv_epi.materials == 10, 2],
                      rv_epi.elements[rv_epi.materials == 10, 1]]).T

        lv_endo_vol = lv_endo.get_volume()
        rv_endo_vol = rv_endo.get_volume()
        lv_epi_vol = lv_epi.get_volume()
        rv_epi_vol = rv_epi.get_volume()

        rv_mass = (rv_epi_vol - rv_endo_vol) * 1.05  # mass in grams
        lv_mass = (lv_epi_vol - lv_endo_vol) * 1.05

        # assign values to dict
        results_dict['lv_vol'] = round(lv_endo_vol, precision)
        results_dict['rv_vol'] = round(rv_endo_vol, precision)
        results_dict['lv_epivol'] = round(lv_epi_vol, precision)
        results_dict['rv_epivol'] = round(rv_epi_vol, precision)
        results_dict['lv_mass'] = round(lv_mass, precision)
        results_dict['rv_mass'] = round(rv_mass, precision)

    return results_dict 

def find_unloaded_pcs_by_gradient_descent_mass_volume(
    initial_pc_scores: np.ndarray,
    target_volume: float,
    original_lv_mass: float,         
    reconstruct_shape: callable,
    get_ED_mesh_from_shape: callable,
    find_volume: callable,
    pca: dict,
    learning_rate: float = 0.01,
    max_iterations: int = 200,
    tolerance: float = 0.1,
    epsilon: float = 1e-4,
    mass_constraint_weight: float = 0.5,
    visual: bool = True) -> tuple:

    """    
    Optimize the principal component scores to achieve a target volume while maintaining mass constraints

    """

    print("Starting optimization process with mass constraint...\n")
    pc_scores = np.copy(initial_pc_scores)
    num_pcs = len(pc_scores)
    volume_history = []
    mass_history = []

    for i in range(max_iterations):
        # Calculate the current volume, mass, and errors
        current_mesh = reconstruct_shape(pc_scores, pca)
        ed_mesh = get_ED_mesh_from_shape(current_mesh)
        
        current_properties = find_volume(ed_mesh)
        current_volume = current_properties['lv_vol']
        current_mass = current_properties['lv_mass']

        volume_history.append(current_volume)
        mass_history.append(current_mass)

        volume_error = current_volume - target_volume
        mass_error = current_mass - original_lv_mass
        
        # calculate a single "loss" value that combines both errors.
        # We use the sum of squared errors. Our goal is to make this loss zero.
        loss = (volume_error**2) + mass_constraint_weight * (mass_error**2)

        print(
            f"Iter {i+1:03d}: "
            f"Volume = {current_volume:6.2f} (Err: {volume_error:6.2f}), "
            f"Mass = {current_mass:5.2f} (Err: {mass_error:5.2f}), "
            f"Loss = {loss:.4f}"
        )

        # Check for convergence based on volume error
        if abs(volume_error) < tolerance:
            print(f"\nConvergence achieved! Volume error is within the tolerance of {tolerance} mL.")
            break

        # Calculate the gradient of the COMBINED LOSS with respect to each PC score
        gradient = np.zeros(num_pcs)
        for j in range(num_pcs):
            pc_scores_perturbed = np.copy(pc_scores)
            pc_scores_perturbed[j] += epsilon

            mesh_perturbed = reconstruct_shape(pc_scores_perturbed, pca)
            ed_perturbed = get_ED_mesh_from_shape(mesh_perturbed)
            
            properties_perturbed = find_volume(ed_perturbed)
            
            # Calculate the loss of the perturbed mesh
            volume_error_p = properties_perturbed['lv_vol'] - target_volume
            mass_error_p = properties_perturbed['lv_mass'] - original_lv_mass
            loss_perturbed = (volume_error_p**2) + mass_constraint_weight * (mass_error_p**2)
            
            # Approximate the partial derivative of the combined loss function
            partial_derivative = (loss_perturbed - loss) / epsilon
            gradient[j] = partial_derivative
        
        # Normalize the gradient to prevent excessively large steps
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 0:
            gradient /= gradient_norm

        # Update the PC scores
        pc_scores -= learning_rate * gradient

    else:
        print(f"\nOptimization finished after {max_iterations} iterations without reaching the exact tolerance.")

    print(f"\nFinal optimized PC scores: {np.round(pc_scores, 4)}")
    print(f"\n Difference in z-scores: {np.round(pc_scores - initial_pc_scores, 4)}")

    if visual:
        visualize(volume_history, mass_history, target_volume, original_lv_mass)

    return pc_scores, volume_history, mass_history

def find_unloaded_pcs_by_gradient_descent_volume(initial_pc_scores: np.ndarray,
    target_volume: float,
    reconstruct_shape: callable,
    get_ED_mesh_from_shape: callable,
    find_volume: callable,
    pca: dict,
    learning_rate: float = 0.01,
    max_iterations: int = 200,
    tolerance: float = 0.1,
    epsilon: float = 1e-4
    ) -> np.ndarray:

    """
    Finds the PC scores that produce a target volume using gradient descent.
    Args:
        initial_pc_scores (np.ndarray): The starting PC scores (e.g., for the ED shape).
        target_volume (float): The desired unloaded volume (e.g., in mL).
        reconstruct_mesh_func (callable): Your function to build a mesh from PC scores.
        calculate_volume_func (callable): Your function to find the volume of a mesh.
        learning_rate (float): Step size for each iteration. May need tuning.
        max_iterations (int): Maximum number of iterations to prevent infinite loops.
        tolerance (float): The acceptable volume difference to stop optimization.
        epsilon (float): A small value used for numerically calculating the gradient.

    Returns:
        np.ndarray: The optimized PC scores for the unloaded shape.
    """
    print("Starting optimization process...\n")
    pc_scores = np.copy(initial_pc_scores)
    num_pcs = len(pc_scores)

    start_time = time.time()

    for i in range(max_iterations):
        # 1. Calculate the current volume and error
        current_mesh = reconstruct_shape(pc_scores, pca)
        ed_mesh = get_ED_mesh_from_shape(current_mesh)
        current_volume = find_volume(ed_mesh)['lv_vol']
        error = current_volume - target_volume
        print(f"Iteration {i+1:03d}: Current Volume = {current_volume:7.2f} mL, Target = {target_volume:7.2f} mL, Error = {error:7.2f} mL")

        # 2. Check for convergence
        if abs(error) < tolerance:
            print(f"\nConvergence achieved! Final volume is within the tolerance of {tolerance} mL.")
            end_time = time.time()
            break

        # 3. Calculate the gradient of the volume with respect to each PC score
        # This tells us how a small change in each PC affects the volume
        gradient = np.zeros(num_pcs)
        for j in range(num_pcs):
            # Perturb the j-th PC score by a small amount (epsilon)
            pc_scores_perturbed = np.copy(pc_scores)
            pc_scores_perturbed[j] += epsilon
            # Calculate the volume of the perturbed mesh
            mesh_perturbed = reconstruct_shape(pc_scores_perturbed, pca)
            ed_perturbed = get_ED_mesh_from_shape(mesh_perturbed)
            volume_perturbed = find_volume(ed_perturbed)

            # # Approximate the partial derivative (how much the volume changed)
            partial_derivative = (volume_perturbed['lv_vol'] - current_volume) / epsilon
            gradient[j] = partial_derivative

        # Normalize the gradient to prevent excessively large steps
        gradient_norm = np.linalg.norm(gradient)
        if gradient_norm > 0:
            gradient /= gradient_norm

        # 4. Update the PC scores
        # We move the scores "downhill" along the gradient to reduce the error.
        # The step size is proportional to the learning rate and the error itself.
        update_step = learning_rate * error * gradient
        pc_scores -= update_step

    else:
        # This block executes if the for loop finishes without a 'break'
        end_time = time.time()
        print(f"\nOptimization finished after {max_iterations} iterations without reaching the exact tolerance.")

    print(f"\nFinal optimized PC scores: {np.round(pc_scores, 4)}")

    # --- Verification ---
    print("\n--- Final State ---")
    print(f"Optimization took {end_time - start_time:.2f} seconds.")
    # Reconstruct the final mesh and verify its volume
    final_unloaded_mesh = reconstruct_shape(pc_scores, pca)
    final_ed_mesh = get_ED_mesh_from_shape(final_unloaded_mesh)
    final_volume = find_volume(final_ed_mesh)

    print(f"\n Difference in z-scores: {np.round(pc_scores - initial_pc_scores, 4)}")

    print(f"Final calculated volume: {final_volume['lv_vol']:.2f} mL")
    print(f"Target volume: {target_volume:.2f} mL")
    print(f"Difference: {abs(final_volume['lv_vol'] - target_volume):.4f} mL")

    return pc_scores

def visualize(volume_history, mass_history, target_volume, target_mass):
    import matplotlib.pyplot as plt
    # Create a figure with two subplots, sharing the same x-axis (Iteration)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 10),
        sharex=True
    )
    fig.suptitle('Optimization Progress Over Iterations', fontsize=16)

    # volume history
    ax1.plot(volume_history, marker='.', linestyle='-', color='b', label='LV Volume')
    # Add a horizontal line for the target volume
    ax1.axhline(y=target_volume, color='r', linestyle='--', label=f'Target Volume ({target_volume:.2f} mL)')
    ax1.set_ylabel('Volume (mL)')
    ax1.legend()
    ax1.grid(True)

    # mass history
    ax2.plot(mass_history, marker='.', linestyle='-', color='g', label='LV Mass')
    # Add a horizontal line for the original mass
    ax2.axhline(y=target_mass, color='r', linestyle='--', label=f'Original Mass ({target_mass:.2f} g)')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Mass (g)')
    ax2.legend()
    ax2.grid(True)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust for suptitle
    plt.show()

    print("Plot displayed.")    

if __name__ == "__main__":
    parser = ArgumentParser(description="Load PCA, connectivity indices, and virtual cohort data.")
    parser.add_argument("--pca_path", type=str, default=None, help="Path to the PCA .mat file.")
    parser.add_argument("--connectivity_path", type=str, default=None, help="Path to the connectivity indices file.")
    parser.add_argument("--virtual_cohort_path", type=str, default=None, help="Path to the virtual cohort data file.")
    
    args = parser.parse_args()

    pca = shape.load_pca(args.pca_path)
    et_indices = shape.load_connectivity(args.connectivity_path)
    virtual_cohort = shape.load_virtual_cohort(args.virtual_cohort_path)

    # prompt for a specific patient
    patient_id = int(input("Enter patient ID (0-based index): "))
    pc_columns = [col for col in virtual_cohort.columns if col.startswith('PC')]
    pc_scores = virtual_cohort[pc_columns]
    patient = list(pc_scores.loc[patient_id,:])
    patient = pc_scores.loc[patient_id].to_numpy()  # Example: get the specified patient's PC scores
    patient_shape = shape.reconstruct_shape(score = patient, atlas = pca)
    patient_ed = shape.get_ED_mesh_from_shape(patient_shape)
    patient_es = shape.get_ES_mesh_from_shape(patient_shape)

    # test calculating the volumes
    volume = find_volume(patient_ed)

    target_volume = virtual_cohort["Unloaded Volume"].iloc[patient_id]
    target_mass = volume["lv_mass"]

    print(f"Target LV Volume: {target_volume:.2f} mL")
    print(f"Original LV Mass:    {target_mass:.2f} g\n")

    # store test which is a dictionary as a dataframe
    df = pd.DataFrame([volume])
    print(df)

    # prompt user for optimization choice
    choice = input("Do you want to optimize the PC scores for volume (v) or mass + volume (m)? ").strip().lower()

    if choice == "mass_volume":
        # Optimize for mass + volume
        unloaded_pc_scores, volume_history, mass_history = find_unloaded_pcs_by_gradient_descent_mass_volume(
            initial_pc_scores=patient,
            target_volume=target_volume,
            original_lv_mass=target_mass,
            reconstruct_shape=shape.reconstruct_shape,
            get_ED_mesh_from_shape=shape.get_ED_mesh_from_shape,
            find_volume=find_volume,
            pca=pca,
            learning_rate=0.05,
            max_iterations=200,
            tolerance=1,
            epsilon=0.5,
            mass_constraint_weight=0.1  # tune this parameter, higher values put more emphasis on mass constraint (higher values will converge slower)
        )
        pass
    elif choice == "volume":
        # Optimize for volume only
        unloaded_pc_scores = find_unloaded_pcs_by_gradient_descent_volume(
            initial_pc_scores=patient,
            target_volume=target_volume,
            reconstruct_shape=shape.reconstruct_shape,
            get_ED_mesh_from_shape=shape.get_ED_mesh_from_shape,
            find_volume=find_volume,
            pca = pca,
            learning_rate=0.05,
            max_iterations=200,
            tolerance= 0.1,
            epsilon=0.5)
    else:
        print("Invalid choice. Please enter 'v' or 'm'.")

    # Reconstruct the unloaded shape from the optimized PC scores
    unloaded_shape = shape.reconstruct_shape(unloaded_pc_scores, pca)
    unloaded_ed = shape.get_ED_mesh_from_shape(unloaded_shape)

