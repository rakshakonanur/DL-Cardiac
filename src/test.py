import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shape
import volume
import scipy.io

mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
pca = mat_data['pca200'][0, 0]
score =[1.699781624, 0.6334225628, -0.9943332855, -0.4799184724, 0.7173195775, 1.452603454, -0.488874658, 1.039413574, 1.314017295, -0.07953502141, -0.09158254309, -0.829926654, 0.2845153771, -2.116230972, -2.450380377, -0.7292193965, -1.950670665, -0.1086362837, -0.6646924126, -0.50757885, -0.2505143608, 1.587090814, -1.204133202, 0.4817391909, -1.081587619]

# Predicted values: [-0.14505391, -0.12284824, -0.27698756,  0.0537183,  -0.09379592, -0.29158689,
#  -0.13056078,  0.42569234,  0.07263599,  0.75241125]
# True values: [-0.34616572, -0.25487567, -0.39251532, -0.11185141, -0.30102752, -0.47441795,
#  -0.06028185,  0.46325936,  0.2368012,   0.66624435]
# Visualizing Sample 8:


patient_shape = shape.reconstruct_shape(score = score, atlas = pca, num_scores=25)
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

ukb.surface.main(case="both", folder=Path(f"./"), custom_points=points)

