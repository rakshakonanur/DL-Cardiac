import sys
from pathlib import Path
import numpy as np
import pandas as pd
import shape
import volume
import scipy.io
import pyvista as pv

mat_data = scipy.io.loadmat("../refs/BioBank_EDES_200.mat")
pca = mat_data['pca200'][0, 0]
score =[1.699781624,	0.6334225628,	-0.9943332855,	-0.4799184724,	0.7173195775,	1.452603454,	-0.488874658,	1.039413574,1.314017295,	-0.07953502141,	-0.09158254309,	-0.829926654,	0.2845153771,	-2.116230972,	-2.450380377,	-0.7292193965,	-1.950670665,	-0.1086362837,	-0.6646924126,	-0.50757885,	-0.2505143608,	1.587090814,	-1.204133202,	0.4817391909,	-1.081587619]  
#  1.60607557 -0.46302547  0.49196294 -0.41271044  3.62954191
#   0.81359985  2.02878289 -1.2762582  -0.06864807 -0.572285   -1.08695108
#  -2.02635609 -0.71833556  2.06402615  0.63695993  0.50659826  0.48768577
#  -1.63475333 -2.02845304  1.63402761 -0.01689359 -0.14334314  2.87849535
#   0.8600064  -3.57116416  0.05024592 -2.90231443 -1.62678455 -2.18983888
#  -2.51409335 -0.3498052   0.47257564 -1.34881235 -1.41716271 -1.28724838
#   0.49082867  2.33275107 -2.61322292  1.13634136 -0.68638634  0.8306947
#  -4.25398435 -0.65739964 -2.55988432  1.27365109 -1.69053872  1.38939185
#  -0.13468672 -1.84461583 -3.9618903   0.49000683  2.51726488 -0.01333288
#  -0.91036181 -1.09680046  1.1719509  -1.78415234  0.89953925 -0.23301243
#  -1.67503254  0.97469806  1.53491577  3.71535936 -1.46870818  3.03941012
#   2.82464956 -1.28300843 -1.88754198  1.04205451  0.01432886 -0.06185553
#   0.79550966  0.34072352  1.75686562  1.942785   -4.47993466 -3.31695202
#   1.92932492 -0.35448343 -1.85139179  2.27722054  0.45644296 -0.60210102
#  -1.05045807  3.4664435   0.63854649 -2.21044321 -3.09105763  1.25934184
#  -0.50956443 -0.59862248 -2.32419397  0.5553144   0.10011225  0.39072027
#   4.20049055 -0.33790268 -0.70913903 -1.59162141  1.13607995 -4.49483898
#   5.04819283 -0.93451198  2.22638597  0.16055883 -1.9812856   0.11250971
#  -0.83086108 -1.5588856   3.26924207  2.00676229 -2.23477835 -0.95188819
#  -2.04788855 -1.89507944 -2.80691801  1.97570485 -0.777179   -1.35969505
#  -1.92819288 -0.72572625 -0.03977579 -2.24232557 -0.84501274  1.36181311
#   1.35872031  1.41317012 -0.58930937  0.03873035 -2.16545007  1.69622923
#  -1.19303689 -1.83443373  3.69700255 -4.85243024 -3.04612866  0.12063451
#  -1.86961336  4.05933158 -1.93634954  2.65432306 -0.13748226 -1.80293023
#  -2.63566779  1.97860354  4.30029465 -1.21029891 -0.39037623 -0.1048946
#  -5.40626129 -4.42451602 -0.41595893 -3.81333783 -4.928527   -1.08066028
#  -0.44527546  1.52536448  2.17687494  0.28882551 -4.10846578 -0.41315589
#   5.86122195  0.75793837 -1.7639902   1.92511458 -0.19884613  0.62578054
#   0.39986299 -5.08755965  3.14467464  1.80874731  0.62886561 -1.98981732
#   2.94086661  4.46300407]

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

# ukb.surface.main(case="both", folder=Path(f"./"), custom_points=points)

