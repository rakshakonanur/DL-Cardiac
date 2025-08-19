# DL-Cardiac

Repository to generate and simulate patient-specific cardiac mechanics using anonymized data from the UK Biobank. The code also can be used to generate training data for machine-learning applications.

## Get Started:

1. In an empty folder (assume DL-Cardiac), clone this repository using:
```bash
 git clone https://github.com/rakshakonanur/DL-Cardiac.git
```
2. Create a "clones" folder, which will contain editable versions of the libraries used.
```bash
 mkdir clones
 cd clones
```
3. Follow installation instructions from this repo: https://github.com/rakshakonanur/rk-sscp25-deep-learning-cardiac-mechanics.git
4. Inside the "clones" folder, clone the following repos:
   - https://github.com/annaqi13/sscp25.git
   - https://github.com/UOA-Heart-Mechanics-Research/biv-me.git
   - https://github.com/rakshakonanur/rk-ukb-atlas.git (This is my version- so must clone this!)
   - https://github.com/rakshakonanur/fenicsx-pulse.git
5. Create a new folder in the parent directory called "refs", add the following two files:
   - https://drive.google.com/file/d/1nGlaEU_l6eJrSsk2DGX7Uqq0DgcDJU31/view
   - Copy the cohort data file from this directory: sscp25/Data/virtual_cohort_data.xlsx
6. Sample code can be run using:
```bash
python3 sample.py
```

## Running Code:
1. To create meshes for the ED, ES, and undefoxrmed ED, in the src/ folder, run:
```bash
python3 mesh.py
```
This by-default creates meshes for all three cases.

2. The simulation.py code is set to generate training data for a particular patient. To run simulations on a singular case, use command-line arguments. For example:
```bash
python3 simulation.py --single_case True --PLV 15.0 --PRV 3.0 --Ta 120.0 --N 200
```
Excluding these command-line arguments will result in the running all pressures/material properties combination for that patient. This script can be run in parallel.
Final results are recommended to be visualized in Paraview.

3. To run simulations in batch, adjust the ED values, material properties, and patient ID in batch.py. The code will automatically loop through all combinations.
```bash
python3 batch.py
```

4. To extract the PCA scores from the deformed meshes, either manually point to the ED and ES files in the main function of extract_pca.py and run:
```bash
python3 extract_pca.py
```
Or, if batch-extraction is required, move all patients to datasets/final/ (sister of clones and src). Then run:
```bash
python3 postprocessing.py
```
This will automatically extract the PCA scores from the simulation results, and combine them into a large table. Details of volume/mass are also included.

5. To run the Machine Learning code, run:
```bash
python3 ML.py
```
The code currently trains on all patients exluding patient_id=5. This is used for testing. A random material/pressure condition is chosen, and its true/predicted deformed PCA are displayed. These PCA scores can be visualized by copying them into test.py and running:
```bash
python3 test.py
```

## Updating Code:

To make sure code is up-to-date, follow either of the following:
1. Navigate to the parent directory DL-Cardiac (above src and clones) and update the src code using:
```bash
git pull
```
Repeat this command in each of the cloned repositories.

2. To automatically update this repo (still testing):
In the src/ folder, run:
```bash
python3 update.py
```
