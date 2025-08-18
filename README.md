# DL-Cardiac

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
3. For generating the final training dataset, the simulation results must be moved to datasets/ (same level as clones and src). 

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
