# DL-Cardiac

## Steps to run simulations with custom geometries:

1. In an empty folder (assume DL-Cardiac), clone this repository using:
```bash
 git clone https://github.com/rakshakonanur/DL-Cardiac.git
```
2. Create a "clones" folder, which will contain editable versions of the libraries used.
'''bash
mkdir clones
cd clones
'''
3. Follow installation instructions from Herik's repo: https://github.com/ComputationalPhysiology/sscp25-deep-learning-cardiac-mechanics
4. Inside the "clones" folder, clone the following repos:
   - https://github.com/ComputationalPhysiology/cardiac-geometriesx.git
   - https://github.com/annaqi13/sscp25.git
   - https://github.com/UOA-Heart-Mechanics-Research/biv-me.git
   - https://github.com/rakshakonanur/rk-ukb-atlas.git (This is my version- so must clone this!)
5. Create a new folder in the parent directory, add the following two files:
   - https://drive.google.com/file/d/1nGlaEU_l6eJrSsk2DGX7Uqq0DgcDJU31/view
   - Copy the cohort data file from this directory: sscp25/Data/virtual_cohort_data.xlsx
6. Code should run by running sample.py

## New instructions for performing simulations on unloaded geometry:

1. Remove the old sscp25-deep-learning-cardiac-mechanics folder from clones/ and clone my fork instead (inside of clones/):
``` bash
git clone https://github.com/rakshakonanur/rk-sscp25-deep-learning-cardiac-mechanics.git
```
2. Navigate to the clones/rk-ukb-atlas and update it using git pull:
```bash
git pull
```
3. Navigate to the parent directory DL-Cardiac (above src and clones) and update the src code using:
```bash
git pull
```
4. To create meshes for the ED, ES, and undeformed ED, run:
```bash
python3 mesh.py
```
This by-default creates meshes for all three cases.
5. To run simulations on any/all cases, run:
```bash
python3 simulation.py
```
This should run the undeformed ED case by default.

## Test to check if repos on computer are out-of-date
In the src directory run:
```bash
python3 update.py
```
