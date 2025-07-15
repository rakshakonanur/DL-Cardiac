# DL-Cardiac

## Steps to run simulations with custom geometries:

1. Follow installation instructions from Herik's repo: https://github.com/ComputationalPhysiology/sscp25-deep-learning-cardiac-mechanics
2. In the same environment, clone this repository to your desired folder using git clone https://github.com/rakshakonanur/DL-Cardiac.git
3. Create a "clones" folder, which will contain editable versions of the libraries used.
4. In this folder, clone the following repos:
   a. https://github.com/ComputationalPhysiology/cardiac-geometriesx.git
   b. https://github.com/annaqi13/sscp25.git
   c. https://github.com/UOA-Heart-Mechanics-Research/biv-me.git
   d. https://github.com/rakshakonanur/rk-ukb-atlas.git (This is my version- so must clone this!)
5. Create a new folder in the parent directory, add the following two files:
   a. https://drive.google.com/file/d/1nGlaEU_l6eJrSsk2DGX7Uqq0DgcDJU31/view
   b. Copy the cohort data file from this directory: sscp25/Data/virtual_cohort_data.xlsx
6. Code should run by running sample.py
