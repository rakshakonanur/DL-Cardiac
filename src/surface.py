import sys
import os
from mpi4py import MPI
from pathlib import Path

# Add the parent directory of 'ukb-atlas' to the path
sys.path.append("/mnt/c/Users/rkona/Documents/SSCP25/Cardiac_Deep_Learning/ukb-atlas/src")

import ukb
from ukb import atlas
from ukb import surface
from ukb import mesh

comm = MPI.COMM_WORLD
mode = -1
outdir = Path("data-full") / f"mode_{mode}"
outdir.mkdir(parents=True, exist_ok=True)
ukb.surface.main(
    folder=outdir,
    cache_dir = outdir, 
    mode=-1,
    case="ED",
)
ukb.mesh.main(
    folder=f"data-full/mode_{mode}",
    case="ED",
    char_length_max=5.0,
    char_length_min=5.0,
    clipped=False,
)
