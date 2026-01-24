#!/bin/bash
#===============================
# Submission Script —  (EMD + NEMD)
#==============================================================
#$ -S /bin/bash
#$ -pe mpi-48 48
#$ -q hpc
#$ -o Log_Files/$JOB_NAME.o$JOB_ID.$TASK_ID
#$ -e Log_Files/$JOB_NAME.e$JOB_ID.$TASK_ID
#$ -cwd
#==============================================================

module purge
module load conda
conda activate polymer-workflow

module load lammps
export MKL_INTERFACE_LAYER=""
export OMP_NUM_THREADS=1

export NSLOTS="$NSLOTS"
initial_dir="$(pwd)"

# SMILES CSV (for array jobs)
SMILES_FILE="SMILES.csv"

#=== USAGE EXAMPLE === 
#  Run from SMILES.csv (array jobs), e.g. first 10 rows (after header): 
# qsub -t 1-10 submit.sh 
#==============================================================
# --- USER CONFIG (CP ) ---
#==============================================================

DO_APG=0        # Amorphous generation (Gasteiger only)
DO_OPT=1        # Optimization / Equilibration
DO_EMD=1        # Cp from EMD
DO_NEMD=1       # Cp from NEMD (NEW)

CSV_START_LINE=2   # 2 = skip header; set to 1 if no header

# Export config
export SMILES_FILE CSV_START_LINE
export DO_APG DO_OPT DO_EMD DO_NEMD
export initial_dir

# Run
source Files/files.sh
