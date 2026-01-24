# POLY-CP-TL

This repository contains all data, simulation workflows, and machine-learning scripts used in the paper:

**“Transfer Learning for Polymer Specific Heat Capacity Prediction”**  

The repository is  to calculate **specific heat capacity ($C_p$)** using molecular dynamics (MD), group contribution (GC), and train using machine-learning (ML).

---

---

## Data

The `data/` directory contains the datasets used for model training and evaluation:

- `data/experimental/Cp_exp.csv`  
  Experimental $C_p$ values used for fine-tuning and evaluation.

- `data/gc/Cp_gc.csv`  
  Group-contribution–derived $C_p$ values used as low-fidelity proxy data.

- `data/md/Cp_md.csv`  
  Molecular-dynamics–derived $C_p$ values used as low-fidelity proxy data.

All datasets contain polymer identifiers and SMILES strings consistent with the manuscript.

---

## Molecular Dynamics Workflow

`md_workflow/`

This folder contains all scripts and input files required to compute polymer $C_p$ using molecular dynamics:

- **NEMD (non-equilibrium MD)**: primary method used for all reported MD $C_p$ values  
- **EMD (equilibrium MD)**: included for benchmarking and methodological comparison only

### Contents
- `lammps_inputs/` – LAMMPS input scripts for EMD and NEMD calculations  
- `python/` – Python scripts to run simulations and post-process enthalpy data  

All MD data used for ML pretraining are generated using **NEMD**, as explicitly stated in the manuscript.

---

## Group Contribution Workflow

`gc_workflow/`

This folder implements the group-contribution method used to estimate polymer $C_p$ as a low-fidelity proxy.

### Contents
- `create_group.csv` – Functional group definitions and parameters  
- `calculate_cp.py` – Script to compute $C_p$ from polymer SMILES  

The GC method is used only for pretraining and validation purposes and is not calibrated using experimental data.

---

## Machine Learning Workflow

`ml_workflow/`

This folder contains all machine-learning and transfer-learning scripts used in the paper.

### Contents
- `train_mlp.py`  
  Nested cross-validation pipeline for MLP models with fingerprint and descriptor inputs, including:
  - experimental-only training
  - low-fidelity pretraining (MD or GC)
  - transfer learning with fine-tuning on experimental data


The ML workflow implements:
- five-fold outer cross-validation
- three-fold inner cross-validation for hyperparameter optimization
- strict exclusion of experimental polymers from low-fidelity pretraining data
- no use of bias-corrected low-fidelity labels for training

---

