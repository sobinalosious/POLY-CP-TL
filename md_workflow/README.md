# Polymer MD Workflow: Specific Heat Capacity 

This workflow automates the generation of polymer structures from SMILES, performs molecular dynamics (MD) simulations using LAMMPS, and computes Specific Heat Capacity

The workflow is designed for an HPC cluster with:
- Sun Grid Engine (SGE) or compatible scheduler
- LAMMPS installed as a module
- Conda for Python environment management


---

## 1. Create the Conda Environment

Load the conda module on your HPC:

```bash
module load conda
```

Create the environment:

```bash
conda env create -f polymer-workflow.yml
```

Activate it:

```bash
conda activate polymer-workflow
```
---

## 2. Install PySIMM


conda activate polymer-workflow
git clone https://github.com/polysimtools/pysimm.git
cd pysimm
pip install .
Add the following line to your ~/.bashrc file:
export LAMMPS_EXEC="export LAMMPS_EXEC=/usr/bin/lmp"   (change as per your path )
source ~/.bashrc

OR

Inside the active conda environment:

git clone https://github.com/polysimtools/pysimm
python pysimm/complete_install.py --pysimm $PWD

PySIMM handles polymer chain construction, random-walk polymerization, GAFF2 parameter assignment, RESP/Gasteiger charges, and LAMMPS data file generation.


---

## 3. Load LAMMPS on HPC

LAMMPS is loaded using your cluster's module system. For example:

```bash
module load lammps
```

LAMMPS does not need to be installed inside the conda environment.

---


## 5. Using the Submission Script

To run the first 50 polymers:

```bash
qsub -t 1-50 submit.sh
```

Logs are stored in:

```text
Log_Files/
```

---

## 6. Property Toggles

Inside `submit.sh`, you will find:

```bash
DO_APG=
DO_OPT=
DO_EMD=0        # Cp from EMD
DO_NEMD=1       # Cp from NEMD
```

Set each option to:
- `1` = enable
- `0` = disable

## 7. Directory Structure (Auto-created)

The workflow will create folders similar to:

```text
POLYMER_DATA/RESULTS/
    RHO_MD.csv
    CP_MD_EMD.csv
    CP_MD_NEMD.csv



```

Each PID has its own subfolder under `POLYMER_DATA`.

---

