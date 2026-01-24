#!/bin/bash
#==============================================================
# Per-PID workflow (sourced by submit.sh) — CP ONLY (EMD + NEMD)
#==============================================================

initial_dir="${initial_dir:-$PWD}"

# ---------------- Helpers ----------------
is_true() {
  case "$(echo "${1:-0}" | tr '[:upper:]' '[:lower:]')" in
    1|y|yes|true|on) return 0 ;; *) return 1 ;;
  esac
}

#==============================================================
# --- Job submission / PID resolution (no nested qsub) ---
#==============================================================

if [[ -n "${PID:-}" ]]; then
  echo "[INFO] PID already set (${PID}); running workflow."
else
  if [[ -z "${SGE_TASK_ID:-}" ]]; then
    echo "[ERROR] No PID and no SGE_TASK_ID found."
    echo "       Submit as an array job, e.g.: qsub -t 1-N submit.sh"
    exit 1
  fi

  line=$((CSV_START_LINE + SGE_TASK_ID - 1))
  PID=$(sed -n "${line}p" "$SMILES_FILE" | cut -d',' -f1 | tr -d '\r')
  echo "[INFO] CSV mode: ${SMILES_FILE} line ${line} → PID=${PID}"
fi

if [[ -z "$PID" ]]; then
  echo "[ERROR] No PID resolved! Exiting."
  exit 1
fi

#==============================================================
# --- Workflow (per PID) — CP ONLY ---
#==============================================================

# Step 1: Amorphous Polymer Generation (APG) — Gasteiger only
if is_true "$DO_APG"; then
  echo "[RUN ] APG for $PID with GASTEIGER charges (APG_gas.py)"
  cd 1.AmorphousGeneration || exit 1
  python "APG_gas.py" "$PID" "$NSLOTS"
  cd "$initial_dir" || exit 1
else
  echo "[SKIP] APG disabled (DO_APG=$DO_APG)"
fi

# Step 2: Optimization / Equilibration
if is_true "$DO_OPT"; then
  echo "[RUN ] Optimization for $PID"
  mkdir -p "POLYMER_DATA/OPTIMIZATION/$PID"
  cd "POLYMER_DATA/OPTIMIZATION/$PID" || exit 1

  mpirun -n "$NSLOTS" lmp -in "../../../2.Simulations/lammps_eq1.in" -var name "$PID"
  mpirun -n "$NSLOTS" lmp -in "../../../2.Simulations/lammps_eq2.in" -var name "$PID"

  # Kept (optional) because your OPT step previously updated RESULTS
  python "../../../3.Analysis/calc_density.py" "$PID"
  python "../../../3.Analysis/update_dens.py" "$PID"

  cd "$initial_dir" || exit 1
else
  echo "[SKIP] Optimization disabled (DO_OPT=$DO_OPT)"
fi

# Step 3: EMD → Cp
if is_true "$DO_EMD"; then
  echo "[RUN ] EMD (Cp) for $PID"
  mkdir -p "POLYMER_DATA/EMD/$PID"
  cd "POLYMER_DATA/EMD/$PID" || exit 1

  mpirun -n "$NSLOTS" lmp -in "../../../2.Simulations/lammps_emd_cp.in" -var name "$PID"

  # Cp post-process
  python "../../../3.Analysis/calc_emd_cp.py" "$PID"
  python "../../../3.Analysis/update_emd_cp.py" "$PID"

  cd "$initial_dir" || exit 1
else
  echo "[SKIP] EMD disabled (DO_EMD=$DO_EMD)"
fi

# Step 4: NEMD → Cp  

if is_true "$DO_NEMD"; then
  echo "[RUN ] NEMD (Cp) for $PID"

  mkdir -p "POLYMER_DATA/NEMD_CP/$PID"
  cd "POLYMER_DATA/NEMD_CP/$PID" || exit 1

  # LAMMPS NEMD run for Cp
  mpirun -n "$NSLOTS" lmp -in "../../../2.Simulations/lammps_nemd_cp.in" -var name "$PID"

  # Post-process + update Cp (NEMD)
  python "../../../3.Analysis/calc_nemd_cp.py" "$PID"
  python "../../../3.Analysis/update_nemd_cp.py" "$PID"

  cd "$initial_dir" || exit 1
else
  echo "[SKIP] NEMD disabled (DO_NEMD=$DO_NEMD)"
fi

echo "CP workflow complete for PID=$PID (EMD=$DO_EMD, NEMD=$DO_NEMD)"
