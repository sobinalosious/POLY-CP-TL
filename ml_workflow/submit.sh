#!/bin/bash
#$ -pe smp 4
#$ -l gpu_card=1
#$ -q gpu
#$ -cwd
#$ -j y
#$ -o logs1/$JOB_NAME.$JOB_ID.log

# -------------------------
# Environment
# -------------------------
module load cuda/11.8
conda activate torch_molecule

echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# -------------------------
# Inputs
# -------------------------
EXP_DATA_PATH="./DATA/CP_EXP.csv"
TARGET_COL="Cp"
SMILES_COL="SMILES"

# Results root
RESULTS_ROOT="./results_mlp_GC"
#RESULTS_ROOT="./results_mlp_MD"

mkdir -p logs
mkdir -p "$RESULTS_ROOT"

# -------------------------
# Normalize boolean env vars
# -------------------------
USE_DESC_FLAG=""
if [[ "${USE_DESCRIPTORS}" == "true" || "${USE_DESCRIPTORS}" == "True" || "${USE_DESCRIPTORS}" == "1" ]]; then
  USE_DESC_FLAG="--use_descriptors"
fi

# Stage flags
EXP_FLAG=""
PRE_FLAG=""
FT_FLAG=""

if [[ "${DO_EXP_NESTED}" == "true" || "${DO_EXP_NESTED}" == "True" || "${DO_EXP_NESTED}" == "1" ]]; then
  EXP_FLAG="--do_exp_nested"
fi
if [[ "${DO_PRETRAIN}" == "true" || "${DO_PRETRAIN}" == "True" || "${DO_PRETRAIN}" == "1" ]]; then
  PRE_FLAG="--do_pretrain"
fi
if [[ "${DO_FT_NESTED}" == "true" || "${DO_FT_NESTED}" == "True" || "${DO_FT_NESTED}" == "1" ]]; then
  FT_FLAG="--do_ft_nested"
fi

# Optional: reset head + freezing (adjust if you want)
FREEZE_MODE="none"
RESET_HEAD=""


# -------------------------
# Safety checks for required env vars
# -------------------------
if [[ -z "${FP_METHOD}" ]]; then
  echo "[ERROR] FP_METHOD is not set."
  exit 1
fi

if [[ -z "${LOW_DATA_PATH}" ]]; then
  echo "[ERROR] LOW_DATA_PATH is not set (passed via qsub -v)."
  exit 1
fi

# For PE runs, require PE_MODEL_PATH
PE_FLAG=""
if [[ "${FP_METHOD}" == "pe" ]]; then
  if [[ -z "${PE_MODEL_PATH}" ]]; then
    echo "[ERROR] FP_METHOD=pe but PE_MODEL_PATH is not set."
    exit 1
  fi
  PE_FLAG="--pe_model_path ${PE_MODEL_PATH}"
else
  # Still pass it if provided (harmless), but keep clean:
  if [[ -n "${PE_MODEL_PATH}" ]]; then
    PE_FLAG="--pe_model_path ${PE_MODEL_PATH}"
  fi
fi

# -------------------------
# Echo config
# -------------------------
echo "🔎 FP_METHOD       : ${FP_METHOD}"
echo "🔎 USE_DESCRIPTORS : ${USE_DESCRIPTORS}"
echo "🔎 LOW_DATA_PATH   : ${LOW_DATA_PATH}"
echo "🔎 PE_MODEL_PATH   : ${PE_MODEL_PATH}"
echo "🔎 OUTER/INNER     : ${OUTER_FOLDS}/${INNER_FOLDS}"
echo "🔎 SEED            : ${SEED}"
echo "🔎 EXP_TRIALS      : ${EXP_TRIALS}"
echo "🔎 PRETRAIN_TRIALS : ${PRETRAIN_TRIALS}"
echo "🔎 FT_TRIALS       : ${FT_TRIALS}"
echo "🔎 DO_EXP_NESTED   : ${DO_EXP_NESTED}"
echo "🔎 DO_PRETRAIN     : ${DO_PRETRAIN}"
echo "🔎 DO_FT_NESTED    : ${DO_FT_NESTED}"
echo "🔎 FREEZE_MODE     : ${FREEZE_MODE}"

# Optional debug: versions
python -c "import numpy as np; import rdkit; print('rdkit', rdkit.__version__); print('numpy', np.__version__)"

# -------------------------
# Run
# -------------------------
python train_cp_nestedcv_mlp_fpdesc.py \
  --exp_data_path "$EXP_DATA_PATH" \
  --low_data_path "$LOW_DATA_PATH" \
  --smiles_col "$SMILES_COL" \
  --target_col "$TARGET_COL" \
  --fp_method "$FP_METHOD" \
  $USE_DESC_FLAG \
  $PE_FLAG \
  --outer_folds "$OUTER_FOLDS" \
  --inner_folds "$INNER_FOLDS" \
  --seed "$SEED" \
  --device auto \
  --results_root "$RESULTS_ROOT" \
  --exp_trials "$EXP_TRIALS" \
  --pretrain_trials "$PRETRAIN_TRIALS" \
  --ft_trials "$FT_TRIALS" \
  $EXP_FLAG \
  $PRE_FLAG \
  $FT_FLAG \
  --freeze_mode "$FREEZE_MODE" \
  $RESET_HEAD
