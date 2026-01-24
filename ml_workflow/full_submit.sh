#!/bin/bash

# Fingerprint methods (must match python choices)
# choices: morgan rdkit maccs topologicaltorsion atompair pe
fp_methods=("morgan" "rdkit" "maccs" "topologicaltorsion"  "atompair" "pe")

# Descriptor options (true/false)
use_descriptors_options=("true" "false")

# LOW datasets to sweep (add as many as you want)
low_data_paths=(
  "../data/md/CP_MD.csv" "../data/gc/CP_GC.csv"
)

# PE model path to sweep (optional list; typically just one)
pe_model_paths=(
  "../data/pe/POLYINFO_PI1M.pkl"
)

# Common experiment settings
OUTER_FOLDS=5
INNER_FOLDS=3
SEED=301
#42,6,1206,99,301
# Optuna trials
EXP_TRIALS=50
PRETRAIN_TRIALS=50
FT_TRIALS=200

# Which stages to run
DO_EXP_NESTED="true"
DO_PRETRAIN="true"
DO_FT_NESTED="true"

# Loop through all combinations and submit jobs
for fp_method in "${fp_methods[@]}"; do
  for use_desc in "${use_descriptors_options[@]}"; do
    for low_path in "${low_data_paths[@]}"; do
      for pe_path in "${pe_model_paths[@]}"; do

        # Make low path job-safe label (remove slashes/dots)
        low_tag=$(echo "$low_path" | sed 's|/|_|g; s|\.|_|g')

        job_name="MLP_${fp_method}_${use_desc}_${low_tag}"
        echo "Submitting: $job_name"

        qsub -N "$job_name" \
          -v FP_METHOD="$fp_method",USE_DESCRIPTORS="$use_desc",LOW_DATA_PATH="$low_path",PE_MODEL_PATH="$pe_path",OUTER_FOLDS="$OUTER_FOLDS",INNER_FOLDS="$INNER_FOLDS",SEED="$SEED",EXP_TRIALS="$EXP_TRIALS",PRETRAIN_TRIALS="$PRETRAIN_TRIALS",FT_TRIALS="$FT_TRIALS",DO_EXP_NESTED="$DO_EXP_NESTED",DO_PRETRAIN="$DO_PRETRAIN",DO_FT_NESTED="$DO_FT_NESTED" \
          submit.sh

      done
    done
  done
done
