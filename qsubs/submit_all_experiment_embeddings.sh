#!/bin/bash
# submit_all_experiment_embeddings.sh
# Submit one job per (experiment, dataset) to LSF

set -euo pipefail

# ------------------------------------------------------------------
# Experiment configuration
# ------------------------------------------------------------------
BASE_DATA_DIR="/sc/arion/projects/DiseaseGeneCell/Huang_lab_project/wangcDrugRepoProject/EnzymaticReactionPrediction/Regression_Data/exp_of_catpred_MPEK_EITLEM_inhouse_dataset/experiments"

OUTPUT_DIR="output/data"
LOG_DIR="logs"
LOG_LEVEL="DEBUG"
N_SAMPLES=0
NROWS=10
SEED=42

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# List of experiments you want to process
EXPERIMENTS=(
  dataset_catpred_kcat
  dataset_catpred_ki
  dataset_catpred_km
  dataset_EITLEM_kcat
  dataset_EITLEM_kkm
  dataset_EITLEM_km
  dataset_inhouse_kd
  dataset_MPEK_kcat
)

# Helper: given an experiment name, return one or more data basenames
get_data_basenames_for_experiment() {
    local exp="$1"
    case "$exp" in
        dataset_catpred_kcat)
            echo "kcat_with_features.joblib"
            ;;
        dataset_catpred_ki)
            echo "ki_with_features.joblib"
            ;;
        dataset_catpred_km)
            echo "kcat_data_with_features.joblib"   # adjust if needed
            ;;
        dataset_EITLEM_kcat)
            echo "data_km_with_features.joblib"     # adjust if needed
            ;;
        dataset_EITLEM_kkm)
            echo "kkm_data_with_features.joblib"
            ;;
        dataset_EITLEM_km)
            echo "km_data_with_features.joblib"
            ;;
        dataset_inhouse_kd)
            echo "data.joblib"
            ;;
        dataset_MPEK_kcat)
            # Special case: two datasets
            echo "data_kcat_with_features.joblib data_km_with_features.joblib"
            ;;
        *)
            echo "Unknown EXPERIMENT: $exp" >&2
            return 1
            ;;
    esac
}

# ------------------------------------------------------------------
# Main submission loop
# ------------------------------------------------------------------
for EXPERIMENT in "${EXPERIMENTS[@]}"; do
    # Resolve default(s) from EXPERIMENT
    read -r -a DATA_BASENAMES <<<"$(get_data_basenames_for_experiment "$EXPERIMENT")"

    for DATA_BASENAME in "${DATA_BASENAMES[@]}"; do
        DATA_FN="${BASE_DATA_DIR}/${EXPERIMENT}/A01_dataset/${DATA_BASENAME}"

        DATA_STEM="${DATA_BASENAME##*/}"
        DATA_STEM="${DATA_STEM%.joblib}"
        DATA_STEM="${DATA_STEM// /_}"

        OUTPUT_FN="${OUTPUT_DIR}/${EXPERIMENT}_${DATA_STEM}_with_features_and_embeddings"
        LOG_FN="${LOG_DIR}/extract_embeddings_${EXPERIMENT}_${DATA_STEM}.log"

        JOB_NAME="emb_${EXPERIMENT}_${DATA_STEM}"

        echo "Submitting job ${JOB_NAME}"
        echo "  Data fn:   ${DATA_FN}"
        echo "  Output fn: ${OUTPUT_FN}"
        echo "  Log fn:    ${LOG_FN}"

        # Submit one job per (experiment, dataset)
        bsub -J "$JOB_NAME" run_experiment_embeddings.sh \
            --data_fn "$DATA_FN" \
            --output_fn "$OUTPUT_FN" \
            --log_fn "$LOG_FN" \
            --log_level "$LOG_LEVEL" \
            --n_samples "$N_SAMPLES" \
            --nrows "$NROWS" \
            --seed "$SEED"
    done
done
