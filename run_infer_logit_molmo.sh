#!/bin/bash

set -e

TEXT="Is the red sphere closer to the camera than the blue sphere? Answer with yes or no."

for SPLIT in h11 h12 h13; do
    echo "========== Molmo-7B-O | $SPLIT =========="
    # python /app/molmo/infer_logit.py \
        # --model-path "allenai/Molmo-7B-O-0924" \
        # --text "$TEXT" \
        # --media-dir "/app/blender/$SPLIT" \
        # --output-csv "logit_results_molmo_${SPLIT}.csv"

    # echo "========== Molmo-7B-O 80K (local) | $SPLIT =========="
    # python /app/molmo/infer_logit.py \
        # --model-path "/app/DATA/data_scale_exp_80k/unshared" \
        # --text "$TEXT" \
        # --media-dir "/app/blender/$SPLIT" \
        # --output-csv "logit_results_molmo_80k_${SPLIT}.csv"

    echo "========== Molmo-7B-O 400K (local) | $SPLIT =========="
    python /app/molmo/infer_logit.py \
        --model-path "/app/DATA/data_scale_exp_400k/unshared" \
        --text "$TEXT" \
        --media-dir "/app/blender/$SPLIT" \
        --output-csv "logit_results_molmo_400k_${SPLIT}.csv"

    echo "========== Molmo-7B-O 800K (local) | $SPLIT =========="
    python /app/molmo/infer_logit.py \
        --model-path "/app/DATA/data_scale_exp_800k/unshared" \
        --text "$TEXT" \
        --media-dir "/app/blender/$SPLIT" \
        --output-csv "logit_results_molmo_800k_${SPLIT}.csv"

    # echo "========== Molmo-7B-O 2M (local) | $SPLIT =========="
    # python /app/molmo/infer_logit.py \
        # --model-path "/app/DATA/data_scale_exp_2m/unshared" \
        # --text "$TEXT" \
        # --media-dir "/app/blender/$SPLIT" \
        # --output-csv "logit_results_molmo_2m_${SPLIT}.csv"
done

echo "All done."
