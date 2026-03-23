#!/bin/bash

set -euo pipefail

IMAGE_ROOT="/app/blender"
VQA_JSON="/app/blender/output/phasevar_5/vqa_obj1.json"

VARIANTS="obj1_closer obj2_closer obj1_farther obj2_farther"

# model_path|suffix|label|gpu
MODELS=(
    "Qwen/Qwen2.5-VL-3B-Instruct|_3b|Qwen2.5-VL-3B|0"
    "Qwen/Qwen2.5-VL-7B-Instruct||Qwen2.5-VL-7B|1"
    "/app/qwen/models/80k/checkpoint-1250|_80k|Qwen2.5-VL-3B-80k|2"
    "/app/qwen/models/400k/checkpoint-6250|_400k|Qwen2.5-VL-3B-400k|3"
    "/app/qwen/models/800k/checkpoint-12500|_800k|Qwen2.5-VL-3B-800k|4"
    "/app/qwen/models/2m/checkpoint-31252|_2m|Qwen2.5-VL-3B-2m|5"
)

run_model() {
    local MODEL_PATH="$1"
    local SUFFIX="$2"
    local LABEL="$3"
    local GPU="$4"

    local BASE_CSV="logit_results_qwen_vqa_phase${SUFFIX}.csv"

    # echo "========== ${LABEL} | phasevar (all variants) | GPU ${GPU} =========="
    # CUDA_VISIBLE_DEVICES=$GPU python /app/qwen/infer_logit_vqa.py \
        # --model-path "$MODEL_PATH" \
        # --vqa-json "$VQA_JSON" \
        # --image-root "$IMAGE_ROOT" \
        # --output-csv "$BASE_CSV"

    echo "========== Plotting per-variant heatmaps =========="
    local VARIANT_CSVS=""
    for V in $VARIANTS; do
        local V_CSV="logit_results_qwen_vqa_phase${SUFFIX}_${V}.csv"
        VARIANT_CSVS="$VARIANT_CSVS $V_CSV"
        python /app/VILA/plot_heatmap_vqa.py \
            --input "$V_CSV" \
            --output "logit_heatmap_qwen_vqa_phase${SUFFIX}_${V}.png" \
            --title "${LABEL} | ${V}"
    done

    echo "========== Plotting aggregate heatmap =========="
    python /app/VILA/plot_heatmap_vqa_agg.py \
        --inputs $VARIANT_CSVS \
        --output "logit_heatmap_qwen_vqa_phase${SUFFIX}_agg.png" \
        --title "${LABEL} | Mean P(correct)"
}

# Launch all models in parallel, one GPU each
pids=()

for entry in "${MODELS[@]}"; do
    IFS='|' read -r MODEL_PATH SUFFIX LABEL GPU <<< "$entry"
    run_model "$MODEL_PATH" "$SUFFIX" "$LABEL" "$GPU" \
        > "output_qwen_vqa_phase${SUFFIX}.log" 2>&1 &
    pids+=($!)
    echo "Launched ${LABEL} on GPU ${GPU} (PID: ${pids[-1]})"
done

echo "All jobs launched (PIDs: ${pids[*]}). Waiting..."

failed=0
for i in "${!MODELS[@]}"; do
    IFS='|' read -r _ SUFFIX LABEL _ <<< "${MODELS[$i]}"
    if wait "${pids[$i]}"; then
        echo "=== Done: ${LABEL} ==="
    else
        echo "=== FAILED: ${LABEL} (see output_qwen_vqa_phase${SUFFIX}.log) ===" >&2
        failed=1
    fi
done

if [ "$failed" -eq 1 ]; then
    echo "Some runs failed." >&2
    exit 1
fi

echo "All done."
