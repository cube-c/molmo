#!/bin/bash

set -e

for SPLIT in h11 h12 h13; do
    python /app/VILA/plot_heatmap.py \
        --input "logit_results_molmo_${SPLIT}.csv" \
        --output "logit_heatmap_molmo_${SPLIT}.png" \
        --title "Molmo-7B-O | $SPLIT — Yes - No Logit Diff"

    python /app/VILA/plot_heatmap.py \
        --input "logit_results_molmo_80k_${SPLIT}.csv" \
        --output "logit_heatmap_molmo_80k_${SPLIT}.png" \
        --title "Molmo-7B-O 80K (local) | $SPLIT — Yes - No Logit Diff"

    python /app/VILA/plot_heatmap.py \
        --input "logit_results_molmo_400k_${SPLIT}.csv" \
        --output "logit_heatmap_molmo_400k_${SPLIT}.png" \
        --title "Molmo-7B-O 400K (local) | $SPLIT — Yes - No Logit Diff"

    python /app/VILA/plot_heatmap.py \
        --input "logit_results_molmo_800k_${SPLIT}.csv" \
        --output "logit_heatmap_molmo_800k_${SPLIT}.png" \
        --title "Molmo-7B-O 800K (local) | $SPLIT — Yes - No Logit Diff"

    python /app/VILA/plot_heatmap.py \
        --input "logit_results_molmo_2m_${SPLIT}.csv" \
        --output "logit_heatmap_molmo_2m_${SPLIT}.png" \
        --title "Molmo-7B-O 2M (local) | $SPLIT — Yes - No Logit Diff"
done

python /app/VILA/plot_logit_stats.py \
    --csv-dir . \
    --prefix molmo \
    --output logit_stats_molmo.png \
    --title "Molmo Logit Diff (Yes − No) by Model & Split"

echo "All done."
