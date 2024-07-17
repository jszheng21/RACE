#!/bin/bash

model=$1
output_path_root=$2
backend=$3

# For `Maintainability (MI Metric)`
dataset="classeval"
dim="maintainability_mi"

python -m race.codegen.generate_pipeline pipeline_classeval \
    --generation_strategy "holistic" \
    --model ${model} \
    --dataset ${dataset} \
    --root ${output_path_root} \
    --dim ${dim} \
    --backend ${backend} \
    --n_samples 1 \
    --temperature 0 \
    --greedy True \
    --base_url ${API_BASE} \
    --api_key ${API_KEY}

# For `Maintainability (Modularity)`
dataset="leetcode"
dims=( \
    "maintainability_module_count_1" \
    "maintainability_module_count_2" \
    "maintainability_module_count_3" \
)

for dim in "${dims[@]}"; do
    echo "$dim"
    python -m race.codegen.generate_pipeline pipeline_leetcode \
        --model ${model} \
        --dataset ${dataset} \
        --root ${output_path_root} \
        --dim ${dim} \
        --backend ${backend} \
        --n_samples 1 \
        --temperature 0 \
        --greedy True \
        --base_url ${API_BASE} \
        --api_key ${API_KEY}
done
