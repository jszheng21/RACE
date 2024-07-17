#!/bin/bash

model=$1
output_path_root=$2
backend=$3

dim="correctness"

# For humaneval and mbpp
datasets=( \
    "humaneval" \
    "mbpp" \
)
for dataset in "${datasets[@]}"; do
    echo "$dataset"
    python -m race.codegen.generate_pipeline pipeline_simple \
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

# For classeval
dataset="classeval"
echo "$dataset"
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

# For leetcode
dataset="leetcode"
echo "$dataset"
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

# For leetcode_efficiency
dataset="leetcode_efficiency"
echo "$dataset"
python -m race.codegen.generate_pipeline pipeline_leetcode_efficiency \
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
