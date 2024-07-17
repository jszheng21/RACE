#!/bin/bash

model=$1
output_path_root=$2
backend=$3

dataset="leetcode_efficiency"
dim="complexity"

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
