#!/bin/bash

model=$1
output_path_root=$2
backend=$3

dataset="humaneval"

dims=( \
    "readability_name_camel" \
    "readability_name_snake" \
    "readability_name_function_camel" \
    "readability_name_function_snake" \
    "readability_name_var_camel" \
    "readability_name_var_snake" \
    "readability_length_setting_1" \
    "readability_length_setting_2" \
    "readability_length_setting_3" \
    "readability_comment_by_function" \
    "readability_comment_by_line" \
)

for dim in "${dims[@]}"; do
    echo "$dim"
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
