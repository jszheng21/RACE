#!/bin/bash

model=$1
output_path_root=$2

dataset="humaneval"
dims=(
    "correctness" \
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

echo "Testing ${model} on Readability"
for dim in "${dims[@]}"; do
    echo "$dim"
    docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_evalplus \
        --dataset ${dataset} \
        --samples /data/${output_path_root}/${dataset}_${dim}_${model}_parsed.jsonl
done

dataset="mbpp"
dim="correctness"

docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_evalplus \
    --dataset ${dataset} \
    --samples /data/${output_path_root}/${dataset}_${dim}_${model}_parsed.jsonl
