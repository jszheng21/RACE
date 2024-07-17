#!/bin/bash

model=$1
output_path_root=$2

echo "Testing ${model} on Efficiency"
dims=("complexity" "correctness")
for dim in "${dims[@]}"; do
    echo "$dim"
    docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_leetcode_style test_pipeline_complexity \
        --model_name ${model} \
        --evaluation_test_case_path "/data/data/leetcode_efficiency/complexity_evaluation_test_cases.jsonl" \
        --evaluation_efficiency_data_path "/data/data/leetcode_efficiency/complexity_evaluation_data.jsonl" \
        --generated_data_path "/data/${output_path_root}/leetcode_efficiency_${dim}_${model}_parsed.jsonl" \
        --result_path "/data/${output_path_root}/leetcode_efficiency_${dim}_${model}_parsed_results.jsonl" \
        --temp_path "/data/${output_path_root}" \
        --timeout 90
done