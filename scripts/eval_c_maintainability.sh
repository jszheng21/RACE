#!/bin/bash

model=$1
output_path_root=$2

echo "Testing ${model} on Maintainability (Modularity)"
dims=("maintainability_module_count_1" "maintainability_module_count_2" "maintainability_module_count_3" "correctness")
for dim in "${dims[@]}"; do
    echo "$dim"
    docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_leetcode_style test_pipeline_simple \
        --model_name ${model} \
        --evaluation_test_case_path "/data/data/leetcode/evaluation_tests.jsonl" \
        --generated_data_path "/data/${output_path_root}/leetcode_${dim}_${model}_parsed.jsonl" \
        --result_path "/data/${output_path_root}/leetcode_${dim}_${model}_parsed_results.jsonl" \
        --temp_path "/data/${output_path_root}"
done

echo "Testing ${model} on Maintainability (MI)"
dims=("maintainability_mi" "correctness")
for dim in "${dims[@]}"; do
    echo "$dim"
    docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_classeval test_pipeline \
        --model_name ${model} \
        --generated_data_path "/data/${output_path_root}/classeval_${dim}_${model}.jsonl" \
        --root "/data/${output_path_root}"
    
    docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_classeval evaluate_pipeline \
        --model_name ${model} \
        --generated_data_path "/data/${output_path_root}/classeval_${dim}_${model}.jsonl" \
        --root "/data/${output_path_root}"
done