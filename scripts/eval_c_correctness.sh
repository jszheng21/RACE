#!/bin/bash

model=$1
output_path_root=$2


echo "Testing ${model} on HumanEval+"
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_evalplus \
    --dataset humaneval \
    --samples /data/${output_path_root}/humaneval_correctness_${model}_parsed.jsonl


echo "Testing ${model} on MBPP+"
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_evalplus \
    --dataset mbpp \
    --samples /data/${output_path_root}/mbpp_correctness_${model}_parsed.jsonl


echo "Testing ${model} on LeetCode"
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_leetcode_style test_pipeline_simple \
    --model_name ${model} \
    --evaluation_test_case_path "/data/data/leetcode/evaluation_tests.jsonl" \
    --generated_data_path "/data/${output_path_root}/leetcode_correctness_${model}_parsed.jsonl" \
    --result_path "/data/${output_path_root}/leetcode_correctness_${model}_parsed_results.jsonl" \
    --temp_path "/data/${output_path_root}"


echo "Testing ${model} on ClassEval"
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_classeval test_pipeline \
    --model_name ${model} \
    --generated_data_path "/data/${output_path_root}/classeval_correctness_${model}.jsonl" \
    --root "/data/${output_path_root}"

docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_classeval evaluate_pipeline \
    --model_name ${model} \
    --generated_data_path "/data/${output_path_root}/classeval_correctness_${model}.jsonl" \
    --root "/data/${output_path_root}"


echo "Testing ${model} on LeetCode_Efficiency"
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_leetcode_style test_pipeline_complexity \
    --model_name ${model} \
    --evaluation_test_case_path "/data/data/leetcode_efficiency/complexity_evaluation_test_cases.jsonl" \
    --evaluation_efficiency_data_path "/data/data/leetcode_efficiency/complexity_evaluation_data.jsonl" \
    --generated_data_path "/data/${output_path_root}/leetcode_efficiency_correctness_${model}_parsed.jsonl" \
    --result_path "/data/${output_path_root}/leetcode_efficiency_correctness_${model}_parsed_results.jsonl" \
    --temp_path "/data/${output_path_root}" \
    --timeout 90
