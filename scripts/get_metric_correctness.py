import json
import fire
import os
from race.codeeval.evaluate_pipeline_classeval import EvaluatePipeline
from race.codeeval.evaluate_pipeline_leetcode_style import EvaluateLeetcodeStylePipeline


def output_correctness(model, output_path_root):
    final_results = {model: {'correctness': {}}}
    
    dim = 'correctness'
    
    # 1. humaneval+
    dataset = 'humaneval'
    humaneval_correct_cnt = 0
    humaneval_total_cnt = 0
    
    input_results_file = os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed_eval_results.json')
    with open(input_results_file, 'r') as f:
        results = json.load(f)
        for key in results['eval'].keys():
            is_pass = results['eval'][key][0]['base_status'] == 'pass' and results['eval'][key][0]['plus_status'] == 'pass'
            if is_pass:
                humaneval_correct_cnt += 1
            humaneval_total_cnt += 1
                
    final_results[model]['correctness'][dataset] = round(humaneval_correct_cnt / humaneval_total_cnt * 100, 1)

    # 2. mbpp+
    dataset = 'mbpp'
    mbpp_correct_cnt = 0
    mbpp_total_cnt = 0
    
    input_results_file = os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed_eval_results.json')
    with open(input_results_file, 'r') as f:
        results = json.load(f)
        for key in results['eval'].keys():
            is_pass = results['eval'][key][0]['base_status'] == 'pass' and results['eval'][key][0]['plus_status'] == 'pass'
            if is_pass:
                mbpp_correct_cnt += 1
            mbpp_total_cnt += 1
    
    final_results[model]['correctness'][dataset] = round(mbpp_correct_cnt / mbpp_total_cnt * 100, 1)

    # 3. classeval
    dataset = 'classeval'
    
    pipeline = EvaluatePipeline(model_name=model,
                                generated_data_path=os.path.join(output_path_root, f'{dataset}_{dim}_{model}.jsonl'),
                                root=output_path_root)
    model_list = [model]
    final_results[model]['correctness'][dataset] = round((pipeline.cal_metrics_pass_at_k(model_list, 1, 1)[model]['class_success'] * 100), 1)

    # 4. leetcode
    dataset = 'leetcode'
    
    pipeline = EvaluateLeetcodeStylePipeline(model_name=model,
                                             evaluation_test_case_path='data/leetcode/evaluation_tests.jsonl',
                                             generated_data_path=os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed.jsonl'),
                                             result_path=os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed_results.jsonl'),
                                             temp_path=output_path_root)
    
    final_results[model]['correctness'][dataset] = round(pipeline.evaluate_pipeline_correctness(), 1)
    
    # 5. leetcode_efficiency
    dataset = 'leetcode_efficiency'
    
    pipeline = EvaluateLeetcodeStylePipeline(model_name=model,
                                             evaluation_test_case_path='data/leetcode_efficiency/complexity_evaluation_test_cases.jsonl',
                                             evaluation_efficiency_data_path='data/leetcode_efficiency/complexity_evaluation_data.jsonl', 
                                             generated_data_path=os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed.jsonl'),
                                             result_path=os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed_results.jsonl'),
                                             temp_path=output_path_root)
    final_results[model]['correctness'][dataset] = round(pipeline.evaluate_pipeline_correctness(), 1)
    
    values = final_results[model]['correctness'].values()
    final_results[model]['correctness']['Correctness'] = round(sum(values) / len(values), 1)
    print(final_results)
    
    return final_results

if __name__ == '__main__':
    fire.Fire(output_correctness)
    
