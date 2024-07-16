import os
import fire
from race.codeeval.evaluate_pipeline_leetcode_style import EvaluateLeetcodeStylePipeline


def output_efficiency(model, output_path_root):
    final_results = {model: {'efficiency': {}}}
    
    dim = 'correctness'
    pipeline = EvaluateLeetcodeStylePipeline(model_name=model,
                                             evaluation_test_case_path='data/leetcode_efficiency/complexity_evaluation_test_cases.jsonl',
                                             evaluation_efficiency_data_path='data/leetcode_efficiency/complexity_evaluation_data.jsonl', 
                                             generated_data_path=os.path.join(output_path_root, f'leetcode_efficiency_{dim}_{model}_parsed.jsonl'),
                                             result_path=os.path.join(output_path_root, f'leetcode_efficiency_{dim}_{model}_parsed_results.jsonl'),
                                             temp_path=output_path_root)
    result_p = pipeline.evaluate_pipeline_correctness()

    dim = 'complexity'
    pipeline = EvaluateLeetcodeStylePipeline(model_name=model,
                                             evaluation_test_case_path='data/leetcode_efficiency/complexity_evaluation_test_cases.jsonl',
                                             evaluation_efficiency_data_path='data/leetcode_efficiency/complexity_evaluation_data.jsonl',
                                             generated_data_path=os.path.join(output_path_root, f'leetcode_efficiency_{dim}_{model}_parsed.jsonl'),
                                             result_path=os.path.join(output_path_root, f'leetcode_efficiency_{dim}_{model}_parsed_results.jsonl'),
                                             temp_path=output_path_root)
    _, result_ni_t, result_ni_s = pipeline.evaluate_pipeline_complexity()

    final_results[model]['efficiency']['E*'] = result_p
    final_results[model]['efficiency']['E_NI_T'] = result_ni_t
    final_results[model]['efficiency']['E_NI_S'] = result_ni_s
    final_results[model]['efficiency']['Efficiency'] = round((result_ni_t + result_ni_s) / 2, 1)

    print(final_results)


if __name__ == '__main__':
    fire.Fire(output_efficiency)

