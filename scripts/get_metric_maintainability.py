import os
import fire
from race.codeeval.evaluate_pipeline_classeval import EvaluatePipeline
from race.codeeval.evaluate_pipeline_leetcode_style import EvaluateLeetcodeStylePipeline


def output_maintainability_mi(model, output_path_root):
    final_results = {model: {'maintainability': {}}}
    
    dim = 'correctness'
    pipeline = EvaluatePipeline(model_name=model,
                                generated_data_path=os.path.join(output_path_root, f'classeval_{dim}_{model}.jsonl'),
                                root=output_path_root)
    model_list = [model]
    result_p = round((pipeline.cal_metrics_pass_at_k(model_list, 1, 1)[model]['class_success'] * 100), 1)
    
    dim = 'maintainability_mi'
    pipeline = EvaluatePipeline(model_name=model,
                                generated_data_path=os.path.join(output_path_root, f'classeval_{dim}_{model}.jsonl'),
                                root=output_path_root)
    model_list = [model]
    result_mi = pipeline.evaluate_pipeline_mi()
    
    final_results[model]['maintainability']['MI*'] = result_p
    final_results[model]['maintainability']['MI'] = result_mi
        
    print(final_results)
        

def output_maintainability_module_count(model, output_path_root):
    dims = [
        'correctness',
        'maintainability_module_count_1', 
        'maintainability_module_count_2', 
        'maintainability_module_count_3',
    ]

    final_results = {model: {'maintainability': {}}}

    result_p = 0
    total_result_p_if = 0
    for i in range(len(dims)):
        dim = dims[i]
        pipeline = EvaluateLeetcodeStylePipeline(model_name=model,
                                                 evaluation_test_case_path='data/leetcode/evaluation_tests.jsonl',
                                                 generated_data_path=os.path.join(output_path_root, f'leetcode_{dim}_{model}_parsed.jsonl'),
                                                 result_path=os.path.join(output_path_root, f'leetcode_{dim}_{model}_parsed_results.jsonl'),
                                                 temp_path=output_path_root)
    
        
        if dim == 'correctness':
            result_p = pipeline.evaluate_pipeline_correctness()
        else:
            _, _, result_p_if = pipeline.evaluate_pipeline_maintainability_module_count(int(dim[-1]))
            total_result_p_if += result_p_if
        
    final_results[model]['maintainability']['MC*'] = round(result_p, 1)
    final_results[model]['maintainability']['MC'] = round(total_result_p_if / 3, 1)
        
    print(final_results)


def output_maintainability(model, output_path_root):
    output_maintainability_mi(model, output_path_root)
    output_maintainability_module_count(model, output_path_root)


if __name__ == '__main__':
    fire.Fire(output_maintainability)

