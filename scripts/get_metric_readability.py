import json
import fire
from functools import partial
from race.codeeval.metrics_utils import *


metrics_mapping = {
    'readability_name_camel': metrics_for_readability_camel,
    'readability_name_snake': metrics_for_readability_snake,
    'readability_name_function_camel': metrics_for_readability_function_camel,
    'readability_name_function_snake': metrics_for_readability_function_snake,
    'readability_name_var_camel': metrics_for_readability_var_camel,
    'readability_name_var_snake': metrics_for_readability_var_snake,
    'readability_length_setting_1': partial(metrics_for_readability_length, max_line_length=60, max_function_length=20),
    'readability_length_setting_2': partial(metrics_for_readability_length, max_line_length=70, max_function_length=30),
    'readability_length_setting_3': partial(metrics_for_readability_length, max_line_length=79, max_function_length=40),
    'readability_comment_by_function': metrics_for_readability_comment_by_function,
    'readability_comment_by_line': metrics_for_readability_comment_by_line,
    'readability_arg_count': metrics_for_readability_arg_count,
    'maintainability_loop_for': partial(metrics_for_maintainability_loop, loop_type='for'),
    'maintainability_loop_while': partial(metrics_for_maintainability_loop, loop_type='while')
}


def output_readability(model, output_path_root):
    dims = [
        'correctness', 
        'readability_name_camel', 
        'readability_name_snake', 
        'readability_name_function_camel', 
        'readability_name_function_snake', 
        'readability_name_var_camel', 
        'readability_name_var_snake', 
        'readability_length_setting_1', 
        'readability_length_setting_2', 
        'readability_length_setting_3', 
        'readability_comment_by_function', 
        'readability_comment_by_line', 
    ]
    
    dataset = 'humaneval'

    final_results = {model: {'readability': {}}}
        
    overall = {'correctness': 0, 'readability_name': 0, 'readability_length': 0, 'readability_comment': 0}
    overall_cnt = {'correctness': 0, 'readability_name': 0, 'readability_length': 0, 'readability_comment': 0}
    
    overall_p = {'readability_name': 0, 'readability_length': 0, 'readability_comment': 0}
    overall_p_cnt = {'readability_name': 0, 'readability_length': 0, 'readability_comment': 0}
    
    overall_if = {'readability_name': 0, 'readability_length': 0, 'readability_comment': 0}
    overall_if_cnt = {'readability_name': 0, 'readability_length': 0, 'readability_comment': 0}

    for dim in dims:
        correct_r_p_cnt = 0
        correct_p_cnt = 0
        correct_if_cnt = 0
        correct_pif_cnt = 0
        
        overall_total_cnt = 0
        
        input_file = os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed.jsonl')
        input_results_file = os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed_eval_results.json')
        
        generated_data = []
        generated_data_result_mapping = {}
        
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)
                generated_data.append(line)
                
        overall_total_cnt += len(generated_data)
                
        with open(input_results_file, 'r') as f:
            results = json.load(f)
            for key in results['eval'].keys():
                is_pass = results['eval'][key][0]['base_status'] == 'pass' and results['eval'][key][0]['plus_status'] == 'pass'
                generated_data_result_mapping[results['eval'][key][0]['task_id']] = is_pass
                if is_pass:
                    correct_r_p_cnt += 1
        
        if dim != 'correctness':
            for line in generated_data:
                if generated_data_result_mapping[line['task_id']]:
                    correct_p_cnt += 1
                
                if metrics_mapping[dim](line['solution']):
                    correct_if_cnt += 1
                    if generated_data_result_mapping[line['task_id']]:
                        correct_pif_cnt += 1
        
        for key in overall.keys():
            if key in dim:
                if dim == 'correctness':
                    overall[key] += round(correct_r_p_cnt / overall_total_cnt * 100, 1)
                    overall_cnt[key] += 1
                else:
                    overall[key] += round(correct_pif_cnt / overall_total_cnt * 100, 1)
                    overall_cnt[key] += 1
                    
                    overall_p[key] += round(correct_p_cnt / overall_total_cnt * 100, 1)
                    overall_p_cnt[key] += 1
                    
                    overall_if[key] += round(correct_if_cnt / overall_total_cnt * 100, 1)
                    overall_if_cnt[key] += 1

                break
            
    
    tmp_dict = {}
    for key in overall.keys():            
        if key in overall_p_cnt and overall_p_cnt[key] > 0:
            if key == 'readability_name':
                latest_key = 'RN_p'
            elif key == 'readability_length':
                latest_key = 'RL_p'
            elif key == 'readability_comment':
                latest_key = 'RC_p'
            
            tmp_dict[latest_key] = round(overall_p[key] / overall_p_cnt[key], 1)
            
        if key in overall_if_cnt and overall_if_cnt[key] > 0:
            if key == 'readability_name':
                latest_key = 'RN_if'
            elif key == 'readability_length':
                latest_key = 'RL_if'
            elif key == 'readability_comment':
                latest_key = 'RC_if'
                
            tmp_dict[latest_key] = round(overall_if[key] / overall_if_cnt[key], 1)
        
        if overall_cnt[key] > 0:
            if key == 'correctness':
                latest_key = 'R*'
            elif key == 'readability_name':
                latest_key = 'RN'
            elif key == 'readability_length':
                latest_key = 'RL'
            elif key == 'readability_comment':
                latest_key = 'RC'
                
            tmp_dict[latest_key] = round(overall[key] / overall_cnt[key], 1)
    
    dataset = 'mbpp'
    dim = 'correctness'
    
    input_results_file = os.path.join(output_path_root, f'{dataset}_{dim}_{model}_parsed_eval_results.json')
    with open(input_results_file, 'r') as f:
        results = json.load(f)
        mbpp_total_cnt = 0
        mbpp_correct_cnt = 0
        for key in results['eval'].keys():
            is_pass = results['eval'][key][0]['base_status'] == 'pass' and results['eval'][key][0]['plus_status'] == 'pass'
            generated_data_result_mapping[results['eval'][key][0]['task_id']] = is_pass
            if is_pass:
                mbpp_correct_cnt += 1
            mbpp_total_cnt += 1
    tmp_dict['MBPP*'] = round(mbpp_correct_cnt / mbpp_total_cnt * 100, 1)
            
    tmp_dict['Readability'] = round((tmp_dict['RN'] + tmp_dict['RL'] + tmp_dict['RC']) / 3, 1)
    
    final_results[model]['readability'] = tmp_dict
    print(final_results)


if __name__ == '__main__':
    fire.Fire(output_readability)
    
