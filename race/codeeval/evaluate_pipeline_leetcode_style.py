import os
import json
import fire
from collections import defaultdict
from race.codeeval.human_eval.evaluation import evaluate_functional_correctness
from race.codeeval.metrics_utils import metrics_for_maintainability_module_count
from race.codeeval.execution_utils import test_time_and_memory_usage


class EvaluateLeetcodeStylePipeline():
    def __init__(
        self, 
        model_name, 
        evaluation_test_case_path, 
        generated_data_path,
        result_path,
        temp_path,
        evaluation_efficiency_data_path=None,
        root=None,
        timeout=60,
        num_process_evaluate=16,
    ):
        self.model_name = model_name
        self.evaluation_test_case_path = evaluation_test_case_path
        self.generated_data_path = generated_data_path
        self.result_path = result_path
        self.temp_path = temp_path
        self.evaluation_efficiency_data_path = evaluation_efficiency_data_path
        self.root = root
        self.timeout = timeout
        self.num_process_evaluate = num_process_evaluate

    def load_problems(self, evaluation_test_case_path):
        if not evaluation_test_case_path.endswith('.jsonl'):
            return
        
        print(f'Reading `problems` from {evaluation_test_case_path}')

        problems = {}
        with open(evaluation_test_case_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                problems[line['task_id']] = line
        
        return problems

    def load_generated_data_list(self, generated_data_path):
        if not generated_data_path.endswith('.jsonl'):
            return
        
        print(f'Reading `generated_data_list` from {generated_data_path}')

        return [json.loads(line) for line in open(generated_data_path, 'r')]

    def save_result(self, result, type):
        save_path = os.path.join(self.root, f'{self.model_name}_{type}.json')
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

            print(f'Written to {save_path}')
            
    def test_pipeline_simple(self):
        self.generated_data_list = self.load_generated_data_list(self.generated_data_path)
        self.problems = self.load_problems(self.evaluation_test_case_path)
        print(f"Loaded {len(self.problems)} cases.")
        
        for line in self.generated_data_list:
            line['generation'] = line['solution']
            del line['solution']

        self.score = evaluate_functional_correctness(
            samples_list=self.generated_data_list,
            problems=self.problems,
            tmp_dir=self.temp_path,
            timeout=self.timeout,
            result_path=self.result_path
        )

    def test_pipeline_complexity(self):
        test_time_and_memory_usage(self.evaluation_test_case_path, self.generated_data_path, self.result_path, self.timeout)

    def evaluate_pipeline(self):
        hardness_results = defaultdict(int)
        for result in [json.loads(line) for line in open(self.result_path, 'r')]:
            problem = self.problems[result['task_id']]

            hardness = problem['meta']['difficulty']
            hardness_results[hardness] += 1
            hardness_results[hardness + "_correct"] += result['passed']

        print("=" * 100)
        print("Pass@1: {:.3f}".format(self.score["pass@1"]))

        for key in ["Easy", "Medium", "Hard"]:
            if key.endswith("_correct") or key not in hardness_results:
                continue
            acc = hardness_results[key + "_correct"] / hardness_results[key]
            print("{}: {:.3f}({}/{})".format(key, acc, hardness_results[key+"_correct"], hardness_results[key]))
        
    def evaluate_pipeline_complexity(self):
        results = [json.loads(line) for line in open(self.result_path, 'r')]
        
        evaluations = []
        with open(self.evaluation_efficiency_data_path, 'r') as f:
            for line in f:
                line = json.loads(line)
                evaluations.append(line)
        
        total_time_cnt = 0
        total_space_cnt = 0
        total_cnt = len(results)
        
        p_cnt = 0
        total_time_NI = 0
        total_space_NI = 0
        
        for i in range(len(results)):
            if 'time' in results[i]['instruction'] and 'space' in results[i]['instruction']:
                total_time_cnt += 1
                total_space_cnt += 1
            elif 'time' in results[i]['instruction']:
                total_time_cnt += 1
            elif 'space' in results[i]['instruction']:
                total_space_cnt += 1
            
            if results[i]['passed']:
                p_cnt += 1
                
                if 'time' in results[i]['instruction'] and 'space' in results[i]['instruction']:
                    T_hat = results[i]['running_time']
                    S_hat = results[i]['peak_memory_usage']
                    
                    T_1 = evaluations[i]['T_1']
                    T_2 = evaluations[i]['T_2']
                    
                    S_1 = evaluations[i]['S_1']
                    S_2 = evaluations[i]['S_2']
                    
                    to_reverse = 1
                    if T_1 > T_2:
                        to_reverse = -1
                        
                    time_NI = 100 * max(min(1 - (T_hat - T_1) / (to_reverse * (T_2 - T_1)), 1), 0)
                    space_NI = 100 * max(min(1 - (S_hat - S_1) / (to_reverse * (S_1 - S_2)), 1), 0)  
                    total_time_NI += time_NI
                    total_space_NI += space_NI
                    
                    # TODO
                    # print(f"{results[i]['task_id']:-^100}")
                    # print(f'{T_hat} {S_hat}')
                    # print(f'{T_1} {T_2}  {S_1} {S_2}')
                    # print(f'{time_NI} {space_NI}')
                elif 'time' in results[i]['instruction']:
                    T_1 = evaluations[i]['T_1']
                    T_2 = evaluations[i]['T_2']
                    
                    T_hat = results[i]['running_time']
                    time_NI = 100 * max(min(1 - (T_hat - T_1) / (T_2 - T_1), 1), 0)
                    total_time_NI += time_NI
                elif 'time' in results[i]['instruction']:
                    T_1 = evaluations[i]['T_1']
                    T_2 = evaluations[i]['T_2']
                    
                    S_hat = results[i]['peak_memory_usage']
                    space_NI = 100 * max(min(1 - (S_hat - S_1) / (S_2 - S_1), 1), 0)
                    total_space_NI += space_NI
        
        print(f'P rate: {p_cnt / total_cnt:.3f} ({p_cnt}/{total_cnt})')
        print(f'Time NI value: {total_time_NI / total_time_cnt:.3f}')
        print(f'Space NI value: {total_space_NI / total_space_cnt:.3f}')

        return round(p_cnt / total_cnt * 100, 1), round(total_time_NI / total_time_cnt, 1), round(total_space_NI / total_space_cnt, 1)
            
    def evaluate_pipeline_correctness(self):
        print(f'Reading from {self.result_path} ...')
        results = [json.loads(line) for line in open(self.result_path, 'r')]

        p_cnt = 0
        for result in results:
            if result['passed']:
                p_cnt += 1
        
        print(f'P rate for efficiency: {p_cnt / len(results):.3f} ({p_cnt}/{len(results)})')
        
        return round(p_cnt / len(results) * 100, 1)
    
    def evaluate_pipeline_maintainability_module_count(self, cnt):
        print(f'Reading from {self.result_path} ...')
        results = [json.loads(line) for line in open(self.result_path, 'r')]

        p_cnt = 0
        p_if_cnt = 0
        if_cnt = 0
        for result in results:
            module_count = metrics_for_maintainability_module_count(result['generation'])
            if module_count == cnt:
                if_cnt += 1
            
            if result['passed']:
                p_cnt += 1

            if result['passed'] and module_count == cnt:
                p_if_cnt += 1
        
        print(f'P rate for efficiency: {p_cnt / len(results):.3f} ({p_cnt}/{len(results)})')
        print(f'IF rate for efficiency: {if_cnt / len(results):.3f} ({if_cnt}/{len(results)})')
        print(f'P. IF rate for efficiency: {p_if_cnt / len(results):.3f} ({p_if_cnt}/{len(results)})')
        
        return round(p_cnt / len(results) * 100, 1), round(if_cnt / len(results) * 100, 1), round(p_if_cnt / len(results) * 100, 1)


if __name__ == '__main__':
    fire.Fire(EvaluateLeetcodeStylePipeline)
