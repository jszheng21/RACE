import time
import importlib
import unittest
import json
import re
import os
import fire
from scipy.special import comb
from func_timeout import func_set_timeout
from race.codeeval.metrics_utils import metrics_for_maintainability_cohesion, metrics_for_maintainability_mi


class EvaluatePipeline():
    def __init__(self, model_name, generated_data_path, root):
        self.model_name = model_name
        self.generated_data_path = generated_data_path
        self.generated_data = self.load_generated_data(generated_data_path)
        self.root = root

    def load_generated_data(self, generated_data_path):
        generated_data = {}
        if generated_data_path.endswith('.jsonl'):
            with open(generated_data_path, 'r') as f:
                for line in f:
                    line = json.loads(line)
                    generated_data[line['task_id']] = line['cont']

        return generated_data

    def get_leading_spaces(self, string):
        return len(string) - len(string.lstrip())

    def parse_code(self, text):
        # Extract response from chat completion
        text = text.rstrip()
        # output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
        output_split_identifier_list = ["### Response:", "@@ Response:"]
        for identifier in output_split_identifier_list:
            if identifier in text:
                text = text.split(identifier)[1]
                break

        # Extract code part from response
        pattern_list = [r"```python(.*?)```", r"```ruby(.*?)```", r"```scss(.*?)```",
                        r"```python(.*)", r"```(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]", r"(.*)\[/PYTHON\]", r"\[PYTHON\](.*)"]  # TODO
        for pattern in pattern_list:
            try:
                code = re.findall(pattern, text, re.S)[0]
                return code
            except:
                continue

        # Try to extract code from text-like response without explicit code part
        code_list = text.split("\n")
        removed_lines = []
        for code_line in code_list:
            if code_line.strip().startswith('class'):
                break
            elif not code_line.strip().startswith('import') and not code_line.strip().startswith('from'):
                removed_lines.append(code_line)
        code_list = [line for line in code_list if line not in removed_lines]
        text = '\n'.join(code_list)

        wrong_indent_flag = False
        for code_line in text.split("\n"):
            if code_line.strip().startswith('class'):
                class_signature_line_leading_spaces = self.get_leading_spaces(code_line)
                if class_signature_line_leading_spaces != 0:
                    wrong_indent_flag = True
                break
        if wrong_indent_flag:
            final_code_line_list = []
            for code_line in text.split("\n"):
                cur_leading_spaces = self.get_leading_spaces(code_line)
                # Keep the relative indentation unchanged
                final_code_line_list.append(' ' * (cur_leading_spaces - class_signature_line_leading_spaces) + code_line.lstrip())
            text = '\n'.join(final_code_line_list)
        return text
    
    def add_static_statement(self, code):
        filtered_code_list = []
        for line in code.split('\n'):
            if '@staticmethod' in line:
                continue
            filtered_code_list.append(line)
        code = '\n'.join(filtered_code_list)
        final_code_list = []
        for line in code.split('\n'):
            if line.strip().startswith('def ') and 'self' not in line and 'cls' not in line and self.get_leading_spaces(line) == 4:
                final_code_list.append('    @staticmethod')
            final_code_list.append(line)
        return '\n'.join(final_code_list)

    def gen_task_code_dict(self):
        task_code_dict = {}
        for task_id, cont in self.generated_data.items():
            task_code_dict[task_id] = []
            for predict in cont['predict']:
                predict = self.parse_code(predict)
                predict = self.add_static_statement(predict)
                predict = '\n'.join(cont['import_statement']) + '\n' + predict

                task_code_dict[task_id].append(predict)
        
        print(f"Loaded {len(task_code_dict)} generated samples.")

        return task_code_dict

    def gen_py_file(self, test_code_name, task_code_list, test_code):
        """
        Merge task code and test code as the final test code, and save as a py file
        """
        cnt = 0
        for code_snippet in task_code_list:
            test_code_py = code_snippet + '\n' + test_code
            with open(f'{test_code_name}_{cnt}.py', 'w', encoding='utf-8') as f:
                f.write(test_code_py)
            cnt += 1

    @func_set_timeout(5)
    def run_unit_test(self, test_code, test_class):
        print(test_code)

        try:
            module = importlib.import_module(test_code)

            filename = os.path.split(self.generated_data_path)[1]
            filename_base = os.path.splitext(filename)[0]
            log_path = os.path.join(self.root, filename_base + "_log_data.log")
            with open(log_path, 'a', encoding='utf-8') as f:
                test_suite = unittest.TestLoader().loadTestsFromTestCase(getattr(module, test_class))
                test_result = unittest.TextTestRunner(stream = f).run(test_suite)

            return test_result
        except Exception as e:
            print(e)

    def test(self, code_num, test_code_name, test_classes):
        result = {}
        for i in range(code_num):
            test_code = test_code_name + '_' + str(i)
            result[test_code] = {}

            for test_class in test_classes:
                res_item = {}
                try:
                    res = self.run_unit_test(test_code, test_class)
                    res_item['errors'] = len(res.errors)
                    res_item['failures'] = len(res.failures)
                    res_item['testsRun'] = res.testsRun
                    result[test_code][test_class] = res_item
                except:
                    res_item['errors'] = 0
                    res_item['failures'] = 0
                    res_item['testsRun'] = 0
                    result[test_code][test_class] = res_item

        return result
    
    def save_result(self, result, type):
        filename = os.path.split(self.generated_data_path)[1]
        filename_base = os.path.splitext(filename)[0]
        save_path = os.path.join(self.root, f'{filename_base}_{type}_result.json')
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

    def test_pipeline(self):        
        # Get generated task code dict
        task_code_dict = self.gen_task_code_dict()

        # Get the final test code which is stored in py file
        for task_id in task_code_dict:
            test_code = self.generated_data[task_id]['test']
            task_code_list = task_code_dict[task_id]
            self.gen_py_file(task_id, task_code_list, test_code)

        # Run unit test on specific py files
        result_dict = {}
        for task_id in task_code_dict:
            task_code_list = task_code_dict[task_id]
            try:
                result = self.test(len(task_code_list), 
                                   task_id,
                                   self.generated_data[task_id]['test_classes'])
                result_dict[task_id] = result
            except:
                continue

        # Save result
        self.save_result(result_dict, "class")

    def evaluate_pipeline_mi(self):
        filename = os.path.split(self.generated_data_path)[1]
        filename_base = os.path.splitext(filename)[0]
        
        file_path = os.path.join(self.root, f'{filename_base}_detailed_result.json')
        with open(file_path, 'r') as f:
            test_result = json.load(f)
        
        task_code_dict = self.gen_task_code_dict()

        result_dict = {}

        global_mi_p = 0
        global_cnt_p = 0
        
        global_mi = 0
        global_cnt = 0
        for task_id, codes in task_code_dict.items():
            acc_mi = 0
            for code in codes:
                try:
                    mi = metrics_for_maintainability_mi(code)
                    if test_result[self.model_name][task_id]['TestClass']['class_success'] != 0:
                        global_cnt_p += 1
                        global_mi_p += mi
                except:
                    mi = 0
                acc_mi += mi
            global_mi += acc_mi
            global_cnt += len(codes)
            result_dict[task_id] = round(acc_mi / len(codes), 2)
        
        print(f'Global average mi: {global_mi / global_cnt:.2f}')
        print(f'Global average mi for passed tests: {global_mi_p / global_cnt_p:.2f} ({global_cnt_p})')

        self.save_result(result_dict, "mi")
        
        return round(global_mi_p / global_cnt_p, 1)
    
    def evaluate_pipeline_cohesion(self):
        task_code_dict = self.gen_task_code_dict()

        result_dict = {}

        global_cohesion = 0
        global_cnt = 0
        local_cnt = 0
        for task_id, codes in task_code_dict.items():
            acc_cohesion = 0
            for code in codes:
                try:
                    cohesion = metrics_for_maintainability_cohesion(code)
                    if cohesion != -1:
                        local_cnt += 1
                        acc_cohesion += cohesion
                except:
                    pass
                
            global_cohesion += acc_cohesion
            result_dict[task_id] = round(acc_cohesion / local_cnt, 2)
            global_cnt += local_cnt
        
        print(f'Global average mi: {global_cohesion / global_cnt:.2f}')

        self.save_result(result_dict, "cohesion")

    def evaluate_pipeline(self):
        result_dict = {}
        model_name = self.model_name
        
        filename = os.path.split(self.generated_data_path)[1]
        filename_base = os.path.splitext(filename)[0]
        model_result_path = os.path.join(self.root, f'{filename_base}_class_result.json')
        with open(model_result_path, 'r') as f:
            model_result = json.load(f)

        result_dict[model_name] = {}
        for task in model_result:
            result_dict[model_name][task] = {}
            for test_num in model_result[task]:
                temp_result = {"success": 0, "partial_success": 0, "fail": 0, "error": 0}
                for test_class in model_result[task][test_num]:
                    if test_class not in result_dict[model_name][task]:
                        result_dict[model_name][task][test_class] = {}
                        result_dict[model_name][task]["TestClass"] = {}
                        result_dict[model_name][task]["TestClass"]["ClassEachTestResult"] = []
                        result_dict[model_name][task][test_class]['success'] = 0
                        result_dict[model_name][task][test_class]['partial_success'] = 0
                        result_dict[model_name][task][test_class]['fail'] = 0
                        result_dict[model_name][task][test_class]['error'] = 0
                        result_dict[model_name][task][test_class]["EachTestResult"] = []
                        result_dict[model_name][task]["TestClass"]["class_success"] = 0
                        result_dict[model_name][task]["TestClass"]["class_partial_success"] = 0
                        result_dict[model_name][task]["TestClass"]["class_fail"] = 0
                    test_answer = self.get_test_answer(model_result[task][test_num][test_class])
                    result_dict[model_name][task][test_class][test_answer] += 1
                    result_dict[model_name][task][test_class]["EachTestResult"].append(test_answer)
                    temp_result[test_answer] += 1

                if temp_result['success'] == len(model_result[task][test_num]):
                    result_dict[model_name][task]["TestClass"]["class_success"] += 1
                    result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append("class_success")
                elif temp_result['fail'] == 0 and temp_result['error'] == 0:
                    result_dict[model_name][task]["TestClass"]["class_partial_success"] += 1
                    result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append("class_partial_success")
                else:
                    result_dict[model_name][task]["TestClass"]["class_fail"] += 1
                    result_dict[model_name][task]["TestClass"]["ClassEachTestResult"].append("class_fail")

        self.save_result(result_dict, "detailed")
    
    def get_test_answer(self, test_result):
        if test_result['testsRun'] == 0 or test_result['errors'] == test_result['testsRun']:
            return 'error'
        if test_result['errors'] + test_result['failures'] == 0:
            return 'success'
        if test_result['errors'] + test_result['failures'] < test_result['testsRun']:
            return 'partial_success'
        return 'fail'

    def cal_pass_at_k(self, n, k, k_success):
        total_combinations = comb(k, n)
        if k - k_success >= n:
            without_k_success_combinations = comb(k - k_success, n)
        else:
            without_k_success_combinations = 0

        with_k_success_combinations = total_combinations - without_k_success_combinations

        pass_at_k = with_k_success_combinations / total_combinations

        return pass_at_k

    def cal_metrics_pass_at_k(self, model_list, n, k):
        filename = os.path.split(self.generated_data_path)[1]
        filename_base = os.path.splitext(filename)[0]
        
        file_path = os.path.join(self.root, f'{filename_base}_detailed_result.json')
        with open(file_path, 'r') as f:
            test_result = json.load(f)

        result = {}

        for model_name in model_list:
            sum_num = 0
            success_num = 0
            class_success_num = 0
            class_num = 0
            partial_success_num = 0
            partial_success_class_num = 0
            for task in test_result[model_name]:
                class_num += 1
                for test_class in test_result[model_name][task]:
                    try:
                        if test_result[model_name][task][test_class]['success'] != 0:
                            pass_at_k = self.cal_pass_at_k(
                                n, k, test_result[model_name][task][test_class]['success'])
                            success_num += pass_at_k
                        if test_result[model_name][task][test_class]['success'] + test_result[model_name][task][test_class]['partial_success'] != 0:
                            pass_at_k = self.cal_pass_at_k(
                                n, k, test_result[model_name][task][test_class]['success'] + test_result[model_name][task][test_class]['partial_success'])
                            partial_success_num += pass_at_k
                        sum_num += 1
                    except:
                        if test_result[model_name][task][test_class]['class_success'] != 0:
                            pass_at_k = self.cal_pass_at_k(
                                n, k, test_result[model_name][task][test_class]['class_success'])
                            class_success_num += pass_at_k
                        k_success = test_result[model_name][task][test_class]['class_success'] + \
                            test_result[model_name][task][test_class]['class_partial_success']
                        if k_success != 0:
                            pass_at_k = self.cal_pass_at_k(n, k, k_success)
                            partial_success_class_num += pass_at_k

            result[model_name] = {"fun_success": round(success_num / sum_num, 3), "class_success": round(class_success_num / class_num, 3),
                                  "fun_partial_success": round(partial_success_num / sum_num, 3), "class_partial_success": round(partial_success_class_num / class_num, 3)}

        return result


if __name__ == '__main__':
    fire.Fire(EvaluatePipeline)
