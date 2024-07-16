import os
import tqdm
import ast
import re
from typing import List

from race.codegen.utils import add_desc_to_init, extract_method_code, get_method_signature, post_process, write_jsonl
from race.codegen.model import make_model
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
)


class GeneratePipeline():
    def __init__(
        self,
        model: str,
        dataset: str,
        root: str,
        dim: str,
        bs: int = 1,
        n_samples: int = 1,
        temperature: float = 0.0,
        resume: bool = True,
        greedy: bool = True,
        id_range: List = None,
        version: str = "default",
        backend: str = "vllm",
        base_url: str = None,
        api_key: str = None,
        tp: int = 1
    ):
        """
        `n_samples`: Number of samples to generate
        `batch_size`: Number of samples generated in a single generation process
        """
        assert dataset in ["humaneval", 
                           "mbpp", 
                           "classeval", 
                           "leetcode", 
                           "leetcode_efficiency"], f"Invalid dataset {dataset}"
        assert backend in ["vllm", "hf", "openai", "genai"], f"Invalid backend {backend}"

        self.model_name = model
        self.dataset_name = dataset
        self.root = root
        self.dim = dim
        self.bs = bs
        self.n_samples = n_samples
        self.temperature = temperature
        self.resume = resume
        self.greedy = greedy
        self.id_range = id_range
        self.version = version
        self.backend = backend
        self.base_url = base_url
        self.tp = tp

        # Prepare dataset and model
        self.dataset = self.load_dataset()

        if self.dataset_name == 'humaneval':
            self.preprocess_humaneval_prompt()
        elif self.dataset_name == 'mbpp':
            self.process_mbpp_prompt()

        self.model = make_model(
            model=model,
            backend=backend,
            dataset=dataset,
            batch_size=bs,
            temperature=temperature,
            tp=tp,
            base_url=base_url,
            api_key=api_key,
        )
        self.model_symbol = self.generate_model_symbol()

        # Make work directory
        os.makedirs(self.root, exist_ok=True)
        
        if '/' in self.model_name:
            self.model_name = self.model_name.split('/')[-1]

    def load_dataset(self):
        if self.dataset_name == "humaneval":
            from race.dataloader import get_human_eval_plus

            dataset = get_human_eval_plus(version=self.version)
        elif self.dataset_name == "mbpp":
            from race.dataloader import get_mbpp_plus

            dataset = get_mbpp_plus(version=self.version)
        elif self.dataset_name == "classeval":
            from race.dataloader import get_class_eval

            dataset = get_class_eval()
        elif self.dataset_name == "leetcode":
            from race.dataloader import get_leetcode

            dataset = get_leetcode()
        elif self.dataset_name == "leetcode_efficiency":
            from race.dataloader import get_leetcode_efficiency

            dataset = get_leetcode_efficiency('complexity_evaluation_data.jsonl')

        return dataset

    def generate_model_symbol(self):
        if 'codellama' in self.model_name.lower() and 'python' in self.model_name.lower():
            symbol = 'codellama_python'
        elif 'codellama' in self.model_name.lower() and 'instruct' in self.model_name.lower():
            symbol = 'codellama_instruct'
        elif 'deepseek' in self.model_name.lower():
            symbol = 'deepseekcoder'
        else:
            symbol = 'default'
        
        print(f'model symbol: {symbol}')
        return symbol
    
    def preprocess_humaneval_prompt(self):
        for key in self.dataset.keys():
            code = self.dataset[key]['prompt']
            parsed_code = ast.parse(code)

            empty_function_docstring = None
            non_empty_function_code = None

            for node in ast.walk(parsed_code):
                if isinstance(node, ast.FunctionDef):
                    function_body = node.body
                    
                    is_empty_function = False
                    if len(function_body) == 0:
                        is_empty_function = True
                    elif len(function_body) == 1:
                        if isinstance(function_body[0], ast.Expr) and isinstance(function_body[0].value, ast.Str):
                            is_empty_function = True
                        elif isinstance(function_body[0], ast.Pass):
                            is_empty_function = True
                    elif len(function_body) == 2:
                        if isinstance(function_body[0], ast.Expr) and isinstance(function_body[0].value, ast.Str) and isinstance(function_body[1], ast.Pass):
                            is_empty_function = True

                    if is_empty_function:
                        empty_function_docstring = ast.get_docstring(node)
                    else:
                        start_line = node.lineno - 1
                        end_line = node.end_lineno
                        non_empty_function_code = "\n".join(code.splitlines()[start_line:end_line])

            content = empty_function_docstring
            index = empty_function_docstring.find('>>>')
            if index != -1:
                content = empty_function_docstring[:index].strip()
            else:
                splitted_content = empty_function_docstring.split('\n')
                for i in range(len(splitted_content)):
                    if 'example' in splitted_content[i].lower():
                        content = '\n'.join(splitted_content[:i]).strip()
                        break
            content = content.replace('_', ' ')

            if non_empty_function_code is not None:
                final_content = content + f'\n\nGiven the following code:\n\n```python\n{non_empty_function_code}\n```'
            else:
                final_content = content
            
            self.dataset[key]['prompt'] = final_content
    
    def process_mbpp_prompt(self):
        for key in self.dataset.keys():
            prompt = self.dataset[key]['prompt']
            prompt = re.findall(f'"""(.*?)assert', prompt, re.S)[0].strip()

            self.dataset[key]['prompt'] = prompt
    
    def pipeline_classeval(self, generation_strategy):
        """
        Based on https://github.com/FudanSELab/ClassEval/blob/master/generation/inference_pipeline.py
        """
        assert generation_strategy in ('holistic', 'incremental', 'compositional'), "Invalid generation strategy"

        error_task_id_list = []
        if generation_strategy == 'holistic':
            result = []
            for _, cont in tqdm.tqdm(self.dataset.items()):
                pred = []
                try:
                    for _ in range(self.n_samples):
                        outputs = self.model.codegen(None, 
                                                     self.dim, 
                                                     do_sample=not self.greedy, 
                                                     num_samples=self.n_samples,
                                                     strategy='classeval_holistic',
                                                     info=cont,
                                                     model_symbol=self.model_symbol)
                        pred.append(outputs)
                    cont['predict'] = pred
                    result.append({'task_id': cont['task_id'], 'cont': cont})

                except Exception as e:
                    print(e)
                    print("IDX: ", cont['task_id'])
                    error_task_id_list.append(cont['task_id'])

        elif generation_strategy == 'incremental':
            result = []
            for _, cont in tqdm.tqdm(self.dataset.items()):
                cont['predict'] = []
                cont['raw_output'] = []
                for _ in range(self.n_samples):
                    pred = []
                    raw_output = []
                    try:
                        class_name = cont['class_name']
                        methods_info = cont['methods_info']
                        imports = '\n'.join(cont['import_statement'])
                        class_init = add_desc_to_init(cont['class_description'], cont['class_constructor'])
                        class_text = imports + '\n' + class_init
                        for method in methods_info:
                            # construct prompt
                            method_name = method['method_name']
                            inst = f"please complete {method_name} method in the following class {class_name}\n\n"
                            class_text_desc = class_text + "\n\n    " + method['method_description']

                            # generate model output
                            outputs = self.model.codegen(None,
                                                         self.dim,
                                                         do_sample=not self.greedy,
                                                         num_samples=self.n_samples,
                                                         strategy='classeval_incremental',
                                                         info={"instruction": inst, "skeleton": class_text_desc},
                                                         model_symbol=self.model_symbol)
                            raw_output.append(outputs)

                            # extract valid generated code
                            generated_method_code = extract_method_code(outputs, method_name)
                            class_text += '\n\n' + generated_method_code
                            pred.append(class_text)

                        cont['predict'].append(pred)
                        cont['raw_output'].append(raw_output)
                        
                    except Exception as e:
                        print(e)
                        print("IDX: ", cont['task_id'])
                        error_task_id_list.append(cont['task_id'])

                result.append({'task_id': cont['task_id'], 'cont': cont})

        elif generation_strategy == 'compositional':
            result = []
            for _, cont in tqdm.tqdm(self.dataset.items()):
                cont['predict'] = []
                for _ in range(self.n_samples):
                    pred = []
                    try:
                        class_name = cont['class_name']
                        methods_info = cont['methods_info']
                        imports = '\n'.join(cont['import_statement'])
                        class_init = add_desc_to_init(cont['class_description'], cont['class_constructor'])
                        for method_to_generate in methods_info:
                            class_text = imports + '\n' + class_init
                            # gather each method's signature to consruct class level skeleton
                            for method in methods_info:
                                if method['method_name'] == method_to_generate['method_name']:
                                    continue
                                class_text += get_method_signature(method['method_description'], method['method_name']) + "\n        pass\n\n"
                            # construct prompt
                            method_name = method_to_generate['method_name']
                            inst = f"please complete {method_name} method in the following class {class_name}\n\n"
                            class_text_desc = class_text + "\n\n    " + method_to_generate['method_description']

                            # generate model output
                            outputs = self.model.codegen(None,
                                                         self.dim,
                                                         do_sample=not self.greedy,
                                                         num_samples=self.n_samples,
                                                         strategy='classeval_compositional',
                                                         info={"instruction": inst, "skeleton": class_text_desc},
                                                         model_symbol=self.model_symbol)
                            pred.append(outputs)

                        cont['predict'].append(pred)
                        
                    except Exception as e:
                        print(e)
                        print("IDX: ", cont['task_id'])
                        error_task_id_list.append(cont['task_id'])

                result.append({'task_id': cont['task_id'], 'cont': cont})
        else:
            print("Unknown Generation Strategy")
            return
        
        print("error_task_id_list: ", error_task_id_list)

        post_process(result, generation_strategy)
        file_name = os.path.join(self.root, f'{self.dataset_name}_{self.dim}_{self.model_name}.jsonl')
        write_jsonl(file_name, result)
    
    def pipeline_leetcode(self):
        samples = [
            dict(task_id=task_id, solution=self.model.codegen(problem["prompt_sft"], 
                                                              self.dim, 
                                                              do_sample=not self.greedy, 
                                                              num_samples=self.n_samples,
                                                              strategy='leetcode',
                                                              model_symbol=self.model_symbol))
            for task_id, problem in tqdm.tqdm(self.dataset.items())
        ]
        
        file_name = os.path.join(self.root, f'{self.dataset_name}_{self.dim}_{self.model_name}.jsonl')
        write_jsonl(file_name, samples)

    def pipeline_leetcode_efficiency(self):
        samples = []
        for (task_id, _), problem in tqdm.tqdm(self.dataset.items()):
            samples.append(dict(task_id=task_id, 
                                instruction=problem['instruction'],
                                solution=self.model.codegen(problem["prompt_sft"], 
                                                            self.dim, 
                                                            do_sample=not self.greedy, 
                                                            num_samples=self.n_samples,
                                                            strategy='leetcode_efficiency',
                                                            model_symbol=self.model_symbol,
                                                            instruction=problem['instruction'])))
        
        file_name = os.path.join(self.root, f'{self.dataset_name}_{self.dim}_{self.model_name}.jsonl')
        write_jsonl(file_name, samples)

    def pipeline_simple(self):
        samples = [
            dict(task_id=task_id, solution=self.model.codegen(problem["prompt"], 
                                                              self.dim, 
                                                              do_sample=not self.greedy, 
                                                              num_samples=self.n_samples))
            for task_id, problem in tqdm.tqdm(self.dataset.items())
        ]
        
        file_name = os.path.join(self.root, f'{self.dataset_name}_{self.dim}_{self.model_name}.jsonl')
        write_jsonl(file_name, samples)

    def pipeline(self):
        identifier = self.model_name.replace("/", "--") + f"_{self.backend}_{self.dim}_temp_{self.temperature}"
        self.workdir = os.path.join(self.root, self.dataset_name, identifier)
        os.makedirs(self.workdir, exist_ok=True)

        with Progress(
            TextColumn(f"{self.dataset} •" + "[progress.percentage]{task.percentage:>3.0f}%"),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
        ) as p:
            for task_id, task in p.track(self.dataset.items()):
                # Check if task_id is in id_range
                if self.id_range is not None:
                    task_id_num = int(task_id.split("/")[1])
                    low, high = self.id_range
                    if task_id_num < low or task_id_num >= high:
                        p.console.print(f"Skipping {task_id} as it is not in {self.id_range}")
                        continue

                # Create project directory
                p_name = task_id.replace("/", "_")
                os.makedirs(os.path.join(self.workdir, p_name), exist_ok=True)
                log = f"Codegen: {p_name} @ {self.model}"
                n_existing = 0
                if self.resume:
                    # count existing .py files
                    n_existing = len(
                        [
                            f
                            for f in os.listdir(os.path.join(self.workdir, p_name))
                            if f.endswith(".py")
                        ]
                    )
                    if n_existing > 0:
                        log += f" (resuming from {n_existing})"

                nsamples = self.n_samples - n_existing
                p.console.print(log)

                # Generate code
                sidx = self.n_samples - nsamples
                while sidx < self.n_samples:
                    outputs = self.model.codegen(
                        task["prompt"],
                        self.dim,
                        do_sample=not self.greedy,
                        num_samples=self.n_samples - sidx,
                    )
                    assert outputs, "No outputs from model!"
                    for impl in outputs:
                        try:
                            with open(
                                os.path.join(self.workdir, p_name, f"{sidx}.py"),
                                "w",
                                encoding="utf-8",
                            ) as f:
                                if self.model.is_direct_completion():
                                    f.write(task["prompt"] + impl)
                                else:
                                    f.write(impl)
                        except UnicodeEncodeError:
                            continue
                        sidx += 1


if __name__ == "__main__":
    from fire import Fire

    Fire(GeneratePipeline)

