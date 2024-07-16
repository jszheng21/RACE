import os
import json
import zlib
import pickle
import base64
import fire
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor, as_completed
from race.codeeval.execution_utils import run_single_evaluation
from race.codeeval.metrics_utils import compute_metrics_from_results

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class EvaluateAppsStylePipeline():
    def __init__(
        self, 
        model_name, 
        evaluation_data_path, 
        generated_data_path,
        root,
        timeout=6,
        num_process_evaluate=16,
    ):
        self.model_name = model_name
        self.test_samples_list = self.load_test_samples_list(evaluation_data_path)
        self.codes_list = self.load_codes_list(generated_data_path)
        self.root = root
        self.timeout = timeout
        self.num_process_evaluate = num_process_evaluate

        print(f"Loaded {len(self.test_samples_list)} cases.")
        print(f"{' A case of test samples ':=^50}")
        print(self.test_samples_list[0])
        print(f"{' A case of generated code ':=^50}")
        print(self.codes_list[0])

    def load_test_samples_list(self, evaluation_data_path):
        if not evaluation_data_path.endswith('.jsonl'):
            return
        
        print(f'Reading `test_samples_list` from {evaluation_data_path}')
        
        with open(evaluation_data_path, 'r') as f:
            evaluation_data = [json.loads(line) for line in f]
            evaluation_data = sorted(evaluation_data, key=lambda x: x['question_id'])

            test_samples_list = []
            for data in evaluation_data:
                public_test_cases = json.loads(data['public_test_cases'])  # a list of dict

                try:
                    private_test_cases = json.loads(data['private_test_cases'])
                except:
                    private_test_cases = json.loads(
                        pickle.loads(
                            zlib.decompress(
                                base64.b64decode(data['private_test_cases'].encode("utf-8"))
                            )
                        )
                    )  # a list of dict

                inputs = [item['input'] for item in public_test_cases + private_test_cases]
                outputs = [item['output'] for item in public_test_cases + private_test_cases]
                fn_name = json.loads(data['metadata']).get('func_name', None)

                test_samples_list.append({'input_output': json.dumps({'inputs': inputs, 
                                                                      'outputs': outputs, 
                                                                      'fn_name': fn_name})})

            return test_samples_list

    def load_codes_list(self, generated_data_path):
        if not generated_data_path.endswith('.json'):
            return
        
        print(f'Reading `codes_list` from {generated_data_path}')
        
        with open(generated_data_path, 'r') as f:
            generated_data = json.load(f)
            generated_data = sorted(generated_data, key=lambda x: x['question_id'])

            codes_list = []
            for data in generated_data:
                codes_list.append(data['code_list'])
            
            return codes_list

    def save_result(self, result, type):
        save_path = os.path.join(self.root, f'{self.model_name}_{type}.json')
        with open(save_path, 'w') as f:
            json.dump(result, f, indent=4, sort_keys=True)

            print(f'Written to {save_path}')

    def test_pipeline(self, debug=False):
        """
        Run unit test for each task parallelly.

        Returns:
            results: dictionary of results, key is the problem index, value is a list of results for each generation
                [-2] = compile error
                [-1] = runtime error
                [False] = failed test case
                [True] = passed test case
        """
        inputs = [
            [(self.test_samples_list[index], self.codes_list[index], self.timeout, debug), index]
            for index in range(len(self.test_samples_list))
        ]

        with tqdm(total=len(inputs)) as pbar:
            with ProcessPoolExecutor(max_workers=1 if debug else self.num_process_evaluate) as executor:
                futures = {
                    executor.submit(run_single_evaluation, arg): index
                    for arg, index in inputs
                }

                results = {}
                metadata = {}
                for future in as_completed(futures):
                    index = futures[future]
                    results[index], metadata[index] = future.result()
                    pbar.update(1)

        assert len(results) == len(inputs), f"results = {len(results)} inputs = {len(inputs)} {results=}"

        metadata = self.post_process_metadata(metadata)

        self.save_result(results, 'results')
        self.save_result(metadata, 'metadata')

    def evaluate_pipeline(self, k_list=[1, 5]):
        model_result_path = os.path.join(self.root, f'{self.model_name}_results.json')
        with open(model_result_path, 'r') as f:
            results = json.load(f)

            metrics = compute_metrics_from_results(results, k_list=k_list)
            self.save_result(metrics, 'metrics')

    def post_process_metadata(self, metadata):
        for key in metadata.keys():
            if type(metadata[key]) is not list:
                metadata[key] = [json.dumps(metadata[key])]
            else:
                metadata[key] = [json.dumps(x) for x in metadata[key]]

        return metadata


if __name__ == '__main__':
    fire.Fire(EvaluateAppsStylePipeline)
