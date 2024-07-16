from race.codegen.generate_pipeline import GeneratePipeline
import fire


def generate(model, root, backend, API_BASE=None, API_KEY=None):
    # For `Readability`
    dims_dict = {
        'humaneval': [
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
            'readability_comment_by_line'
        ],
        'mbpp': [
            'correctness', 
        ]
    }

    for dataset, dims in dims_dict.items():
        print(dataset)
        for dim in dims:
            print(dim)
            pipeline = GeneratePipeline(model=model,
                                        dataset=dataset,
                                        root=root,
                                        dim=dim,
                                        backend=backend,
                                        n_samples=1,
                                        temperature=0,
                                        greedy=True,
                                        base_url=API_BASE,
                                        api_key=API_KEY,)


            pipeline.pipeline_simple()
            

if __name__ == '__main__':
    fire.Fire(generate)