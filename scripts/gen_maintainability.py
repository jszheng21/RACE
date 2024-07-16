from race.codegen.generate_pipeline import GeneratePipeline
import fire


def generate(model, root, backend, API_BASE=None, API_KEY=None):
    # For `Maintainability (MI Metric)`
    dims = ['correctness', 'maintainability_mi']

    for dim in dims:
        print(dim)
        pipeline = GeneratePipeline(model=model,
                                    dataset='classeval',
                                    root=root,
                                    dim=dim,
                                    backend=backend,
                                    n_samples=1,
                                    temperature=0,
                                    greedy=True,
                                    base_url=API_BASE,
                                    api_key=API_KEY,)

        pipeline.pipeline_classeval(generation_strategy='holistic')

    # For `Maintainability (Modularity)`
    dims = ['correctness',
            'maintainability_module_count_1',
            'maintainability_module_count_2',
            'maintainability_module_count_3']

    for dim in dims:
        print(dim)
        pipeline = GeneratePipeline(model=model,
                                    dataset='leetcode',
                                    root=root,
                                    dim=dim,
                                    backend=backend,
                                    n_samples=1,
                                    temperature=0,
                                    greedy=True,
                                    base_url=API_BASE,
                                    api_key=API_KEY,)

        pipeline.pipeline_leetcode()
        
        
if __name__ == '__main__':
    fire.Fire(generate)