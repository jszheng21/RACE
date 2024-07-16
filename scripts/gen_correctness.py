from race.codegen.generate_pipeline import GeneratePipeline
import fire


def generate(model, root, backend, API_BASE=None, API_KEY=None):
    datasets = ['humaneval', 'mbpp', 'classeval', 'leetcode', 'leetcode_efficiency']
    dim = 'correctness'

    for dataset in datasets:
        print(dataset)
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