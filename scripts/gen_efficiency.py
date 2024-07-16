from race.codegen.generate_pipeline import GeneratePipeline
import fire


def generate(model, root, backend, API_BASE=None, API_KEY=None):
    # For `Efficiency`
    dims = ['correctness', 'complexity']

    for dim in dims:
        print(dim)
        pipeline = GeneratePipeline(model=model,
                                    dataset='leetcode_efficiency',
                                    root=root,
                                    dim=dim,
                                    backend=backend,
                                    n_samples=1,
                                    temperature=0,
                                    greedy=True,
                                    base_url=API_BASE,
                                    api_key=API_KEY,)

        pipeline.pipeline_leetcode_efficiency()
        

if __name__ == '__main__':
    fire.Fire(generate)