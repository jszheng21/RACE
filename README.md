# RACE(🏎️)

<p align="center">
    <a href="https://huggingface.co/spaces/jszheng/RACE_leaderboard"><img src="https://img.shields.io/badge/%F0%9F%8F%86-leaderboard-8A2BE2"></a>
    <a href="https://github.com/jszheng21/RACE/LICENSE"><img src="https://img.shields.io/pypi/l/evalplus"></a>
</p>


<p align="center">
    <a href="#-quick-start">🔥Quick Start</a> •
    <a href="#-useful-tools">🔨Tools</a> •
    <a href="#-citation">📜Citation</a> •
    <a href="#-acknowledgement">🙏Acknowledgement</a>
</p>


## 🏎️ About

RACE is a multi-dimensional benchmark for code generation that focuses on **R**eadability, m**A**intainability, **C**orrectness, and **E**fficiency. Its goal is to evaluate LLM's ability to generate code that is correct and meets the requirements of real-world development scenarios. The benchmark is designed with various real-world demands across different **_demand-dependent_** dimensions, making it more applicable to practical scenarios. To facilitate the evaluation of RACE, we provide easy-to-use evaluation scripts, while evaluating in a virtualized environment ensures the security of executing the code.

![overview](assets/race_overview.jpg)

The overall evaluation pipeline is shown as the above. Firstly, we summarize multiple representative factors for each dimension based on their respective quality definitions. Secondly, we design several reasonable customized requirements for each factor and integrate them into task descriptions, requiring the model to generate code that is both correct and meets these requirements. Finally, leveraging static analysis and runtime monitoring techniques, we develop evaluation metrics tailored to each factor to achieve accurate and efficient evaluation. 

## 🔥 Quick Start

To start, please run the following to prepare the environment:

```bash
pip install -e .
```

### Code Generation

Take `Readability` as an example, use the following command to generate code samples from a model, which are saved in the JSON Lines (jsonl) format.

```bash
python scripts/gen_readability.py \
    --model ${model}$ \
    --root "./outputs" \
    --backend [openai|vllm] \
    --API_BASE ${API_BASE} \
    --API_KEY ${API_KEY}
```

<details><summary>⏬ More commands for other dimensions.<i>:: click to expand ::</i></summary>
<div>

```bash
# For `Maintainability`
python scripts/gen_maintainability.py \
    --model ${model}$ \
    --root "./outputs" \
    --backend [openai|vllm] \
    --API_BASE ${API_BASE} \
    --API_KEY ${API_KEY}

# For `Efficiency`
python scripts/gen_efficiency.py \
    --model ${model}$ \
    --root "./outputs" \
    --backend [openai|vllm] \
    --API_BASE ${API_BASE} \
    --API_KEY ${API_KEY}
```

</div>
</details>


### Code Post-processing

Use the following command to read the code sample file, extract valid code from the model generation responses, and save them to a file with a `parsed` suffix.

```bash
python scripts/parse_generated_file.py \
    --generated_file_path ${generated_file_path} \
    --model ${model}$
```


### Code Evaluation

First, build the docker image as an environment for evaluating the code by execution.

```bash
docker build --rm -f "./Dockerfile" -t race:latest "."
```

Then, using code readability as an example, test the correctness of the LLM-generated code based on test cases.

```bash
./eval_readability.sh ${model} ${root}
```

<details><summary>⏬ More commands for other dimensions.<i>:: click to expand ::</i></summary>
<div>

```bash
# For `Readability`
./eval_maintainability.sh ${model} ${root}

# For `Efficiency`
./eval_efficiency.sh ${model} ${root}
```

More details about testing a single model on a single factor.

```bash
# For `Readability`
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_evalplus \
    --dataset [humaneval|mbpp] \
    --samples "/data/outputs/${parsed_generated_file}$"

# For `Maintainability (MI Metric)`
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_classeval test_pipeline \
    --model_name ${model} \
    --generated_data_path "/data/outputs/${generated_file}$" \
    --root "/data/outputs"

# For `Maintainability (Modularity)`
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_leetcode_style test_pipeline_simple \
    --model_name ${model} \
    --evaluation_test_case_path "/data/data/leetcode/evaluation_tests.jsonl" \
    --generated_data_path "/data/outputs/${parsed_generated_file}$" \
    --result_path "/data/outputs/${results_file}$" \
    --temp_path "/data/outputs"

# For `Efficiency`
docker run -v $(pwd):/data race:latest race.codeeval.evaluate_pipeline_leetcode_style test_pipeline_complexity \
    --model_name ${model} \
    --evaluation_test_case_path "/data/data/leetcode_efficiency/complexity_evaluation_test_cases.jsonl" \
    --evaluation_efficiency_data_path "/data/data/leetcode_efficiency/complexity_evaluation_data.jsonl" \
    --generated_data_path "/data/outputs/${parsed_generated_file}$" \
    --result_path "/data/outputs/${results_file}$" \
    --temp_path "/data/outputs"
```

</div>
</details>

Finally, get the evaluation results based on specific metrics. Take `Readability` as an example:

```bash
python scripts/get_metric_readability.py \
    --model ${model} \
    --output_path_root ${root}
```

<details><summary>⏬ More commands for other dimensions.<i>:: click to expand ::</i></summary>
<div>

```bash
# For `Maintainability`
python scripts/get_metric_maintainability.py \
    --model ${model} \
    --output_path_root ${root}

# For `Efficiency`
python scripts/get_metric_efficiency.py \
    --model ${model} \
    --output_path_root ${root}
```

</div>
</details>


## Issues

In order to use vllm to accelerate the DeepSeek-Coder-V2 inference process, additional branch versions need to be installed (https://github.com/zwd003/vllm)


## 📜 Citation

```bibtex

```

## 🙏 Acknowledgement

- [HumanEval](https://github.com/openai/human-eval)
- [MBPP](https://github.com/google-research/google-research/tree/master/mbpp)
- [EvalPlus](https://github.com/evalplus/evalplus)