from transformers import AutoTokenizer


# some random words which serves as the splitter
_MAGIC_SPLITTER_ = "-[[]]-this-is-really-our-highest-priority-[[]]-"

# `correctness`, `readability`, and `maintainability_loop`
EVAL_INSTRUCTIONS = {
    "correctness": "Please generate the Python code to solve the following problem.\n\nProblem:\n\n{problem}",
    "readability_name_camel": "Please generate the Python code to solve the following problem, and use camel case for both function names and variable names.\n\nProblem:\n\n{problem}",
    "readability_name_snake": "Please generate the Python code to solve the following problem, and use snake case for both function names and variable names.\n\nProblem:\n\n{problem}",
    "readability_name_function_camel": "Please generate the Python code to solve the following problem, and use camel case for function names.\n\nProblem:\n\n{problem}",
    "readability_name_function_snake": "Please generate the Python code to solve the following problem, and use snake case for function names.\n\nProblem:\n\n{problem}",
    "readability_name_var_camel": "Please generate the Python code to solve the following problem, and use camel case for variable names.\n\nProblem:\n\n{problem}",
    "readability_name_var_snake": "Please generate the Python code to solve the following problem, and use snake case for variable names.\n\nProblem:\n\n{problem}",
    "readability_length_setting_1": "Please generate the Python code to solve the following problem, where each line is less than 60 characters long and each function is less than 20 lines long.\n\nProblem:\n\n{problem}",
    "readability_length_setting_2": "Please generate the Python code to solve the following problem, where each line is less than 70 characters long and each function is less than 30 lines long.\n\nProblem:\n\n{problem}",
    "readability_length_setting_3": "Please generate the Python code to solve the following problem, where each line is less than 79 characters long and each function is less than 40 lines long.\n\nProblem:\n\n{problem}",
    "readability_comment_by_function": "Please generate the Python code to solve the following problem, and add the necessary docstring for each function.\n\nProblem:\n\n{problem}",
    "readability_comment_by_line": "Please generate the Python code to solve the following problem, and add comments for each line in each function.\n\nProblem:\n\n{problem}",
    "maintainability_loop_for": "Please generate the Python code to solve the following problem, and just use the for statement to implement the desired loop structures.\n\nProblem:\n\n{problem}",
    "maintainability_loop_while": "Please generate the Python code to solve the following problem, and just use the while statement to implement the desired loop structures.\n\nProblem:\n\n{problem}",
}

EVAL_INSTRUCTIONS_FOR_DIRECT_COMPLETION = {
    "correctness": "Please generate the code to solve the following problem.\n\n{problem}",
    "readability_name_camel": "Please generate the Python code to solve the following problem, and use camel case for both function names and variable names.\n\n{problem}",
    "readability_name_snake": "Please generate the Python code to solve the following problem, and use snake case for both function names and variable names.\n\n{problem}",
    "readability_name_function_camel": "Please generate the Python code to solve the following problem, and use camel case for function names.\n\n{problem}",
    "readability_name_function_snake": "Please generate the Python code to solve the following problem, and use snake case for function names.\n\n{problem}",
    "readability_name_var_camel": "Please generate the Python code to solve the following problem, and use camel case for variable names.\n\n{problem}",
    "readability_name_var_snake": "Please generate the Python code to solve the following problem, and use snake case for variable names.\n\n{problem}",
    "readability_length_setting_1": "Please generate the Python code to solve the following problem, where each line is less than 60 characters long and each function is less than 20 lines long.\n\n{problem}",
    "readability_length_setting_2": "Please generate the Python code to solve the following problem, where each line is less than 70 characters long and each function is less than 30 lines long.\n\n{problem}",
    "readability_length_setting_3": "Please generate the Python code to solve the following problem, where each line is less than 79 characters long and each function is less than 40 lines long.\n\n{problem}",
    "readability_comment_by_function": "Please generate the Python code to solve the following problem, and add the necessary docstring for each function.\n\n{problem}",
    "readability_comment_by_line": "Please generate the Python code to solve the following problem, and add comments for each line in each function.\n\n{problem}",
    "maintainability_loop_for": "Please generate the Python code to solve the following problem, and just use the for statement to implement the desired loop structures.\n\n{problem}",
    "maintainability_loop_while": "Please generate the Python code to solve the following problem, and just use the while statement to implement the desired loop structures.\n\n{problem}",
}

# `maintainability_mi`
EVAL_INSTRUCTIONS_FOR_CLASSEVAL = {
    "correctness": "Please complete the class {class_name} in the following code.\n\n```python\n{skeleton}\n```",
    "maintainability_mi": "Please complete the class {class_name} in the following code, and ensure that the code has good maintainability. Code maintainability refers to how easy it is to support and change the code.\n\n```python\n{skeleton}\n```",
}

# `maintainability_module_count`
EVAL_INSTRUCTIONS_FOR_LEETCODE = {
    "correctness": "Please complete the code below to solve above problem.",
    "maintainability_module_count_1": "Please complete the code below to solve above problem, and use only the given function.",
    "maintainability_module_count_2": "Please complete the code below to solve above problem, and use only the given function and one addition sub-function.",
    "maintainability_module_count_3": "Please complete the code below to solve above problem, and use only the given function and two addition sub-functions."
}

# `efficiency`
EVAL_INSTRUCTIONS_FOR_LEETCODE_EFFICIENCY = {
    "correctness": "Please complete the code below to solve above problem.",
    "complexity": "Please complete the code below to solve above problem, and make sure that {instruction}.",
}


def construct_prompt_evalplus(prompt: str, tokenizer: AutoTokenizer):
    # directly return prompt if it does not have a tokenizer.chat_template
    if tokenizer.chat_template is None:
        return prompt

    prompt = f"""\
Please provide a self-contained Python script that solves the following problem in a markdown code block:
```
{prompt.strip()}
```
"""
    response = f"""\
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
{_MAGIC_SPLITTER_}
```
"""
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
    ).split(_MAGIC_SPLITTER_)[0]
    return prompt


def construct_prompt_customized(prompt: str, dim: str, tokenizer: AutoTokenizer):
    system_prompt = "You are a helpful assistant good at coding."
    user_prompt = EVAL_INSTRUCTIONS[dim].format(problem=prompt.strip())

    if tokenizer.chat_template is None:
        response = "\n\nBelow is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python\n"""

        prompt = user_prompt + response
    else:
        response = f"""\
Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:
```python
{_MAGIC_SPLITTER_}
```
"""

        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": response}
            ],
            tokenize=False,
        ).split(_MAGIC_SPLITTER_)[0]

    return prompt


def construct_prompt_leetcode_efficiency(prompt: str, dim: str, tokenizer: AutoTokenizer, kwargs):
    system_prompt = "You are a helpful assistant good at coding."
    model_symbol = kwargs['model_symbol']
    
    if dim == 'correctness':
        replace_content = EVAL_INSTRUCTIONS_FOR_LEETCODE_EFFICIENCY[dim]
    else:
        replace_content = EVAL_INSTRUCTIONS_FOR_LEETCODE_EFFICIENCY[dim].format(instruction=kwargs['instruction'])
    user_prompt = prompt.strip().replace('Please complete the code below to solve above prblem:', replace_content)
        
    if 'codellama' in model_symbol:
        user_prompt = user_prompt.replace('```python', '[PYTHON]')
        user_prompt = user_prompt.replace('```', '[/PYTHON]')

    if tokenizer.chat_template is None:
        # Original
        # if 'codellama_python' in model_symbol:
        #     user_prompt = user_prompt.replace('```python', '[PYTHON]')
        # prompt = user_prompt.replace('        \n```', '')
            
        # Wizard-version
        response = f"\n\nBelow is a Python script that fulfills the requirements:\n```python\n"""
        
        if model_symbol == 'codellama_python':
            response = response.replace('```python', '[PYTHON]')
        
        prompt = user_prompt + response
    else:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt


def construct_prompt_leetcode(prompt: str, dim: str, tokenizer: AutoTokenizer, model_symbol):
    system_prompt = "You are a helpful assistant good at coding."
    
    replace_content = EVAL_INSTRUCTIONS_FOR_LEETCODE[dim]
    if len(replace_content) > 0:
        user_prompt = prompt.strip().replace('Please complete the code below to solve above prblem:', replace_content)
    else:
        user_prompt = prompt.strip()
    
    if 'codellama' in model_symbol:
        user_prompt = user_prompt.replace('```python', '[PYTHON]')
        user_prompt = user_prompt.replace('```', '[/PYTHON]')

    if tokenizer.chat_template is None:
        # Original
        # if 'codellama_python' in model_symbol:
        #     user_prompt = user_prompt.replace('```python', '[PYTHON]')
        # prompt = user_prompt.replace('        \n```', '')
            
        # Wizard-version
        response = f"\n\nBelow is a Python script that fulfills the requirements:\n```python\n"""
        
        if model_symbol == 'codellama_python':
            response = response.replace('```python', '[PYTHON]')
        
        prompt = user_prompt + response
    else:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt


def construct_prompt_customized_direct_completion(prompt: str, dim: str, tokenizer: AutoTokenizer):
    assert tokenizer.chat_template is None, "Tokenizer.chat_template must be None for direct completion"

    prompt = EVAL_INSTRUCTIONS_FOR_DIRECT_COMPLETION[dim].format(problem=prompt.strip())
    return prompt


def constrct_prompt_classeval(strategy: str, info: dict, dim: str, tokenizer: AutoTokenizer, model_symbol):
    prompt = ""
    if strategy == 'holistic':
        instruction = EVAL_INSTRUCTIONS_FOR_CLASSEVAL[dim].format(class_name=info['class_name'], 
                                                                  skeleton=info['skeleton'])

    elif strategy == 'incremental' or strategy == 'compositional':
        instruction = info['instruction'] + info['skeleton']
    
    if 'codellama' in model_symbol:
        instruction = instruction.replace('```python', '[PYTHON]')
        instruction = instruction.replace('```', '[/PYTHON]')

    if tokenizer.chat_template is None:
        if strategy == 'holistic':
            response = f"\n\nBelow is a Python script with the {info['class_name']} class:\n```python\n"""
        else:
            response = f"\n\nBelow is a Python script with a self-contained class:\n```python\n"""
        
        if model_symbol == 'codellama_python':
            response = response.replace('```python', '[PYTHON]')

        prompt = instruction + response
    else:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": instruction},
            ],
            tokenize=False,
            add_generation_prompt=True
        )

    return prompt


def construct_prompt(
    prompt: str, 
    strategy: str,
    dim: str, 
    tokenizer: AutoTokenizer,
    **kwargs
) -> str:

    if strategy == 'evalplus':
        prompt = construct_prompt_evalplus(prompt, tokenizer)
    elif strategy == 'customized':
        prompt = construct_prompt_customized(prompt, dim, tokenizer)
    elif strategy == 'customized_direct':
        prompt = construct_prompt_customized_direct_completion(prompt, dim, tokenizer)
    elif strategy == 'classeval_holistic':
        prompt = constrct_prompt_classeval('holistic', kwargs['info'], dim, tokenizer, kwargs['model_symbol'])
    elif strategy == 'classeval_incremental':
        prompt = constrct_prompt_classeval('incremental', kwargs['info'], dim, tokenizer, kwargs['model_symbol'])
    elif strategy == 'classeval_compositional':
        prompt = constrct_prompt_classeval('compositional', kwargs['info'], dim, tokenizer, kwargs['model_symbol'])
    elif strategy == 'leetcode':
        prompt = construct_prompt_leetcode(prompt, dim, tokenizer, kwargs['model_symbol'])
    elif strategy == 'leetcode_efficiency':
        prompt = construct_prompt_leetcode_efficiency(prompt, dim, tokenizer, kwargs)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return prompt


def construct_prompt_simple(
    prompt: str,
    strategy: str,
    dim: str,
    **kwargs
) -> str:
    
    if strategy == 'customized':
        prompt = EVAL_INSTRUCTIONS[dim].format(problem=prompt.strip())
    elif strategy == 'customized_direct':
        prompt = EVAL_INSTRUCTIONS_FOR_DIRECT_COMPLETION[dim].format(problem=prompt.strip())
    elif strategy == 'classeval_holistic':
        instruction = EVAL_INSTRUCTIONS_FOR_CLASSEVAL[dim].format(class_name=kwargs['info']['class_name'], skeleton=kwargs['info']['skeleton'])
        prompt = instruction
    elif strategy == 'classeval_incremental' or strategy == 'classeval_compositional':
        instruction = kwargs['info']['instruction'] + kwargs['info']['skeleton']
        prompt = instruction
    elif strategy == 'leetcode':
        replace_content = EVAL_INSTRUCTIONS_FOR_LEETCODE[dim]
        prompt = prompt.strip().replace('Please complete the code below to solve above prblem:', replace_content)
    elif strategy == 'leetcode_efficiency':
        if dim == 'correctness':
            replace_content = EVAL_INSTRUCTIONS_FOR_LEETCODE_EFFICIENCY[dim]
        else:
            replace_content = EVAL_INSTRUCTIONS_FOR_LEETCODE_EFFICIENCY[dim].format(instruction=kwargs['instruction'])
        prompt = prompt.strip().replace('Please complete the code below to solve above prblem:', replace_content)
    else:
        raise ValueError(f"Invalid strategy: {strategy}")

    return prompt


