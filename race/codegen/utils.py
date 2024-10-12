import openai
import re
import os
import gzip
import json
from tenacity import retry, stop_after_attempt, wait_exponential
from openai.types.chat import ChatCompletion
from typing import Dict, Iterable
from warnings import warn

try:
    import google.generativeai as genai
except ImportError:
    warn("google.generativeai will not work. Fix by `pip install google`")


def write_jsonl(
    filename: str, data: Iterable[Dict], append: bool = False, drop_builtin: bool = True
):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    if drop_builtin:
                        x = {k: v for k, v in x.items() if not k.startswith("_")}
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                if drop_builtin:
                    x = {k: v for k, v in x.items() if not k.startswith("_")}
                fp.write((json.dumps(x) + "\n").encode("utf-8"))

    print(f'Written to {filename}')


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=20))
def retry_get_genai_response(
    model,
    user_prompt,
    temperature
):
    try:
        return model.generate_content(user_prompt,
                                      generation_config=genai.GenerationConfig(temperature=temperature))
    except Exception as e:
        print(e)
        raise TimeoutError


def get_genai_response(*args, **kwargs):
    res = None
    while res is None or len(res.parts) == 0:
        try:
            res = retry_get_genai_response(*args, **kwargs)
        except Exception as e:
            print(f"Error occur: {e}")

    return res.text


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, max=20))
def retry_get_openai_response(
    client: openai.Client,
    model: str,
    user_prompt: str,
    system_prompt: str = "",
    max_tokens: int = None,
    temperature: float = 0,
    n: int = 1,
    **kwargs
) -> ChatCompletion:
    messages = []
    if len(system_prompt) > 0:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    stops = []
    
    try:
        ret = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            n=n,
            stop=stops,
            **kwargs
        )
        
        if ret.choices is None:
            raise ValueError("No response from OpenAI")
        return ret
    except Exception as e:
        print(e)
        raise TimeoutError


def get_openai_response(*args, **kwargs) -> ChatCompletion:
    res = None
    while res is None:
        try:
            res = retry_get_openai_response(*args, **kwargs)
        except Exception as e:
            print(f"Error occur: {e}")

    return res


def get_leading_spaces(string):
    return len(string) - len(string.lstrip())


def get_method_signature(code, method_name):
    method_def_prefix = "def " + method_name + '('
    code_segment = code.split('):')
    for segment in code_segment:
        if method_def_prefix in segment:
            return "    " + segment + "):"
    return ""


def add_desc_to_init(self, desc, class_init):
    class_init_list = class_init.split('\n')
    class_init_list[0] += " \n" + desc
    class_init = '\n'.join(class_init_list)
    return class_init


def extract_method_code(code, method_name):
    # extract code of method {method_name} from {code}
    output_split_identifier_list = ["### Response:", "@@ Response:", "[/INST]"]
    for identifier in output_split_identifier_list:
        if identifier in code:
            code = code.split(identifier)[1]
            break

    pattern_list = [r"```python(.*?)```", r"\[PYTHON\](.*?)\[/PYTHON\]"]
    for pattern in pattern_list:
        code_part = re.findall(pattern, code, re.S)
        if code_part:
            code = code_part[0]
            break

    code_list = code.split('\n')

    method_code_list = []
    method_def_prefix = "def " + method_name + '('
    skip_line_list = ["```", '\r']
    # extract generated method code corresponding method_name, the strategy is to find the line
    # has "def methodname(...)" and following lines have more leading spaces than the first "def" line
    for i, line in enumerate(code_list):
        if method_def_prefix in line:
            method_code_list = code_list[i:]
            break

    if len(method_code_list) == 0:
        return ""

    for i, line in enumerate(method_code_list):
        if line in skip_line_list:
            method_code_list[i] = ""
    
    if get_leading_spaces(method_code_list[1]) - get_leading_spaces(method_code_list[0]) > 4:
        method_code_list[0] = " " * 4 + method_code_list[0]

    first_line_leading_space = get_leading_spaces(method_code_list[0])
    for i, line in enumerate(method_code_list[1:]):
        if get_leading_spaces(line) <= first_line_leading_space and len(line) > 0:
            method_code_list = method_code_list[:i + 1]
            break

    for i, line in enumerate(method_code_list):
        method_code_list[i] = ' ' * (4 - first_line_leading_space) + line
    
    if 'self' not in method_code_list[0] and 'cls' not in method_code_list[0]:
        method_code_list.insert(0, ' ' * 4 + "@staticmethod")

    line_notation_mark = 0
    for line in method_code_list:
        if line == " " * 8 + "\"\"\"" or line == " " * 4 + "\"\"\"":
            line_notation_mark = line_notation_mark + 1
    if line_notation_mark % 2 == 1:
        method_code_list.append(" " * 8 + "\"\"\"")
        method_code_list.append(" " * 8 + "pass")

    method_code = '\n'.join(method_code_list)
    method_code = method_code.rstrip() + '\n'
    return method_code

def post_process(result, generation_strategy):
    """
    Only for `incremental` and `compositional` generation strategies
    """
    if generation_strategy == 'incremental':
        for cont in result:
            pred = []
            for result in cont['predict']:
                pred.append(result[-1])
            cont['predict'] = pred
    elif generation_strategy == 'compositional':
        for cont in result:
            cont['raw_output'] = cont['predict'].copy()
        for cont in result:
            cont['predict'] = []
            for raw_output in cont['raw_output']:
                class_code = '\n'.join(cont['import_statement']) + '\n' + cont['class_constructor']
                for i in range(len(raw_output)):
                    method_name = cont['methods_info'][i]['method_name']
                    code = raw_output[i]
                    method_code = extract_method_code(code, method_name)
                    class_code += '\n\n' + method_code
                cont['predict'].append(class_code)
