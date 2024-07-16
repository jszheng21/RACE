import re
import fire


def parse_code(text, only_second_stage=False, for_completion_model=False):
    final_text = ''

    # ----------------- 1st stage -----------------
    # Extract code part from response
    if not only_second_stage:
        pattern_list = [r"```python3(.*?)```", r"```python3(.*)", 
                        r"```python(.*?)```", r"```python(.*)", 
                        r"\[PYTHON\](.*?)\[/PYTHON\]", r"(.*?)\[/PYTHON\]", 
                        r"\[PYTHON\](.*)"]
        if for_completion_model:
            pattern_list = [r"```python3(.*?)```", r"```python3(.*)", 
                            r"```python(.*?)```", r"```python(.*)", 
                            r"(.*?)\[/PYTHON\]", r"\[PYTHON\](.*?)\[/PYTHON\]", 
                            r"\[PYTHON\](.*)"]
        
        for pattern in pattern_list:
            try:
                code = re.findall(pattern, text, re.S)[0]
                if len(code) > 0:
                    text = code
                    break
            except:
                continue    

    # ----------------- 2nd stage -----------------
    splitted_text = text.split('\n')
    r_line_idx = -1

    # Follow the comments to find the test examples
    for i in reversed(range(len(splitted_text))):
        if splitted_text[i].strip().startswith('#') and 'test' in splitted_text[i].lower():
            r_line_idx = i
        elif splitted_text[i].strip().startswith('#') and 'example' in splitted_text[i].lower():
            r_line_idx = i
    
    # Find the test examples with print function
    if r_line_idx == -1:
        last_flag = 0
        for i in reversed(range(len(splitted_text))):
            if splitted_text[i].strip().startswith('print('):
                last_flag = 1

            if last_flag == 1 and len(splitted_text[i].strip()) == 0:
                r_line_idx = i + 1
                break

    # Find the main function
    if r_line_idx == -1:
        for i in reversed(range(len(splitted_text))):
            if splitted_text[i].strip() == 'if __name__ == "__main__":':
                r_line_idx = i
                break
    
    # Filter the code
    if r_line_idx != -1:
        final_text = '\n'.join(splitted_text[: r_line_idx]).strip()
    else:
        final_text = text.strip()

    return final_text


if __name__ == '__main__':
    fire.Fire(parse_code)

