from race.codeprocess.parser import parse_code
import json
import fire
import os


def parse(generated_file_path, model):
    directory = os.path.dirname(generated_file_path)
    file_name, file_extension = os.path.splitext(os.path.basename(generated_file_path))
    
    parsed_file_path = os.path.join(directory, f'{file_name}_parsed{file_extension}')
    
    f_w = open(parsed_file_path, 'w')
    with open(generated_file_path, 'r') as f:
        for line in f:
            line = json.loads(line)

            for_completion_model = False
            if 'python' in model.lower():
                for_completion_model = True

            only_second_stage = False
            if 'wizard' in model.lower():
                only_second_stage = True
                
            line['solution'] = parse_code(line['solution'], only_second_stage, for_completion_model)
            
            f_w.write(json.dumps(line) + '\n')
    f_w.close()
    
    print(f'Output to {parsed_file_path}')
    
    
if __name__ == '__main__':
    fire.Fire(parse)

