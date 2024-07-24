import json
import fire
from get_metric_correctness import output_correctness
from get_metric_readability import output_readability
from get_metric_maintainability import output_maintainability
from get_metric_efficiency import output_efficiency


def output_overall(model, output_path_root):
    r1 = output_correctness(model, output_path_root)
    r2 = output_readability(model, output_path_root)
    r3 = output_maintainability(model, output_path_root)
    r4 = output_efficiency(model, output_path_root)

    final_results = {model: {**r1[model], **r2[model], **r3[model], **r4[model]}}
    
    total_overall = 0
    for key in final_results[model].keys():
        total_overall += final_results[model][key][key.capitalize()]
    
    final_results[model]['overall'] = {}
    final_results[model]['overall']['RACE Score'] = round(total_overall / 4, 1)
    
    print(json.dumps(final_results, indent=4))

if __name__ == '__main__':
    fire.Fire(output_overall)
