import json
import os
from typing import Dict

from race.dataloader.utils import (
    CACHE_DIR,
    make_cache,
)


def get_class_eval(local_path=None) -> Dict[str, Dict]:
    """Get ClassEval from FudanSELab's github repo and return as a list of parsed dicts.

    Returns:
        List[Dict[str, str]]: List of dicts with keys:

        'skeleton'
        'test'
        'solution_code'
        'import_statement'
        'class_description'
        'class_name'
        'test_classes'
        'class_constructor'
        'fields'
        'methods_info'
    """
    if local_path is None:
        # Check if data file exists in CACHE_DIR
        url = 'https://github.com/FudanSELab/ClassEval/raw/master/data/ClassEval_data.json'
        data_path = os.path.join(CACHE_DIR, "ClassEval_data.json")

        make_cache(url, data_path)
    else:
        assert os.path.exists(local_path), f"File not found: {local_path}"
        print(f"Loading dataset from {local_path}")
        data_path = local_path
        

    data = json.load(open(data_path, 'r', encoding='utf-8'))
    return {task["task_id"]: task for task in data}
