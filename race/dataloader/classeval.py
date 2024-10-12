import json
import os
from typing import Dict
from pathlib import Path

from race.dataloader.utils import (
    CACHE_DIR,
    make_cache,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "classeval"

def get_class_eval(local_classeval_file=None) -> Dict[str, Dict]:
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
    if local_classeval_file is None:
        # Check if data file exists in CACHE_DIR
        
        # Due to changes in the ClassEval_data version, use a fixed and uniform version
        # url_latest = 'https://github.com/FudanSELab/ClassEval/raw/master/data/ClassEval_data.json'
        url_fix_version = 'https://github.com/FudanSELab/ClassEval/raw/8ac9909c134b3cd760774cb7bb8d34202f47786a/data/ClassEval_data.json'
        data_path = os.path.join(CACHE_DIR, "ClassEval_data.json")

        make_cache(url_fix_version, data_path)
    else:
        local_path = os.path.join(DATA_DIR, local_classeval_file)
        
        assert os.path.exists(local_path), f"File not found: {local_path}"
        print(f"Loading dataset from {local_path}")
        data_path = local_path
        

    data = json.load(open(data_path, 'r', encoding='utf-8'))
    return {task["task_id"]: task for task in data}
