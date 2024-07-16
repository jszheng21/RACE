import json
import os
from typing import Dict

from race.dataloader.utils import (
    CACHE_DIR,
    make_cache,
)


def get_leetcode(local_path=None) -> Dict[str, Dict]:
    """
    Get LeetCode Contest benchmark from DeepSeek-Coder's github repo and return as a list of parsed dicts.
    """
    if local_path is None:
        # Check if data file exists in CACHE_DIR
        url = 'https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/Evaluation/LeetCode/data/20240121-Jul.jsonl'
        data_path = os.path.join(CACHE_DIR, "20240121-Jul.jsonl")

        make_cache(url, data_path)
    else:
        assert os.path.exists(local_path), f"File not found: {local_path}"
        print(f"Loading dataset from {local_path}")
        data_path = local_path
        
    if data_path.endswith(".json"):
        data = json.load(open(data_path, 'r', encoding='utf-8'))
    elif data_path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(data_path, 'r', encoding='utf-8')]

    return {task["task_id"]: task for task in data}
