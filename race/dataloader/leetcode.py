import json
import os
from typing import Dict
from pathlib import Path

from race.dataloader.utils import (
    CACHE_DIR,
    make_cache,
)

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "leetcode"

def get_leetcode(local_leetcode_file=None) -> Dict[str, Dict]:
    """
    Get LeetCode Contest benchmark from DeepSeek-Coder's github repo and return as a list of parsed dicts.
    """
    if local_leetcode_file is None:
        # Check if data file exists in CACHE_DIR
        url = 'https://github.com/deepseek-ai/DeepSeek-Coder/raw/main/Evaluation/LeetCode/data/20240121-Jul.jsonl'
        data_path = os.path.join(CACHE_DIR, "20240121-Jul.jsonl")

        make_cache(url, data_path)
    else:
        local_path = os.path.join(DATA_DIR, local_leetcode_file)
        
        assert os.path.exists(local_path), f"File not found: {local_path}"
        print(f"Loading dataset from {local_path}")
        data_path = local_path
        
    if data_path.endswith(".json"):
        data = json.load(open(data_path, 'r', encoding='utf-8'))
    elif data_path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(data_path, 'r', encoding='utf-8')]

    return {task["task_id"]: task for task in data}
