import json
import os
from typing import Dict
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent.parent / "data" / "leetcode_efficiency"


def get_leetcode_efficiency(local_leetcode_file) -> Dict[str, Dict]:
    """
    Get LeetCode benchmark.
    """
    local_path = os.path.join(DATA_DIR, local_leetcode_file)
    
    assert os.path.exists(local_path), f"File not found: {local_path}"
    print(f"Loading dataset from {local_path}")
    data_path = local_path
        
    if data_path.endswith(".json"):
        data = json.load(open(data_path, 'r', encoding='utf-8'))
    elif data_path.endswith(".jsonl"):
        data = [json.loads(line) for line in open(data_path, 'r', encoding='utf-8')]

    return {(task["task_id"], task["instruction"]): task for task in data}
