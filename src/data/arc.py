import os
from .base import load_jsonl

# Expect files:
#   {data_dir}/arc_train.jsonl
#   {data_dir}/arc_test.jsonl
# Format per line:
#   {"question": "...", "choices": {"A":"...","B":"...","C":"...","D":"..."},"answer":"A"}

def load_split(data_dir: str, split: str) -> list[dict]:
    path = os.path.join(data_dir, f"arc_{split}.jsonl")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing dataset file: {path}")
    return load_jsonl(path)
