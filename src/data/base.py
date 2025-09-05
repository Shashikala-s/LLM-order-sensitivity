from dataclasses import dataclass
from typing import Dict, List
import json

@dataclass
class MCQItem:
    question: str
    choices: Dict[str, str]   # {"A": "...", "B": "...", "C": "...", "D": "..."}
    answer: str               # "A" | "B" | "C" | "D"

def load_jsonl(path: str) -> List[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            items.append(json.loads(line))
    return items
