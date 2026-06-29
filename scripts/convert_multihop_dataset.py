"""
Convert GPT-generated multi-hop questions to the project's QA format.

Reads  : datasets/ai_ml/multihop_qa_raw.json
Writes : datasets/ai_ml/multihop_qa.json

Also fixes known corpus title mismatches:
  "Anthropic" -> "Constitutional AI"  (no standalone Anthropic doc exists)

Run from repo root:
    python scripts/convert_multihop_dataset.py
"""

import json
from pathlib import Path

RAW_PATH    = Path("datasets/ai_ml/multihop_qa_raw.json")
OUTPUT_PATH = Path("datasets/ai_ml/multihop_qa.json")

# Titles that don't exist in the corpus -> map to the correct doc title
TITLE_FIXES = {
    "Anthropic": "Constitutional AI",
}

def fix_titles(titles: list[str]) -> list[str]:
    return [TITLE_FIXES.get(t, t) for t in titles]

with open(RAW_PATH, "r", encoding="utf-8") as f:
    raw = json.load(f)

converted = []
for item in raw:
    converted.append({
        "id":                str(item["id"]),
        "question":          item["question"],
        "answer":            item["ground_truth"],
        "supporting_titles": fix_titles(item["supporting_titles"]),
        "difficulty":        item.get("difficulty", "multi-hop"),
    })

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(converted, f, indent=2, ensure_ascii=False)

print(f"Converted {len(converted)} questions -> {OUTPUT_PATH}")
for q in converted:
    print(f"  [{q['id']:>2}] [{q['difficulty']:<15}] {q['question'][:65]}...")