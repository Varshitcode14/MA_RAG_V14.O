# tests/find_rowling.py

import json

count = 0

with open("corpus/hotpot_corpus_small.jsonl","r",encoding="utf-8") as f:
    for line in f:
        if "Rowling" in line:
            count += 1

print(count)