import json

INPUT = "corpus/hotpot_corpus.jsonl"
OUTPUT = "corpus/hotpot_corpus_small.jsonl"

LIMIT = 5000

with open(INPUT, "r", encoding="utf-8") as fin, \
     open(OUTPUT, "w", encoding="utf-8") as fout:

    for i, line in enumerate(fin):
        if i >= LIMIT:
            break
        fout.write(line)

print(f"Saved first {LIMIT} docs.")