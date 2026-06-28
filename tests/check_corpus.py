import json

with open("corpus/hotpot_corpus.jsonl", "r", encoding="utf-8") as f:
    for i in range(5):
        row = json.loads(next(f))
        print("\n")
        print(row["title"])
        print("-" * 50)
        print(row["text"][:500])