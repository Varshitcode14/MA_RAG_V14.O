import json
from pathlib import Path

DATASET_PATH = "datasets/hotpotqa/train.json"
OUTPUT_PATH = "corpus/hotpot_corpus.jsonl"

with open(DATASET_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

documents = {}

for sample in data:
    titles = sample["context"]["title"]
    sentences = sample["context"]["sentences"]

    for title, sent_list in zip(titles, sentences):
        text = " ".join(sent_list).strip()

        if title not in documents:
            documents[title] = text

Path("corpus").mkdir(exist_ok=True)

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for title, text in documents.items():
        record = {
            "title": title,
            "text": text
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

print(f"Unique documents: {len(documents)}")
print(f"Saved to: {OUTPUT_PATH}")