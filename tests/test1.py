import json

with open("datasets/hotpotqa/train.json", "r", encoding="utf-8") as f:
    data = json.load(f)

sample = data[0]

print(sample.keys())
print(sample)