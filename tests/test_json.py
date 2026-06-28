import json

with open("datasets/hotpotqa/dev.json","r",encoding="utf-8") as f:
    data = json.load(f)

print(type(data))
print(len(data))
print(data[0].keys())
print(data[0]["question"])
print(data[0]["answer"])