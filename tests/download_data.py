from datasets import load_dataset
import json
from pathlib import Path

# -----------------
# HotpotQA
# -----------------
hotpot_dir = Path("datasets/hotpotqa")
hotpot_dir.mkdir(parents=True, exist_ok=True)

print("Downloading HotpotQA...")

hotpot_train = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="train")
hotpot_dev = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")

with open(hotpot_dir / "train.json", "w", encoding="utf-8") as f:
    json.dump(list(hotpot_train), f)

with open(hotpot_dir / "dev.json", "w", encoding="utf-8") as f:
    json.dump(list(hotpot_dev), f)

print("HotpotQA downloaded.")