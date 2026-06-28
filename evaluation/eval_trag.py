import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

import json
import time
import re

from T_RAG.T_main import TRAG

NUM_SAMPLES = 20

RESULTS_DIR = "results"
RESULTS_FILE = f"{RESULTS_DIR}/trag_preds.json"

Path(RESULTS_DIR).mkdir(
    exist_ok=True
)


def normalize_answer(s):
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.split())
    return s


def exact_match(pred, gold):
    return (
        normalize_answer(pred)
        ==
        normalize_answer(gold)
    )


print("Loading T-RAG...")

rag = TRAG()

with open(
    "datasets/hotpotqa/dev.json",
    "r",
    encoding="utf-8"
) as f:
    data = json.load(f)

samples = data[:NUM_SAMPLES]

predictions = []

correct = 0

start_time = time.time()

for idx, sample in enumerate(
    samples,
    start=1
):

    question = sample["question"]
    gold = sample["answer"]

    try:

        pred, _ = rag.answer(question)

        is_correct = exact_match(
            pred,
            gold
        )

        if is_correct:
            correct += 1

        row = {
            "id": sample["id"],
            "question": question,
            "gold": gold,
            "prediction": pred,
            "correct": is_correct
        }

        predictions.append(row)

        with open(
            RESULTS_FILE,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                predictions,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(
            f"[{idx}/{NUM_SAMPLES}] "
            f"EM={is_correct}"
        )

    except Exception as e:

        print(
            f"ERROR Q{idx}: {e}"
        )

elapsed = (
    time.time()
    -
    start_time
)

em = (
    correct
    /
    NUM_SAMPLES
) * 100

print("\n" + "="*60)
print(f"EM: {em:.2f}%")
print(
    f"Correct: {correct}/{NUM_SAMPLES}"
)
print(
    f"Runtime: {elapsed:.2f} sec"
)