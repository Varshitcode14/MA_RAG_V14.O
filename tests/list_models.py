import json
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

results = {
    "groq": [],
    "cerebras": []
}


# ---------- GROQ ----------
try:
    from groq import Groq

    key = os.getenv("GROQ_API_KEYS").split(",")[0].strip()
    client = Groq(api_key=key)

    models = client.models.list()

    results["groq"] = sorted([m.id for m in models.data])

    print("\n=== GROQ MODELS ===")
    for model in results["groq"]:
        print(model)

except Exception as e:
    print(f"\nGroq Error: {e}")


# ---------- CEREBRAS ----------
try:
    from cerebras.cloud.sdk import Cerebras

    key = os.getenv("CEREBRAS_API_KEYS").split(",")[0].strip()
    client = Cerebras(api_key=key)

    models = client.models.list()

    results["cerebras"] = sorted([m.id for m in models.data])

    print("\n=== CEREBRAS MODELS ===")
    for model in results["cerebras"]:
        print(model)

except Exception as e:
    print(f"\nCerebras Error: {e}")


# ---------- SAVE ----------
output_file = Path(__file__).parent / "models_available.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nSaved to: {output_file}")