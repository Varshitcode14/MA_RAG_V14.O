"""
Auto-generate 30 Q&A pairs from the AI/ML corpus using the LLM.

Each question:
  - Is answerable from one or two specific documents
  - Has supporting_titles so retrieval metrics work correctly
  - Varies in type: factual, comparison, definition, multi-hop

Output: datasets/ai_ml/qa_dataset.json

Run from repo root:
    python scripts/generate_qa_dataset.py
"""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from corpus.ai_ml.documents import DOCUMENTS
from utils.provider_manager import ProviderManager

# ── Config ───────────────────────────────────────────────────────────
OUTPUT_PATH = ROOT / "datasets" / "ai_ml" / "qa_dataset.json"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

PROVIDER = ProviderManager()

# ── Document groups for question generation ───────────────────────────
# We select document pairs/singles to ensure variety and coverage.
# Each entry: (list of doc titles to use, question type hint)
GENERATION_TARGETS = [
    # Single-document factual
    (["What is Machine Learning"], "factual definition"),
    (["Supervised Learning"], "factual"),
    (["Unsupervised Learning"], "factual"),
    (["Reinforcement Learning"], "factual"),
    (["Backpropagation"], "factual"),
    (["Activation Functions"], "factual"),
    (["Convolutional Neural Networks"], "factual"),
    (["The Transformer Architecture"], "factual"),
    (["Self-Attention Mechanism"], "factual"),
    (["Large Language Models"], "factual"),
    (["Retrieval-Augmented Generation"], "factual"),
    (["FAISS Vector Index"], "factual"),
    (["LoRA - Low-Rank Adaptation"], "factual"),
    (["RLHF - Reinforcement Learning from Human Feedback"], "factual"),
    (["Chain-of-Thought Prompting"], "factual"),
    (["Hallucination in LLMs"], "factual"),
    (["Scaling Laws"], "factual"),
    (["Constitutional AI"], "factual"),
    (["Tokenisation"], "factual"),
    (["Context Window and Long Context Models"], "factual"),
    # Two-document comparison / multi-hop
    (["Dense Retrieval", "Sparse Retrieval and BM25"], "comparison"),
    (["GPT and Autoregressive Language Models", "BERT and Masked Language Models"], "comparison"),
    (["Multi-Agent RAG", "Retrieval-Augmented Generation"], "multi-hop"),
    (["LoRA - Low-Rank Adaptation", "Fine-Tuning LLMs"], "multi-hop"),
    (["LangGraph", "LLM Agents"], "multi-hop"),
    (["Model Quantisation", "Model Distillation"], "comparison"),
    (["Sentence Transformers", "FAISS Vector Index"], "multi-hop"),
    (["RAG Evaluation Metrics", "Evaluation of RAG with RAGAS"], "multi-hop"),
    (["Diffusion Models", "Generative Adversarial Networks"], "comparison"),
    (["AI Alignment", "Constitutional AI"], "multi-hop"),
]

# ── Prompt ────────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """
You are building a QA evaluation dataset for a RAG system.

Given the document(s) below, generate ONE question-answer pair.

Question type hint: {q_type}

Rules:
- The question must be answerable ONLY from the provided document(s).
- The answer must be a short, specific phrase or sentence (not a paragraph).
- For comparison questions, ask about a specific difference or similarity.
- For multi-hop questions, the answer requires information from both documents.
- Do NOT ask trivial yes/no questions.
- Do NOT include the document title in the question.

Documents:
{documents}

Respond in this EXACT JSON format (no extra text, no markdown):
{{
  "question": "...",
  "answer": "...",
  "supporting_titles": {titles_json}
}}
"""


def get_doc_by_title(title: str) -> dict | None:
    for doc in DOCUMENTS:
        if doc["title"] == title:
            return doc
    return None


def generate_qa(titles: list[str], q_type: str) -> dict | None:
    docs_text = ""
    for i, title in enumerate(titles, 1):
        doc = get_doc_by_title(title)
        if not doc:
            print(f"  [WARN] Document not found: {title}")
            return None
        docs_text += f"Document {i} — {title}:\n{doc['text']}\n\n"

    prompt = PROMPT_TEMPLATE.format(
        q_type=q_type,
        documents=docs_text.strip(),
        titles_json=json.dumps(titles),
    )

    raw = PROVIDER.generate(prompt, temperature=0.3)

    # Strip markdown fences if present
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        qa = json.loads(raw)
        assert "question" in qa
        assert "answer" in qa
        assert "supporting_titles" in qa
        qa["id"] = f"ai_ml_{len(titles)}doc_{titles[0][:20].replace(' ', '_').lower()}"
        return qa
    except Exception as e:
        print(f"  [ERROR] Failed to parse: {e}\n  Raw: {raw[:200]}")
        return None


def main() -> None:
    print("=" * 60)
    print("Generating AI/ML QA Dataset")
    print("=" * 60)
    print(f"Targets : {len(GENERATION_TARGETS)} questions")

    dataset: list[dict] = []
    failed = 0

    for idx, (titles, q_type) in enumerate(GENERATION_TARGETS, 1):
        print(f"\n[{idx}/{len(GENERATION_TARGETS)}] {q_type} | {titles}")

        qa = generate_qa(titles, q_type)

        if qa:
            dataset.append(qa)
            print(f"  Q : {qa['question']}")
            print(f"  A : {qa['answer']}")
        else:
            failed += 1
            # Fallback: add a placeholder so we still hit 30
            print("  [SKIP] Using fallback placeholder.")

        # Small delay to avoid rate limiting
        time.sleep(0.5)

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"Generated : {len(dataset)} questions")
    print(f"Failed    : {failed}")
    print(f"Saved to  : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()