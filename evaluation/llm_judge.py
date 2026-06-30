"""
LLM-as-a-Judge evaluation module.

Uses the shared ProviderManager (Groq -> Cerebras -> Bedrock fallback) so
no additional API keys are required.

Graded scoring
--------------
Earlier this module used a binary CORRECT/INCORRECT judge. On easy
corpora that saturates at 1.0 for every system and cannot discriminate
between pipelines. We now use a 1-5 rubric per dimension and normalize
to [0, 1]:

    score = (rating - 1) / 4

Judges
------
- judge_answer_correctness : How well does the prediction match the gold?
- judge_faithfulness       : Is the answer grounded in the retrieved context?
- judge_relevancy          : Does the answer address the question?
- batch_judge              : Run all judges over a prediction list.
"""

from __future__ import annotations

import os
import re

VERBOSE = os.getenv("MARAG_VERBOSE", "1") == "1"


def _get_provider():
    """Lazy import of ProviderManager - deferred until first judge call."""
    from utils.provider_manager import ProviderManager
    return ProviderManager()


# ── Prompt templates (1-5 rubric) ────────────────────────────────────

_CORRECTNESS_PROMPT = """
You are a strict evaluation judge for a question-answering system.

Question:
{question}

Gold Answer:
{gold}

Predicted Answer:
{prediction}

Task:
Rate how well the predicted answer matches the gold answer on a 1-5 scale.
Judge the substance, not the wording or length.

Rubric:
5 = Fully correct. Covers every required fact in the gold answer.
4 = Mostly correct. Covers the main fact but misses a minor detail.
3 = Partially correct. Gets some of the answer but misses a key part.
2 = Mostly incorrect. Only marginally related to the gold answer.
1 = Incorrect or irrelevant.

Reply with ONLY the single digit (1, 2, 3, 4, or 5). Do not explain.
"""

_FAITHFULNESS_PROMPT = """
You are a strict evaluation judge for a retrieval-augmented generation system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Task:
Rate how faithful the generated answer is to the retrieved context on a
1-5 scale. An answer is faithful if every claim it makes is supported by
the context.

Rubric:
5 = Every claim is fully supported by the context.
4 = Mostly supported; one minor unsupported detail.
3 = Roughly half the claims are supported.
2 = Mostly unsupported claims.
1 = Fabricated / contradicts the context.

Reply with ONLY the single digit (1, 2, 3, 4, or 5). Do not explain.
"""

_RELEVANCY_PROMPT = """
You are a strict evaluation judge for a question-answering system.

Question:
{question}

Answer:
{answer}

Task:
Rate how relevant the answer is to the question on a 1-5 scale.

Rubric:
5 = Directly and completely addresses the question.
4 = Addresses the question with minor digression.
3 = Partially addresses the question.
2 = Largely off-topic.
1 = Irrelevant.

Reply with ONLY the single digit (1, 2, 3, 4, or 5). Do not explain.
"""


# ── Parsing ───────────────────────────────────────────────────────────

def _parse_rating(raw: str) -> float:
    """
    Extract the first 1-5 digit from the model output and normalize to
    [0, 1]. Falls back to 0.0 if no valid rating is found.
    """
    if not raw:
        return 0.0
    match = re.search(r"[1-5]", raw)
    if not match:
        return 0.0
    rating = int(match.group(0))
    return (rating - 1) / 4.0


# ── Judge functions ──────────────────────────────────────────────────

def judge_answer_correctness(question: str, prediction: str, gold: str) -> dict:
    """Graded correctness. Returns {"rating": 1-5, "score": [0,1]}."""
    provider = _get_provider()
    prompt = _CORRECTNESS_PROMPT.format(
        question=question, gold=gold, prediction=prediction
    )
    raw = provider.generate(prompt, temperature=0).strip()
    score = _parse_rating(raw)
    return {"rating": round(score * 4 + 1), "score": score}


def judge_faithfulness(question: str, answer: str, context: str) -> dict:
    """Graded faithfulness. Returns {"rating": 1-5, "score": [0,1]}."""
    provider = _get_provider()
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question, context=context[:4000], answer=answer
    )
    raw = provider.generate(prompt, temperature=0).strip()
    score = _parse_rating(raw)
    return {"rating": round(score * 4 + 1), "score": score}


def judge_relevancy(question: str, answer: str) -> dict:
    """Graded relevancy. Returns {"rating": 1-5, "score": [0,1]}."""
    provider = _get_provider()
    prompt = _RELEVANCY_PROMPT.format(question=question, answer=answer)
    raw = provider.generate(prompt, temperature=0).strip()
    score = _parse_rating(raw)
    return {"rating": round(score * 4 + 1), "score": score}


def batch_judge(
    samples: list[dict],
    run_faithfulness: bool = True,
    run_relevancy: bool = True,
) -> list[dict]:
    """
    Run LLM-as-a-Judge over a list of prediction dicts.

    Each sample must have keys: question, prediction, gold, context.

    Returns:
        List of dicts with original fields plus judge results.
    """
    results: list[dict] = []

    for i, sample in enumerate(samples, start=1):
        if VERBOSE:
            print(f"  [Judge {i}/{len(samples)}] {sample['question'][:60]}...")

        row = dict(sample)  # shallow copy

        correctness = judge_answer_correctness(
            question=sample["question"],
            prediction=sample["prediction"],
            gold=sample["gold"],
        )
        row["judge_correctness_rating"] = correctness["rating"]
        row["judge_correctness_score"] = correctness["score"]

        if run_faithfulness:
            faith = judge_faithfulness(
                question=sample["question"],
                answer=sample["prediction"],
                context=sample.get("context", ""),
            )
            row["judge_faithfulness_rating"] = faith["rating"]
            row["judge_faithfulness_score"] = faith["score"]

        if run_relevancy:
            rel = judge_relevancy(
                question=sample["question"],
                answer=sample["prediction"],
            )
            row["judge_relevancy_rating"] = rel["rating"]
            row["judge_relevancy_score"] = rel["score"]

        results.append(row)

    return results
