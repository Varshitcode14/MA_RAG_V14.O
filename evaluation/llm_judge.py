"""
LLM-as-a-Judge evaluation module.

Uses the shared ProviderManager (Groq → Cerebras fallback) so no
additional API keys are required.

Judges
------
- judge_answer_correctness : Is the prediction correct given the gold answer?
- judge_faithfulness       : Is the answer grounded in the retrieved context?
- judge_relevancy          : Does the answer address the question?
- batch_judge              : Run all judges over a prediction list.
"""

from __future__ import annotations


def _get_provider():
    """Lazy import of ProviderManager — deferred until first judge call."""
    from utils.provider_manager import ProviderManager
    return ProviderManager()


# ── Prompt templates ─────────────────────────────────────────────────

_CORRECTNESS_PROMPT = """
You are an evaluation judge for a question-answering system.

Question:
{question}

Gold Answer:
{gold}

Predicted Answer:
{prediction}

Task:
Decide whether the predicted answer is CORRECT given the gold answer.
A prediction is correct if it conveys the same core information,
even if worded differently.

Reply with EXACTLY one of:
CORRECT
INCORRECT

Do not explain.
"""

_FAITHFULNESS_PROMPT = """
You are an evaluation judge for a retrieval-augmented generation system.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

Task:
Decide whether the generated answer is FAITHFUL to the retrieved context.
An answer is faithful if every claim it makes can be traced to the context.

Reply with EXACTLY one of:
FAITHFUL
UNFAITHFUL

Do not explain.
"""

_RELEVANCY_PROMPT = """
You are an evaluation judge for a question-answering system.

Question:
{question}

Answer:
{answer}

Task:
Decide whether the answer is RELEVANT to the question.
An answer is relevant if it directly addresses what was asked.

Reply with EXACTLY one of:
RELEVANT
IRRELEVANT

Do not explain.
"""


# ── Judge functions ──────────────────────────────────────────────────

def judge_answer_correctness(
    question: str,
    prediction: str,
    gold: str,
) -> dict[str, str | float]:
    """
    Ask the LLM whether the prediction matches the gold answer.

    Returns:
        {"verdict": "CORRECT"|"INCORRECT", "score": 1.0|0.0}
    """
    provider = _get_provider()
    prompt = _CORRECTNESS_PROMPT.format(
        question=question,
        gold=gold,
        prediction=prediction,
    )
    raw = provider.generate(prompt, temperature=0).strip().upper()
    verdict = "CORRECT" if "CORRECT" in raw else "INCORRECT"
    return {"verdict": verdict, "score": 1.0 if verdict == "CORRECT" else 0.0}


def judge_faithfulness(
    question: str,
    answer: str,
    context: str,
) -> dict[str, str | float]:
    """
    Ask the LLM whether the answer is grounded in the context.

    Returns:
        {"verdict": "FAITHFUL"|"UNFAITHFUL", "score": 1.0|0.0}
    """
    provider = _get_provider()
    prompt = _FAITHFULNESS_PROMPT.format(
        question=question,
        context=context[:3000],  # guard against very long contexts
        answer=answer,
    )
    raw = provider.generate(prompt, temperature=0).strip().upper()
    verdict = "FAITHFUL" if "FAITHFUL" in raw else "UNFAITHFUL"
    return {"verdict": verdict, "score": 1.0 if verdict == "FAITHFUL" else 0.0}


def judge_relevancy(
    question: str,
    answer: str,
) -> dict[str, str | float]:
    """
    Ask the LLM whether the answer is relevant to the question.

    Returns:
        {"verdict": "RELEVANT"|"IRRELEVANT", "score": 1.0|0.0}
    """
    provider = _get_provider()
    prompt = _RELEVANCY_PROMPT.format(
        question=question,
        answer=answer,
    )
    raw = provider.generate(prompt, temperature=0).strip().upper()
    verdict = "RELEVANT" if "RELEVANT" in raw else "IRRELEVANT"
    return {"verdict": verdict, "score": 1.0 if verdict == "RELEVANT" else 0.0}


def batch_judge(
    samples: list[dict],
    run_faithfulness: bool = True,
    run_relevancy: bool = True,
) -> list[dict]:
    """
    Run LLM-as-a-Judge over a list of prediction dicts.

    Each sample must have keys: question, prediction, gold, context.

    Args:
        samples:            List of prediction dicts.
        run_faithfulness:   Include faithfulness judge (uses more tokens).
        run_relevancy:      Include relevancy judge.

    Returns:
        List of dicts with original fields plus judge results.
    """
    results: list[dict] = []

    for i, sample in enumerate(samples, start=1):
        print(f"  [Judge {i}/{len(samples)}] {sample['question'][:60]}...")

        row = dict(sample)  # shallow copy

        correctness = judge_answer_correctness(
            question=sample["question"],
            prediction=sample["prediction"],
            gold=sample["gold"],
        )
        row["judge_correctness"] = correctness["verdict"]
        row["judge_correctness_score"] = correctness["score"]

        if run_faithfulness:
            faithfulness = judge_faithfulness(
                question=sample["question"],
                answer=sample["prediction"],
                context=sample.get("context", ""),
            )
            row["judge_faithfulness"] = faithfulness["verdict"]
            row["judge_faithfulness_score"] = faithfulness["score"]

        if run_relevancy:
            relevancy = judge_relevancy(
                question=sample["question"],
                answer=sample["prediction"],
            )
            row["judge_relevancy"] = relevancy["verdict"]
            row["judge_relevancy_score"] = relevancy["score"]

        results.append(row)

    return results