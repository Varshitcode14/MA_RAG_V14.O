"""
Prompts for Traditional RAG pipeline.

Kept separate so they can be tuned independently
without touching the retrieval or generation logic.
"""

TRADITIONAL_RAG_PROMPT = """
You are a question answering system.

Use ONLY the provided context to answer the question.

If the answer is not present in the context, reply exactly:
I don't know.

Rules:
- Do NOT hallucinate facts.
- Be concise and direct.
- Return ONLY the answer, no explanation.

Context:
{context}

Question:
{question}

Answer:
"""