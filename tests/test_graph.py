from MA_RAG.pipeline import run

question = (
    "What is Retrieval-Augmented Generation and "
    "how does retrieval improve large language models?"
)

result = run(question)

print(result)