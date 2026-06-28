from MA_RAG.pipeline import run


question = (
    "Where was the only European Cup Final in which Jupp Heynckes played held?"
)

result = run(question)

print(result)