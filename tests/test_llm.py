import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).resolve().parents[1])
)

from T_RAG.llm.T_llm import TLLM

llm = TLLM()

response = llm.generate(
    question="Who wrote Harry Potter?",
    context="""
Harry Potter is a series of fantasy novels.

J. K. Rowling is the author of Harry Potter.
"""
)

print(response)