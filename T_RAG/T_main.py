import sys
import time

sys.path.append("T_RAG/retrieval")
sys.path.append("T_RAG/llm")

from T_RAG.retrieval.T_retriever import TRetriever
from T_RAG.llm.T_llm import TLLM


class TRAG:

    def __init__(self):

        self.retriever = TRetriever(
            faiss_path="T_RAG/indexes/hotpot_cosine.faiss",
            metadata_path="T_RAG/indexes/metadata.pkl",
            cosine=True
        )

        self.llm = TLLM()

    def answer(self, question):

        total_start = time.perf_counter()

        # -----------------------------
        # Retrieval
        # -----------------------------
        retrieval_start = time.perf_counter()

        docs = self.retriever.search(
            question,
            k=3
        )

        retrieval_time = time.perf_counter() - retrieval_start

        context = "\n\n".join(
            [
                f"Title: {d['title']}\n{d['text']}"
                for d in docs
            ]
        )

        # -----------------------------
        # Generation
        # -----------------------------
        generation_start = time.perf_counter()

        answer = self.llm.generate(
            question=question,
            context=context
        )

        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        result = {

            "question": question,

            "answer": answer,

            "retrieved_docs": docs,

            "retrieved_titles": [
                d["title"] for d in docs
            ],

            "context": context,

            "retrieval_time": retrieval_time,

            "generation_time": generation_time,

            "total_time": total_time,

            # For compatibility with MA-RAG

            "history": [],

            "reasoning_steps": 1
        }

        return result


if __name__ == "__main__":

    rag = TRAG()

    question = input("Question: ")

    result = rag.answer(question)

    print("\n")
    print("=" * 80)
    print("RESULT")
    print("=" * 80)

    print(f"Question : {result['question']}")
    print(f"Answer   : {result['answer']}")

    print("\nRetrieved Documents")

    for i, title in enumerate(result["retrieved_titles"], 1):

        print(f"{i}. {title}")

    print("\nTiming")

    print(f"Retrieval : {result['retrieval_time']:.3f} sec")
    print(f"Generation: {result['generation_time']:.3f} sec")
    print(f"Total     : {result['total_time']:.3f} sec")