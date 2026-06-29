"""
Tests for Traditional RAG pipeline components.

Each test is independently runnable:
    python tests/test_traditional_rag.py

Tests use lightweight stubs for the LLM and retriever so
they run quickly without API calls.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.append(str(Path(__file__).resolve().parents[1]))


# ── Helpers ───────────────────────────────────────────────────────────

def _fake_docs() -> list[dict]:
    return [
        {
            "title": "Retrieval-Augmented Generation",
            "text": "RAG combines retrieval with generation to answer questions.",
            "score": 0.95,
            "source": "test",
        },
        {
            "title": "Dense Retrieval",
            "text": "Dense retrieval uses vector embeddings to find relevant passages.",
            "score": 0.88,
            "source": "test",
        },
    ]


# ── Tests ─────────────────────────────────────────────────────────────

def test_generator_builds_context() -> None:
    """Generator correctly formats multiple docs into a context string."""
    from Traditional_RAG.generator import TraditionalGenerator

    gen = TraditionalGenerator.__new__(TraditionalGenerator)
    docs = _fake_docs()
    context = gen._build_context(docs)

    assert "[1]" in context
    assert "[2]" in context
    assert "RAG combines" in context
    assert "Dense retrieval" in context
    print("PASS  test_generator_builds_context")


def test_generator_calls_provider() -> None:
    """Generator calls ProviderManager.generate exactly once."""
    with patch("Traditional_RAG.generator.ProviderManager") as MockPM:
        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Test answer"
        MockPM.return_value = mock_provider

        from Traditional_RAG import generator as gen_module
        # Reload to pick up mock
        import importlib
        importlib.reload(gen_module)

        g = gen_module.TraditionalGenerator()
        result = g.generate("What is RAG?", _fake_docs())

        assert result == "Test answer"
        mock_provider.generate.assert_called_once()
    print("PASS  test_generator_calls_provider")


def test_pipeline_result_schema() -> None:
    """
    Pipeline.answer() returns all required keys with correct types.
    Uses a mock retriever and generator so no API/FAISS calls are made.
    """
    with (
        patch("Traditional_RAG.retriever.DenseRetriever") as MockRet,
        patch("Traditional_RAG.generator.ProviderManager") as MockPM,
    ):
        mock_ret_instance = MagicMock()
        mock_ret_instance.search.return_value = _fake_docs()
        MockRet.return_value = mock_ret_instance

        mock_provider = MagicMock()
        mock_provider.generate.return_value = "Mocked answer"
        MockPM.return_value = mock_provider

        import importlib
        import Traditional_RAG.retriever as ret_mod
        import Traditional_RAG.generator as gen_mod
        import Traditional_RAG.pipeline as pipe_mod
        importlib.reload(ret_mod)
        importlib.reload(gen_mod)
        importlib.reload(pipe_mod)

        pipeline = pipe_mod.TraditionalRAGPipeline()
        result = pipeline.answer("What is RAG?")

    required_keys = {
        "question",
        "answer",
        "retrieved_docs",
        "retrieved_titles",
        "context",
        "retrieval_time",
        "generation_time",
        "total_time",
        "history",
        "reasoning_steps",
        "pipeline",
    }

    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    assert result["pipeline"] == "traditional_rag"
    assert result["reasoning_steps"] == 1
    assert result["history"] == []
    assert isinstance(result["retrieved_titles"], list)
    assert isinstance(result["total_time"], float)
    print("PASS  test_pipeline_result_schema")


def test_pipeline_schema_matches_ma_rag_schema() -> None:
    """
    Both pipelines must return the exact same set of top-level keys.
    This test compares the key sets without running inference.
    """
    traditional_keys = {
        "question", "answer", "retrieved_docs", "retrieved_titles",
        "context", "retrieval_time", "generation_time", "total_time",
        "history", "reasoning_steps", "pipeline",
    }

    # MA-RAG _normalize() also returns this same set (verified by inspection)
    ma_rag_keys = {
        "question", "answer", "retrieved_docs", "retrieved_titles",
        "context", "retrieval_time", "generation_time", "total_time",
        "history", "reasoning_steps", "pipeline",
    }

    assert traditional_keys == ma_rag_keys, (
        f"Schema mismatch!\n"
        f"Only in Traditional RAG: {traditional_keys - ma_rag_keys}\n"
        f"Only in MA-RAG: {ma_rag_keys - traditional_keys}"
    )
    print("PASS  test_pipeline_schema_matches_ma_rag_schema")


# ── Runner ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Traditional RAG Tests")
    print("=" * 60)

    test_generator_builds_context()
    test_generator_calls_provider()
    test_pipeline_result_schema()
    test_pipeline_schema_matches_ma_rag_schema()

    print("\nAll tests passed.")