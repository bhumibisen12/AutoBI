"""Unit tests for Step 5: knowledge base coverage, review flags, Chroma RAG, and eval hit-rate@3."""

from __future__ import annotations

from pathlib import Path

import httpx
from pydantic import SecretStr

from core.config import Settings
from core.rag import (
    LocalHashEmbedder,
    OpenAICompatibleEmbedder,
    build_knowledge_index,
    chunk_knowledge_base,
    create_embedder,
    evaluate_retrieval,
    load_dax_entries,
    load_eval_questions,
    load_kpi_entries,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "evals" / "rag_eval_questions.yaml"


def test_knowledge_base_domains_patterns_and_human_review_flags() -> None:
    """Every KPI/DAX entry has a plain-language definition and needs_human_review=True."""
    kpi_entries = load_kpi_entries(KNOWLEDGE_DIR)
    dax_entries = load_dax_entries(KNOWLEDGE_DIR)

    expected_domains = {
        "sales",
        "finance",
        "hr",
        "healthcare",
        "education",
        "operations",
        "marketing",
        "agriculture_commodity_prices",
    }
    observed_domains = {entry.domain for entry, _ in kpi_entries}
    assert observed_domains == expected_domains

    for entry, path in kpi_entries:
        assert entry.needs_human_review is True, f"{entry.id} in {path} missing review flag"
        assert len(entry.plain_language_definition.strip()) >= 15

    expected_dax_families = {
        "totals",
        "divide_ratios",
        "time_intelligence",
        "rolling_averages",
        "percent_of_total",
        "ranking",
        "running_total",
        "distinct_count",
        "top_n",
    }
    observed_families = {entry.pattern_family for entry, _ in dax_entries}
    assert observed_families == expected_dax_families

    for entry, path in dax_entries:
        assert entry.needs_human_review is True, f"{entry.id} in {path} missing review flag"
        assert len(entry.plain_language_definition.strip()) >= 15

    chunks = chunk_knowledge_base(KNOWLEDGE_DIR)
    visual_chunks = [c for c in chunks if c.category == "visual_rules"]
    assert len(visual_chunks) >= 6


def test_chroma_index_retrieve_filtering_and_persistence(tmp_path: Path) -> None:
    """ChromaDB index retrieves chunks with source IDs, supports filters, and persists cleanly."""
    persist_dir = tmp_path / "chroma_store"
    index = build_knowledge_index(
        KNOWLEDGE_DIR,
        embedder=LocalHashEmbedder(),
        persist_dir=persist_dir,
        collection_name="test_kb",
    )
    assert index.count >= 50

    dax_hits = index.retrieve(
        "safe division ratio avoiding zero errors",
        k=3,
        category="dax_patterns",
    )
    assert len(dax_hits) == 3
    assert all(hit.category == "dax_patterns" for hit in dax_hits)
    assert dax_hits[0].source_id == "dax.divide_ratios.safe_divide"
    assert dax_hits[0].needs_human_review is True

    hr_hits = index.retrieve("employee turnover attrition", k=2, domain="hr")
    assert len(hr_hits) == 2
    assert all(hit.domain == "hr" for hit in hr_hits)
    assert any(hit.source_id == "kpi.hr.attrition_rate" for hit in hr_hits)

    # Re-open persisted Chroma store without re-indexing
    reopened = build_knowledge_index(
        KNOWLEDGE_DIR,
        embedder=LocalHashEmbedder(),
        persist_dir=persist_dir,
        collection_name="test_kb",
    )
    assert reopened.count == index.count


def test_embedder_factory_and_openai_embedder_mock_transport() -> None:
    """Embedder factory respects config and OpenAICompatibleEmbedder works over httpx transport."""
    mock_settings = Settings(_env_file=None, llm_provider="mock")
    assert isinstance(create_embedder(mock_settings), LocalHashEmbedder)

    def _handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer sk-embed-test1234567890"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            },
        )

    with httpx.Client(transport=httpx.MockTransport(_handler)) as http_client:
        embedder = OpenAICompatibleEmbedder(
            api_key=SecretStr("sk-embed-test1234567890"),
            model="text-embedding-3-small",
            http_client=http_client,
        )
        vecs = embedder.embed_texts(["gross margin", "rolling average"])
        assert vecs == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert embedder.dimension == 3


def test_retrieval_eval_20_questions_hit_rate_at_3() -> None:
    """20-question retrieval benchmark achieves strong hit-rate@3 over the local knowledge base."""
    questions = load_eval_questions(EVAL_QUESTIONS_PATH)
    assert len(questions) == 20

    index = build_knowledge_index(KNOWLEDGE_DIR, embedder=LocalHashEmbedder())
    report = evaluate_retrieval(index, questions, k=3)

    assert report.total_questions == 20
    assert report.k == 3
    assert report.hit_rate_at_k >= 0.85, (
        f"Expected hit-rate@3 >= 0.85, got {report.hit_rate_at_k:.2%}; "
        f"misses={[r.question_id for r in report.results if not r.hit_at_k]}"
    )

