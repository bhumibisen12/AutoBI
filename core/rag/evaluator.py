"""Retrieval evaluation harness reporting hit-rate@k and MRR@k over benchmark questions."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from core.rag.index import KnowledgeIndex
from core.schemas.rag import (
    RetrievalEvalCaseResult,
    RetrievalEvalQuestion,
    RetrievalEvalReport,
)


def load_eval_questions(questions_path: Path | str) -> list[RetrievalEvalQuestion]:
    """Load and validate retrieval evaluation questions from a YAML file."""
    raw = yaml.safe_load(Path(questions_path).read_text(encoding="utf-8")) or {}
    return [RetrievalEvalQuestion.model_validate(item) for item in raw.get("questions", [])]


def _matches_expected(source_id: str, expected_sources: Sequence[str]) -> bool:
    """Return True if ``source_id`` matches any expected source ID exactly or by prefix."""
    for expected in expected_sources:
        if source_id == expected or source_id.startswith(f"{expected}#"):
            return True
    return False


def evaluate_retrieval(
    index: KnowledgeIndex,
    questions: Sequence[RetrievalEvalQuestion],
    *,
    k: int = 3,
) -> RetrievalEvalReport:
    """Evaluate ``index.retrieve(query, k=k)`` across benchmark questions and compute hit-rate@k."""
    results: list[RetrievalEvalCaseResult] = []
    hits = 0
    rr_sum = 0.0

    for question in questions:
        retrieved = index.retrieve(question.query, k=k)
        retrieved_sources = [chunk.source_id for chunk in retrieved]

        first_hit_rank: int | None = None
        for rank, src_id in enumerate(retrieved_sources, start=1):
            if _matches_expected(src_id, question.expected_sources):
                first_hit_rank = rank
                break

        hit = first_hit_rank is not None
        rr = round(1.0 / first_hit_rank, 4) if first_hit_rank is not None else 0.0
        if hit:
            hits += 1
            rr_sum += rr

        results.append(
            RetrievalEvalCaseResult(
                question_id=question.id,
                query=question.query,
                expected_sources=list(question.expected_sources),
                retrieved_sources=retrieved_sources,
                hit_at_k=hit,
                reciprocal_rank=rr,
            )
        )

    total = len(questions)
    hit_rate = round(hits / total, 4) if total > 0 else 0.0
    mrr = round(rr_sum / total, 4) if total > 0 else 0.0

    return RetrievalEvalReport(
        k=k,
        total_questions=total,
        hits=hits,
        hit_rate_at_k=hit_rate,
        mrr_at_k=mrr,
        results=results,
    )

