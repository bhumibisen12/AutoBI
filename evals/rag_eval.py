"""CLI runner for evaluating AutoBI's local RAG retrieval hit-rate@3."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.rag import (  # noqa: E402
    LocalHashEmbedder,
    build_knowledge_index,
    evaluate_retrieval,
    load_eval_questions,
)
from core.schemas import RetrievalEvalReport  # noqa: E402


def run_eval(k: int = 3) -> RetrievalEvalReport:
    """Build the local knowledge index and run the 20-question retrieval benchmark."""
    index = build_knowledge_index(
        PROJECT_ROOT / "knowledge",
        embedder=LocalHashEmbedder(),
    )
    questions = load_eval_questions(PROJECT_ROOT / "evals" / "rag_eval_questions.yaml")
    return evaluate_retrieval(index, questions, k=k)


def main() -> int:
    """Execute the RAG evaluation and print per-question hits plus hit-rate@3."""
    report = run_eval(k=3)
    print(f"RAG Retrieval Evaluation (k={report.k}, questions={report.total_questions})")
    print("-" * 80)
    for case in report.results:
        status = "HIT " if case.hit_at_k else "MISS"
        print(
            f"[{status}] {case.question_id}: expected={case.expected_sources} | "
            f"top{report.k}={case.retrieved_sources}"
        )
    print("-" * 80)
    print(
        f"Hit-Rate@{report.k}: {report.hits}/{report.total_questions} "
        f"({report.hit_rate_at_k * 100:.1f}%) | MRR@{report.k}: {report.mrr_at_k:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

