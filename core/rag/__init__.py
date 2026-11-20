"""Local RAG indexing, embedding, retrieval, and evaluation utilities."""

from core.rag.chunker import chunk_knowledge_base, load_dax_entries, load_kpi_entries
from core.rag.embeddings import (
    BaseEmbedder,
    EmbeddingError,
    LocalHashEmbedder,
    OpenAICompatibleEmbedder,
    SentenceTransformerEmbedder,
    create_embedder,
)
from core.rag.evaluator import evaluate_retrieval, load_eval_questions
from core.rag.index import KnowledgeIndex, build_knowledge_index

__all__ = [
    "BaseEmbedder",
    "EmbeddingError",
    "KnowledgeIndex",
    "LocalHashEmbedder",
    "OpenAICompatibleEmbedder",
    "SentenceTransformerEmbedder",
    "build_knowledge_index",
    "chunk_knowledge_base",
    "create_embedder",
    "evaluate_retrieval",
    "load_dax_entries",
    "load_eval_questions",
    "load_kpi_entries",
]

