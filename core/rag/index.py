"""ChromaDB-backed vector index and retriever over the local AutoBI knowledge base."""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import chromadb

from core.config import Settings, get_settings
from core.logging import get_logger
from core.rag.chunker import chunk_knowledge_base
from core.rag.embeddings import BaseEmbedder, create_embedder
from core.schemas.rag import KnowledgeCategory, KnowledgeChunk, RetrievedChunk

logger = get_logger("rag")


class KnowledgeIndex:
    """ChromaDB vector index supporting metadata-filtered semantic retrieval."""

    def __init__(
        self,
        *,
        embedder: BaseEmbedder | None = None,
        persist_dir: Path | None = None,
        collection_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Initialize an in-memory or persistent ChromaDB collection."""
        self._embedder = embedder or create_embedder(settings or get_settings())
        self._persist_dir = persist_dir
        if persist_dir is not None:
            persist_dir.mkdir(parents=True, exist_ok=True)
            self._client: Any = chromadb.PersistentClient(path=str(persist_dir))
            resolved_name = collection_name or "autobi_knowledge"
        else:
            self._client = chromadb.Client()
            resolved_name = collection_name or f"autobi_kb_{uuid.uuid4().hex[:8]}"

        self._collection_name = resolved_name
        self._collection = self._client.get_or_create_collection(
            name=resolved_name,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def count(self) -> int:
        """Return total number of chunks currently indexed in ChromaDB."""
        return int(self._collection.count())

    def index_chunks(self, chunks: Sequence[KnowledgeChunk]) -> int:
        """Embed and upsert ``chunks`` into the ChromaDB collection."""
        if not chunks:
            return 0

        texts = [chunk.content for chunk in chunks]
        embeddings = self._embedder.embed_texts(texts)
        ids = [chunk.chunk_id for chunk in chunks]
        metadatas: list[dict[str, str | int | float | bool]] = []

        for chunk in chunks:
            meta: dict[str, str | int | float | bool] = {
                "source_id": chunk.source_id,
                "source_file": chunk.source_file,
                "category": chunk.category,
                "domain": chunk.domain or "",
                "title": chunk.title,
                "needs_human_review": chunk.needs_human_review,
            }
            for k, v in chunk.metadata.items():
                if k not in meta:
                    meta[k] = v
            metadatas.append(meta)

        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info(
            "Indexed %d knowledge chunks into Chroma collection '%s'",
            len(chunks),
            self._collection_name,
        )
        return len(chunks)

    def retrieve(
        self,
        query: str,
        k: int = 5,
        *,
        category: KnowledgeCategory | None = None,
        domain: str | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the top-``k`` most relevant knowledge chunks with source IDs and scores."""
        if k < 1 or self.count == 0:
            return []

        query_embedding = self._embedder.embed_query(query)
        where: dict[str, Any] | None = None
        clauses: list[dict[str, Any]] = []
        if category is not None:
            clauses.append({"category": category})
        if domain is not None:
            clauses.append({"domain": domain})
        if len(clauses) == 1:
            where = clauses[0]
        elif len(clauses) > 1:
            where = {"$and": clauses}

        n_results = min(k, self.count)
        raw = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids_list = raw.get("ids", [[]])[0]
        docs_list = raw.get("documents", [[]])[0]
        metas_list = raw.get("metadatas", [[]])[0]
        dists_list = raw.get("distances", [[]])[0]

        results: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            ids_list,
            docs_list,
            metas_list,
            dists_list,
            strict=True,
        ):
            meta_dict = dict(meta or {})
            similarity = max(-1.0, min(1.0, round(1.0 - float(dist), 4)))
            extra_meta = {
                mk: mv
                for mk, mv in meta_dict.items()
                if mk
                not in {
                    "source_id",
                    "source_file",
                    "category",
                    "domain",
                    "title",
                    "needs_human_review",
                }
            }
            results.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    source_id=str(meta_dict.get("source_id", chunk_id)),
                    source_file=str(meta_dict.get("source_file", "")),
                    category=meta_dict["category"],
                    domain=str(meta_dict["domain"]) if meta_dict.get("domain") else None,
                    title=str(meta_dict.get("title", chunk_id)),
                    content=str(doc),
                    score=similarity,
                    needs_human_review=bool(meta_dict.get("needs_human_review", True)),
                    metadata=extra_meta,
                )
            )
        return results


def build_knowledge_index(
    knowledge_dir: Path | str = "knowledge",
    *,
    embedder: BaseEmbedder | None = None,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
    settings: Settings | None = None,
) -> KnowledgeIndex:
    """Chunk the local ``knowledge/`` directory and build a ready-to-query ``KnowledgeIndex``."""
    chunks = chunk_knowledge_base(knowledge_dir)
    index = KnowledgeIndex(
        embedder=embedder,
        persist_dir=persist_dir,
        collection_name=collection_name,
        settings=settings,
    )
    index.index_chunks(chunks)
    return index

