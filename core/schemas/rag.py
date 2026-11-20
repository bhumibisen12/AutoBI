"""Pydantic contracts for the local RAG knowledge base, retrieval results, and evaluation."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

KnowledgeCategory = Literal["kpi_library", "dax_patterns", "visual_rules"]


class KPIEntry(BaseModel):
    """Structured KPI knowledge entry loaded from ``knowledge/kpi_library/*.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    domain: str = Field(min_length=1)
    name: str = Field(min_length=1)
    plain_language_definition: str = Field(min_length=1)
    formula: str = Field(min_length=1)
    required_column_roles: list[str] = Field(min_length=1)
    format_hint: Literal["currency", "percent", "integer", "decimal", "ratio"] = "decimal"
    needs_human_review: bool = Field(
        default=True,
        description="Must remain True until reviewed and approved by a human domain reviewer.",
    )


class DAXPatternEntry(BaseModel):
    """Structured DAX pattern entry loaded from ``knowledge/dax_patterns/*.yaml``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    pattern_family: str = Field(min_length=1)
    name: str = Field(min_length=1)
    plain_language_definition: str = Field(min_length=1)
    dax_template: str = Field(min_length=1)
    sql_verification_pattern: str = Field(default="")
    required_column_roles: list[str] = Field(min_length=1)
    pitfalls: list[str] = Field(default_factory=list)
    needs_human_review: bool = Field(
        default=True,
        description="Must remain True until reviewed and approved by a human domain reviewer.",
    )


class KnowledgeChunk(BaseModel):
    """Single indexed knowledge chunk ready for embedding and vector storage."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    category: KnowledgeCategory
    domain: str | None = None
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    needs_human_review: bool = True
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    """Knowledge chunk returned by ``retrieve(query, k)`` with similarity score and source ID."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    category: KnowledgeCategory
    domain: str | None = None
    title: str = Field(min_length=1)
    content: str = Field(min_length=1)
    score: float = Field(ge=-1.0, le=1.0)
    needs_human_review: bool = True
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)


class RetrievalEvalQuestion(BaseModel):
    """Single benchmark question for evaluating RAG retrieval quality."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_sources: list[str] = Field(min_length=1)
    category: KnowledgeCategory


class RetrievalEvalCaseResult(BaseModel):
    """Per-question outcome of a retrieval evaluation run."""

    model_config = ConfigDict(extra="forbid")

    question_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    expected_sources: list[str]
    retrieved_sources: list[str]
    hit_at_k: bool
    reciprocal_rank: float = Field(ge=0.0, le=1.0)


class RetrievalEvalReport(BaseModel):
    """Aggregate RAG retrieval evaluation report (hit-rate@k and MRR@k)."""

    model_config = ConfigDict(extra="forbid")

    k: int = Field(ge=1)
    total_questions: int = Field(ge=0)
    hits: int = Field(ge=0)
    hit_rate_at_k: float = Field(ge=0.0, le=1.0)
    mrr_at_k: float = Field(ge=0.0, le=1.0)
    results: list[RetrievalEvalCaseResult] = Field(default_factory=list)

