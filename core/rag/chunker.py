"""Knowledge base loader and chunker for YAML (KPIs, DAX patterns) and Markdown visual rules."""

from __future__ import annotations

import re
from pathlib import Path

import yaml

from core.schemas.rag import DAXPatternEntry, KnowledgeChunk, KPIEntry

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Convert a heading string into a stable lowercase identifier slug."""
    return _SLUG_RE.sub("_", text.strip().lower()).strip("_") or "section"


def _relative_posix(path: Path, root: Path) -> str:
    """Return a clean POSIX path relative to ``root.parent`` (e.g. ``knowledge/...``)."""
    try:
        return path.resolve().relative_to(root.resolve().parent).as_posix()
    except ValueError:
        return path.name


def load_kpi_entries(knowledge_dir: Path) -> list[tuple[KPIEntry, Path]]:
    """Load and Pydantic-validate every KPI YAML file under ``knowledge/kpi_library/``."""
    kpi_dir = knowledge_dir / "kpi_library"
    if not kpi_dir.is_dir():
        return []
    entries: list[tuple[KPIEntry, Path]] = []
    for yaml_path in sorted(kpi_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        for item in raw.get("kpis", []):
            entries.append((KPIEntry.model_validate(item), yaml_path))
    return entries


def load_dax_entries(knowledge_dir: Path) -> list[tuple[DAXPatternEntry, Path]]:
    """Load and Pydantic-validate every DAX pattern YAML under ``knowledge/dax_patterns/``."""
    dax_dir = knowledge_dir / "dax_patterns"
    if not dax_dir.is_dir():
        return []
    entries: list[tuple[DAXPatternEntry, Path]] = []
    for yaml_path in sorted(dax_dir.glob("*.yaml")):
        raw = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        for item in raw.get("patterns", []):
            entries.append((DAXPatternEntry.model_validate(item), yaml_path))
    return entries


def _chunk_markdown_file(
    md_path: Path,
    knowledge_dir: Path,
    *,
    max_chars: int = 1200,
) -> list[KnowledgeChunk]:
    """Split a visual-rules Markdown document by ``## `` sections with title context preserved."""
    text = md_path.read_text(encoding="utf-8").strip()
    if not text:
        return []

    lines = text.splitlines()
    doc_title = md_path.stem.replace("_", " ").title()
    sections: list[tuple[str, list[str]]] = []
    current_heading = "Overview"
    current_lines: list[str] = []

    for line in lines:
        if line.startswith("# ") and not sections and not current_lines:
            doc_title = line[2:].strip()
            continue
        if line.startswith("## "):
            if current_lines:
                sections.append((current_heading, current_lines))
            current_heading = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)
    if current_lines:
        sections.append((current_heading, current_lines))

    rel_file = _relative_posix(md_path, knowledge_dir)
    chunks: list[KnowledgeChunk] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            continue
        slug = _slugify(heading)
        source_id = f"visual.{md_path.stem}#{slug}"
        full_text = f"{doc_title} — {heading}\n{body}"

        if len(full_text) <= max_chars:
            chunks.append(
                KnowledgeChunk(
                    chunk_id=source_id,
                    source_id=source_id,
                    source_file=rel_file,
                    category="visual_rules",
                    domain="visual_design",
                    title=f"{doc_title}: {heading}",
                    content=full_text,
                    needs_human_review=False,
                    metadata={"rule_file": md_path.stem, "section": slug},
                )
            )
        else:
            # Sub-chunk long sections cleanly by paragraph boundaries.
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            buf: list[str] = []
            part_idx = 1
            for para in paragraphs:
                candidate = "\n\n".join([*buf, para])
                if buf and len(candidate) + len(doc_title) + len(heading) + 10 > max_chars:
                    part_id = f"{source_id}_part{part_idx}"
                    chunks.append(
                        KnowledgeChunk(
                            chunk_id=part_id,
                            source_id=source_id,
                            source_file=rel_file,
                            category="visual_rules",
                            domain="visual_design",
                            title=f"{doc_title}: {heading} (Part {part_idx})",
                            content=f"{doc_title} — {heading}\n" + "\n\n".join(buf),
                            needs_human_review=False,
                            metadata={"rule_file": md_path.stem, "section": slug},
                        )
                    )
                    part_idx += 1
                    buf = [para]
                else:
                    buf.append(para)
            if buf:
                part_id = f"{source_id}_part{part_idx}"
                chunks.append(
                    KnowledgeChunk(
                        chunk_id=part_id,
                        source_id=source_id,
                        source_file=rel_file,
                        category="visual_rules",
                        domain="visual_design",
                        title=f"{doc_title}: {heading} (Part {part_idx})",
                        content=f"{doc_title} — {heading}\n" + "\n\n".join(buf),
                        needs_human_review=False,
                        metadata={"rule_file": md_path.stem, "section": slug},
                    )
                )
    return chunks


def chunk_knowledge_base(
    knowledge_dir: Path | str,
    *,
    max_markdown_chars: int = 1200,
) -> list[KnowledgeChunk]:
    """Load and chunk all KPI YAML, DAX pattern YAML, and visual rule Markdown files."""
    root = Path(knowledge_dir)
    chunks: list[KnowledgeChunk] = []

    # 1. KPI library chunks
    for kpi, yaml_path in load_kpi_entries(root):
        roles = ", ".join(kpi.required_column_roles)
        content = (
            f"KPI: {kpi.name} (Domain: {kpi.domain})\n"
            f"Definition: {kpi.plain_language_definition}\n"
            f"Formula: {kpi.formula}\n"
            f"Required column roles: {roles}\n"
            f"Format: {kpi.format_hint}"
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=kpi.id,
                source_id=kpi.id,
                source_file=_relative_posix(yaml_path, root),
                category="kpi_library",
                domain=kpi.domain,
                title=f"{kpi.domain.title()} KPI: {kpi.name}",
                content=content,
                needs_human_review=kpi.needs_human_review,
                metadata={"format_hint": kpi.format_hint, "formula": kpi.formula},
            )
        )

    # 2. DAX pattern chunks
    for dax, yaml_path in load_dax_entries(root):
        roles = ", ".join(dax.required_column_roles)
        pitfalls = "; ".join(dax.pitfalls) if dax.pitfalls else "None"
        content = (
            f"DAX Pattern: {dax.name} (Family: {dax.pattern_family})\n"
            f"Definition: {dax.plain_language_definition}\n"
            f"DAX Template:\n{dax.dax_template.strip()}\n"
            f"Required column roles: {roles}\n"
            f"SQL Verification: {dax.sql_verification_pattern}\n"
            f"Pitfalls: {pitfalls}"
        )
        chunks.append(
            KnowledgeChunk(
                chunk_id=dax.id,
                source_id=dax.id,
                source_file=_relative_posix(yaml_path, root),
                category="dax_patterns",
                domain=dax.pattern_family,
                title=f"DAX Pattern: {dax.name}",
                content=content,
                needs_human_review=dax.needs_human_review,
                metadata={"pattern_family": dax.pattern_family},
            )
        )

    # 3. Visual design rule chunks
    visual_dir = root / "visual_rules"
    if visual_dir.is_dir():
        for md_path in sorted(visual_dir.glob("*.md")):
            chunks.extend(_chunk_markdown_file(md_path, root, max_chars=max_markdown_chars))

    return chunks

