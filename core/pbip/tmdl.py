"""TMDL semantic model generator (database, model, expressions, relationships, tables)."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Sequence
from pathlib import Path

from core.pbip.templates import SCHEMA_PBISM
from core.schemas.dax import DAXMeasureSpec
from core.schemas.pbip import DataSourceFormat
from core.schemas.semantic import (
    DataTypeBI,
    SemanticColumn,
    SemanticModel,
    SemanticTable,
)

_TMDL_NAMESPACE = uuid.UUID("7f3b5c42-48b2-4e0c-9c74-96f9f9d60101")
_PLAIN_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_M_TYPE_MAP: dict[DataTypeBI, str] = {
    "int64": "Int64.Type",
    "double": "type number",
    "string": "type text",
    "dateTime": "type datetime",
    "boolean": "type logical",
}


def stable_guid(key: str) -> str:
    """Return a deterministic UUIDv5 string so generated TMDL files stay reproducible."""
    return str(uuid.uuid5(_TMDL_NAMESPACE, key))


def _tmdl_ident(name: str) -> str:
    """Quote a TMDL identifier in single quotes when it contains spaces or special characters."""
    if _PLAIN_IDENT_RE.match(name):
        return name
    escaped = name.replace("'", "''")
    return f"'{escaped}'"


def _render_m_partition(
    table: SemanticTable,
    *,
    source_format: DataSourceFormat,
) -> list[str]:
    """Render a Power Query M import partition reading from the ``DataFolderPath`` parameter."""
    type_pairs = ", ".join(
        f'{{"{col.name}", {_M_TYPE_MAP.get(col.data_type, "type any")}}}'
        for col in table.columns
    )
    ext = "parquet" if source_format == "parquet" else "csv"
    file_expr = f'DataFolderPath & "/{table.table_name}.{ext}"'

    if source_format == "parquet":
        return [
            f"\tpartition {_tmdl_ident(table.table_name + '-Partition')} = m",
            "\t\tmode: import",
            "\t\tsource =",
            "\t\t\tlet",
            f"\t\t\t\tSource = Parquet.Document(File.Contents({file_expr})),",
            f'\t\t\t\t#"Changed Type" = Table.TransformColumnTypes(Source, {{{type_pairs}}})',
            "\t\t\tin",
            '\t\t\t\t#"Changed Type"',
        ]

    col_count = len(table.columns)
    return [
        f"\tpartition {_tmdl_ident(table.table_name + '-Partition')} = m",
        "\t\tmode: import",
        "\t\tsource =",
        "\t\t\tlet",
        (
            f"\t\t\t\tSource = Csv.Document(File.Contents({file_expr}), "
            f'[Delimiter=",", Columns={col_count}, Encoding=65001, QuoteStyle=QuoteStyle.Csv]),'
        ),
        '\t\t\t\t#"Promoted Headers" = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),',
        (
            '\t\t\t\t#"Changed Type" = Table.TransformColumnTypes('
            f'#"Promoted Headers", {{{type_pairs}}})'
        ),
        "\t\t\tin",
        '\t\t\t\t#"Changed Type"',
    ]


def _render_column_block(table: SemanticTable, col: SemanticColumn) -> list[str]:
    """Render a single TMDL column definition block."""
    lines = [
        f"\tcolumn {_tmdl_ident(col.name)}",
        f"\t\tdataType: {col.data_type}",
    ]
    if table.is_date_table and col.name == table.date_column:
        lines.append("\t\tisKey")
    if col.is_hidden:
        lines.append("\t\tisHidden")
    if col.format_string:
        lines.append(f"\t\tformatString: {col.format_string}")
    lines.append(f"\t\tlineageTag: {stable_guid(f'col:{table.table_name}.{col.name}')}")
    lines.append(f"\t\tsummarizeBy: {col.summarize_by}")
    lines.append(f"\t\tsourceColumn: {col.name}")
    if table.is_date_table and col.name == table.date_column:
        lines.append("\t\tannotation UnderlyingDateTimeDataType = Date")
    lines.append("")
    return lines


def _render_measure_block(table_name: str, measure: DAXMeasureSpec) -> list[str]:
    """Render a single TMDL measure block supporting single- and multi-line DAX expressions."""
    expr_lines = measure.expression.strip().splitlines()
    lines: list[str] = []
    if measure.description:
        clean_desc = measure.description.replace("\n", " ").strip()
        lines.append(f"\t/// {clean_desc}")
    if len(expr_lines) == 1:
        lines.append(f"\tmeasure {_tmdl_ident(measure.name)} = {expr_lines[0]}")
    else:
        lines.append(f"\tmeasure {_tmdl_ident(measure.name)} =")
        for expr_line in expr_lines:
            lines.append(f"\t\t\t{expr_line}")
    lines.append(f"\t\tformatString: {measure.format_string}")
    if measure.display_folder:
        lines.append(f"\t\tdisplayFolder: {measure.display_folder}")
    lines.append(f"\t\tlineageTag: {stable_guid(f'measure:{table_name}.{measure.name}')}")
    lines.append("")
    return lines


def _render_table_tmdl(
    table: SemanticTable,
    table_measures: Sequence[DAXMeasureSpec],
    *,
    source_format: DataSourceFormat,
) -> str:
    """Render the complete ``tables/<table>.tmdl`` file content."""
    lines: list[str] = []
    if table.description:
        lines.append(f"/// {table.description.replace(chr(10), ' ').strip()}")
    lines.append(f"table {_tmdl_ident(table.table_name)}")
    if table.is_date_table:
        lines.append("\tdataCategory: Time")
    lines.append(f"\tlineageTag: {stable_guid(f'table:{table.table_name}')}")
    lines.append("")

    for measure in table_measures:
        lines.extend(_render_measure_block(table.table_name, measure))

    for col in table.columns:
        lines.extend(_render_column_block(table, col))

    lines.extend(_render_m_partition(table, source_format=source_format))
    lines.append("")
    return "\n".join(lines)


def write_tmdl_semantic_model(
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    semantic_model_dir: Path,
    *,
    source_format: DataSourceFormat = "parquet",
    data_folder_param_default: str = "../data",
) -> list[Path]:
    """Write a complete TMDL semantic model folder under ``semantic_model_dir``."""
    definition_dir = semantic_model_dir / "definition"
    tables_dir = definition_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. definition.pbism
    pbism_path = semantic_model_dir / "definition.pbism"
    pbism_payload = {
        "$schema": SCHEMA_PBISM,
        "version": "4.2",
        "settings": {},
    }
    pbism_path.write_text(json.dumps(pbism_payload, indent=2) + "\n", encoding="utf-8")
    written.append(pbism_path)

    # 2. definition/database.tmdl
    db_path = definition_dir / "database.tmdl"
    db_path.write_text("database\n\tcompatibilityLevel: 1567\n", encoding="utf-8")
    written.append(db_path)

    # 3. definition/model.tmdl
    model_lines = [
        "model Model",
        "\tculture: en-US",
        "\tdefaultPowerBIDataSourceVersion: powerBI_V3",
        "\tsourceQueryCulture: en-US",
        "\tdiscourageImplicitMeasures",
        "",
    ]
    for tbl in model.tables:
        model_lines.append(f"ref table {_tmdl_ident(tbl.table_name)}")
    model_lines.append("")
    model_path = definition_dir / "model.tmdl"
    model_path.write_text("\n".join(model_lines), encoding="utf-8")
    written.append(model_path)

    # 4. definition/expressions.tmdl (portable DataFolderPath M parameter)
    escaped_default = data_folder_param_default.replace('"', '""')
    expr_lines = [
        (
            f'expression DataFolderPath = "{escaped_default}" '
            'meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]'
        ),
        f"\tlineageTag: {stable_guid('expr:DataFolderPath')}",
        "",
    ]
    expr_path = definition_dir / "expressions.tmdl"
    expr_path.write_text("\n".join(expr_lines), encoding="utf-8")
    written.append(expr_path)

    # 5. definition/relationships.tmdl (when relationships exist)
    if model.relationships:
        rel_lines: list[str] = []
        for rel in model.relationships:
            rel_id = stable_guid(f"rel:{rel.name}")
            rel_lines.append(f"relationship {rel_id}")
            rel_lines.append(
                f"\tfromColumn: {_tmdl_ident(rel.from_table)}.{_tmdl_ident(rel.from_column)}"
            )
            rel_lines.append(
                f"\ttoColumn: {_tmdl_ident(rel.to_table)}.{_tmdl_ident(rel.to_column)}"
            )
            rel_lines.append("\tcrossFilteringBehavior: oneDirection")
            if not rel.is_active:
                rel_lines.append("\tisActive: false")
            rel_lines.append("")
        rel_path = definition_dir / "relationships.tmdl"
        rel_path.write_text("\n".join(rel_lines), encoding="utf-8")
        written.append(rel_path)

    # 6. definition/tables/<table>.tmdl
    measures_by_table: dict[str, list[DAXMeasureSpec]] = {}
    for m in measures:
        measures_by_table.setdefault(m.table_name, []).append(m)

    for tbl in model.tables:
        tbl_measures = measures_by_table.get(tbl.table_name, [])
        tmdl_text = _render_table_tmdl(tbl, tbl_measures, source_format=source_format)
        tbl_path = tables_dir / f"{tbl.table_name}.tmdl"
        tbl_path.write_text(tmdl_text, encoding="utf-8")
        written.append(tbl_path)

    return written

