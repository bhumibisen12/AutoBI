"""Companion bundle exports: measures.dax, data dictionary, verification, README, and fallback."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from core.schemas.dax import DAXMeasureSpec, VerificationReport
from core.schemas.pbip import PBIPBuildConfig
from core.schemas.report_spec import ReportSpec
from core.schemas.semantic import SemanticModel
from core.security import neutralize_dataframe_for_export


def export_measures_dax(measures: Sequence[DAXMeasureSpec], output_path: Path) -> Path:
    """Write all generated DAX measures to a standalone ``measures.dax`` script."""
    lines = [
        "// AutoBI Generated DAX Measures",
        "// All formulas were instantiated from allowlisted templates (never LLM free-form DAX).",
        "",
    ]
    for m in measures:
        lines.append(
            f"// Table: {m.table_name} | Format: {m.format_string} | Template: {m.template_id}"
        )
        if m.description:
            lines.append(f"// Description: {m.description}")
        lines.append(f"{m.name} =")
        for expr_line in m.expression.strip().splitlines():
            lines.append(f"    {expr_line}")
        lines.append("")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def export_data_dictionary_excel(
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    output_path: Path,
) -> Path:
    """Export a formula-injection-neutralised multi-sheet Excel data dictionary."""
    tables_df = pd.DataFrame(
        [
            {
                "table_name": t.table_name,
                "kind": t.kind,
                "is_date_table": t.is_date_table,
                "primary_key": t.primary_key or "",
                "row_count": t.row_count,
                "column_count": len(t.columns),
                "description": t.description,
            }
            for t in model.tables
        ]
    )

    columns_df = pd.DataFrame(
        [
            {
                "table_name": t.table_name,
                "column_name": c.name,
                "friendly_name": c.friendly_name,
                "role": c.role,
                "data_type": c.data_type,
                "format_string": c.format_string or "",
                "summarize_by": c.summarize_by,
                "is_key": c.is_key,
                "description": c.description,
            }
            for t in model.tables
            for c in t.columns
        ]
    )

    measures_df = pd.DataFrame(
        [
            {
                "measure_name": m.name,
                "table_name": m.table_name,
                "template_id": m.template_id,
                "format_string": m.format_string,
                "display_folder": m.display_folder,
                "description": m.description,
                "dax_expression": m.expression,
            }
            for m in measures
        ]
    )

    rels_df = pd.DataFrame(
        [
            {
                "relationship_name": r.name,
                "from_table": r.from_table,
                "from_column": r.from_column,
                "to_table": r.to_table,
                "to_column": r.to_column,
                "cardinality": r.cardinality,
                "cross_filter_direction": r.cross_filter_direction,
            }
            for r in model.relationships
        ]
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        neutralize_dataframe_for_export(tables_df).to_excel(
            writer, sheet_name="Tables", index=False
        )
        neutralize_dataframe_for_export(columns_df).to_excel(
            writer, sheet_name="Columns", index=False
        )
        neutralize_dataframe_for_export(measures_df).to_excel(
            writer, sheet_name="Measures", index=False
        )
        neutralize_dataframe_for_export(rels_df).to_excel(
            writer, sheet_name="Relationships", index=False
        )
    return output_path


def export_verification_reports(
    report: VerificationReport,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write ``verification_report.json`` and human-readable ``verification_report.md``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "verification_report.json"
    json_path.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")

    unverified = [r for r in report.results if r.status == "UNVERIFIED"]
    failed = [r for r in report.results if r.status == "FAILED"]

    md_lines = [
        "# AutoBI DAX Verification Report",
        "",
        f"- **Batch ID**: `{report.batch_id or 'N/A'}`",
        f"- **Total Measures**: {report.total_measures}",
        f"- **VERIFIED**: {report.verified_count}",
        f"- **UNVERIFIED**: {report.unverified_count}",
        f"- **FAILED**: {report.failed_count}",
        "",
    ]

    if unverified:
        md_lines.append("## UNVERIFIED Measures (Manual Review Required)")
        md_lines.append("")
        for item in unverified:
            md_lines.append(f"- **{item.measure_name}** (`{item.template_id}`): {item.notes}")
        md_lines.append("")
    else:
        md_lines.append("## UNVERIFIED Measures\n\nNone — all measures were reconciled.\n")

    if failed:
        md_lines.append("## FAILED Measures")
        md_lines.append("")
        for item in failed:
            errs = "; ".join(item.validation_errors)
            md_lines.append(f"- **{item.measure_name}**: {errs}")
        md_lines.append("")

    md_lines.append("## Detailed Reconciliation")
    md_lines.append("")
    md_lines.append("| Measure | Status | Reconciled Result | Notes |")
    md_lines.append("| --- | --- | --- | --- |")
    for item in report.results:
        val_str = f"{item.result:,.4f}" if item.result is not None else "NULL / N/A"
        md_lines.append(f"| {item.measure_name} | {item.status} | {val_str} | {item.notes} |")

    md_path = output_dir / "verification_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return json_path, md_path


def _format_bytes(size: int) -> str:
    """Format byte counts into readable KB/MB strings."""
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def export_bundle_readme(
    bundle_root: Path,
    *,
    config: PBIPBuildConfig,
    data_size_bytes: int,
    large_data_warning: str | None,
    unverified_measures: Sequence[str],
) -> Path:
    """Write a portable ``README.md`` with file sizes, large-data warning, and open checklist."""
    file_entries: list[tuple[str, int]] = []
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file() and path.name != "README.md":
            rel = path.relative_to(bundle_root).as_posix()
            file_entries.append((rel, path.stat().st_size))

    lines = [
        f"# {config.project_name} — AutoBI Power BI Project Bundle",
        "",
        "> **IMPORTANT:** This PBIP project was validated against offline JSON schemas and TMDL",
        "> symbol checks, **not** opened inside a live Power BI Desktop instance. Follow the",
        "> manual verification checklist below on your machine.",
        "",
    ]

    if large_data_warning:
        lines.extend([f"> **LARGE DATA WARNING:** {large_data_warning}", ""])

    lines.extend(
        [
            "## Manual Checklist to Open in Power BI Desktop",
            "",
            "1. Unzip this entire archive to a local folder "
            "(for example, `C:\\Reports\\AutoBI_Project`).",
            "2. Ensure you have a recent **Power BI Desktop** with **PBIR** and **TMDL** "
            "project preview features enabled (*File -> Options -> Preview features*).",
            f"3. Double-click `pbip/{config.project_name}.pbip` to open the project.",
            "4. Go to **Home -> Transform data -> Edit parameters**, update `DataFolderPath` from "
            f"`{config.data_folder_param_default}` to your unzipped absolute `data` folder path "
            "(for example, `C:/Reports/AutoBI_Project/data`), and click **Apply changes**.",
            "5. Click **Refresh** and verify tables, `dim_date` date-table marking, and 1:* "
            "relationships in **Model view**.",
            "6. Inspect every visual in **Report view** and compare KPI numbers against "
            "`verification_report.md`.",
            "7. If Power BI Desktop reports a schema mismatch on any visual, use the fallback "
            "artifacts (`measures.dax` and `data_dictionary.xlsx`) included in this bundle.",
            "",
            "## Unverified Measures",
            "",
            (
                "\n".join(f"- `{m}`" for m in unverified_measures)
                if unverified_measures
                else "None — all generated DAX measures reconciled against DuckDB SQL."
            ),
            "",
            "## Included Files and Sizes",
            "",
            f"- **Curated Data Total Size**: {_format_bytes(data_size_bytes)}",
            "",
            "| File | Size |",
            "| --- | --- |",
        ]
    )

    for rel, sz in file_entries:
        lines.append(f"| `{rel}` | {_format_bytes(sz)} |")

    readme_path = bundle_root / "README.md"
    readme_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return readme_path


def export_fallback_build_guide(
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    report_spec: ReportSpec,
    output_path: Path,
    *,
    fallback_reason: str,
) -> Path:
    """Write a clear step-by-step manual Power BI build guide when PBIP export falls back."""
    lines = [
        f"# Fallback Manual Power BI Build Guide — {report_spec.report_title}",
        "",
        f"**Why fallback export was generated:** {fallback_reason}",
        "",
        "## Step 1 — Load Curated Data Tables",
        "",
        "Open Power BI Desktop -> **Get Data** -> load each file from the bundled `data/` folder:",
        "",
    ]
    for tbl in model.tables:
        lines.append(
            f"- `{tbl.table_name}.parquet` (or `{tbl.table_name}.csv`) — {tbl.row_count} rows"
        )

    lines.extend(["", "## Step 2 — Configure Date Table & Relationships", ""])
    if model.date_table_name:
        lines.append(
            f"- Mark `{model.date_table_name}` as a Date Table using column `date` "
            "(**Table tools -> Mark as date table**)."
        )
    for rel in model.relationships:
        lines.append(
            f"- Relationship: `{rel.from_table}[{rel.from_column}]` -> "
            f"`{rel.to_table}[{rel.to_column}]` ({rel.cardinality}, single direction)."
        )

    lines.extend(
        [
            "",
            "## Step 3 — Paste DAX Measures",
            "",
            f"Create the {len(measures)} measures listed in `measures.dax` "
            "(see `data_dictionary.xlsx` for format strings and descriptions).",
            "",
            "## Step 4 — Build Report Pages",
            "",
        ]
    )
    for page in report_spec.pages:
        lines.append(f"### Page: {page.title}")
        if page.slicers:
            slicers_str = ", ".join(f"{s.table_name}.{s.column_name}" for s in page.slicers)
            lines.append(f"- Slicers: {slicers_str}")
        for vis in page.visuals:
            meas_str = ", ".join(vis.measures) if vis.measures else "none"
            cat_str = (
                f"{vis.category_field.table_name}.{vis.category_field.column_name}"
                if vis.category_field
                else "N/A"
            )
            lines.append(
                f"- **{vis.title}** (`{vis.visual_type}`): "
                f"category={cat_str}, measures=[{meas_str}]"
            )
        lines.append("")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
