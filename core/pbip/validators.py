"""JSON-schema and cross-reference validator for generated PBIP / TMDL / PBIR projects."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator

from core.pbip.templates import (
    PAGE_JSON_SCHEMA,
    PAGES_JSON_SCHEMA,
    PBIP_FILE_JSON_SCHEMA,
    PBIR_JSON_SCHEMA,
    PBISM_JSON_SCHEMA,
    REPORT_JSON_SCHEMA,
    THEME_JSON_SCHEMA,
    VERSION_JSON_SCHEMA,
    VISUAL_JSON_SCHEMA,
)
from core.schemas.pbip import PBIPValidationIssue, PBIPValidationResult

_REF_TABLE_RE = re.compile(r"^ref table (?:'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))\s*$")
_COLUMN_DECL_RE = re.compile(r"^\tcolumn (?:'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))\s*$")
_MEASURE_DECL_RE = re.compile(r"^\tmeasure (?:'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))\s*=")


def _validate_json_file(
    file_path: Path,
    schema: dict[str, Any],
    issues: list[PBIPValidationIssue],
    rel_base: Path,
) -> dict[str, Any] | None:
    """Parse ``file_path`` as JSON and validate against ``schema`` via ``Draft7Validator``."""
    rel = file_path.relative_to(rel_base).as_posix() if file_path.exists() else file_path.name
    if not file_path.is_file():
        issues.append(PBIPValidationIssue(file_path=rel, message="Required file is missing."))
        return None
    try:
        data = json.loads(file_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        issues.append(PBIPValidationIssue(file_path=rel, message=f"Invalid JSON: {exc}"))
        return None

    validator = Draft7Validator(schema)
    for err in sorted(validator.iter_errors(data), key=lambda e: list(e.path)):
        loc = ".".join(str(p) for p in err.path) or "<root>"
        issues.append(
            PBIPValidationIssue(
                file_path=rel,
                message=f"Schema error at '{loc}': {err.message}",
            )
        )
    return data if isinstance(data, dict) else None


def _parse_tmdl_symbols(
    tables_dir: Path,
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Extract declared column and measure names per table from ``tables/*.tmdl``."""
    columns_by_table: dict[str, set[str]] = {}
    measures_by_table: dict[str, set[str]] = {}
    if not tables_dir.is_dir():
        return columns_by_table, measures_by_table

    for tmdl_file in sorted(tables_dir.glob("*.tmdl")):
        table_name = tmdl_file.stem
        cols: set[str] = set()
        measures: set[str] = set()
        for line in tmdl_file.read_text(encoding="utf-8").splitlines():
            col_match = _COLUMN_DECL_RE.match(line)
            if col_match:
                cols.add((col_match.group(1) or col_match.group(2)).replace("''", "'"))
            meas_match = _MEASURE_DECL_RE.match(line)
            if meas_match:
                measures.add((meas_match.group(1) or meas_match.group(2)).replace("''", "'"))
        columns_by_table[table_name] = cols
        measures_by_table[table_name] = measures
    return columns_by_table, measures_by_table


def _validate_visual_bindings(
    visual_data: dict[str, Any],
    rel_path: str,
    columns_by_table: dict[str, set[str]],
    measures_by_table: dict[str, set[str]],
    issues: list[PBIPValidationIssue],
) -> None:
    """Verify every column/measure referenced inside ``visual.json`` exists in TMDL."""
    query_state = visual_data.get("visual", {}).get("query", {}).get("queryState", {})
    if not isinstance(query_state, dict):
        return

    for role_name, role_obj in query_state.items():
        if not isinstance(role_obj, dict):
            continue
        for proj in role_obj.get("projections", []):
            field = proj.get("field", {}) if isinstance(proj, dict) else {}
            if "Column" in field:
                entity = field["Column"].get("Expression", {}).get("SourceRef", {}).get("Entity")
                prop = field["Column"].get("Property")
                if entity not in columns_by_table:
                    issues.append(
                        PBIPValidationIssue(
                            file_path=rel_path,
                            message=(
                                f"Visual role '{role_name}' references unknown table '{entity}'."
                            ),
                        )
                    )
                elif prop not in columns_by_table[entity]:
                    issues.append(
                        PBIPValidationIssue(
                            file_path=rel_path,
                            message=(
                                f"Visual role '{role_name}' references unknown column "
                                f"'{entity}.{prop}'."
                            ),
                        )
                    )
            elif "Measure" in field:
                entity = field["Measure"].get("Expression", {}).get("SourceRef", {}).get("Entity")
                prop = field["Measure"].get("Property")
                if entity not in measures_by_table or prop not in measures_by_table[entity]:
                    issues.append(
                        PBIPValidationIssue(
                            file_path=rel_path,
                            message=(
                                f"Visual role '{role_name}' references unknown measure "
                                f"'{entity}.{prop}'."
                            ),
                        )
                    )


def validate_pbip_project(pbip_root: Path, project_name: str) -> PBIPValidationResult:
    """Validate a generated PBIP project folder against official JSON schemas and TMDL symbols."""
    issues: list[PBIPValidationIssue] = []
    files_checked = 0

    pbip_file = pbip_root / f"{project_name}.pbip"
    report_dir = pbip_root / f"{project_name}.Report"
    model_dir = pbip_root / f"{project_name}.SemanticModel"

    # 1. Validate .pbip entry file
    files_checked += 1
    pbip_data = _validate_json_file(pbip_file, PBIP_FILE_JSON_SCHEMA, issues, pbip_root)
    if pbip_data is not None:
        report_rel = pbip_data["artifacts"][0]["report"]["path"]
        if not (pbip_root / report_rel).is_dir():
            issues.append(
                PBIPValidationIssue(
                    file_path=pbip_file.name,
                    message=f"Artifact report path '{report_rel}' does not resolve to a directory.",
                )
            )

    # 2. Validate SemanticModel (definition.pbism + TMDL files)
    files_checked += 1
    _validate_json_file(model_dir / "definition.pbism", PBISM_JSON_SCHEMA, issues, pbip_root)

    tmdl_def = model_dir / "definition"
    for required_tmdl in ("database.tmdl", "model.tmdl", "expressions.tmdl"):
        req_path = tmdl_def / required_tmdl
        files_checked += 1
        if not req_path.is_file():
            issues.append(
                PBIPValidationIssue(
                    file_path=f"{project_name}.SemanticModel/definition/{required_tmdl}",
                    message="Required TMDL file is missing.",
                )
            )

    expr_path = tmdl_def / "expressions.tmdl"
    if expr_path.is_file() and "DataFolderPath" not in expr_path.read_text(encoding="utf-8"):
        issues.append(
            PBIPValidationIssue(
                file_path=expr_path.relative_to(pbip_root).as_posix(),
                message="expressions.tmdl is missing the portable DataFolderPath M parameter.",
            )
        )

    tables_dir = tmdl_def / "tables"
    columns_by_table, measures_by_table = _parse_tmdl_symbols(tables_dir)

    model_tmdl_path = tmdl_def / "model.tmdl"
    if model_tmdl_path.is_file():
        for line in model_tmdl_path.read_text(encoding="utf-8").splitlines():
            ref_match = _REF_TABLE_RE.match(line.strip())
            if ref_match:
                ref_tbl = (ref_match.group(1) or ref_match.group(2)).replace("''", "'")
                if ref_tbl not in columns_by_table:
                    issues.append(
                        PBIPValidationIssue(
                            file_path=model_tmdl_path.relative_to(pbip_root).as_posix(),
                            message=f"model.tmdl references missing table file '{ref_tbl}.tmdl'.",
                        )
                    )
        files_checked += len(columns_by_table)

    # 3. Validate Report (definition.pbir, version, report, theme, pages, visuals)
    files_checked += 1
    pbir_data = _validate_json_file(
        report_dir / "definition.pbir", PBIR_JSON_SCHEMA, issues, pbip_root
    )
    if pbir_data is not None:
        sm_rel = pbir_data["datasetReference"]["byPath"]["path"]
        resolved_sm = (report_dir / sm_rel).resolve()
        if not resolved_sm.is_dir():
            issues.append(
                PBIPValidationIssue(
                    file_path=f"{project_name}.Report/definition.pbir",
                    message=f"datasetReference byPath '{sm_rel}' does not exist.",
                )
            )

    rep_def = report_dir / "definition"
    files_checked += 1
    _validate_json_file(rep_def / "version.json", VERSION_JSON_SCHEMA, issues, pbip_root)

    files_checked += 1
    _validate_json_file(rep_def / "report.json", REPORT_JSON_SCHEMA, issues, pbip_root)

    theme_path = report_dir / "StaticResources" / "RegisteredResources" / "autobi_theme.json"
    files_checked += 1
    _validate_json_file(theme_path, THEME_JSON_SCHEMA, issues, pbip_root)

    pages_root = rep_def / "pages"
    files_checked += 1
    pages_data = _validate_json_file(
        pages_root / "pages.json", PAGES_JSON_SCHEMA, issues, pbip_root
    )

    if pages_data is not None:
        for page_id in pages_data.get("pageOrder", []):
            page_dir = pages_root / page_id
            files_checked += 1
            _validate_json_file(page_dir / "page.json", PAGE_JSON_SCHEMA, issues, pbip_root)

            visuals_dir = page_dir / "visuals"
            if visuals_dir.is_dir():
                for vis_file in sorted(visuals_dir.glob("*/visual.json")):
                    files_checked += 1
                    vis_data = _validate_json_file(vis_file, VISUAL_JSON_SCHEMA, issues, pbip_root)
                    if vis_data is not None:
                        _validate_visual_bindings(
                            vis_data,
                            vis_file.relative_to(pbip_root).as_posix(),
                            columns_by_table,
                            measures_by_table,
                            issues,
                        )

    has_errors = any(issue.severity == "error" for issue in issues)
    return PBIPValidationResult(
        is_valid=not has_errors,
        files_checked=files_checked,
        issues=issues,
    )

