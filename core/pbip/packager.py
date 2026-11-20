"""PBIP project orchestrator and portable ZIP packager with automatic fallback export."""

from __future__ import annotations

import json
import re
import shutil
import zipfile
from collections.abc import Sequence
from pathlib import Path

from core.logging import get_logger
from core.pbip.artifacts import (
    export_bundle_readme,
    export_data_dictionary_excel,
    export_fallback_build_guide,
    export_measures_dax,
    export_verification_reports,
)
from core.pbip.pbir import write_pbir_report
from core.pbip.templates import SCHEMA_PBIP
from core.pbip.tmdl import write_tmdl_semantic_model
from core.pbip.validators import validate_pbip_project
from core.schemas.dax import DAXMeasureSpec, VerificationReport
from core.schemas.pbip import PBIPBuildConfig, PBIPExportResult, PBIPValidationResult
from core.schemas.report_spec import ReportSpec
from core.schemas.semantic import SemanticModel
from core.storage import read_curated_table

_SAFE_PROJECT_RE = re.compile(r"[^A-Za-z0-9_-]+")
logger = get_logger("pbip")


def _safe_project_name(raw_name: str) -> str:
    """Sanitize a project name for filesystem and PBIP artifact paths."""
    cleaned = _SAFE_PROJECT_RE.sub("_", raw_name.strip()).strip("_")
    return cleaned or "AutoBI_Project"


def _copy_curated_data_files(
    model: SemanticModel,
    target_data_dir: Path,
) -> int:
    """Copy curated Parquet files (plus CSV copies for portability) under bundle ``data/``."""
    target_data_dir.mkdir(parents=True, exist_ok=True)
    total_bytes = 0

    if model.curated_dir is None or not model.curated_dir.is_dir():
        return 0

    for tbl in model.tables:
        src_parquet = model.curated_dir / f"{tbl.table_name}.parquet"
        if not src_parquet.is_file():
            continue
        dst_parquet = target_data_dir / f"{tbl.table_name}.parquet"
        shutil.copy2(src_parquet, dst_parquet)
        total_bytes += dst_parquet.stat().st_size

        # Also export UTF-8 CSV copy so users on older Desktop builds can switch easily.
        df = read_curated_table(model.curated_dir, tbl.table_name)
        dst_csv = target_data_dir / f"{tbl.table_name}.csv"
        df.to_csv(dst_csv, index=False, encoding="utf-8")
        total_bytes += dst_csv.stat().st_size

    return total_bytes


def _zip_directory(source_dir: Path, zip_path: Path, root_arc_prefix: str) -> tuple[int, list[str]]:
    """Package every file under ``source_dir`` into ``zip_path`` cleanly."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    archived_files: list[str] = []
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(source_dir.rglob("*")):
            if not file_path.is_file():
                continue
            rel = file_path.relative_to(source_dir).as_posix()
            arcname = f"{root_arc_prefix}/{rel}"
            zf.write(file_path, arcname=arcname)
            archived_files.append(rel)
    return zip_path.stat().st_size, archived_files


def _write_pbip_entry_file(pbip_root: Path, project_name: str) -> Path:
    """Write ``<ProjectName>.pbip`` pointing to ``<ProjectName>.Report``."""
    pbip_path = pbip_root / f"{project_name}.pbip"
    payload = {
        "$schema": SCHEMA_PBIP,
        "version": "1.0",
        "artifacts": [{"report": {"path": f"{project_name}.Report"}}],
        "settings": {"enableAutoRecovery": True},
    }
    pbip_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return pbip_path


def build_pbip_package(
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
    report_spec: ReportSpec,
    verification_report: VerificationReport,
    output_dir: Path | str,
    *,
    config: PBIPBuildConfig | None = None,
    force_fallback: bool = False,
) -> PBIPExportResult:
    """Generate, schema-validate, and zip a complete portable PBIP bundle (with fallback export).

    The output ZIP always contains:
    - ``pbip/`` (TMDL semantic model + PBIR report) when generation & schema validation succeed
    - ``data/`` (curated Parquet + CSV files for portability across machines)
    - ``measures.dax``
    - ``data_dictionary.xlsx`` (formula-injection neutralised)
    - ``verification_report.json`` and ``verification_report.md`` (listing UNVERIFIED measures)
    - ``README.md`` (file sizes, large-data warning, and manual open checklist)

    If PBIP generation or schema validation fails (or ``force_fallback=True``), falls back
    cleanly to a ZIP containing ``measures.dax``, ``data_dictionary.xlsx``, verification reports,
    curated ``data/``, and ``STEP_BY_STEP_BUILD_GUIDE.md``.
    """
    cfg = config or PBIPBuildConfig()
    project_name = _safe_project_name(cfg.project_name)
    out_root = Path(output_dir)
    bundle_dir = out_root / project_name
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    unverified = [r.measure_name for r in verification_report.results if r.status == "UNVERIFIED"]

    # Always prepare portable curated data + companion artifacts
    data_size_bytes = _copy_curated_data_files(model, bundle_dir / "data")
    large_data_warning: str | None = None
    threshold_bytes = int(cfg.large_data_warning_mb * 1024 * 1024)
    if data_size_bytes > threshold_bytes:
        mb = data_size_bytes / (1024 * 1024)
        large_data_warning = (
            f"Bundled curated data is {mb:.2f} MB (exceeding {cfg.large_data_warning_mb:.1f} MB). "
            "Consider storing data in a shared data lake or database instead of a local ZIP."
        )

    export_measures_dax(measures, bundle_dir / "measures.dax")
    export_data_dictionary_excel(model, measures, bundle_dir / "data_dictionary.xlsx")
    export_verification_reports(verification_report, bundle_dir)

    pbip_root = bundle_dir / "pbip"
    validation_result: PBIPValidationResult | None = None
    fallback_reason: str | None = None
    pbip_file_path: Path | None = None

    if force_fallback:
        fallback_reason = "Fallback export explicitly requested."
    else:
        try:
            pbip_root.mkdir(parents=True, exist_ok=True)
            pbip_file_path = _write_pbip_entry_file(pbip_root, project_name)
            sm_dir = pbip_root / f"{project_name}.SemanticModel"
            rep_dir = pbip_root / f"{project_name}.Report"

            write_tmdl_semantic_model(
                model,
                measures,
                sm_dir,
                source_format=cfg.source_format,
                data_folder_param_default=cfg.data_folder_param_default,
            )
            write_pbir_report(
                report_spec,
                measures,
                rep_dir,
                semantic_model_rel_path=f"../{project_name}.SemanticModel",
            )

            validation_result = validate_pbip_project(pbip_root, project_name)
            if not validation_result.is_valid:
                err_summary = "; ".join(
                    f"{iss.file_path}: {iss.message}" for iss in validation_result.issues[:3]
                )
                fallback_reason = f"PBIP schema validation failed: {err_summary}"
                shutil.rmtree(pbip_root, ignore_errors=True)
                pbip_file_path = None
        except Exception as exc:
            fallback_reason = f"PBIP generation error: {exc}"
            shutil.rmtree(pbip_root, ignore_errors=True)
            pbip_file_path = None

    if fallback_reason is not None:
        export_fallback_build_guide(
            model,
            measures,
            report_spec,
            bundle_dir / "STEP_BY_STEP_BUILD_GUIDE.md",
            fallback_reason=fallback_reason,
        )
        export_bundle_readme(
            bundle_dir,
            config=cfg.model_copy(update={"project_name": project_name}),
            data_size_bytes=data_size_bytes,
            large_data_warning=large_data_warning,
            unverified_measures=unverified,
        )
        zip_path = out_root / f"{project_name}_FALLBACK.zip"
        total_size, files_list = _zip_directory(bundle_dir, zip_path, project_name)
        logger.warning("Generated fallback bundle for '%s': %s", project_name, fallback_reason)
        return PBIPExportResult(
            project_name=project_name,
            mode="fallback",
            zip_path=zip_path,
            output_dir=bundle_dir,
            pbip_file_path=None,
            files_generated=files_list,
            total_size_bytes=total_size,
            data_size_bytes=data_size_bytes,
            large_data_warning=large_data_warning,
            unverified_measures=unverified,
            validation_result=validation_result,
            fallback_reason=fallback_reason,
        )

    export_bundle_readme(
        bundle_dir,
        config=cfg.model_copy(update={"project_name": project_name}),
        data_size_bytes=data_size_bytes,
        large_data_warning=large_data_warning,
        unverified_measures=unverified,
    )
    zip_path = out_root / f"{project_name}_PBIP.zip"
    total_size, files_list = _zip_directory(bundle_dir, zip_path, project_name)
    logger.info(
        "Generated validated PBIP bundle '%s' (%d files, %d bytes)",
        project_name,
        len(files_list),
        total_size,
    )
    return PBIPExportResult(
        project_name=project_name,
        mode="pbip",
        zip_path=zip_path,
        output_dir=bundle_dir,
        pbip_file_path=pbip_file_path,
        files_generated=files_list,
        total_size_bytes=total_size,
        data_size_bytes=data_size_bytes,
        large_data_warning=large_data_warning,
        unverified_measures=unverified,
        validation_result=validation_result,
        fallback_reason=None,
    )

