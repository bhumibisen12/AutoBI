"""Data-quality gate evaluation for the ETL pipeline (no DAX or report-spec logic)."""

from __future__ import annotations

from collections.abc import Sequence

from core.schemas.etl import (
    QualityGateCheck,
    QualityGateConfig,
    TableReconciliation,
    ValidationReport,
)
from core.schemas.profiling import ProfileReport


class QualityGateFailedError(Exception):
    """Raised when one or more data-quality gates fail and halt the ETL pipeline."""

    def __init__(self, report: ValidationReport) -> None:
        """Store the complete ``ValidationReport`` alongside a concise failure summary."""
        self.report = report
        failed_summary = "; ".join(chk.message for chk in report.failed_checks[:5])
        super().__init__(
            f"Quality gates failed for batch '{report.batch_id}' "
            f"({len(report.failed_checks)} failed check(s)): {failed_summary}"
        )


def evaluate_quality_gates(
    *,
    batch_id: str,
    curated_profile: ProfileReport,
    reconciliations: Sequence[TableReconciliation],
    config: QualityGateConfig | None = None,
) -> ValidationReport:
    """Evaluate configurable quality gates over curated table profiles and reconciliations.

    Gates checked:
    - Row reconciliation (``rows_read == rows_staged == curated + rejected + duplicates``)
    - Non-empty curated source tables
    - Duplicate row percentage (``max_duplicate_pct``)
    - Invalid-type / quarantined row percentage (``max_invalid_type_pct``)
    - Column null percentage (``max_null_pct``)
    - Remaining mixed-type column percentage (``max_invalid_type_pct``)
    - Numeric outlier percentage via IQR / robust z-score (``max_outlier_pct``)
    """
    cfg = config or QualityGateConfig()
    checks: list[QualityGateCheck] = []

    for recon in reconciliations:
        table = recon.table_name

        # 1. Exact row reconciliation gate
        if cfg.require_reconciliation:
            accounted = recon.rows_curated + recon.rows_rejected + recon.rows_dropped_duplicates
            passed = recon.is_reconciled
            checks.append(
                QualityGateCheck(
                    gate_name="row_reconciliation",
                    table_name=table,
                    passed=passed,
                    observed_value=float(accounted),
                    threshold_value=float(recon.rows_read),
                    message=(
                        f"Reconciled {table}: read={recon.rows_read}, staged={recon.rows_staged}, "
                        f"curated={recon.rows_curated}, rejected={recon.rows_rejected}, "
                        f"duplicates={recon.rows_dropped_duplicates}."
                        if passed
                        else f"Reconciliation mismatch on {table}: read={recon.rows_read}, "
                        f"staged={recon.rows_staged}, accounted={accounted}."
                    ),
                )
            )

        # 2. Non-empty curated table gate
        if cfg.require_non_empty:
            passed = recon.rows_curated > 0
            checks.append(
                QualityGateCheck(
                    gate_name="non_empty_curated",
                    table_name=table,
                    passed=passed,
                    observed_value=float(recon.rows_curated),
                    threshold_value=1.0,
                    message=(
                        f"Curated table '{table}' retains {recon.rows_curated} row(s)."
                        if passed
                        else f"Curated table '{table}' is empty after quarantine/deduplication."
                    ),
                )
            )

        # 3. Duplicate percentage gate
        dup_pct = (
            round((recon.rows_dropped_duplicates / recon.rows_staged) * 100.0, 2)
            if recon.rows_staged > 0
            else 0.0
        )
        dup_passed = dup_pct <= cfg.max_duplicate_pct
        checks.append(
            QualityGateCheck(
                gate_name="max_duplicate_pct",
                table_name=table,
                passed=dup_passed,
                observed_value=dup_pct,
                threshold_value=cfg.max_duplicate_pct,
                message=(
                    f"Duplicate rate for '{table}' is {dup_pct:.2f}% "
                    f"(threshold <= {cfg.max_duplicate_pct:.2f}%)."
                ),
            )
        )

        # 4. Rejected / invalid-type percentage gate
        rej_pct = (
            round((recon.rows_rejected / recon.rows_staged) * 100.0, 2)
            if recon.rows_staged > 0
            else 0.0
        )
        rej_passed = rej_pct <= cfg.max_invalid_type_pct
        checks.append(
            QualityGateCheck(
                gate_name="max_invalid_type_pct",
                table_name=table,
                passed=rej_passed,
                observed_value=rej_pct,
                threshold_value=cfg.max_invalid_type_pct,
                message=(
                    f"Invalid-type quarantine rate for '{table}' is {rej_pct:.2f}% "
                    f"(threshold <= {cfg.max_invalid_type_pct:.2f}%)."
                ),
            )
        )

    # 5. Per-column curated profile gates (nulls, remaining mixed types, outliers)
    for table_prof in curated_profile.tables:
        if table_prof.table_name == "dim_date":
            continue
        for col_prof in table_prof.columns:
            null_passed = col_prof.null_percentage <= cfg.max_null_pct
            checks.append(
                QualityGateCheck(
                    gate_name="max_null_pct",
                    table_name=table_prof.table_name,
                    column_name=col_prof.column_name,
                    passed=null_passed,
                    observed_value=col_prof.null_percentage,
                    threshold_value=cfg.max_null_pct,
                    message=(
                        f"Null rate for '{table_prof.table_name}.{col_prof.column_name}' is "
                        f"{col_prof.null_percentage:.2f}% (threshold <= {cfg.max_null_pct:.2f}%)."
                    ),
                )
            )

            for issue in col_prof.issues:
                if issue.issue_type == "mixed_types":
                    mixed_passed = issue.affected_percentage <= cfg.max_invalid_type_pct
                    checks.append(
                        QualityGateCheck(
                            gate_name="mixed_types_pct",
                            table_name=table_prof.table_name,
                            column_name=col_prof.column_name,
                            passed=mixed_passed,
                            observed_value=issue.affected_percentage,
                            threshold_value=cfg.max_invalid_type_pct,
                            message=(
                                "Mixed-type rate for "
                                f"'{table_prof.table_name}.{col_prof.column_name}' "
                                f"is {issue.affected_percentage:.2f}% "
                                f"(threshold <= {cfg.max_invalid_type_pct:.2f}%)."
                            ),
                        )
                    )
                elif issue.issue_type in {"outliers_iqr", "outliers_robust_z"}:
                    outlier_passed = issue.affected_percentage <= cfg.max_outlier_pct
                    checks.append(
                        QualityGateCheck(
                            gate_name=issue.issue_type,
                            table_name=table_prof.table_name,
                            column_name=col_prof.column_name,
                            passed=outlier_passed,
                            observed_value=issue.affected_percentage,
                            threshold_value=cfg.max_outlier_pct,
                            message=(
                                f"Outlier rate ({issue.issue_type}) for "
                                f"'{table_prof.table_name}.{col_prof.column_name}' is "
                                f"{issue.affected_percentage:.2f}% "
                                f"(threshold <= {cfg.max_outlier_pct:.2f}%)."
                            ),
                        )
                    )

    all_passed = all(chk.passed for chk in checks)
    return ValidationReport(batch_id=batch_id, passed=all_passed, checks=checks)

