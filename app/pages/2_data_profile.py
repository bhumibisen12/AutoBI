"""Page 2: ETL batch log, quality gates, profile summary, editable column confirmation, lineage."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from app.components import (  # noqa: E402
    LOGO_PATH,
    init_session,
    load_session_state,
    render_pii_warnings,
    render_sidebar_shell,
    render_top_workspace_bar,
)
from core.schemas.orchestrator import ColumnConfirmationEdit  # noqa: E402

COLUMN_ROLES = ["dimension", "measure", "date", "id", "text"]


def main() -> None:
    """Render Page 2 — ETL Batch Log, Quality Gates, Profile, Column Roles Editor, and Lineage."""
    st.set_page_config(
        page_title="2. Data Profile | AutoBI Studio",
        page_icon=str(LOGO_PATH),
        layout="wide",
    )
    session_id, settings, orchestrator, worker = init_session()
    state = load_session_state(orchestrator, session_id)
    render_sidebar_shell(session_id, settings, orchestrator, state, worker)

    render_top_workspace_bar(
        "Data Profile & Column Confirmation",
        "Inspect ETL reconciliation, quality gates, semantic types, lineage, and confirm columns.",
        state,
    )

    if state.error_message:
        st.error(f"Pipeline stopped at `{state.failed_node}`: {state.error_message}")

    if state.etl_result is None or state.profile_report is None:
        st.info("Upload data and run ETL on **1. Upload** first.")
        st.page_link("pages/1_upload.py", label="Go to 1. Upload", icon="📤")
        return

    etl = state.etl_result
    batch = etl.batch_log

    st.subheader("ETL Batch Log & Row Reconciliation")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Batch Status", batch.status.upper())
    c2.metric("Rows Read", f"{batch.rows_read:,}")
    c3.metric("Rows Staged", f"{batch.rows_staged:,}")
    c4.metric("Rows Curated", f"{batch.rows_curated:,}")
    c5.metric("Rows Rejected", f"{batch.rows_rejected:,}")

    rec_rows = [
        {
            "Table": r.table_name,
            "Read": r.rows_read,
            "Staged": r.rows_staged,
            "Curated": r.rows_curated,
            "Rejected": r.rows_rejected,
            "Dropped Duplicates": r.rows_dropped_duplicates,
            "Balanced": "PASS" if r.is_reconciled else "MISMATCH",
        }
        for r in etl.reconciliations
    ]
    st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True)

    st.subheader("Quality Gate Results")
    gate_rows = [
        {
            "Table": chk.table_name,
            "Column": chk.column_name or "(table)",
            "Gate": chk.gate_name,
            "Observed": round(chk.observed_value, 4),
            "Threshold": round(chk.threshold_value, 4),
            "Passed": "PASS" if chk.passed else "FAIL",
            "Message": chk.message,
        }
        for chk in etl.validation_report.checks
    ]
    if etl.validation_report.passed:
        st.success("All configured ETL quality gates passed.")
    else:
        st.error("One or more quality gates failed.")
    st.dataframe(pd.DataFrame(gate_rows), use_container_width=True, hide_index=True)

    if state.pii_reports:
        with st.expander("PII Scan Summary", expanded=False):
            render_pii_warnings(state.pii_reports)

    st.divider()
    st.subheader("Dataset Profile Summary")
    for tbl in state.profile_report.tables:
        with st.expander(
            f"Table `{tbl.table_name}` — {tbl.row_count:,} rows, {tbl.column_count} columns "
            f"(duplicates: {tbl.duplicate_row_count})",
            expanded=True,
        ):
            col_rows = [
                {
                    "Column": col.column_name,
                    "Semantic Type": col.semantic_type,
                    "Null %": f"{col.null_percentage:.1f}%",
                    "Distinct": col.distinct_count,
                    "Min": (
                        f"{col.numeric_stats.min:,.2f}"
                        if col.numeric_stats is not None
                        else (col.date_range.min_date if col.date_range else "")
                    ),
                    "Max": (
                        f"{col.numeric_stats.max:,.2f}"
                        if col.numeric_stats is not None
                        else (col.date_range.max_date if col.date_range else "")
                    ),
                    "Issues": ", ".join(i.issue_type for i in col.issues) or "none",
                }
                for col in tbl.columns
            ]
            st.dataframe(pd.DataFrame(col_rows), use_container_width=True, hide_index=True)

    if state.profile_report.candidate_relationships:
        st.markdown("**Detected Candidate Relationships**")
        rel_rows = [
            {
                "From": f"{r.left_table}.{r.left_column}",
                "To": f"{r.right_table}.{r.right_column}",
                "Overlap": f"{r.value_overlap:.0%}",
                "Confidence": f"{r.confidence:.0%}",
                "Cardinality": r.cardinality_hint,
            }
            for r in state.profile_report.candidate_relationships
        ]
        st.dataframe(pd.DataFrame(rel_rows), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Confirm Column Roles & Friendly Names (Human Gate)")
    st.caption(
        "Review the inferred column roles and business names below. Edits here are carried "
        "into the Power BI Semantic Model and Data Dictionary."
    )

    understanding = state.dataset_understanding
    if understanding is not None and understanding.columns:
        initial_rows = [
            {
                "table_name": c.table_name,
                "column_name": c.column_name,
                "role": c.role,
                "friendly_name": c.friendly_name,
                "description": c.description,
            }
            for c in understanding.columns
        ]
        edited_df = st.data_editor(
            pd.DataFrame(initial_rows),
            use_container_width=True,
            hide_index=True,
            disabled=["table_name", "column_name"],
            column_config={
                "role": st.column_config.SelectboxColumn("Role", options=COLUMN_ROLES),
                "friendly_name": st.column_config.TextColumn("Friendly Name"),
                "description": st.column_config.TextColumn("Description"),
            },
            key="profile_column_editor",
        )

        col_save, col_next = st.columns([2, 1])
        if col_save.button("Save Confirmed Column Roles", type="primary", use_container_width=True):
            edits = [
                ColumnConfirmationEdit(
                    table_name=str(row["table_name"]),
                    column_name=str(row["column_name"]),
                    role=str(row["role"]),  # type: ignore[arg-type]
                    friendly_name=str(row["friendly_name"]).strip() or str(row["column_name"]),
                    description=str(row["description"] or ""),
                )
                for _, row in edited_df.iterrows()
            ]
            st.session_state["confirmed_column_edits"] = [e.model_dump() for e in edits]
            st.success(f"Saved {len(edits)} confirmed column definition(s) for Step 3.")

        col_next.page_link(
            "pages/3_prompt_builder.py",
            label="Continue to 3. KPI & Design Builder ->",
            icon="🎯",
        )

    st.divider()
    with st.expander("Transformation & Column-Level Lineage Log", expanded=False):
        if etl.lineage_graph.events:
            lin_rows = [
                {
                    "Step": ev.step_index,
                    "Operation": ev.operation,
                    "Table": ev.table_name,
                    "Sources": ", ".join(ev.source_columns),
                    "Targets": ", ".join(ev.target_columns),
                    "Rows In -> Out": f"{ev.rows_before} -> {ev.rows_after}",
                    "Note": ev.note,
                }
                for ev in etl.lineage_graph.events
            ]
            st.dataframe(pd.DataFrame(lin_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No lineage events recorded.")


if __name__ == "__main__":
    main()

