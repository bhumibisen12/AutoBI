"""Pydantic contracts for the LangGraph orchestrator, human gates, patches, and background jobs."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field

from core.schemas.dax import (
    DAXMeasureSpec,
    MetricSelectionPlan,
    SelectedMetricSpec,
    VerificationReport,
)
from core.schemas.etl import ETLRunResult
from core.schemas.insights import InsightsRequest, InsightsSummary
from core.schemas.llm import ColumnRole, DatasetUnderstanding, UserIntent
from core.schemas.pbip import DataSourceFormat, PBIPExportResult
from core.schemas.profiling import ProfileReport
from core.schemas.report_spec import DraftVisualSpec, ReportSpec, SlicerSpec, VisualType
from core.schemas.security import PIIReport
from core.schemas.semantic import SemanticModel

WorkflowNodeName = Literal[
    "etl_pipeline",
    "profile",
    "understand",
    "human_confirm",
    "semantic_model",
    "dax_reconciliation",
    "insights",
    "report_spec",
    "human_review",
    "pbip_generation",
]

WorkflowStatus = Literal[
    "idle",
    "running",
    "awaiting_human_confirm",
    "awaiting_human_review",
    "completed",
    "failed",
]

NodeExecutionStatus = Literal["completed", "paused", "failed", "skipped"]

JobState = Literal["queued", "running", "completed", "failed"]

ReportPatchOpType = Literal[
    "rename_report",
    "rename_page",
    "add_page",
    "merge_to_single_page",
    "update_visual_title",
    "update_visual_style",
    "change_visual_type",
    "remove_visual",
    "add_visual",
    "add_slicer",
    "set_theme",
]


class NodeExecutionRecord(BaseModel):
    """Duration and outcome log entry for a single orchestrator node execution."""

    node: WorkflowNodeName
    status: NodeExecutionStatus
    started_at: datetime
    finished_at: datetime
    duration_ms: float = Field(ge=0.0)
    message: str = ""


class ColumnConfirmationEdit(BaseModel):
    """User-confirmed column role, friendly display name, and description."""

    table_name: str = Field(min_length=1)
    column_name: str = Field(min_length=1)
    role: ColumnRole
    friendly_name: str = Field(min_length=1)
    description: str = ""


class DesignOptions(BaseModel):
    """User design and execution preferences selected before semantic/report generation."""

    project_name: str = Field(default="AutoBI_Report", min_length=1)
    theme_name: str = Field(default="Auto (Best for Domain)", min_length=1)
    audience: str = Field(default="business analysts", min_length=1)
    source_format: DataSourceFormat = "parquet"
    run_insights: bool = False
    insights_request: InsightsRequest | None = None


class HumanConfirmPayload(BaseModel):
    """Human confirmation gate payload approving column roles, KPIs, and design options."""

    confirmed: bool = True
    column_edits: list[ColumnConfirmationEdit] = Field(default_factory=list)
    selected_metrics: list[SelectedMetricSpec] | None = None
    design_options: DesignOptions = Field(default_factory=DesignOptions)
    notes: str = ""


class ReportPatchOp(BaseModel):
    """Single declarative, allowlisted refinement operation applied to a ``ReportSpec``."""

    op: ReportPatchOpType
    report_title: str | None = None
    page_id: str | None = None
    page_title: str | None = None
    subtitle: str | None = None
    visual_id: str | None = None
    target_visual_type: VisualType | None = None
    apply_to_all: bool = False
    new_title: str | None = None
    new_visual_type: VisualType | None = None
    visual: DraftVisualSpec | None = None
    slicer: SlicerSpec | None = None
    theme_name: str | None = None
    title_align: Literal["left", "center", "right"] | None = None
    title_position: Literal["top", "bottom"] | None = None
    title_font_size: int | None = None
    custom_colors: list[str] | None = None
    show_data_labels: bool | None = None
    show_legend: bool | None = None
    sort_descending: bool | None = None
    background_color: str | None = None


class ReportSpecPatch(BaseModel):
    """Validated bundle of allowlisted refinement operations for a ``ReportSpec``."""

    summary: str = Field(min_length=1)
    operations: list[ReportPatchOp] = Field(default_factory=list)


class HumanReviewPayload(BaseModel):
    """Human review gate payload approving the previewed report before PBIP generation."""

    approved: bool = True
    refinement_notes: str = ""


class BackgroundJobStatus(BaseModel):
    """Status snapshot for a background worker task so the Streamlit UI never blocks."""

    job_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    stage: str = Field(min_length=1)
    status: JobState = "queued"
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    message: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    error: str | None = None


class PipelineSessionState(BaseModel):
    """Full persisted orchestrator state for a single isolated user session."""

    session_id: str = Field(min_length=1)
    status: WorkflowStatus = "idle"
    current_node: WorkflowNodeName | None = None
    failed_node: WorkflowNodeName | None = None
    error_message: str | None = None

    input_files: list[str] = Field(default_factory=list)
    db_source_label: str | None = None
    user_prompt: str = ""

    pii_reports: dict[str, PIIReport] = Field(default_factory=dict)
    etl_result: ETLRunResult | None = None
    profile_report: ProfileReport | None = None
    dataset_understanding: DatasetUnderstanding | None = None
    user_intent: UserIntent | None = None
    proposed_metrics: MetricSelectionPlan | None = None

    human_confirm: HumanConfirmPayload | None = None

    semantic_model: SemanticModel | None = None
    dax_measures: list[DAXMeasureSpec] = Field(default_factory=list)
    verification_report: VerificationReport | None = None
    insights_summary: InsightsSummary | None = None
    report_spec: ReportSpec | None = None
    preview_image_paths: list[str] = Field(default_factory=list)

    human_review: HumanReviewPayload | None = None
    pbip_export: PBIPExportResult | None = None

    chat_history: list[dict[str, str]] = Field(default_factory=list)
    node_logs: list[NodeExecutionRecord] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
