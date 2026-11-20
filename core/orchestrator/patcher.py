"""Accessible theme catalog and declarative, validated ReportSpec refinement patcher."""

from __future__ import annotations

import re
from collections.abc import Sequence

from core.llm import BaseLLMClient, LLMPrompt, MockLLMClient
from core.report_spec import finalize_report_spec
from core.schemas.dax import DAXMeasureSpec
from core.schemas.orchestrator import ReportPatchOp, ReportSpecPatch
from core.schemas.report_spec import (
    DraftPageSpec,
    DraftReportSpec,
    DraftVisualSpec,
    FieldBinding,
    ReportSpec,
    SlicerSpec,
    ThemeSpec,
    VisualType,
)
from core.schemas.semantic import SemanticModel

THEME_CATALOG: dict[str, ThemeSpec] = {
    "AutoBI Default": ThemeSpec(
        name="AutoBI Default",
        palette=["#1B4F72", "#2E86C1", "#148F77", "#D68910", "#884EA0", "#C0392B"],
        background_color="#FFFFFF",
        surface_color="#F8F9FA",
        border_color="#E2E8F0",
        text_color="#1C2833",
        muted_text_color="#475569",
    ),
    "Executive Navy": ThemeSpec(
        name="Executive Navy",
        palette=["#0F2C59", "#2B5876", "#3E7CB1", "#C89D54", "#6C757D", "#A63D40"],
        background_color="#FFFFFF",
        surface_color="#F4F6F9",
        border_color="#CBD5E1",
        text_color="#0F172A",
        muted_text_color="#475569",
    ),
    "Emerald Slate": ThemeSpec(
        name="Emerald Slate",
        palette=["#0D6E6E", "#198754", "#20A39E", "#E09F3E", "#4A5568", "#BC4749"],
        background_color="#FFFFFF",
        surface_color="#F5F7F6",
        border_color="#CBD5E1",
        text_color="#1A202C",
        muted_text_color="#4A5568",
    ),
    "Rose Pink": ThemeSpec(
        name="Rose Pink",
        palette=["#BE185D", "#EC4899", "#F472B6", "#9D174D", "#6366F1", "#F59E0B"],
        background_color="#FFFBFD",
        surface_color="#FDF2F8",
        border_color="#FBCFE8",
        text_color="#3B0764",
        muted_text_color="#831843",
    ),
    "Royal Purple": ThemeSpec(
        name="Royal Purple",
        palette=["#5B21B6", "#7C3AED", "#A78BFA", "#EC4899", "#0EA5E9", "#F59E0B"],
        background_color="#FFFFFF",
        surface_color="#F5F3FF",
        border_color="#DDD6FE",
        text_color="#1E1B4B",
        muted_text_color="#5B21B6",
    ),
    "Sunset Coral": ThemeSpec(
        name="Sunset Coral",
        palette=["#C2410C", "#EA580C", "#F97316", "#0D9488", "#2563EB", "#7C3AED"],
        background_color="#FFFFFF",
        surface_color="#FFF7ED",
        border_color="#FED7AA",
        text_color="#1C1917",
        muted_text_color="#7C2D12",
    ),
    "Midnight Slate": ThemeSpec(
        name="Midnight Slate",
        palette=["#38BDF8", "#818CF8", "#34D399", "#F472B6", "#FBBF24", "#FB7185"],
        background_color="#0F172A",
        surface_color="#1E293B",
        border_color="#334155",
        text_color="#F8FAFC",
        muted_text_color="#94A3B8",
    ),
    "High Contrast": ThemeSpec(
        name="High Contrast",
        palette=["#003F5C", "#BC5090", "#FF6361", "#FFA600", "#58508D", "#00876C"],
        background_color="#FFFFFF",
        surface_color="#F8FAFC",
        border_color="#1E293B",
        text_color="#000000",
        muted_text_color="#334155",
    ),
}

_COLOR_THEME_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (("pink", "rose", "magenta", "blush"), "Rose Pink"),
    (("purple", "violet", "lavender", "indigo"), "Royal Purple"),
    (("coral", "orange", "sunset", "warm"), "Sunset Coral"),
    (("dark", "midnight", "black", "night"), "Midnight Slate"),
    (("emerald", "green", "teal"), "Emerald Slate"),
    (("navy", "blue", "executive", "corporate", "professional"), "Executive Navy"),
    (("contrast", "accessible"), "High Contrast"),
]

_VISUAL_TYPE_ALIASES: dict[str, VisualType] = {
    "bar": "bar_chart",
    "bar chart": "bar_chart",
    "column": "column_chart",
    "column chart": "column_chart",
    "line": "line_chart",
    "line chart": "line_chart",
    "area": "area_chart",
    "area chart": "area_chart",
    "donut": "donut_chart",
    "donut chart": "donut_chart",
    "pie": "pie_chart",
    "pie chart": "pie_chart",
    "scatter": "scatter_chart",
    "scatter chart": "scatter_chart",
    "scatter plot": "scatter_chart",
    "table": "table",
    "matrix": "matrix",
    "card": "kpi_card",
    "kpi card": "kpi_card",
    "kpi": "kpi_card",
}

_NAMED_COLORS: dict[str, str] = {
    "black": "#111827",
    "white": "#FFFFFF",
    "red": "#DC2626",
    "blue": "#2563EB",
    "navy": "#0F2C59",
    "green": "#16A34A",
    "emerald": "#059669",
    "teal": "#0D9488",
    "cyan": "#06B6D4",
    "pink": "#EC4899",
    "rose": "#BE185D",
    "purple": "#7C3AED",
    "violet": "#6D28D9",
    "indigo": "#4F46E5",
    "orange": "#EA580C",
    "coral": "#F97316",
    "yellow": "#EAB308",
    "gold": "#D97706",
    "gray": "#6B7280",
    "grey": "#6B7280",
    "silver": "#CBD5E1",
    "maroon": "#881337",
    "brown": "#92400E",
}


def _extract_colors(text: str) -> list[str]:
    """Extract hex codes and named CSS colors mentioned in ``text`` in order of appearance."""
    found: list[tuple[int, str]] = []
    for m in re.finditer(r"#[0-9a-fA-F]{6}\b", text):
        found.append((m.start(), m.group(0).upper()))
    lowered = text.lower()
    for name, hex_code in _NAMED_COLORS.items():
        for m in re.finditer(rf"\b{re.escape(name)}\b", lowered):
            found.append((m.start(), hex_code))
    found.sort(key=lambda item: item[0])
    deduped: list[str] = []
    for _, code in found:
        if code not in deduped:
            deduped.append(code)
    return deduped


def _detect_targeted_visual_type(lowered: str) -> VisualType | None:
    """Detect if the user explicitly targeted a specific visual type (e.g. 'pie chart')."""
    # Check longer multi-word aliases first so 'bar chart' wins over 'bar'.
    for alias in sorted(_VISUAL_TYPE_ALIASES.keys(), key=len, reverse=True):
        if re.search(rf"\b{re.escape(alias)}s?\b", lowered):
            return _VISUAL_TYPE_ALIASES[alias]
    return None


DOMAIN_THEME_MAP: dict[str, str] = {
    "sales": "Executive Navy",
    "finance": "Executive Navy",
    "healthcare": "Emerald Slate",
    "clinical": "Emerald Slate",
    "human_resources": "Royal Purple",
    "hr": "Royal Purple",
    "education": "Royal Purple",
    "agriculture": "Emerald Slate",
    "operations": "AutoBI Default",
    "marketing": "Sunset Coral",
    "retail": "Sunset Coral",
}


def choose_theme_for_domain(business_domain: str) -> ThemeSpec:
    """Select an executive, domain-appropriate ``ThemeSpec`` automatically."""
    key = business_domain.strip().lower()
    chosen_name = DOMAIN_THEME_MAP.get(key, "Executive Navy")
    return THEME_CATALOG[chosen_name].model_copy(deep=True)


def get_theme_by_name(theme_name: str, *, business_domain: str = "general_analytics") -> ThemeSpec:
    """Resolve a theme name (or ``'Auto (Best for Domain)'``) to a ``ThemeSpec``."""
    cleaned = theme_name.strip().lower()
    if cleaned in {"auto", "auto (best for domain)"}:
        return choose_theme_for_domain(business_domain)
    for key, theme in THEME_CATALOG.items():
        if key.lower() == cleaned:
            return theme.model_copy(deep=True)
    return THEME_CATALOG["AutoBI Default"].model_copy(deep=True)


def _report_to_draft(report_spec: ReportSpec) -> DraftReportSpec:
    """Convert a finalized ``ReportSpec`` back into an unpositioned ``DraftReportSpec``."""
    draft_pages: list[DraftPageSpec] = []
    for page in report_spec.pages:
        draft_visuals = [
            DraftVisualSpec(
                visual_type=vis.visual_type,
                title=vis.title,
                category_field=vis.category_field,
                secondary_field=vis.secondary_field,
                columns=list(vis.columns),
                measures=list(vis.measures),
                format=vis.format,
                alt_text=vis.alt_text,
            )
            for vis in page.visuals
        ]
        draft_pages.append(
            DraftPageSpec(
                title=page.title,
                subtitle=page.subtitle,
                slicers=list(page.slicers),
                visuals=draft_visuals,
            )
        )
    return DraftReportSpec(
        report_title=report_spec.report_title,
        pages=draft_pages,
        rationale="Patched from finalized ReportSpec.",
    )


def _build_draft_visual_for_add(
    vis_type: VisualType,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
) -> DraftVisualSpec | None:
    """Build a schema-valid ``DraftVisualSpec`` for a chat 'add <chart>' command."""
    if not measures:
        return None
    fact_tables = [t for t in model.tables if t.kind == "fact"] or [
        t for t in model.tables if not t.is_date_table
    ]
    if not fact_tables:
        return None
    fact = fact_tables[0]
    dim_cols = [c for c in fact.columns if c.role == "dimension"]
    primary_m = measures[0].name
    second_m = measures[1].name if len(measures) > 1 else primary_m

    if vis_type == "kpi_card":
        return DraftVisualSpec(visual_type="kpi_card", title=primary_m, measures=[primary_m])
    if vis_type in {"line_chart", "area_chart"} and model.date_table_name:
        return DraftVisualSpec(
            visual_type=vis_type,
            title=f"{primary_m} Trend",
            category_field=FieldBinding(table_name=model.date_table_name, column_name="year_month"),
            measures=[primary_m],
        )
    if vis_type == "scatter_chart" and len(measures) >= 2 and dim_cols:
        return DraftVisualSpec(
            visual_type="scatter_chart",
            title=f"{primary_m} vs {second_m}",
            category_field=FieldBinding(table_name=fact.table_name, column_name=dim_cols[0].name),
            measures=[primary_m, second_m],
        )
    if vis_type in {"table", "matrix"} and dim_cols:
        cols = [FieldBinding(table_name=fact.table_name, column_name=c.name) for c in dim_cols[:2]]
        return DraftVisualSpec(
            visual_type=vis_type,
            title=f"{fact.table_name.replace('_', ' ').title()} {vis_type.title()}",
            category_field=cols[0],
            secondary_field=cols[1] if len(cols) > 1 and vis_type == "matrix" else None,
            columns=cols,
            measures=[m.name for m in measures[:2]],
        )
    if dim_cols:
        dim = dim_cols[0]
        return DraftVisualSpec(
            visual_type=vis_type,
            title=f"{primary_m} by {dim.friendly_name}",
            category_field=FieldBinding(table_name=fact.table_name, column_name=dim.name),
            measures=[primary_m],
        )
    return None


def _deterministic_patch_from_message(
    user_message: str,
    report_spec: ReportSpec,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec] = (),
) -> ReportSpecPatch:
    """Parse natural-language report refinements into declarative ``ReportSpecPatch`` ops."""
    msg = user_message.strip()
    lowered = msg.lower()
    ops: list[ReportPatchOp] = []
    first_page_id = report_spec.pages[0].page_id if report_spec.pages else "page_1"

    # 0. Per-visual or all-charts styling (title alignment, custom chart colors, labels, legend)
    target_vtype = _detect_targeted_visual_type(lowered)
    mentioned_colors = _extract_colors(msg)
    is_chart_specific_color = bool(
        target_vtype is not None
        and mentioned_colors
        and any(w in lowered for w in ("color", "colour", "palette", "black", "white", "only"))
        and "theme" not in lowered
    )

    style_kwargs: dict[str, object] = {}
    mentions_title = any(w in lowered for w in ("title", "heading", "name", "chart", "visual"))
    if re.search(r"\b(?:center|centre|centered|centred|middle)\b", lowered) and mentions_title:
        style_kwargs["title_align"] = "center"
    elif re.search(r"\bright\b", lowered) and ("title" in lowered or "align" in lowered):
        style_kwargs["title_align"] = "right"
    elif re.search(r"\bleft\b", lowered) and ("title" in lowered or "align" in lowered):
        style_kwargs["title_align"] = "left"

    if re.search(r"\b(?:below|bottom|under|underneath)\b", lowered) and mentions_title:
        style_kwargs["title_position"] = "bottom"
    elif re.search(r"\b(?:above|top)\b", lowered) and ("title" in lowered or "name" in lowered):
        style_kwargs["title_position"] = "top"

    if is_chart_specific_color:
        # When user asks for e.g. 'black and white' on pie/donut chart, pad monochrome shades
        # if more than 2 slices exist while keeping black & white first.
        colors_list = list(mentioned_colors)
        if set(colors_list) == {"#111827", "#FFFFFF"}:
            colors_list = ["#111827", "#FFFFFF", "#4B5563", "#D1D5DB", "#374151", "#9CA3AF"]
        style_kwargs["custom_colors"] = colors_list

    if re.search(r"\b(?:hide|remove|no)\s+(?:the\s+)?(?:data\s+)?labels?\b", lowered):
        style_kwargs["show_data_labels"] = False
    elif re.search(r"\b(?:show|add|enable)\s+(?:the\s+)?(?:data\s+)?labels?\b", lowered):
        style_kwargs["show_data_labels"] = True

    if re.search(r"\b(?:hide|remove|no)\s+(?:the\s+)?legend\b", lowered):
        style_kwargs["show_legend"] = False
    elif re.search(r"\b(?:show|add|enable)\s+(?:the\s+)?legend\b", lowered):
        style_kwargs["show_legend"] = True

    if re.search(r"\bsort\s+asc(?:ending)?\b", lowered):
        style_kwargs["sort_descending"] = False
    elif re.search(r"\bsort\s+desc(?:ending)?\b", lowered):
        style_kwargs["sort_descending"] = True

    if re.search(r"\b(?:bigger|larger|increase)\s+(?:the\s+)?(?:title|font)\b", lowered):
        style_kwargs["title_font_size"] = 18
    elif re.search(r"\b(?:smaller|decrease|compact)\s+(?:the\s+)?(?:title|font)\b", lowered):
        style_kwargs["title_font_size"] = 12

    if style_kwargs:
        force_all = bool(
            re.search(r"\b(?:all\s+(?:visuals|charts|graphs)|every\s+(?:visual|chart))\b", lowered)
        )
        effective_target = None if force_all else target_vtype
        ops.append(
            ReportPatchOp(
                op="update_visual_style",
                target_visual_type=effective_target,
                apply_to_all=(effective_target is None),
                **style_kwargs,  # type: ignore[arg-type]
            )
        )

    # 1. Whole-report theme change (skip if user explicitly asked to color ONLY one chart!)
    if not is_chart_specific_color:
        matched_theme: str | None = None
        for theme_name in THEME_CATALOG:
            if theme_name.lower() in lowered:
                matched_theme = theme_name
                break
        if matched_theme is None:
            for keywords, theme_name in _COLOR_THEME_KEYWORDS:
                if any(re.search(rf"\b{re.escape(kw)}\b", lowered) for kw in keywords):
                    matched_theme = theme_name
                    break
        if matched_theme is not None:
            ops.append(ReportPatchOp(op="set_theme", theme_name=matched_theme))

    # 2. Rename report title: "rename report to X" / "change title to X"
    title_match = re.search(
        r"(?:rename|change|set)\s+(?:the\s+)?(?:report\s+)?title\s+to\s+[\"']?([^\"'\n]+?)[\"']?"
        r"(?=\s+and\s+(?:switch|change|set|add|remove|rename)\b|$)",
        msg,
        flags=re.IGNORECASE,
    )
    if title_match:
        ops.append(ReportPatchOp(op="rename_report", report_title=title_match.group(1).strip()))

    # 3. Rename page: "rename page to X"
    page_rename_match = re.search(
        r"rename\s+(?:the\s+)?page\s+to\s+[\"']?([^\"'\n]+)[\"']?",
        msg,
        flags=re.IGNORECASE,
    )
    if page_rename_match:
        ops.append(
            ReportPatchOp(
                op="rename_page",
                page_id=first_page_id,
                new_title=page_rename_match.group(1).strip(),
            )
        )

    # 4. Add page: "add page <Title>"
    add_page_match = re.search(
        r"add\s+(?:a\s+)?(?:new\s+)?page(?:\s+(?:called|named|titled))?\s+[\"']?([^\"'\n]+)[\"']?",
        msg,
        flags=re.IGNORECASE,
    )
    if add_page_match:
        ops.append(
            ReportPatchOp(
                op="add_page",
                page_title=add_page_match.group(1).strip(),
                subtitle="Added via report refinement chat",
            )
        )

    # 4b. Merge all pages into one single page
    if re.search(
        r"\b(?:one|1|single|ine)\s+page\b|\b(?:merge|combine)\s+(?:all\s+)?pages?\b|"
        r"\ball\s+(?:things|charts|visuals)\b.*\bpage\b",
        lowered,
    ):
        ops.append(ReportPatchOp(op="merge_to_single_page"))

    # 5. Add slicer: "add slicer for <column>"
    slicer_match = re.search(
        r"add\s+(?:a\s+)?slicer\s+(?:for|on)\s+([A-Za-z0-9_ ]+)",
        msg,
        flags=re.IGNORECASE,
    )
    if slicer_match:
        target_col = slicer_match.group(1).strip().lower().replace(" ", "_")
        for tbl in model.tables:
            for col in tbl.columns:
                if (
                    col.name.lower() == target_col
                    or col.name.lower().replace(" ", "_") == target_col
                    or col.friendly_name.lower().replace(" ", "_") == target_col
                ):
                    ops.append(
                        ReportPatchOp(
                            op="add_slicer",
                            page_id=first_page_id,
                            slicer=SlicerSpec(
                                table_name=tbl.table_name,
                                column_name=col.name,
                                title=col.friendly_name,
                                slicer_style="dropdown",
                            ),
                        )
                    )
                    break

    # 6. Add visual: "add pie chart" / "add donut chart" / "add bar chart" / "add scatter chart"
    add_vis_match = re.search(
        r"\badd\s+(?:a\s+|an\s+|new\s+)?(pie\s+chart|donut\s+chart|bar\s+chart|column\s+chart|"
        r"line\s+chart|area\s+chart|scatter\s+chart|kpi\s+card|pie|donut|bar|column|line|"
        r"scatter|table|matrix)\b",
        lowered,
    )
    if add_vis_match:
        vis_key = re.sub(r"\s+", " ", add_vis_match.group(1).strip())
        resolved_type: VisualType | None = (
            "scatter_chart" if "scatter" in vis_key else _VISUAL_TYPE_ALIASES.get(vis_key)
        )
        if resolved_type is not None:
            new_vis = _build_draft_visual_for_add(resolved_type, model, measures)
            if new_vis is not None:
                # Pick a page that still has room under the 8-visual cap.
                target_pid = next(
                    (p.page_id for p in report_spec.pages if len(p.visuals) < 8),
                    first_page_id,
                )
                ops.append(ReportPatchOp(op="add_visual", page_id=target_pid, visual=new_vis))

    # 7. Change visual type: "change <old_type> to <new_type>"
    swap_match = re.search(
        r"(?:change|switch|convert)\s+(?:the\s+)?([a-z ]+?)\s+to\s+(?:a\s+)?([a-z ]+)",
        lowered,
    )
    if swap_match:
        src_key = swap_match.group(1).strip()
        dst_key = swap_match.group(2).strip().rstrip(".")
        dst_type = _VISUAL_TYPE_ALIASES.get(dst_key)
        src_type = _VISUAL_TYPE_ALIASES.get(src_key)
        if dst_type is not None:
            for page in report_spec.pages:
                for vis in page.visuals:
                    if src_type is None or vis.visual_type == src_type:
                        ops.append(
                            ReportPatchOp(
                                op="change_visual_type",
                                page_id=page.page_id,
                                visual_id=vis.visual_id,
                                new_visual_type=dst_type,
                            )
                        )
                        break
                if any(o.op == "change_visual_type" for o in ops):
                    break

    # 8. Remove visual by title substring or type: "remove <target>"
    remove_match = re.search(r"(?:remove|delete)\s+(?:the\s+)?([A-Za-z0-9_ ]+)", msg, re.IGNORECASE)
    if remove_match and not ops:
        target = remove_match.group(1).strip().lower().rstrip(".")
        target_type = _VISUAL_TYPE_ALIASES.get(target)
        for page in report_spec.pages:
            if len(page.visuals) <= 1:
                continue
            for vis in page.visuals:
                if target in vis.title.lower() or (target_type and vis.visual_type == target_type):
                    ops.append(
                        ReportPatchOp(
                            op="remove_visual",
                            page_id=page.page_id,
                            visual_id=vis.visual_id,
                        )
                    )
                    break

    if not ops:
        summary = (
            "No structural patch matched that request. Try: "
            "'use pink theme', 'add donut chart', 'add pie chart', "
            "'change title to Q4 Executive View', 'change bar chart to column chart', "
            "'add slicer for region', or 'add page Details'."
        )
        return ReportSpecPatch(summary=summary, operations=[])

    summary = f"Applied {len(ops)} refinement operation(s): " + ", ".join(op.op for op in ops)
    return ReportSpecPatch(summary=summary, operations=ops)


def propose_report_patch(
    user_message: str,
    report_spec: ReportSpec,
    model: SemanticModel,
    *,
    measures: Sequence[DAXMeasureSpec] = (),
    llm_client: BaseLLMClient | None = None,
) -> ReportSpecPatch:
    """Propose a validated ``ReportSpecPatch`` from a user's chat refinement message."""
    if llm_client is not None and not isinstance(llm_client, MockLLMClient):
        system_prompt = (
            "You translate report refinement chat messages into a declarative ReportSpecPatch "
            "using ONLY allowlisted ops: rename_report, rename_page, add_page, "
            "merge_to_single_page, update_visual_title, update_visual_style (supports "
            "target_visual_type, apply_to_all, title_align ['left'|'center'|'right'], "
            "custom_colors [hex list], show_data_labels, show_legend, sort_descending, "
            "title_font_size, background_color), change_visual_type, remove_visual, "
            "add_visual, add_slicer, set_theme. Never output pixel coordinates or code."
        )
        context = {
            "report_title": report_spec.report_title,
            "available_themes": list(THEME_CATALOG.keys()),
            "available_measures": [m.name for m in measures],
            "pages": [
                {
                    "page_id": p.page_id,
                    "title": p.title,
                    "visuals": [
                        {"visual_id": v.visual_id, "type": v.visual_type, "title": v.title}
                        for v in p.visuals
                    ],
                }
                for p in report_spec.pages
            ],
            "tables": [
                {"table_name": t.table_name, "columns": [c.name for c in t.columns]}
                for t in model.tables
            ],
        }
        prompt = LLMPrompt(
            system_prompt=system_prompt,
            user_prompt=(
                f"User refinement request: {user_message}\n\n"
                f"<UNTRUSTED_REPORT_STATE>\n{context}\n</UNTRUSTED_REPORT_STATE>"
            ),
            temperature=0.0,
        )
        patch, _ = llm_client.generate_structured(prompt, ReportSpecPatch, use_cache=False)
        return patch

    return _deterministic_patch_from_message(user_message, report_spec, model, measures)


def apply_report_patch(
    report_spec: ReportSpec,
    patch: ReportSpecPatch,
    model: SemanticModel,
    measures: Sequence[DAXMeasureSpec],
) -> ReportSpec:
    """Apply an allowlisted ``ReportSpecPatch`` and re-run full validation + 12-col layout."""
    if not patch.operations:
        return report_spec

    draft = _report_to_draft(report_spec)
    theme = report_spec.theme.model_copy(deep=True)
    page_index_by_id = {p.page_id: idx for idx, p in enumerate(report_spec.pages)}
    vis_index_by_page: dict[str, dict[str, int]] = {
        p.page_id: {v.visual_id: v_idx for v_idx, v in enumerate(p.visuals)}
        for p in report_spec.pages
    }

    for op in patch.operations:
        if op.op == "set_theme" and op.theme_name:
            theme = get_theme_by_name(op.theme_name)
        elif op.op == "rename_report" and op.report_title:
            draft.report_title = op.report_title.strip()
        elif op.op == "rename_page" and op.page_id in page_index_by_id and op.new_title:
            draft.pages[page_index_by_id[op.page_id]].title = op.new_title.strip()
        elif op.op == "add_page" and op.page_title:
            default_measure = measures[0].name if measures else ""
            if default_measure:
                draft.pages.append(
                    DraftPageSpec(
                        title=op.page_title.strip(),
                        subtitle=op.subtitle or "",
                        visuals=[
                            DraftVisualSpec(
                                visual_type="kpi_card",
                                title=default_measure,
                                measures=[default_measure],
                            )
                        ],
                    )
                )
        elif op.op == "merge_to_single_page":
            all_visuals = [v for page in draft.pages for v in page.visuals]
            # Keep up to 4 KPI cards and fill remaining slots up to the 8-visual page cap.
            kpis = [v for v in all_visuals if v.visual_type == "kpi_card"][:4]
            others = [v for v in all_visuals if v.visual_type != "kpi_card"]
            merged_visuals = [*kpis, *others[: max(1, 8 - len(kpis))]]
            seen_slicers: set[tuple[str, str]] = set()
            merged_slicers: list[SlicerSpec] = []
            for page in draft.pages:
                for slicer in page.slicers:
                    key = (slicer.table_name, slicer.column_name)
                    if key not in seen_slicers and len(merged_slicers) < 4:
                        seen_slicers.add(key)
                        merged_slicers.append(slicer)
            base_page = draft.pages[0]
            draft.pages = [
                DraftPageSpec(
                    title=base_page.title,
                    subtitle=base_page.subtitle,
                    slicers=merged_slicers,
                    visuals=merged_visuals,
                )
            ]
        elif op.op == "update_visual_title" and op.page_id in vis_index_by_page and op.new_title:
            v_idx = vis_index_by_page[op.page_id].get(op.visual_id or "")
            if v_idx is not None:
                p_idx = page_index_by_id[op.page_id]
                draft.pages[p_idx].visuals[v_idx].title = op.new_title.strip()
        elif op.op == "update_visual_style":
            # Match pie_chart <-> donut_chart interchangeably when user says 'pie chart' on a donut.
            existing_types = {v.visual_type for p in draft.pages for v in p.visuals}
            target_types: set[VisualType] = set()
            if op.target_visual_type is not None:
                target_types.add(op.target_visual_type)
                if op.target_visual_type == "pie_chart" and "pie_chart" not in existing_types:
                    target_types.add("donut_chart")
                elif op.target_visual_type == "donut_chart" and "donut_chart" not in existing_types:
                    target_types.add("pie_chart")

            for p_idx, page in enumerate(draft.pages):
                if op.page_id and page_index_by_id.get(op.page_id) != p_idx:
                    continue
                for v_idx, vis in enumerate(page.visuals):
                    matches_id = (
                        op.visual_id is not None
                        and op.page_id in vis_index_by_page
                        and vis_index_by_page[op.page_id].get(op.visual_id) == v_idx
                    )
                    matches_type = bool(target_types and vis.visual_type in target_types)
                    if not (op.apply_to_all or matches_id or matches_type):
                        continue
                    updates: dict[str, object] = {}
                    if op.title_align is not None:
                        updates["title_align"] = op.title_align
                    if op.title_position is not None:
                        updates["title_position"] = op.title_position
                    if op.title_font_size is not None:
                        updates["title_font_size"] = op.title_font_size
                    if op.custom_colors is not None:
                        updates["custom_colors"] = list(op.custom_colors)
                    if op.show_data_labels is not None:
                        updates["show_data_labels"] = op.show_data_labels
                    if op.show_legend is not None:
                        updates["show_legend"] = op.show_legend
                    if op.sort_descending is not None:
                        updates["sort_descending"] = op.sort_descending
                    if op.background_color is not None:
                        updates["background_color"] = op.background_color
                    if updates:
                        vis.format = vis.format.model_copy(update=updates)
        elif (
            op.op == "change_visual_type"
            and op.page_id in vis_index_by_page
            and op.new_visual_type is not None
        ):
            v_idx = vis_index_by_page[op.page_id].get(op.visual_id or "")
            if v_idx is not None:
                p_idx = page_index_by_id[op.page_id]
                vis = draft.pages[p_idx].visuals[v_idx]
                vis.visual_type = op.new_visual_type
                if op.new_visual_type == "table" and not vis.columns and vis.category_field:
                    vis.columns = [
                        FieldBinding(
                            table_name=vis.category_field.table_name,
                            column_name=vis.category_field.column_name,
                        )
                    ]
        elif op.op == "remove_visual" and op.page_id in vis_index_by_page:
            v_idx = vis_index_by_page[op.page_id].get(op.visual_id or "")
            p_idx = page_index_by_id[op.page_id]
            if v_idx is not None and len(draft.pages[p_idx].visuals) > 1:
                draft.pages[p_idx].visuals.pop(v_idx)
        elif op.op == "add_visual" and op.page_id in page_index_by_id and op.visual is not None:
            p_idx = page_index_by_id[op.page_id]
            draft.pages[p_idx].visuals.append(op.visual)
        elif op.op == "add_slicer" and op.page_id in page_index_by_id and op.slicer is not None:
            p_idx = page_index_by_id[op.page_id]
            existing = {
                (s.table_name, s.column_name) for s in draft.pages[p_idx].slicers
            }
            if (
                (op.slicer.table_name, op.slicer.column_name) not in existing
                and len(draft.pages[p_idx].slicers) < 4
            ):
                draft.pages[p_idx].slicers.append(op.slicer)

    return finalize_report_spec(
        draft,
        model,
        measures,
        theme=theme,
        curated_duckdb_path=model.curated_duckdb_path,
        rag_source_ids=list(report_spec.rag_source_ids),
    )
