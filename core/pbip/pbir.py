"""PBIR report definition writer (report.json, pages, visuals, slicers, and custom theme)."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from core.pbip.templates import (
    SCHEMA_PAGE,
    SCHEMA_PAGES,
    SCHEMA_PBIR,
    SCHEMA_REPORT,
    SCHEMA_VERSION,
    SCHEMA_VISUAL,
    build_powerbi_theme_json,
)
from core.schemas.dax import DAXMeasureSpec
from core.schemas.report_spec import (
    ReportSpec,
    SlicerSpec,
    VisualSpec,
    VisualType,
)

_PBIR_VISUAL_TYPE_MAP: dict[VisualType, str] = {
    "kpi_card": "card",
    "bar_chart": "clusteredBarChart",
    "column_chart": "clusteredColumnChart",
    "line_chart": "lineChart",
    "area_chart": "areaChart",
    "donut_chart": "donutChart",
    "pie_chart": "pieChart",
    "scatter_chart": "scatterChart",
    "table": "tableEx",
    "matrix": "pivotTable",
}


def _col_projection(table_name: str, column_name: str) -> dict[str, Any]:
    """Build a PBIR column field projection."""
    return {
        "field": {
            "Column": {
                "Expression": {"SourceRef": {"Entity": table_name}},
                "Property": column_name,
            }
        },
        "queryRef": f"{table_name}.{column_name}",
        "active": True,
    }


def _measure_projection(table_name: str, measure_name: str) -> dict[str, Any]:
    """Build a PBIR measure field projection."""
    return {
        "field": {
            "Measure": {
                "Expression": {"SourceRef": {"Entity": table_name}},
                "Property": measure_name,
            }
        },
        "queryRef": f"{table_name}.{measure_name}",
    }


def _build_visual_query_state(
    visual: VisualSpec,
    measure_map: dict[str, DAXMeasureSpec],
) -> dict[str, Any]:
    """Map an AutoBI ``VisualSpec`` into PBIR ``queryState`` data role projections."""
    vtype = visual.visual_type
    m_projs = [
        _measure_projection(measure_map[m_name].table_name, m_name)
        for m_name in visual.measures
        if m_name in measure_map
    ]

    if vtype == "kpi_card":
        return {"Values": {"projections": m_projs}}

    if vtype in {
        "bar_chart",
        "column_chart",
        "line_chart",
        "area_chart",
        "donut_chart",
        "pie_chart",
    }:
        state: dict[str, Any] = {}
        if visual.category_field is not None:
            state["Category"] = {
                "projections": [
                    _col_projection(
                        visual.category_field.table_name,
                        visual.category_field.column_name,
                    )
                ]
            }
        if visual.secondary_field is not None:
            state["Series"] = {
                "projections": [
                    _col_projection(
                        visual.secondary_field.table_name,
                        visual.secondary_field.column_name,
                    )
                ]
            }
        if m_projs:
            state["Y"] = {"projections": m_projs}
        return state

    if vtype == "scatter_chart":
        state = {}
        if visual.category_field is not None:
            state["Details"] = {
                "projections": [
                    _col_projection(
                        visual.category_field.table_name,
                        visual.category_field.column_name,
                    )
                ]
            }
        if len(m_projs) >= 1:
            state["X"] = {"projections": [m_projs[0]]}
        if len(m_projs) >= 2:
            state["Y"] = {"projections": [m_projs[1]]}
        return state

    if vtype == "table":
        col_projs = [_col_projection(c.table_name, c.column_name) for c in visual.columns]
        if not col_projs and visual.category_field is not None:
            col_projs.append(
                _col_projection(visual.category_field.table_name, visual.category_field.column_name)
            )
        return {"Values": {"projections": [*col_projs, *m_projs]}}

    if vtype == "matrix":
        state = {}
        row_projs: list[dict[str, Any]] = []
        if visual.category_field is not None:
            row_projs.append(
                _col_projection(visual.category_field.table_name, visual.category_field.column_name)
            )
        elif visual.columns:
            row_projs.append(
                _col_projection(visual.columns[0].table_name, visual.columns[0].column_name)
            )
        if row_projs:
            state["Rows"] = {"projections": row_projs}
        if visual.secondary_field is not None:
            state["Columns"] = {
                "projections": [
                    _col_projection(
                        visual.secondary_field.table_name,
                        visual.secondary_field.column_name,
                    )
                ]
            }
        elif len(visual.columns) >= 2:
            state["Columns"] = {
                "projections": [
                    _col_projection(visual.columns[1].table_name, visual.columns[1].column_name)
                ]
            }
        if m_projs:
            state["Values"] = {"projections": m_projs}
        return state

    return {"Values": {"projections": m_projs}}


def _escaped_dax_literal(text: str) -> str:
    """Wrap a string literal for PBIR formatting property expressions."""
    escaped = text.replace("'", "''")
    return f"'{escaped}'"


def _build_visual_json(
    visual: VisualSpec,
    measure_map: dict[str, DAXMeasureSpec],
    *,
    z_index: int,
    tab_order: int,
) -> dict[str, Any]:
    """Serialize one ``VisualSpec`` into a PBIR ``visual.json`` dictionary."""
    pbir_type = _PBIR_VISUAL_TYPE_MAP[visual.visual_type]
    query_state = _build_visual_query_state(visual, measure_map)

    return {
        "$schema": SCHEMA_VISUAL,
        "name": visual.visual_id,
        "position": {
            "x": visual.pixel_bounds.x,
            "y": visual.pixel_bounds.y,
            "z": z_index,
            "height": visual.pixel_bounds.height,
            "width": visual.pixel_bounds.width,
            "tabOrder": tab_order,
        },
        "visual": {
            "visualType": pbir_type,
            "query": {"queryState": query_state},
            "visualContainerObjects": {
                "title": [
                    {
                        "properties": {
                            "show": {"expr": {"Literal": {"Value": "true"}}},
                            "text": {
                                "expr": {"Literal": {"Value": _escaped_dax_literal(visual.title)}}
                            },
                        }
                    }
                ],
                "general": [
                    {
                        "properties": {
                            "altText": {
                                "expr": {
                                    "Literal": {"Value": _escaped_dax_literal(visual.alt_text)}
                                }
                            }
                        }
                    }
                ],
            },
        },
    }


def _build_slicer_visual_json(
    slicer: SlicerSpec,
    *,
    visual_id: str,
    x: int,
    y: int,
    width: int,
    height: int,
    z_index: int,
    tab_order: int,
) -> dict[str, Any]:
    """Serialize a page ``SlicerSpec`` into a PBIR ``slicer`` visual container."""
    return {
        "$schema": SCHEMA_VISUAL,
        "name": visual_id,
        "position": {
            "x": x,
            "y": y,
            "z": z_index,
            "height": height,
            "width": width,
            "tabOrder": tab_order,
        },
        "visual": {
            "visualType": "slicer",
            "query": {
                "queryState": {
                    "Values": {
                        "projections": [_col_projection(slicer.table_name, slicer.column_name)]
                    }
                }
            },
            "visualContainerObjects": {
                "title": [
                    {
                        "properties": {
                            "show": {"expr": {"Literal": {"Value": "true"}}},
                            "text": {
                                "expr": {"Literal": {"Value": _escaped_dax_literal(slicer.title)}}
                            },
                        }
                    }
                ],
                "general": [
                    {
                        "properties": {
                            "altText": {
                                "expr": {
                                    "Literal": {
                                        "Value": _escaped_dax_literal(
                                            f"Slicer filtering by {slicer.title}."
                                        )
                                    }
                                }
                            }
                        }
                    }
                ],
            },
        },
    }


def write_pbir_report(
    report_spec: ReportSpec,
    measures: Sequence[DAXMeasureSpec],
    report_dir: Path,
    *,
    semantic_model_rel_path: str,
) -> list[Path]:
    """Write a complete PBIR ``.Report`` folder conforming to Microsoft PBIR schemas."""
    definition_dir = report_dir / "definition"
    pages_root = definition_dir / "pages"
    resources_dir = report_dir / "StaticResources" / "RegisteredResources"
    pages_root.mkdir(parents=True, exist_ok=True)
    resources_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    measure_map = {m.name: m for m in measures}

    # 1. definition.pbir
    pbir_path = report_dir / "definition.pbir"
    pbir_payload = {
        "$schema": SCHEMA_PBIR,
        "version": "4.0",
        "datasetReference": {
            "byPath": {"path": semantic_model_rel_path},
        },
    }
    pbir_path.write_text(json.dumps(pbir_payload, indent=2) + "\n", encoding="utf-8")
    written.append(pbir_path)

    # 2. Custom accessible theme under StaticResources/RegisteredResources/
    theme_filename = "autobi_theme.json"
    theme_path = resources_dir / theme_filename
    theme_payload = build_powerbi_theme_json(report_spec.theme)
    theme_path.write_text(json.dumps(theme_payload, indent=2) + "\n", encoding="utf-8")
    written.append(theme_path)

    # 3. definition/version.json
    version_path = definition_dir / "version.json"
    version_payload = {
        "$schema": SCHEMA_VERSION,
        "buildVersion": "2.138.0.0",
        "reportVersion": "3.1.0",
    }
    version_path.write_text(json.dumps(version_payload, indent=2) + "\n", encoding="utf-8")
    written.append(version_path)

    # 4. definition/report.json
    report_json_path = definition_dir / "report.json"
    report_payload = {
        "$schema": SCHEMA_REPORT,
        "themeCollection": {
            "baseTheme": {
                "name": "CY24SU10",
                "reportVersionAtImport": {
                    "visual": "2.4.0",
                    "report": "3.1.0",
                    "page": "2.0.0",
                },
                "type": "SharedResources",
            },
            "customTheme": {
                "name": theme_filename,
                "reportVersionAtImport": {
                    "visual": "2.4.0",
                    "report": "3.1.0",
                    "page": "2.0.0",
                },
                "type": "RegisteredResources",
            },
        },
        "resourcePackages": [
            {
                "name": "RegisteredResources",
                "type": "RegisteredResources",
                "items": [
                    {
                        "name": theme_filename,
                        "path": theme_filename,
                        "type": "CustomTheme",
                    }
                ],
            }
        ],
        "settings": {
            "useStylableVisualContainerHeader": True,
            "defaultDrillFilterOtherVisuals": True,
        },
    }
    report_json_path.write_text(json.dumps(report_payload, indent=2) + "\n", encoding="utf-8")
    written.append(report_json_path)

    # 5. definition/pages/pages.json
    page_ids = [p.page_id for p in report_spec.pages]
    pages_meta_path = pages_root / "pages.json"
    pages_meta_payload = {
        "$schema": SCHEMA_PAGES,
        "pageOrder": page_ids,
        "activePageName": page_ids[0],
    }
    pages_meta_path.write_text(json.dumps(pages_meta_payload, indent=2) + "\n", encoding="utf-8")
    written.append(pages_meta_path)

    # 6. Individual pages and visual containers
    for page in report_spec.pages:
        page_dir = pages_root / page.page_id
        visuals_dir = page_dir / "visuals"
        visuals_dir.mkdir(parents=True, exist_ok=True)

        page_json_path = page_dir / "page.json"
        page_payload = {
            "$schema": SCHEMA_PAGE,
            "name": page.page_id,
            "displayName": page.title,
            "width": report_spec.canvas_width,
            "height": report_spec.canvas_height,
            "displayOption": "FitToPage",
        }
        page_json_path.write_text(json.dumps(page_payload, indent=2) + "\n", encoding="utf-8")
        written.append(page_json_path)

        z_counter = 0
        tab_counter = 0

        # Emit slicers neatly along the reserved top slicer strip (y=84..122, zero overlap)
        if page.slicers:
            gutter = 16
            usable_w = report_spec.canvas_width - 2 * gutter
            n_slicers = len(page.slicers)
            slot_w = max(120, (usable_w - gutter * (n_slicers - 1)) // n_slicers)
            for s_idx, slicer in enumerate(page.slicers, start=1):
                vis_id = f"{page.page_id}_slicer_{s_idx:02d}"
                vis_dir = visuals_dir / vis_id
                vis_dir.mkdir(parents=True, exist_ok=True)
                slicer_x = gutter + (s_idx - 1) * (slot_w + gutter)
                slicer_payload = _build_slicer_visual_json(
                    slicer,
                    visual_id=vis_id,
                    x=slicer_x,
                    y=84,
                    width=slot_w,
                    height=38,
                    z_index=z_counter,
                    tab_order=tab_counter,
                )
                vis_path = vis_dir / "visual.json"
                vis_path.write_text(json.dumps(slicer_payload, indent=2) + "\n", encoding="utf-8")
                written.append(vis_path)
                z_counter += 100
                tab_counter += 100

        for visual in page.visuals:
            vis_dir = visuals_dir / visual.visual_id
            vis_dir.mkdir(parents=True, exist_ok=True)
            vis_payload = _build_visual_json(
                visual,
                measure_map,
                z_index=z_counter,
                tab_order=tab_counter,
            )
            vis_path = vis_dir / "visual.json"
            vis_path.write_text(json.dumps(vis_payload, indent=2) + "\n", encoding="utf-8")
            written.append(vis_path)
            z_counter += 100
            tab_counter += 100

    return written
