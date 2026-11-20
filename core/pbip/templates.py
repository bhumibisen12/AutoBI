"""Official Microsoft PBIP/PBIR/TMDL schema URIs, Draft-07 validation schemas, and theme builder."""

from __future__ import annotations

from typing import Any

from core.schemas.report_spec import ThemeSpec

SCHEMA_PBIP = (
    "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json"
)
SCHEMA_PBISM = (
    "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/"
    "definitionProperties/1.0.0/schema.json"
)
SCHEMA_PBIR = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definitionProperties/2.0.0/schema.json"
)
SCHEMA_VERSION = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/versionMetadata/1.0.0/schema.json"
)
SCHEMA_REPORT = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/report/3.1.0/schema.json"
)
SCHEMA_PAGES = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/pagesMetadata/1.0.0/schema.json"
)
SCHEMA_PAGE = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/page/2.0.0/schema.json"
)
SCHEMA_VISUAL = (
    "https://developer.microsoft.com/json-schemas/fabric/item/report/"
    "definition/visualContainer/2.4.0/schema.json"
)

# Local Draft-07 JSON schemas modeled directly on Microsoft's official schema contracts
# so validation runs reliably offline without live network fetches.
PBIP_FILE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "version", "artifacts"],
    "properties": {
        "$schema": {"const": SCHEMA_PBIP},
        "version": {"type": "string", "minLength": 1},
        "artifacts": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["report"],
                "properties": {
                    "report": {
                        "type": "object",
                        "required": ["path"],
                        "properties": {"path": {"type": "string", "minLength": 1}},
                    }
                },
            },
        },
        "settings": {"type": "object"},
    },
}

PBISM_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "version"],
    "properties": {
        "$schema": {"const": SCHEMA_PBISM},
        "version": {"type": "string", "minLength": 1},
        "settings": {"type": "object"},
    },
}

PBIR_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "version", "datasetReference"],
    "properties": {
        "$schema": {"const": SCHEMA_PBIR},
        "version": {"type": "string", "minLength": 1},
        "datasetReference": {
            "type": "object",
            "required": ["byPath"],
            "properties": {
                "byPath": {
                    "type": "object",
                    "required": ["path"],
                    "properties": {"path": {"type": "string", "minLength": 1}},
                }
            },
        },
    },
}

VERSION_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "buildVersion", "reportVersion"],
    "properties": {
        "$schema": {"const": SCHEMA_VERSION},
        "buildVersion": {"type": "string", "minLength": 1},
        "reportVersion": {"type": "string", "minLength": 1},
    },
}

REPORT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "themeCollection"],
    "properties": {
        "$schema": {"const": SCHEMA_REPORT},
        "themeCollection": {
            "type": "object",
            "required": ["baseTheme"],
            "properties": {
                "baseTheme": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "type": {"type": "string", "enum": ["SharedResources"]},
                        "reportVersionAtImport": {"type": "object"},
                    },
                },
                "customTheme": {
                    "type": "object",
                    "required": ["name", "type"],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "type": {"type": "string", "enum": ["RegisteredResources"]},
                        "reportVersionAtImport": {"type": "object"},
                    },
                },
            },
        },
        "resourcePackages": {"type": "array"},
        "settings": {"type": "object"},
    },
}

PAGES_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "pageOrder", "activePageName"],
    "properties": {
        "$schema": {"const": SCHEMA_PAGES},
        "pageOrder": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "activePageName": {"type": "string", "minLength": 1},
    },
}

PAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "name", "displayName", "width", "height", "displayOption"],
    "properties": {
        "$schema": {"const": SCHEMA_PAGE},
        "name": {"type": "string", "minLength": 1},
        "displayName": {"type": "string", "minLength": 1},
        "width": {"type": "integer", "minimum": 320},
        "height": {"type": "integer", "minimum": 240},
        "displayOption": {"type": "string", "enum": ["FitToPage", "FitToWidth", "ActualSize"]},
    },
}

VISUAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["$schema", "name", "position", "visual"],
    "properties": {
        "$schema": {"const": SCHEMA_VISUAL},
        "name": {"type": "string", "minLength": 1},
        "position": {
            "type": "object",
            "required": ["x", "y", "z", "height", "width"],
            "properties": {
                "x": {"type": "number", "minimum": 0},
                "y": {"type": "number", "minimum": 0},
                "z": {"type": "integer", "minimum": 0},
                "height": {"type": "number", "exclusiveMinimum": 0},
                "width": {"type": "number", "exclusiveMinimum": 0},
                "tabOrder": {"type": "integer", "minimum": 0},
            },
        },
        "visual": {
            "type": "object",
            "required": ["visualType", "query"],
            "properties": {
                "visualType": {"type": "string", "minLength": 1},
                "query": {
                    "type": "object",
                    "required": ["queryState"],
                    "properties": {"queryState": {"type": "object"}},
                },
                "visualContainerObjects": {"type": "object"},
                "objects": {"type": "object"},
            },
        },
    },
}

THEME_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["name", "dataColors", "background", "foreground"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "dataColors": {
            "type": "array",
            "minItems": 4,
            "items": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
        },
        "background": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
        "foreground": {"type": "string", "pattern": "^#[0-9A-Fa-f]{6}$"},
    },
}


def build_powerbi_theme_json(theme: ThemeSpec) -> dict[str, Any]:
    """Generate a Power BI custom theme JSON payload from AutoBI's ``ThemeSpec``."""
    return {
        "name": theme.name,
        "dataColors": list(theme.palette),
        "background": theme.background_color,
        "foreground": theme.text_color,
        "tableAccent": theme.palette[0],
        "good": "#009E73",
        "neutral": "#E69F00",
        "bad": "#D55E00",
        "textClasses": {
            "title": {
                "fontFace": theme.font_family,
                "fontSize": 13,
                "color": theme.text_color,
            },
            "callout": {
                "fontFace": theme.font_family,
                "fontSize": 22,
                "color": theme.palette[0],
            },
            "header": {
                "fontFace": theme.font_family,
                "fontSize": 11,
                "color": theme.text_color,
            },
            "label": {
                "fontFace": theme.font_family,
                "fontSize": 10,
                "color": theme.muted_text_color,
            },
        },
    }

