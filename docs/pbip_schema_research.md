# Power BI Project (PBIP / TMDL / PBIR) Format Research

This document summarizes the official Microsoft specifications and JSON schemas targeted by `core/pbip`.

## Official Documentation & Schema Sources

- **Power BI Desktop Projects (PBIP) Overview**: https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-overview
- **PBIP Semantic Model Folder (TMDL)**: https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-dataset
- **Tabular Model Definition Language (TMDL)**: https://learn.microsoft.com/en-us/analysis-services/tmdl/tmdl-overview
- **Power BI Enhanced Report Format (PBIR)**: https://learn.microsoft.com/en-us/power-bi/developer/projects/projects-report
- **Official Microsoft JSON Schemas Repository**: https://github.com/microsoft/json-schemas

## Target PBIP Folder Structure

```text
<ProjectName>/
├── data/                                               # Portable curated Parquet/CSV tables
│   ├── <table_1>.parquet
│   ├── <table_1>.csv
│   └── dim_date.parquet
├── pbip/
│   ├── <ProjectName>.pbip
│   ├── <ProjectName>.SemanticModel/
│   │   ├── definition.pbism
│   │   └── definition/
│   │       ├── database.tmdl
│   │       ├── model.tmdl
│   │       ├── expressions.tmdl
│   │       ├── relationships.tmdl
│   │       └── tables/
│   │           ├── <table_1>.tmdl
│   │           └── dim_date.tmdl
│   └── <ProjectName>.Report/
│       ├── definition.pbir
│       ├── StaticResources/
│       │   └── RegisteredResources/
│       │       └── autobi_theme.json
│       └── definition/
│           ├── version.json
│           ├── report.json
│           └── pages/
│               ├── pages.json
│               └── <page_id>/
│                   ├── page.json
│                   └── visuals/
│                       └── <visual_id>/
│                           └── visual.json
├── measures.dax
├── data_dictionary.xlsx
├── verification_report.json
├── verification_report.md
└── README.md
```

## Target JSON Schema Versions

| Artifact | `$schema` URI |
| --- | --- |
| `<ProjectName>.pbip` | `https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json` |
| `definition.pbism` | `https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json` |
| `definition.pbir` | `https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json` |
| `definition/version.json` | `https://developer.microsoft.com/json-schemas/fabric/item/report/definition/versionMetadata/1.0.0/schema.json` |
| `definition/report.json` | `https://developer.microsoft.com/json-schemas/fabric/item/report/definition/report/3.1.0/schema.json` |
| `definition/pages/pages.json` | `https://developer.microsoft.com/json-schemas/fabric/item/report/definition/pagesMetadata/1.0.0/schema.json` |
| `definition/pages/<id>/page.json` | `https://developer.microsoft.com/json-schemas/fabric/item/report/definition/page/2.0.0/schema.json` |
| `definition/pages/<id>/visuals/<id>/visual.json` | `https://developer.microsoft.com/json-schemas/fabric/item/report/definition/visualContainer/2.4.0/schema.json` |

## Known Schema Uncertainties Requiring Manual Desktop Verification

1. **Power Query M relative path resolution**: `File.Contents()` in Power BI Desktop requires an absolute filesystem path at refresh time. AutoBI parameterises the path through `DataFolderPath` in `expressions.tmdl` (default `"../data"`), which should be updated to the unzipped local `data` folder path via *Transform data -> Edit parameters*.
2. **`Parquet.Document` vs `Csv.Document`**: Older Power BI Desktop builds may prefer `Csv.Document` over `Parquet.Document`. Both `.parquet` and `.csv` copies are bundled under `data/`, and `PBIPBuildConfig.source_format` allows switching cleanly.
3. **PBIR visual container minor schema evolution (`2.4.0` vs newer `2.x`)**: Core `position`, `visualType`, and `query.queryState` bindings are well-established, whereas optional `visualContainerObjects` formatting sub-properties (e.g. `title`, `general.altText`, slicer mode) evolve across monthly Desktop updates and should be verified when opening in Power BI Desktop.
4. **Built-in `baseTheme` release tag**: `report.json` references a Power BI Desktop base theme tag (`CY24SU10`) alongside our self-contained custom theme `autobi_theme.json` under `StaticResources/RegisteredResources/`. Desktop may prompt to update the base theme version depending on installed release.
5. **Matrix (`pivotTable`) vs Table (`tableEx`) roles**: `tableEx` projects all columns and measures under `Values`, while `pivotTable` splits `Rows`, `Columns`, and `Values`; exact drill/subtotal formatting options should be verified manually in Desktop.

