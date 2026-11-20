# Sample Datasets (SYNTHETIC)

> **IMPORTANT:** Every dataset in this directory is **100% SYNTHETIC** and was generated solely for AutoBI development, testing, and demonstration. No real customer, employee, or patient records are included.

| File | Domain | Format | Tables / Sheets | Rows |
| --- | --- | --- | --- | --- |
| `SYNTHETIC_sales.csv` | Retail / B2B Sales | CSV (UTF-8) | 1 table | 36 |
| `SYNTHETIC_hr.xlsx` | Human Resources | Excel (`.xlsx`) | 2 sheets (`employees`, `departments`) | 30 (25 + 5) |
| `SYNTHETIC_healthcare.csv` | Healthcare Operations | CSV (UTF-8) | 1 table | 30 |

See `notebooks/01_sample_data_ingestion.ipynb` for a walkthrough that ingests all three datasets into the `raw` and `staging` layers and queries them with DuckDB.

