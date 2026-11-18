# AutoBI

AutoBI generates Power BI projects (PBIP: TMDL semantic model + PBIR report) with verified DAX measures, a data dictionary, a verification report, and an insights summary from uploaded datasets and natural-language prompts.

## Quickstart

1. Copy `.env.example` to `.env` and configure your environment variables.
2. Run tests: `python scripts/test.py` (or `pytest`).
3. Run linter: `python scripts/lint.py` (or `ruff check .`).
4. Launch the UI: `python scripts/run.py` (or `streamlit run app/main.py`).

