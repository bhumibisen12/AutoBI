"""Evaluation dataset builders and ground-truth schemas for AutoBI's 6-dataset eval suite.

All generated datasets are 100% SYNTHETIC and deterministic (fixed RNG seed) so evaluation
metrics and scale benchmarks (including the 200k-row mandi-price CSV) are reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from core.schemas.llm import ColumnRole

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"


@dataclass(frozen=True)
class EvalDatasetSpec:
    """Metadata, input file path, prompt, and ground-truth column roles for one eval dataset."""

    name: str
    domain: str
    path: Path
    prompt: str
    expected_roles: dict[str, dict[str, ColumnRole]]
    row_target: int | None = None


def _generate_synthetic_education_csv(dest: Path) -> Path:
    """Create a deterministic SYNTHETIC education performance dataset."""
    rng = np.random.default_rng(42)
    n = 120
    dates = pd.date_range("2024-01-15", periods=12, freq="MS")
    departments = ["Computer Science", "Mathematics", "Physics", "Economics", "Biology"]
    campuses = ["North", "South", "Online"]
    df = pd.DataFrame(
        {
            "enrollment_id": [f"ENR-{1000 + i}" for i in range(n)],
            "term_date": [dates[i % len(dates)].strftime("%Y-%m-%d") for i in range(n)],
            "department": [departments[i % len(departments)] for i in range(n)],
            "campus": [campuses[i % len(campuses)] for i in range(n)],
            "credits_attempted": rng.integers(9, 19, size=n),
            "tuition_usd": np.round(rng.uniform(2500.0, 9500.0, size=n), 2),
            "gpa": np.round(rng.uniform(2.1, 4.0, size=n), 2),
            "attendance_pct": np.round(rng.uniform(72.0, 99.5, size=n), 1),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def _generate_synthetic_finance_csv(dest: Path) -> Path:
    """Create a deterministic SYNTHETIC corporate finance ledger dataset."""
    rng = np.random.default_rng(77)
    n = 144
    dates = pd.date_range("2024-01-01", periods=12, freq="MS")
    cost_centers = ["Engineering", "Sales", "Marketing", "Operations", "G&A"]
    regions = ["AMER", "EMEA", "APAC"]
    categories = ["Payroll", "Cloud", "Travel", "Software", "Facilities"]
    budget = np.round(rng.uniform(15000.0, 120000.0, size=n), 2)
    actual = np.round(budget * rng.uniform(0.82, 1.15, size=n), 2)
    df = pd.DataFrame(
        {
            "journal_id": [f"JRN-{5000 + i}" for i in range(n)],
            "posting_date": [dates[i % len(dates)].strftime("%Y-%m-%d") for i in range(n)],
            "cost_center": [cost_centers[i % len(cost_centers)] for i in range(n)],
            "region": [regions[i % len(regions)] for i in range(n)],
            "expense_category": [categories[i % len(categories)] for i in range(n)],
            "budget_amount": budget,
            "actual_amount": actual,
            "variance_amount": np.round(actual - budget, 2),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def _generate_synthetic_mandi_200k_csv(dest: Path, *, rows: int = 200_000) -> Path:
    """Create a deterministic SYNTHETIC Indian mandi-price CSV sampled to ``rows`` rows."""
    if dest.is_file():
        # Reuse cached file if row count matches expected scale target.
        with dest.open("rb") as fh:
            line_count = sum(1 for _ in fh) - 1
        if line_count == rows:
            return dest

    rng = np.random.default_rng(2025)
    states = np.array(
        ["Maharashtra", "Uttar Pradesh", "Madhya Pradesh", "Karnataka", "Gujarat", "Punjab"]
    )
    districts = np.array(
        ["Nashik", "Pune", "Indore", "Agra", "Bengaluru", "Rajkot", "Ludhiana", "Nagpur"]
    )
    markets = np.array(
        ["Lasalgaon APMC", "Vashi Mandi", "Indore Mandi", "Azadpur", "Yeshwanthpur", "Gondal"]
    )
    commodities = np.array(["Onion", "Tomato", "Potato", "Wheat", "Soybean", "Cotton", "Rice"])
    date_pool = pd.date_range("2023-01-01", "2024-12-31", freq="D").strftime("%Y-%m-%d").to_numpy()

    modal_price = np.round(rng.uniform(1200.0, 6500.0, size=rows), 2)
    spread = np.round(rng.uniform(80.0, 450.0, size=rows), 2)
    min_price = np.maximum(200.0, np.round(modal_price - spread, 2))
    max_price = np.round(modal_price + spread, 2)
    arrivals = np.round(rng.uniform(5.0, 420.0, size=rows), 2)

    df = pd.DataFrame(
        {
            "lot_id": np.char.add("LOT-", np.arange(1, rows + 1).astype(str)),
            "arrival_date": rng.choice(date_pool, size=rows),
            "state": rng.choice(states, size=rows),
            "district": rng.choice(districts, size=rows),
            "market": rng.choice(markets, size=rows),
            "commodity": rng.choice(commodities, size=rows),
            "min_price": min_price,
            "max_price": max_price,
            "modal_price": modal_price,
            "arrivals_tonnes": arrivals,
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    return dest


def prepare_eval_datasets(
    work_dir: Path,
    *,
    mandi_rows: int = 200_000,
) -> list[EvalDatasetSpec]:
    """Prepare all 6 evaluation datasets (3 bundled + 3 generated synthetic, incl. 200k mandi)."""
    gen_dir = work_dir / "generated_synthetic_datasets"
    edu_path = _generate_synthetic_education_csv(gen_dir / "SYNTHETIC_education.csv")
    fin_path = _generate_synthetic_finance_csv(gen_dir / "SYNTHETIC_finance.csv")
    mandi_path = _generate_synthetic_mandi_200k_csv(
        gen_dir / f"SYNTHETIC_mandi_prices_{mandi_rows}.csv",
        rows=mandi_rows,
    )

    return [
        EvalDatasetSpec(
            name="sales",
            domain="sales",
            path=SAMPLE_DATA_DIR / "SYNTHETIC_sales.csv",
            prompt="Analyze monthly sales revenue, cost, and regional performance.",
            expected_roles={
                "synthetic_sales": {
                    "order_id": "id",
                    "order_date": "date",
                    "region": "dimension",
                    "channel": "dimension",
                    "category": "dimension",
                    "product": "dimension",
                    "quantity": "measure",
                    "unit_price": "measure",
                    "revenue": "measure",
                    "cost": "measure",
                }
            },
        ),
        EvalDatasetSpec(
            name="hr",
            domain="human_resources",
            path=SAMPLE_DATA_DIR / "SYNTHETIC_hr.xlsx",
            prompt="Analyze employee headcount, base salary, performance, and department budgets.",
            expected_roles={
                "synthetic_hr__employees": {
                    "employee_id": "id",
                    "department_id": "id",
                    "role_level": "dimension",
                    "hire_date": "date",
                    "base_salary_usd": "measure",
                    "performance_rating": "measure",
                    "attrition_flag": "dimension",
                    "work_mode": "dimension",
                },
                "synthetic_hr__departments": {
                    "department_id": "id",
                    "department_name": "dimension",
                    "cost_center": "dimension",
                    "headcount_budget": "measure",
                },
            },
        ),
        EvalDatasetSpec(
            name="healthcare",
            domain="healthcare",
            path=SAMPLE_DATA_DIR / "SYNTHETIC_healthcare.csv",
            prompt="Track patient visit volumes, billed amounts, and length of stay by department.",
            expected_roles={
                "synthetic_healthcare": {
                    "visit_id": "id",
                    "visit_date": "date",
                    "department": "dimension",
                    "visit_type": "dimension",
                    "age_group": "dimension",
                    "length_of_stay_days": "measure",
                    "billed_amount_usd": "measure",
                    "readmitted_30d": "dimension",
                    "satisfaction_score": "measure",
                }
            },
        ),
        EvalDatasetSpec(
            name="education",
            domain="education",
            path=edu_path,
            prompt="Monitor tuition revenue, GPA, attendance, and enrollments across departments.",
            expected_roles={
                "synthetic_education": {
                    "enrollment_id": "id",
                    "term_date": "date",
                    "department": "dimension",
                    "campus": "dimension",
                    "credits_attempted": "measure",
                    "tuition_usd": "measure",
                    "gpa": "measure",
                    "attendance_pct": "measure",
                }
            },
        ),
        EvalDatasetSpec(
            name="finance",
            domain="finance",
            path=fin_path,
            prompt="Compare actual spend vs budget variance by cost center and region over time.",
            expected_roles={
                "synthetic_finance": {
                    "journal_id": "id",
                    "posting_date": "date",
                    "cost_center": "dimension",
                    "region": "dimension",
                    "expense_category": "dimension",
                    "budget_amount": "measure",
                    "actual_amount": "measure",
                    "variance_amount": "measure",
                }
            },
        ),
        EvalDatasetSpec(
            name="mandi_prices_200k",
            domain="agriculture",
            path=mandi_path,
            prompt="Analyze commodity modal prices and arrival volumes across states and markets.",
            expected_roles={
                f"synthetic_mandi_prices_{mandi_rows}": {
                    "lot_id": "id",
                    "arrival_date": "date",
                    "state": "dimension",
                    "district": "dimension",
                    "market": "dimension",
                    "commodity": "dimension",
                    "min_price": "measure",
                    "max_price": "measure",
                    "modal_price": "measure",
                    "arrivals_tonnes": "measure",
                }
            },
            row_target=mandi_rows,
        ),
    ]
