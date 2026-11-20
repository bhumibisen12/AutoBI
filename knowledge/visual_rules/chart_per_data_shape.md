# Visual Design Rules: Chart Selection per Data Shape

## Time Series Trends
- Use a **Line Chart** or **Area Chart** whenever the X-axis is a continuous date or time dimension (`dim_date[date]`, `year_month`, `quarter`).
- Limit a single line chart to at most 4–5 series; use small multiples or top-N grouping if more categories exist.
- Never use a pie chart or unordered bar chart to display trends over time.

## Categorical Comparison
- Use a **Clustered Bar Chart (horizontal)** for comparing magnitudes across nominal categories, especially when category labels are long or there are 6–15 categories.
- Sort bars in descending order by measure value so executive readers spot leaders and laggards immediately.
- Use a **Clustered Column Chart (vertical)** only when there are few short labels (<= 6) or ordinal buckets (such as age groups or quarters).

## Part-to-Whole Composition
- Use a **Donut Chart** or **100% Stacked Bar Chart** ONLY when showing proportions of a single positive measure across at most 5 categories.
- Never use pie or donut charts when there are more than 5 slices or negative values; switch to a sorted horizontal bar chart instead.

## Correlation and Two-Measure Relationships
- Use a **Scatter Plot** when comparing two continuous numeric measures across entities (for example, Cost vs Revenue or Length of Stay vs Billed Amount), optionally encoding a third measure as bubble size.

## Headline KPIs and Detailed Lookup
- Use **KPI Cards** for 3–5 headline scalar metrics (such as Total Revenue, Gross Margin %, Attrition Rate, 30-Day Readmission Rate) placed along the top banner.
- Use a **Table** or **Matrix** when users need exact multi-column numbers, rankings, or row-level detail with conditional formatting.

