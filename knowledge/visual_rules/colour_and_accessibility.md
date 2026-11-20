# Visual Design Rules: Colour Palette and Accessibility

## Colorblind-Safe Palettes
- Always default to **colorblind-safe categorical palettes** such as **Okabe-Ito** (`#0072B2`, `#E69F00`, `#009E73`, `#56B4E9`, `#D55E00`, `#CC79A7`) or **Blue-Orange** diverging schemes.
- Never use raw red-green pairs (`#FF0000` vs `#00FF00`) as the sole differentiator for positive vs negative variance; pair blue/orange or teal/vermilion with directional arrows (`▲`/`▼`) or explicit `+`/`-` signs.
- Limit categorical colors in a single chart to **at most 4–5 distinct hues**, grouping remaining minor categories into neutral gray (`#94A3B8`) as "Other".

## Contrast and Typography Accessibility (WCAG AA)
- Enforce **WCAG 2.1 AA contrast ratios**: at least **4.5:1** for normal text (axis labels, card values, table cells) and at least **3.0:1** for large headings and graphical chart elements (bars, lines, focus borders).
- Use clean sans-serif typography (Segoe UI / Inter / DIN), minimum 10–11 pt axis labels, 12–14 pt visual titles, and clear plain-language titles that state the takeaway and units (e.g., "Monthly Revenue (USD)").

## Screen-Reader and Alt-Text Readiness
- Provide concise descriptive **alt text** on every PBIR visual summarizing what metric, dimension, and time grain the chart displays.
- Avoid encoding information with color alone: combine color with direct data labels, line styles, or tooltips.

