# Part 2 – Automation (Summary & Visualizations)

This notebook cell generates summary stats, simple visuals, and **3 actionable insights** from the dataset.

## Files generated (outputs_part2/)
- `summary_stats.csv`
- `missingness_by_column.csv`
- `category_breakdown.csv` (if applicable)
- `chart_missingness_top10.png`
- `chart_hist_<metric>.png`
- `chart_timeseries_<metric>.png` (if applicable)
- `chart_bar_<category>_<metric>.png` (if applicable)
- `insights.txt`

## Notes
- Charts: **matplotlib** only, one chart per figure, no explicit colors.
- Auto-detection: date column (regex), numeric metric (amount-like or highest variance), category (2–10 uniques).
