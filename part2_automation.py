#!/usr/bin/env python3
"""
Part 2 – Automation (Summary & Visualizations)
Usage:
    python part2_automation.py --input path/to/clean.csv --outdir outputs_part2
Notes:
    - Charts: matplotlib only, one chart per figure, no explicit colors.
"""
import os, re, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DATE_LIKE_PAT  = re.compile(r"(date|fecha|created|updated|timestamp|time)", re.I)
CITY_LIKE_PAT  = re.compile(r"(city|ciudad|municipio|billing\\s*city|mailing\\s*city)", re.I)
AMOUNT_LIKE_PAT = re.compile(r"(amount|monto|importe|value|price|revenue|total)", re.I)

def find_first_matching_col(columns, pattern: re.Pattern, preferred_order=None):
    cols = list(columns)
    if preferred_order:
        for name in preferred_order:
            for c in cols:
                if c.strip().lower() == name.strip().lower():
                    return c
    for c in cols:
        if pattern.search(c):
            return c
    return None

def choose_numeric_column(df: pd.DataFrame):
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols:
        return None
    amount_candidates = [c for c in num_cols if AMOUNT_LIKE_PAT.search(c)]
    if amount_candidates:
        best = None; best_score = -1
        for c in amount_candidates:
            nonnull = df[c].notna().sum()
            var = np.nanvar(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
            score = nonnull + var
            if score > best_score:
                best = c; best_score = score
        return best
    variances = [(c, np.nanvar(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))) for c in num_cols]
    variances.sort(key=lambda x: x[1], reverse=True)
    return variances[0][0] if variances else None

def choose_category_column(df: pd.DataFrame, max_uniques=10):
    candidates = []
    for c in df.columns:
        if df[c].dtype == object:
            nun = df[c].nunique(dropna=True)
            if 2 <= nun <= max_uniques:
                candidates.append((c, nun, df[c].notna().sum()))
    city_like = [c for (c,_,_) in candidates if CITY_LIKE_PAT.search(c)]
    if city_like:
        return city_like[0]
    if candidates:
        candidates.sort(key=lambda x: (-x[2], x[1]))
        return candidates[0][0]
    return None

def choose_date_column(df: pd.DataFrame):
    return find_first_matching_col(df.columns, DATE_LIKE_PAT, preferred_order=["Date","Created Date"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to cleaned CSV")
    ap.add_argument("--outdir", default="outputs_part2", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.input)

    summary_path = os.path.join(args.outdir, "summary_stats.csv")
    missing_path = os.path.join(args.outdir, "missingness_by_column.csv")
    cat_summary_path = os.path.join(args.outdir, "category_breakdown.csv")

    desc = df.describe(include=[np.number]).T
    desc.to_csv(summary_path)

    miss = df.isna().mean().sort_values(ascending=False).rename("missing_rate").to_frame()
    miss.to_csv(missing_path)

    date_col = choose_date_column(df)
    num_col  = choose_numeric_column(df)
    cat_col  = choose_category_column(df)

    # Missingness chart
    try:
        top_miss = miss.head(10)
        plt.figure()
        top_miss["missing_rate"].plot(kind="bar")
        plt.title("Top 10 Columns by Missingness")
        plt.ylabel("Missing rate")
        plt.tight_layout()
        plt.savefig(os.path.join(args.outdir, "chart_missingness_top10.png"), dpi=160)
        plt.close()
    except Exception:
        pass

    # Histogram
    if num_col is not None:
        try:
            series = pd.to_numeric(df[num_col], errors="coerce").dropna()
            if series.shape[0] > 0:
                plt.figure()
                plt.hist(series, bins=20)
                plt.title(f"Distribution of {num_col}")
                plt.xlabel(num_col); plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, f"chart_hist_{re.sub('[^A-Za-z0-9_]+','_',num_col)}.png"), dpi=160)
                plt.close()
        except Exception:
            pass

    # Time series
    if date_col is not None and num_col is not None:
        try:
            ts = df[[date_col, num_col]].copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col, num_col])
            if ts.shape[0] > 0:
                ts_grouped = ts.groupby(ts[date_col].dt.date)[num_col].mean()
                if ts_grouped.shape[0] >= 2:
                    plt.figure()
                    ts_grouped.plot()
                    plt.title(f"{num_col} over time (daily mean)")
                    plt.xlabel("Date"); plt.ylabel(num_col)
                    plt.tight_layout()
                    plt.savefig(os.path.join(args.outdir, f"chart_timeseries_{re.sub('[^A-Za-z0-9_]+','_',num_col)}.png"), dpi=160)
                    plt.close()
        except Exception:
            pass

    # Category breakdown
    if cat_col is not None and num_col is not None:
        try:
            grp = df[[cat_col, num_col]].copy()
            grp[num_col] = pd.to_numeric(grp[num_col], errors="coerce")
            agg = grp.groupby(cat_col)[num_col].mean().sort_values(ascending=False).dropna()
            if agg.shape[0] >= 2:
                agg.to_csv(cat_summary_path, header=["mean_value"])
                plt.figure()
                agg.plot(kind="bar")
                plt.title(f"Average {num_col} by {cat_col}")
                plt.xlabel(cat_col); plt.ylabel(f"Avg {num_col}")
                plt.tight_layout()
                plt.savefig(os.path.join(args.outdir, f"chart_bar_{re.sub('[^A-Za-z0-9_]+','_',cat_col)}_{re.sub('[^A-Za-z0-9_]+','_',num_col)}.png"), dpi=160)
                plt.close()
        except Exception:
            pass

    # Insights
    insights = []
    try:
        if not miss.empty:
            top_col, top_rate = miss["missing_rate"].index[0], float(miss["missing_rate"].iloc[0])
            insights.append(f"{top_col} has the highest missingness at {top_rate:.1%}. Prioritize capture/validation to reduce data loss.")
    except Exception:
        pass

    try:
        if date_col is not None and num_col is not None:
            ts = df[[date_col, num_col]].copy()
            ts[date_col] = pd.to_datetime(ts[date_col], errors="coerce")
            ts = ts.dropna(subset=[date_col, num_col])
            ts_grouped = ts.groupby(ts[date_col].dt.date)[num_col].mean()
            if ts_grouped.shape[0] >= 2:
                y = ts_grouped.values.astype(float)
                x = np.arange(len(y), dtype=float)
                slope = ((x - x.mean())*(y - y.mean())).sum() / ((x - x.mean())**2).sum()
                direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "flat"
                insights.append(f"{num_col} shows a {direction} trend over time (slope={slope:.4f}). Investigate seasonality and drivers.")
    except Exception:
        pass

    try:
        if cat_col is not None and num_col is not None:
            grp2 = df[[cat_col, num_col]].copy()
            grp2[num_col] = pd.to_numeric(grp2[num_col], errors="coerce")
            agg2 = grp2.groupby(cat_col)[num_col].mean().sort_values(ascending=False).dropna()
            if agg2.shape[0] >= 1:
                top_cat, top_val = str(agg2.index[0]), float(agg2.iloc[0])
                insights.append(f"Segment '{top_cat}' has the highest average {num_col} ({top_val:.2f}). Prioritize this segment for targeted actions.")
    except Exception:
        pass

    while len(insights) < 3:
        insights.append("Improve data quality and standardization (validations, unique keys, regular ETL) to increase reliability of analytics.")

    insights_path = os.path.join(args.outdir, "insights.txt")
    with open(insights_path, "w", encoding="utf-8") as f:
        for i, line in enumerate(insights, 1):
            f.write(f"{i}. {line}\\n")

if __name__ == "__main__":
    main()
