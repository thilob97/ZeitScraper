"""
Story Lifecycle analysis: half-life of news cycles for top keywords on zeit.de.
Outputs JSON: /opt/data/ZeitScraper/dashboard/lifecycle_data.json
"""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet")
OUT = Path("/opt/data/ZeitScraper/dashboard/lifecycle_data.json")
OUT.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    df = pd.read_parquet(DATA)
    # Normalize published date (drop tz to be safe)
    df["date"] = pd.to_datetime(df["Published"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date", "KeyWords", "Category"]).copy()
    # Keyword parsing: comma separated, strip whitespace, drop empties / very short noise
    bad = {"", "z+", "liveblog:", "liveblog"}
    kw_lists = []
    for s in df["KeyWords"].astype(str):
        parts = [p.strip() for p in s.split(",")]
        parts = [p for p in parts if p and p.lower() not in bad and len(p) <= 60]
        kw_lists.append(parts)
    df["kw_list"] = kw_lists

    # ---- explode to long form: one row per (article, keyword) ----
    long = df[["date", "Category", "kw_list"]].explode("kw_list")
    long = long.rename(columns={"kw_list": "keyword"})
    long = long.dropna(subset=["keyword"])
    long = long[long["keyword"].str.len() > 0]

    # Daily keyword counts
    daily = long.groupby(["keyword", "date"]).size().reset_index(name="count")
    overall_counts = long.groupby("keyword").size().sort_values(ascending=False)
    top50 = overall_counts.head(50).index.tolist()

    # Daily top-100 keywords (for "stays in top-100" measure)
    daily_rank = daily.sort_values(["date", "count"], ascending=[True, False])
    daily_top100: dict = {}
    for d, grp in daily.groupby("date"):
        top = grp.nlargest(100, "count")["keyword"].tolist()
        daily_top100[d] = set(top)

    # Per-keyword daily series for top50
    kw_series: dict = {}
    for kw in top50:
        s = daily[daily["keyword"] == kw].set_index("date")["count"].sort_index()
        # Reindex across full date range so gaps are 0
        full_idx = pd.date_range(s.index.min(), df["date"].max(), freq="D")
        s = s.reindex(full_idx, fill_value=0)
        kw_series[kw] = s

    # ---------------- 1. Half-lives ----------------
    half_lives: dict = {}
    for kw in top50:
        s = kw_series[kw]
        if s.empty:
            continue
        peak_count = int(s.max())
        peak_date = str(s.idxmax().date())
        first_date = s.index.min()
        # days active in top-100 after first appearance
        # Count days from first appearance until last non-zero day
        nonzero = s[s > 0]
        if nonzero.empty:
            continue
        last_date = nonzero.index.max()
        total_days_active = int((last_date - first_date).days) + 1
        # half-life: days after peak until count drops to <=50% of peak
        peak_idx = s.idxmax()
        peak_val = s.max()
        after = s.loc[peak_idx:]
        below = after[after <= 0.5 * peak_val]
        if not below.empty:
            half_life_days = int((below.index[0] - peak_idx).days)
        else:
            half_life_days = -1  # never dropped below half peak
        half_lives[kw] = {
            "peak_count": peak_count,
            "peak_date": peak_date,
            "half_life_days": half_life_days,
            "total_days_active": total_days_active,
        }

    # ---------------- 2. Lifecycle types ----------------
    lifecycle_types: dict = {}
    for kw in top50:
        s = kw_series[kw]
        if s.empty:
            continue
        peak_val = int(s.max())
        peak_idx = s.idxmax()
        nonzero = s[s > 0]
        first_date = nonzero.index.min()
        last_date = nonzero.index.max()
        days_active = int((last_date - first_date).days) + 1
        # Above 50% of peak window
        above_half = s[s >= 0.5 * peak_val]
        days_above_half = int((above_half.index.max() - above_half.index.min()).days) + 1 if not above_half.empty else 0

        # Detect multiple peaks: local maxima separated by dips below 30% of peak
        # Simple approach: find days above 0.5*peak, count distinct runs
        above = (s >= 0.5 * peak_val).astype(int)
        # find runs
        runs = []
        in_run = False
        start = None
        prev_idx = None
        for d, val in above.items():
            if val == 1 and not in_run:
                start = d
                in_run = True
            elif val == 0 and in_run:
                runs.append((start, prev_idx))
                in_run = False
            prev_idx = d
        if in_run:
            runs.append((start, prev_idx))
        # require at least 7-day gaps between runs for cyclical
        cyclical = False
        if len(runs) >= 2:
            gaps = []
            for i in range(1, len(runs)):
                gap = (runs[i][0] - runs[i - 1][1]).days
                gaps.append(gap)
            if any(g >= 7 for g in gaps):
                cyclical = True

        # dying: was big (peak in top-tier) and now gone (last 60 days < 5% peak)
        recent = s.tail(60)
        recent_max = int(recent.max())
        if days_active <= 7:
            ltype = "flash"
        elif cyclical:
            ltype = "cyclical"
        elif days_above_half > 30:
            ltype = "sustained"
        elif recent_max <= 0.05 * peak_val and peak_val > 0:
            ltype = "dying"
        else:
            # default: pick the dominant trait
            if days_above_half >= 14:
                ltype = "sustained"
            else:
                ltype = "flash"
        lifecycle_types[kw] = {
            "type": ltype,
            "peak_date": str(peak_idx.date()),
            "days_active": days_active,
        }

    # ---------------- 3 & 4. Longest / shortest lived ----------------
    lifespans = []
    for kw in top50:
        s = kw_series[kw]
        nonzero = s[s > 0]
        if nonzero.empty:
            continue
        days = int((nonzero.index.max() - nonzero.index.min()).days) + 1
        lifespans.append((kw, days, int(s.max())))
    lifespans.sort(key=lambda x: x[1], reverse=True)
    longest_lived = [{"keyword": k, "days_active": d, "peak_count": p} for k, d, p in lifespans[:20]]
    lifespans.sort(key=lambda x: (x[1], -x[2]))
    shortest_lived = [{"keyword": k, "peak_count": p, "days_active": d} for k, d, p in lifespans[:20]]

    # ---------------- 5. Comeback kings ----------------
    comebacks = []
    for kw in top50:
        s = kw_series[kw]
        peak_val = int(s.max())
        if peak_val < 5:
            continue
        # find runs of nonzero
        above = (s > 0).astype(int)
        runs = []
        in_run = False
        start = None
        prev_idx = None
        for d, val in above.items():
            if val == 1 and not in_run:
                start = d
                in_run = True
            elif val == 0 and in_run:
                runs.append((start, prev_idx))
                in_run = False
            prev_idx = d
        if in_run:
            runs.append((start, prev_idx))
        if len(runs) < 2:
            continue
        # find gap > 30 days between consecutive runs
        for i in range(1, len(runs)):
            gap = (runs[i][0] - runs[i - 1][1]).days
            if gap > 30:
                first_run_max = int(s.loc[runs[i - 1][0]:runs[i - 1][1]].max())
                second_run_max = int(s.loc[runs[i][0]:runs[i][1]].max())
                if first_run_max <= 0:
                    continue
                recovery_pct = round(100 * second_run_max / first_run_max, 1)
                if recovery_pct >= 50:
                    comebacks.append({
                        "keyword": kw,
                        "first_peak": str(s.loc[runs[i - 1][0]:runs[i - 1][1]].idxmax().date()),
                        "gap_days": gap,
                        "second_peak": str(s.loc[runs[i][0]:runs[i][1]].idxmax().date()),
                        "recovery_pct": recovery_pct,
                    })
                break  # only first qualifying gap
    comebacks.sort(key=lambda x: x["recovery_pct"], reverse=True)
    comebacks = comebacks[:20]

    # ---------------- 6. Monthly topic churn ----------------
    # top-100 keywords per month (by count within that month)
    long["month"] = long["date"].dt.to_period("M").astype(str)
    monthly_counts = long.groupby(["month", "keyword"]).size().reset_index(name="count")
    monthly_top100: dict = {}
    for m, grp in monthly_counts.groupby("month"):
        monthly_top100[m] = set(grp.nlargest(100, "count")["keyword"].tolist())
    months = sorted(monthly_top100.keys())
    monthly_churn: dict = {}
    for i, m in enumerate(months):
        cur = monthly_top100[m]
        if i == 0:
            new_topics = len(cur)
            lost_topics = 0
            total = len(cur)
        else:
            prev = monthly_top100[months[i - 1]]
            new_topics = len(cur - prev)
            lost_topics = len(prev - cur)
            total = len(cur)
        monthly_churn[m] = {
            "new_topics": new_topics,
            "lost_topics": lost_topics,
            "total": total,
        }

    # ---------------- 7. Category lifespan ----------------
    # Clean category names (strip Z+ prefix artifacts)
    def clean_cat(c: str) -> str:
        c = re.sub(r"Z\+.*?;\s*", "", c)
        return c.strip()

    long["cat_clean"] = long["Category"].apply(clean_cat)
    # For each category, find span (last - first article date) across whole dataset
    cat_lifespans: dict = defaultdict(list)
    cat_data = long.groupby("cat_clean")
    for cat, grp in cat_data:
        # Per-keyword lifespan within a category to measure "topic" lifespan
        # but the task asks "average lifespan of articles by category"
        # Compute span per keyword within that category, then average across keywords
        for kw, kgrp in grp.groupby("keyword"):
            if len(kgrp) < 3:
                continue
            span = (kgrp["date"].max() - kgrp["date"].min()).days + 1
            cat_lifespans[cat].append(span)
    cat_summary = []
    for cat, spans in cat_lifespans.items():
        if len(spans) < 5:
            continue
        cat_summary.append((cat, float(np.mean(spans)), float(np.median(spans)), len(spans)))
    cat_summary.sort(key=lambda x: x[1], reverse=True)
    category_lifespan = {}
    for cat, avg, med, n in cat_summary[:15]:
        category_lifespan[cat] = {
            "avg_lifespan": round(avg, 1),
            "median_lifespan": round(med, 1),
        }

    out = {
        "half_lives": half_lives,
        "lifecycle_types": lifecycle_types,
        "longest_lived": longest_lived,
        "shortest_lived": shortest_lived,
        "comebacks": comebacks,
        "monthly_churn": monthly_churn,
        "category_lifespan": category_lifespan,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Wrote {OUT}")
    print(f"half_lives: {len(half_lives)}")
    print(f"lifecycle_types: {len(lifecycle_types)}")
    types = Counter(v['type'] for v in lifecycle_types.values())
    print("Lifecycle types:", dict(types))
    print(f"longest_lived: {len(longest_lived)} top: {longest_lived[0] if longest_lived else None}")
    print(f"shortest_lived: {len(shortest_lived)} top: {shortest_lived[0] if shortest_lived else None}")
    print(f"comebacks: {len(comebacks)} top: {comebacks[0] if comebacks else None}")
    print(f"monthly_churn months: {len(monthly_churn)}")
    print(f"category_lifespan: {len(category_lifespan)}")
    print("Categories:", list(category_lifespan.keys())[:15])


if __name__ == "__main__":
    main()