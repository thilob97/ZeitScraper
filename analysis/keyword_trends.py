"""Keyword trend analysis for zeit.de articles (March 2024 - July 2026).

Produces /opt/data/ZeitScraper/analysis/keyword_trends.json with:
  - overall_top_keywords: most frequent keywords across the full corpus
  - monthly_trends: top-N keywords per month with counts
  - rising_falling: keywords whose share grew / shrank most between the first
    and last complete month
  - category_keywords: top keywords per broad category bucket
  - summary: dataset metadata
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

DATA = Path("/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet")
OUT = Path("/opt/data/ZeitScraper/analysis/keyword_trends.json")
TOP_OVERALL = 50
TOP_PER_MONTH = 15
MONTHLY_TOP_FOR_TREND = 200  # track top-N keywords to compute rise/fall

df = pd.read_parquet(DATA)
df["Published"] = pd.to_datetime(df["Published"], utc=True)
# Month period as YYYY-MM
df["month"] = df["Published"].dt.strftime("%Y-%m")

# Split keywords: comma-separated, strip whitespace, drop empties, lowercase for aggregation
def split_kw(s: str):
    parts = re.split(r"\s*,\s*", s)
    out = []
    for p in parts:
        p = p.strip()
        if p:
            out.append(p)
    return out

print("Exploding keywords...")
kw_series = df["KeyWords"].map(split_kw)
df_kw = pd.DataFrame({
    "month": df["month"].values,
    "category": df["Category"].values,
    "keyword": kw_series,
})
df_kw = df_kw.explode("keyword", ignore_index=True)
df_kw["keyword"] = df_kw["keyword"].astype(str)
df_kw = df_kw[df_kw["keyword"].str.len() > 0]

print(f"Exploded rows: {len(df_kw):,}")
print("Computing overall top keywords...")
overall = Counter(df_kw["keyword"]).most_common(TOP_OVERALL)
overall_top = [{"keyword": k, "count": int(c)} for k, c in overall]

print("Computing monthly trends...")
# Per-month totals (article count = unique articles per month, for share)
monthly_article_counts = df.groupby("month").size().to_dict()
monthly_kw_counts = df_kw.groupby(["month", "keyword"]).size().reset_index(name="count")

months = sorted(monthly_article_counts.keys())
monthly_trends = {}
for m in months:
    sub = (monthly_kw_counts[monthly_kw_counts["month"] == m]
           .sort_values("count", ascending=False)
           .head(TOP_PER_MONTH))
    total = monthly_article_counts[m]
    entries = []
    for _, r in sub.iterrows():
        entries.append({"keyword": str(r["keyword"]), "count": int(r["count"]), "share": round(float(r["count"]) / total, 4)})
    monthly_trends[m] = entries

print("Computing rising/falling keywords...")
# Track top-N keywords per month, then compare first vs last complete month share
first_month = months[0]
last_month = months[-1]
# Use first 3 and last 3 months averaged to smooth noise
first_months = months[:3]
last_months = months[-3:] if len(months) >= 6 else months[-2:]

# Global top keywords by total count (restrict trend analysis to these to avoid noise)
top_keywords_for_trend = [k for k, _ in Counter(df_kw["keyword"]).most_common(MONTHLY_TOP_FOR_TREND)]

def month_share(keyword, mlist):
    total = sum(monthly_article_counts[m] for m in mlist)
    if total == 0:
        return 0.0
    cnt = int(monthly_kw_counts[(monthly_kw_counts["month"].isin(mlist)) & (monthly_kw_counts["keyword"] == keyword)]["count"].sum())
    return cnt / total

trend_rows = []
for kw in top_keywords_for_trend:
    s_first = month_share(kw, first_months)
    s_last = month_share(kw, last_months)
    delta = s_last - s_first
    trend_rows.append({"keyword": kw, "share_first": round(s_first, 5), "share_last": round(s_last, 5), "delta": round(delta, 5), "abs_delta": round(abs(delta), 5)})

trend_rows.sort(key=lambda r: r["abs_delta"], reverse=True)
rising = sorted([r for r in trend_rows if r["delta"] > 0], key=lambda r: r["delta"], reverse=True)[:20]
falling = sorted([r for r in trend_rows if r["delta"] < 0], key=lambda r: r["delta"])[:20]

print("Computing category keywords...")
# Category is very granular; group by top-level category (first part of Category, which often begins with broad theme)
df_kw["category"] = df_kw["category"].astype(str)
# Use first token of category as a coarse bucket
df_kw["cat_bucket"] = df_kw["category"].str.split(",").str[0].str.strip()
# The most frequent broad buckets (coarse):
cat_counts = df_kw["cat_bucket"].value_counts().head(15)
category_keywords = {}
for cat in cat_counts.index:
    sub = df_kw[df_kw["cat_bucket"] == cat]
    top = Counter(sub["keyword"]).most_common(10)
    category_keywords[cat] = [{"keyword": k, "count": int(c)} for k, c in top]

summary = {
    "total_articles": int(len(df)),
    "total_keyword_instances": int(len(df_kw)),
    "unique_keywords": int(df_kw["keyword"].nunique()),
    "date_min": str(df["Published"].min()),
    "date_max": str(df["Published"].max()),
    "months_covered": len(months),
    "first_month": first_month,
    "last_month": last_month,
}

result = {
    "summary": summary,
    "overall_top_keywords": overall_top,
    "monthly_trends": monthly_trends,
    "rising_keywords": rising,
    "falling_keywords": falling,
    "category_keywords": category_keywords,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2))
print(f"Wrote {OUT}")
print(json.dumps(summary, ensure_ascii=False, indent=2))