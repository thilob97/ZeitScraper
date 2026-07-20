"""
Category peaks analysis for zeit.de articles.

For each meaningful news category/section we compute:
  * total article count
  * peak day (date with most articles) and count
  * peak week (ISO week with most articles) and count
  * peak month (YYYY-MM with most articles) and count
  * peak hour-of-day (0..23 with most articles) and count
  * peak weekday (Monday..Sunday with most articles) and count

Results are written to analysis/category_peaks.json as a list of dicts,
one per (category, depth) — using both the raw zeit.de "Category" field
(normalised) and the top-level URL section ("news", "politik", ...).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "data" / "processed" / "articles_consolidated.parquet"
OUT = HERE / "category_peaks.json"


def normalise_category(raw: str) -> str | None:
    """Strip the 'Z+ (abopflichtiger Inhalt);\\n ...\\n' paywall prefix and whitespace."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = re.sub(r"^Z\+\s*\(abopflichtiger Inhalt\);?\s*\\n\s*\\n\s*", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def add_peaks(rows: list[dict], grp: pd.DataFrame, label_key: str, label_val: str) -> None:
    if grp.empty:
        return
    n = len(grp)

    # Peak day
    day_counts = grp["Published_Date"].value_counts()
    peak_day = str(day_counts.index[0])
    peak_day_count = int(day_counts.iloc[0])

    # Peak ISO week (year-week)
    iso_week = grp["Published"].dt.isocalendar()
    week_label = iso_week["year"].astype(str) + "-W" + iso_week["week"].astype(str).str.zfill(2)
    week_counts = week_label.value_counts()
    peak_week = str(week_counts.index[0])
    peak_week_count = int(week_counts.iloc[0])

    # Peak month (YYYY-MM already in Published_Month)
    month_counts = grp["Published_Month"].value_counts()
    peak_month = str(month_counts.index[0])
    peak_month_count = int(month_counts.iloc[0])

    # Peak hour-of-day
    hour_counts = grp["Published_Hour"].value_counts()
    peak_hour = int(hour_counts.index[0])
    peak_hour_count = int(hour_counts.iloc[0])

    # Peak weekday (order Mon..Sun)
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd = grp["Published_Weekday"].dropna()
    wd_counts = wd.value_counts()
    peak_weekday = str(wd_counts.index[0])
    peak_weekday_count = int(wd_counts.iloc[0])

    date_min = str(grp["Published"].min().date())
    date_max = str(grp["Published"].max().date())

    rows.append({
        "source": label_key,
        "category": label_val,
        "total_articles": n,
        "first_published": date_min,
        "last_published": date_max,
        "peak_day": peak_day,
        "peak_day_count": peak_day_count,
        "peak_week": peak_week,
        "peak_week_count": peak_week_count,
        "peak_month": peak_month,
        "peak_month_count": peak_month_count,
        "peak_hour": peak_hour,
        "peak_hour_count": peak_hour_count,
        "peak_weekday": peak_weekday,
        "peak_weekday_count": peak_weekday_count,
    })


def main() -> None:
    df = pd.read_parquet(SRC)

    # Normalised category field
    df["Category_clean"] = df["Category"].map(normalise_category)

    # Top-level URL section
    df["Section"] = df["Link"].str.extract(r"zeit\.de/([^/]+)/")

    # Drop rows with no publish date
    df = df.dropna(subset=["Published"]).copy()

    rows: list[dict] = []

    # --- Categories: keep those with >= 100 articles (top meaningful ones) ---
    cat_counts = df["Category_clean"].value_counts()
    keep_cats = cat_counts[cat_counts >= 100].index.tolist()
    for cat in keep_cats:
        sub = df[df["Category_clean"] == cat]
        add_peaks(rows, sub, "category", cat)

    # --- Sections (top-level URL path) ---
    sec_counts = df["Section"].value_counts()
    keep_secs = sec_counts[sec_counts >= 50].index.tolist()
    for sec in keep_secs:
        sub = df[df["Section"] == sec]
        add_peaks(rows, sub, "section", sec)

    # Sort by source then total desc
    rows.sort(key=lambda r: (r["source"], -r["total_articles"]))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote {len(rows)} peak records to {OUT}")
    # Quick sanity print
    print(f"  categories with peaks: {sum(1 for r in rows if r['source']=='category')}")
    print(f"  sections with peaks:    {sum(1 for r in rows if r['source']=='section')}")


if __name__ == "__main__":
    main()