"""
Story Lifecycle analysis: half-life of news cycles for top keywords on zeit.de.

Refined approach:
- Top-50 keywords (the broadest high-volume topics) used for half-life + lifecycle types.
- For "longest-lived" / "shortest-lived" we expand to top-500 keywords so we capture
  both evergreen mainstream topics (longest) and short-lived flash topics (shortest).
- Cyclical detection uses peak-pair analysis (peaks >= 50% of max separated by a
  deep trough), not just zero-gaps (which fire on every recurring topic).
- Category lifespan collapses the messy zeit.de Category column into a canonical
  category (sport, politics, news, crime, weather, ...).
"""
import json
import re
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

DATA = "/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet"
OUT = "/opt/data/ZeitScraper/dashboard/lifecycle_data.json"
from pathlib import Path
Path(OUT).parent.mkdir(parents=True, exist_ok=True)

# Rubric / section tags that should be excluded from "topic" analysis
RUBRIC = {
    "news",
    "politik",
    "sport",
    "kultur",
    "wirtschaft",
    "gesellschaft",
    "leute",
    "wissenschaft",
    "reise",
    "auto",
    "digital",
    "z+",
    "z+ (abopflichtiger inhalt)",
    "deutsche presse-agentur",
    "liveblog:",
    "liveblog",
    "dpa",
    "ergänzend: ahda sport",
}


def is_topic(kw: str) -> bool:
    kw = kw.strip()
    if not kw:
        return False
    if len(kw) > 60:
        return False
    if kw.lower() in RUBRIC:
        return False
    if kw.lower().startswith("z+"):
        return False
    if "spieltag" in kw.lower():
        return False
    if kw.lower().startswith("liveblog"):
        return False
    # pure numbers
    if re.fullmatch(r"[\d.\-]+", kw):
        return False
    return True


def clean_category(c: str) -> str:
    """Collapse the messy zeit.de Category column into canonical categories."""
    if not c or not isinstance(c, str):
        return "Other"
    c = re.sub(r"Z\+.*?;\s*", "", c)
    c = re.sub(r"\s+", " ", c).strip()
    low = c.lower()
    # Match against canonical buckets
    if any(k in low for k in ["fußball", "bundesliga", "champions league", "handball", "sport", "liga"]):
        return "Sport"
    if any(k in low for k in ["politik", "bundestag", "landtag", "bundestagswahl", "regierung", "wahl", "europa", "eu-", "europa"]):
        return "Politik"
    if any(k in low for k in ["kriminalität", "polizei", "justiz", "staatsanwaltschaft", "gericht", "mord", "strafe", "verbrechen"]):
        return "Kriminalität/Justiz"
    if any(k in low for k in ["unfall", "verkehr", "brand", "feuerwehr", "notfall", "wetter", "sturm"]):
        return "Unfälle/Wetter/Verkehr"
    if any(k in low for k in ["ukraine", "russland", "israel", "gaza", "nahost", "krieg", "militär", "nato", "korea", "iran", "syrien"]):
        return "Ausland/Konflikt"
    if any(k in low for k in ["wirtschaft", "börse", "finanzen", "unternehmen", "konsum"]):
        return "Wirtschaft"
    if any(k in low for k in ["kultur", "kunst", "musik", "literatur", "film", "theater", "ausstellung"]):
        return "Kultur"
    if any(k in low for k in ["wissenschaft", "forschung", "gesundheit", "medizin", "krankheit", "impfung"]):
        return "Wissenschaft/Gesundheit"
    if any(k in low for k in ["tier", "hund", "katze"]):
        return "Tiere"
    if any(k in low for k in [" migration", "migration", "flüchtling", "asyl"]):
        return "Migration"
    if any(k in low for k in ["klima", "umwelt", "energie", "co2"]):
        return "Klima/Umwelt"
    if any(k in low for k in ["schule", "bildung", "uni", "studium", "lehrer"]):
        return "Bildung"
    if any(k in low for k in ["medien", "internet", "digital", "social media", "ki ", "künstliche intelligenz"]):
        return "Medien/Digital"
    return "Other/Regional"


def main() -> None:
    df = pd.read_parquet(DATA)
    df["date"] = pd.to_datetime(df["Published"]).dt.tz_localize(None).dt.normalize()
    df = df.dropna(subset=["date", "KeyWords"]).copy()

    kw_lists = []
    for s in df["KeyWords"].astype(str):
        parts = [p.strip() for p in s.split(",") if is_topic(p.strip())]
        kw_lists.append(parts)
    df["kw_list"] = kw_lists

    long = df[["date", "Category", "kw_list"]].explode("kw_list").rename(columns={"kw_list": "keyword"})
    long = long.dropna(subset=["keyword"])
    long = long[long["keyword"].str.len() > 0]

    # Daily counts per keyword
    daily = long.groupby(["keyword", "date"]).size().reset_index(name="count")
    overall = long.groupby("keyword").size().sort_values(ascending=False)

    top50 = overall.head(50).index.tolist()
    # Wider pools for longest-lived (evergreen topics) and shortest-lived/comebacks (flash topics)
    top500 = overall.head(500).index.tolist()
    # Pool with at least 15 mentions total, sorted by total volume — used for flash and comebacks
    pool_15plus = overall[overall >= 15].index.tolist()  # ~13k keywords
    # Reuse the same daily series helper for both pools

    # Full date range
    full_start = df["date"].min()
    full_end = df["date"].max()
    full_idx = pd.date_range(full_start, full_end, freq="D")

    def series_for(kw):
        s = daily[daily["keyword"] == kw].set_index("date")["count"].sort_index()
        return s.reindex(full_idx, fill_value=0)

    # ---------- 1. Half-lives (top-50) ----------
    half_lives = {}
    for kw in top50:
        s = series_for(kw)
        peak_count = int(s.max())
        if peak_count == 0:
            continue
        peak_idx = s.idxmax()
        peak_date = str(peak_idx.date())
        # total_days_active: days from first nonzero to last nonzero (inclusive)
        nonzero = s[s > 0]
        if nonzero.empty:
            continue
        first_date = nonzero.index.min()
        last_date = nonzero.index.max()
        total_days_active = int((last_date - first_date).days) + 1
        # half-life: days after peak until count first drops <=50% of peak
        after = s.loc[peak_idx:].iloc[1:]  # exclude peak day itself
        below = after[after <= 0.5 * peak_count]
        half_life_days = int((below.index[0] - peak_idx).days) if not below.empty else -1
        half_lives[kw] = {
            "peak_count": peak_count,
            "peak_date": peak_date,
            "half_life_days": half_life_days,
            "total_days_active": total_days_active,
        }

    # ---------- 2. Lifecycle types (top-50) ----------
    def find_peaks(s: pd.Series, threshold: float) -> list:
        """Return indices of local maxima >= threshold*max, separated by dips below."""
        max_v = s.max()
        if max_v == 0:
            return []
        thr = threshold * max_v
        # Find runs above thr
        above = (s >= thr).astype(int)
        runs = []
        in_run = False
        start = None
        prev = None
        for d, v in above.items():
            if v == 1 and not in_run:
                start = d
                in_run = True
            elif v == 0 and in_run:
                runs.append((start, prev))
                in_run = False
            prev = d
        if in_run:
            runs.append((start, prev))
        # For each run, find local max within that window
        peak_dates = []
        for r_start, r_end in runs:
            window = s.loc[r_start:r_end]
            peak_dates.append((window.idxmax(), int(window.max())))
        return peak_dates

    lifecycle_types = {}
    for kw in top50:
        s = series_for(kw)
        peak_val = int(s.max())
        if peak_val == 0:
            continue
        nonzero = s[s > 0]
        first_date = nonzero.index.min()
        last_date = nonzero.index.max()
        days_active = int((last_date - first_date).days) + 1
        peak_idx = s.idxmax()

        # Use distinct peak runs above 50% of max with >14-day trough gap (dipping
        # below 25% of peak) for cyclical detection. This avoids misclassifying
        # every recurring beat as cyclical.
        peaks_50 = find_peaks(s, 0.5)
        cyclical = False
        if len(peaks_50) >= 2:
            for i in range(1, len(peaks_50)):
                gap_days = (peaks_50[i][0] - peaks_50[i - 1][0]).days
                if gap_days >= 14:
                    # check the trough between the two peaks drops below 25% of peak_val
                    between = s.loc[peaks_50[i - 1][0]:peaks_50[i][0]]
                    if between.min() <= 0.25 * peak_val:
                        cyclical = True
                        break

        # days above 50% of peak (consecutive window around peak)
        above = (s >= 0.5 * peak_val).astype(int)
        # find the run containing the peak
        runs = []
        in_run = False
        start = None
        prev = None
        for d, v in above.items():
            if v == 1 and not in_run:
                start = d
                in_run = True
            elif v == 0 and in_run:
                runs.append((start, prev))
                in_run = False
            prev = d
        if in_run:
            runs.append((start, prev))
        days_above_half = 0
        for r_start, r_end in runs:
            if r_start <= peak_idx <= r_end:
                days_above_half = (r_end - r_start).days + 1
                break

        # Recent activity (last 60 days)
        recent_max = int(s.tail(60).max())

        # Classification priority
        if days_active <= 7:
            ltype = "flash"
        elif cyclical:
            ltype = "cyclical"
        elif days_above_half > 30:
            ltype = "sustained"
        elif recent_max <= 0.1 * peak_val:
            ltype = "dying"
        elif days_above_half >= 14:
            ltype = "sustained"
        else:
            ltype = "flash"
        lifecycle_types[kw] = {
            "type": ltype,
            "peak_date": str(peak_idx.date()),
            "days_active": days_active,
        }

    # ---------- 3. Longest-lived (top-20 from top-500) ----------
    lifespan_records = []
    for kw in top500:
        s = series_for(kw)
        nonzero = s[s > 0]
        if nonzero.empty:
            continue
        # Only consider keywords that had at least 30 mentions total (skip noise)
        if int(s.sum()) < 30:
            continue
        first_date = nonzero.index.min()
        last_date = nonzero.index.max()
        days_active = int((last_date - first_date).days) + 1
        # Require min peak of 5 to filter pure noise
        peak = int(s.max())
        if peak < 5:
            continue
        lifespan_records.append({"keyword": kw, "days_active": days_active, "peak_count": peak})
    lifespan_records.sort(key=lambda x: (-x["days_active"], -x["peak_count"]))
    longest_lived = lifespan_records[:20]

    # ---------- 4. Shortest-lived flash topics (top-20) ----------
    # Scan wider pool (>=15 mentions total) for keywords that spike then die.
    flash_records = []
    for kw in pool_15plus:
        s = series_for(kw)
        nonzero = s[s > 0]
        if nonzero.empty:
            continue
        peak = int(s.max())
        if peak < 5:
            continue
        first_date = nonzero.index.min()
        last_date = nonzero.index.max()
        days_active = int((last_date - first_date).days) + 1
        total = int(s.sum())
        concentration = peak / max(total, 1)
        # Flash criteria: short active span AND most volume concentrated in the peak day
        if days_active <= 60 and concentration >= 0.1:
            flash_records.append({"keyword": kw, "peak_count": peak, "days_active": days_active, "concentration": round(concentration, 3)})
    # Sort by: shortest days_active first, then highest peak (most notable flash)
    flash_records.sort(key=lambda x: (x["days_active"], -x["peak_count"]))
    shortest_lived = [{"keyword": r["keyword"], "peak_count": r["peak_count"], "days_active": r["days_active"]} for r in flash_records[:20]]

    # ---------- 5. Comeback kings ----------
    # Search the wider pool (>=15 mentions total) for topics that disappeared
    # >30 days then came back with >=50% of original peak.
    comebacks = []
    for kw in pool_15plus:
        s = series_for(kw)
        if int(s.max()) < 5:
            continue
        # find runs of nonzero
        above = (s > 0).astype(int)
        runs = []
        in_run = False
        start = None
        prev = None
        for d, v in above.items():
            if v == 1 and not in_run:
                start = d
                in_run = True
            elif v == 0 and in_run:
                runs.append((start, prev))
                in_run = False
            prev = d
        if in_run:
            runs.append((start, prev))
        if len(runs) < 2:
            continue
        # For each pair of consecutive runs with gap > 30 days, find best recovery
        best = None
        for i in range(1, len(runs)):
            gap = (runs[i][0] - runs[i - 1][1]).days
            if gap > 30:
                first_seg = s.loc[runs[i - 1][0]:runs[i - 1][1]]
                second_seg = s.loc[runs[i][0]:runs[i][1]]
                first_peak = int(first_seg.max())
                second_peak = int(second_seg.max())
                if first_peak < 5:
                    continue
                recovery = 100 * second_peak / first_peak
                if recovery >= 50:
                    if best is None or recovery > best["recovery_pct"]:
                        best = {
                            "keyword": kw,
                            "first_peak": str(first_seg.idxmax().date()),
                            "gap_days": gap,
                            "second_peak": str(second_seg.idxmax().date()),
                            "recovery_pct": round(recovery, 1),
                        }
        if best:
            comebacks.append(best)
    comebacks.sort(key=lambda x: -x["recovery_pct"])
    comebacks = comebacks[:20]

    # ---------- 6. Monthly topic churn ----------
    long["month"] = long["date"].dt.to_period("M").astype(str)
    monthly_counts = long.groupby(["month", "keyword"]).size().reset_index(name="count")
    monthly_top100 = {}
    for m, grp in monthly_counts.groupby("month"):
        monthly_top100[m] = set(grp.nlargest(100, "count")["keyword"].tolist())
    months = sorted(monthly_top100.keys())
    monthly_churn = {}
    for i, m in enumerate(months):
        cur = monthly_top100[m]
        if i == 0:
            new_topics = len(cur)
            lost_topics = 0
        else:
            prev = monthly_top100[months[i - 1]]
            new_topics = len(cur - prev)
            lost_topics = len(prev - cur)
        monthly_churn[m] = {
            "new_topics": new_topics,
            "lost_topics": lost_topics,
            "total": len(cur),
        }

    # ---------- 7. Category lifespan ----------
    long["cat_clean"] = long["Category"].apply(clean_category)
    cat_keyword_spans = defaultdict(list)
    for cat, grp in long.groupby("cat_clean"):
        for kw, kgrp in grp.groupby("keyword"):
            if len(kgrp) < 3:
                continue
            span = (kgrp["date"].max() - kgrp["date"].min()).days + 1
            cat_keyword_spans[cat].append(span)
    cat_summary = []
    for cat, spans in cat_keyword_spans.items():
        if len(spans) < 5:
            continue
        cat_summary.append((cat, float(np.mean(spans)), float(np.median(spans)), len(spans)))
    cat_summary.sort(key=lambda x: -x[1])
    category_lifespan = {}
    for cat, avg, med, n in cat_summary[:15]:
        category_lifespan[cat] = {
            "avg_lifespan": round(avg, 1),
            "median_lifespan": round(med, 1),
            "n_keywords": n,
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
    Path(OUT).write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Summary diagnostics
    print(f"Wrote {OUT}")
    print(f"half_lives: {len(half_lives)}")
    print(f"lifecycle_types: {len(lifecycle_types)}")
    types = Counter(v["type"] for v in lifecycle_types.values())
    print("Lifecycle type distribution:", dict(types))
    print(f"longest_lived: {len(longest_lived)} (top: {longest_lived[0] if longest_lived else None})")
    print(f"shortest_lived: {len(shortest_lived)} (top: {shortest_lived[0] if shortest_lived else None})")
    print(f"comebacks: {len(comebacks)} (top: {comebacks[0] if comebacks else None})")
    print(f"monthly_churn: {len(monthly_churn)} months")
    print(f"category_lifespan: {len(category_lifespan)} categories")
    for c, v in list(category_lifespan.items())[:15]:
        print(f"  {c}: avg={v['avg_lifespan']}d median={v['median_lifespan']}d n={v['n_keywords']}")


if __name__ == "__main__":
    main()