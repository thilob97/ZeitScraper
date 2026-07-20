#!/usr/bin/env python3
"""Agenda setting & lead-lag analysis: who drives the news cycle?"""
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

DATA = Path("/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet")
OUT = Path("/opt/data/ZeitScraper/dashboard/agenda_data.json")
OUT.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load & prep
# ---------------------------------------------------------------------------
df = pd.read_parquet(DATA)
print(f"Loaded {len(df):,} articles")

# Source classification
src = df["Source"].astype(str)
is_dpa = src.str.contains("dpa", case=False, na=False)
is_zeit = src.str.contains("ZEIT|zeit", case=False, na=False) & ~src.str.contains(
    "dpa", case=False, na=False
)

df["src_dpa"] = is_dpa
df["src_zeit"] = is_zeit
print(f"dpa-only: {df['src_dpa'].sum():,}  ZEIT-only: {df['src_zeit'].sum():,}")

# Date handling — use Published, convert to date
df["pub_date"] = pd.to_datetime(df["Published"]).dt.tz_convert(None).dt.normalize()
df["pub_hour"] = df["Published"].dt.hour

# Parse keywords into list
def parse_kw(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(",") if x.strip()]

df["kw_list"] = df["KeyWords"].apply(parse_kw)

# ---------------------------------------------------------------------------
# Compute overall top keywords (full dataset)
# ---------------------------------------------------------------------------
print("Computing keyword frequencies...")
kw_counter = Counter()
for kws in df["kw_list"]:
    kw_counter.update(kws)

# Exclude overly generic single-word tags? The task says top-N keywords, so include all.
TOP20 = [kw for kw, _ in kw_counter.most_common(20)]
TOP30 = [kw for kw, _ in kw_counter.most_common(30)]
print("Top 10:", TOP20[:10])

# ---------------------------------------------------------------------------
# 1. Lead-lag: dpa vs ZEIT first appearance for top-20 keywords
# ---------------------------------------------------------------------------
print("\n=== 1. Lead-lag dpa vs ZEIT ===")
lead_lag = {}
for kw in TOP20:
    dpa_mask = df["src_dpa"] & df["kw_list"].apply(lambda x: kw in x)
    zeit_mask = df["src_zeit"] & df["kw_list"].apply(lambda x: kw in x)
    dpa_dates = df.loc[dpa_mask, "pub_date"]
    zeit_dates = df.loc[zeit_mask, "pub_date"]
    dpa_first = dpa_dates.min() if len(dpa_dates) else None
    zeit_first = zeit_dates.min() if len(zeit_dates) else None
    if dpa_first is not None and zeit_first is not None:
        lag = (dpa_first - zeit_first).days
    elif dpa_first is not None:
        lag = None
    elif zeit_first is not None:
        lag = None
    else:
        lag = None
    lead_lag[kw] = {
        "dpa_first": dpa_first.strftime("%Y-%m-%d") if dpa_first is not None else None,
        "zeit_first": zeit_first.strftime("%Y-%m-%d") if zeit_first is not None else None,
        "lag_days": lag,  # positive = dpa earlier, negative = ZEIT earlier
    }
    print(f"  {kw}: dpa={lead_lag[kw]['dpa_first']} zeit={lead_lag[kw]['zeit_first']} lag={lag}")

# Average lag
lags = [v["lag_days"] for v in lead_lag.values() if v["lag_days"] is not None]
avg_lag = sum(lags) / len(lags) if lags else None
print(f"  Average lag (dpa-first minus zeit-first): {avg_lag}")

# ---------------------------------------------------------------------------
# 2. Morning-to-afternoon propagation (6-9am, 12-15pm, 18-21pm)
# ---------------------------------------------------------------------------
print("\n=== 2. Morning/afternoon/evening ===")
morning_afternoon = {}
for kw in TOP20:
    mask = df["kw_list"].apply(lambda x: kw in x)
    sub = df.loc[mask]
    morning = ((sub["pub_hour"] >= 6) & (sub["pub_hour"] < 9)).sum()
    afternoon = ((sub["pub_hour"] >= 12) & (sub["pub_hour"] < 15)).sum()
    evening = ((sub["pub_hour"] >= 18) & (sub["pub_hour"] < 21)).sum()
    morning_afternoon[kw] = {"morning": int(morning), "afternoon": int(afternoon), "evening": int(evening)}
    print(f"  {kw}: m={morning} a={afternoon} e={evening}")

# ---------------------------------------------------------------------------
# 3. Topic lifespan: days in top-100 daily keywords after first appearing
# ---------------------------------------------------------------------------
print("\n=== 3. Topic lifespan (top-30) ===")

# Build daily keyword counts
daily_kw = defaultdict(Counter)
day_kw_sets = df.groupby("pub_date").apply(
    lambda g: Counter(kw for kws in g["kw_list"] for kw in kws), include_groups=False
)
# day_kw_sets is a Series of Counters indexed by date
date_index = sorted(day_kw_sets.index)
print(f"  {len(date_index)} distinct dates")

# Daily top-100 set
daily_top100 = {}
for d in date_index:
    c = day_kw_sets[d]
    daily_top100[d] = set(kw for kw, _ in c.most_common(100))

# For each top-30 keyword, find first_seen, last_seen in top-100, peak date
topic_lifespan = {}
# We need counts per keyword per day for peak detection
kw_daily_counts = defaultdict(dict)
for d in date_index:
    c = day_kw_sets[d]
    for kw, cnt in c.items():
        kw_daily_counts[kw][d] = cnt

for kw in TOP30:
    counts_by_day = kw_daily_counts.get(kw, {})
    if not counts_by_day:
        continue
    # Days this keyword was in top-100
    in_top100_days = [d for d in date_index if kw in daily_top100[d]]
    if not in_top100_days:
        first_seen = min(counts_by_day.keys())
        last_seen = max(counts_by_day.keys())
        days_active = (last_seen - first_seen).days
    else:
        first_seen = min(in_top100_days)
        last_seen = max(in_top100_days)
        days_active = (last_seen - first_seen).days
    peak_date = max(counts_by_day, key=counts_by_day.get)
    topic_lifespan[kw] = {
        "first_seen": first_seen.strftime("%Y-%m-%d"),
        "last_seen": last_seen.strftime("%Y-%m-%d"),
        "days_active": int(days_active),
        "peak_date": peak_date.strftime("%Y-%m-%d"),
    }
    print(f"  {kw}: first={topic_lifespan[kw]['first_seen']} last={topic_lifespan[kw]['last_seen']} days={days_active} peak={topic_lifespan[kw]['peak_date']}")

# ---------------------------------------------------------------------------
# 4. Second-peak detection: keywords that appear, fade, then reappear (>2x gap)
# ---------------------------------------------------------------------------
print("\n=== 4. Second-peak detection ===")

print("  Building weekly keyword counts...")
weekly_counts = defaultdict(lambda: defaultdict(int))
for d in date_index:
    week_start = d - pd.Timedelta(days=d.weekday())
    c = day_kw_sets[d]
    for kw, cnt in c.items():
        weekly_counts[week_start][kw] += cnt

all_weeks = sorted(weekly_counts.keys())

# Candidate keywords: those that appear in at least 2 distinct weeks with meaningful counts
# Look across ALL keywords (not just top-30) for second peaks, but prioritize top ones
# To keep it tractable, look at keywords that appear in weekly top-200 at least once
kw_total_weeks = defaultdict(int)
for w in all_weeks:
    top_week = set(kw for kw, _ in Counter(weekly_counts[w]).most_common(200))
    for kw in top_week:
        kw_total_weeks[kw] += 1

candidate_kw = [kw for kw, n in kw_total_weeks.items() if n >= 2]
print(f"  Candidate keywords (in weekly top-200 >=2 weeks): {len(candidate_kw)}")

second_peaks = []
for kw in candidate_kw:
    series = np.array([weekly_counts[w].get(kw, 0) for w in all_weeks])
    if series.max() < 5:
        continue
    # Find peaks: local maxima
    peaks = []
    for i in range(1, len(series) - 1):
        if series[i] > series[i-1] and series[i] >= series[i+1] and series[i] >= 3:
            peaks.append((i, series[i]))
    # Also check edges
    if len(series) >= 2:
        if series[0] >= series[1] and series[0] >= 3:
            peaks.insert(0, (0, series[0]))
        if series[-1] >= series[-2] and series[-1] >= 3:
            peaks.append((len(series)-1, series[-1]))
    if len(peaks) < 2:
        continue
    # Find the pair of peaks with a real fade between them:
    # valley must drop to <50% of the lower peak, gap >= 4 weeks (real disappearance)
    best = None
    for i in range(len(peaks)):
        for j in range(i+1, len(peaks)):
            p1_idx, p1_val = peaks[i]
            p2_idx, p2_val = peaks[j]
            valley = series[p1_idx:p2_idx+1].min()
            gap_weeks = p2_idx - p1_idx
            if valley < 0.5 * min(p1_val, p2_val) and gap_weeks >= 4:
                # Score: prefer strong second peaks (real comebacks) with substantial gaps
                # score = second_peak_height * gap_weeks (favors big comeback after long absence)
                score = p2_val * gap_weeks
                if best is None or score > best[5]:
                    best = (p1_idx, p2_idx, gap_weeks, p1_val, p2_val, score)
    if best:
        p1_w = all_weeks[best[0]]
        p2_w = all_weeks[best[1]]
        second_peaks.append({
            "keyword": kw,
            "first_peak": p1_w.strftime("%Y-%m-%d"),
            "second_peak": p2_w.strftime("%Y-%m-%d"),
            "gap_days": int(best[2] * 7),
        })

# Sort by gap_days desc, take top 20
second_peaks.sort(key=lambda x: x["gap_days"], reverse=True)
second_peaks = second_peaks[:20]
for sp in second_peaks:
    print(f"  {sp['keyword']}: {sp['first_peak']} -> {sp['second_peak']} (gap {sp['gap_days']}d)")

# ---------------------------------------------------------------------------
# 5. Topic death: keywords in top-50 for 3+ months then dropped below top-500
# ---------------------------------------------------------------------------
print("\n=== 5. Topic death ===")

# Monthly keyword rankings
monthly_kw = defaultdict(Counter)
for d in date_index:
    month = d.strftime("%Y-%m")
    c = day_kw_sets[d]
    monthly_kw[month].update(c)

all_months = sorted(monthly_kw.keys())

# Build keyword -> monthly rank
kw_month_rank = defaultdict(dict)
for m in all_months:
    c = monthly_kw[m]
    for rank, (kw, cnt) in enumerate(c.most_common(), 1):
        kw_month_rank[kw][m] = rank

# Find keywords that were in top-50 for 3+ consecutive months, then dropped below top-500
# Build a sorted list of all month timestamps for consecutive-month checks
month_ts = [pd.Timestamp(m + "-01") for m in all_months]
month_to_idx = {m: i for i, m in enumerate(all_months)}

dead_topics = []
for kw, ranks in kw_month_rank.items():
    # Find longest run of CONSECUTIVE calendar months where rank <= 50
    # ranks only contains months where the keyword appeared; missing months = absent (>500)
    best_run_start = None
    best_run_len = 0
    run_start = None
    run_len = 0
    for i, m in enumerate(all_months):
        rank = ranks.get(m)  # None if absent
        in_top50 = rank is not None and rank <= 50
        if in_top50:
            if run_start is None:
                run_start = m
                run_len = 1
            else:
                # consecutive calendar month
                prev_idx = month_to_idx[m] - 1
                if prev_idx >= 0 and all_months[prev_idx] in ranks and ranks.get(all_months[prev_idx], 999) <= 50:
                    run_len += 1
                else:
                    # gap in top-50; close run
                    if run_len > best_run_len:
                        best_run_len = run_len
                        best_run_start = run_start
                    run_start = m
                    run_len = 1
        else:
            if run_len > best_run_len:
                best_run_len = run_len
                best_run_start = run_start
            run_start = None
            run_len = 0
    if run_len > best_run_len:
        best_run_len = run_len
        best_run_start = run_start

    if best_run_len >= 3:
        peak_month = best_run_start
        # Last month the keyword appeared at all
        months_sorted = sorted(ranks.keys())
        last_month = months_sorted[-1]
        # Check dropped: in the 3 most recent calendar months, rank > 500 or absent
        recent = all_months[-3:]
        dropped = all(ranks.get(m, 99999) > 500 for m in recent)
        if dropped:
            dead_topics.append({
                "keyword": kw,
                "peak_month": peak_month,
                "last_month": last_month,
                "months_active": int(best_run_len),
            })

dead_topics.sort(key=lambda x: x["months_active"], reverse=True)
dead_topics = dead_topics[:20]
for dt in dead_topics:
    print(f"  {dt['keyword']}: peak={dt['peak_month']} last={dt['last_month']} months={dt['months_active']}")

# ---------------------------------------------------------------------------
# 6. Early indicator patterns: keywords spiking 7 days before known events
# ---------------------------------------------------------------------------
print("\n=== 6. Early indicators ===")

# Known event dates
events = [
    {"event": "Bundestagswahl 2025", "date": pd.Timestamp("2025-02-23")},
    {"event": "US Presidential Election (Trump)", "date": pd.Timestamp("2024-11-05")},
    {"event": "Israel-Iran direct attack (Oct 2024)", "date": pd.Timestamp("2024-10-01")},
    {"event": "Israel-Iran escalation (Apr 2024)", "date": pd.Timestamp("2024-04-13")},
    {"event": "EU Parliament Election 2024", "date": pd.Timestamp("2024-06-09")},
    {"event": "Trump inauguration 2025", "date": pd.Timestamp("2025-01-20")},
    {"event": "Munich Security Conference 2025", "date": pd.Timestamp("2025-02-14")},
    {"event": "Israel-Hamas ceasefire Jan 2025", "date": pd.Timestamp("2025-01-19")},
]

early_indicators = []
for ev in events:
    event_date = ev["date"]
    event_name = ev["event"]
    # Look at 14 days before event (to capture 1-2 weeks before)
    window_start = event_date - pd.Timedelta(days=14)
    window_end = event_date - pd.Timedelta(days=1)
    mask = (df["pub_date"] >= window_start) & (df["pub_date"] <= window_end)
    window_df = df.loc[mask]
    if len(window_df) == 0:
        continue
    # Count keywords in this window
    w_counter = Counter()
    for kws in window_df["kw_list"]:
        w_counter.update(kws)
    # Compare with baseline (30 days before window) to find spikes
    baseline_start = window_start - pd.Timedelta(days=30)
    baseline_end = window_start - pd.Timedelta(days=1)
    base_mask = (df["pub_date"] >= baseline_start) & (df["pub_date"] <= baseline_end)
    base_df = df.loc[base_mask]
    b_counter = Counter()
    for kws in base_df["kw_list"]:
        b_counter.update(kws)
    # Normalize by article count
    base_norm = len(base_df) if len(base_df) > 0 else 1
    window_norm = len(window_df) if len(window_df) > 0 else 1
    # Find keywords with high relative increase
    spikes = []
    for kw, cnt in w_counter.most_common(200):
        base_rate = b_counter.get(kw, 0) / base_norm
        window_rate = cnt / window_norm
        if base_rate > 0:
            ratio = window_rate / base_rate
        else:
            ratio = window_rate * 10  # new keyword
        if cnt >= 5 and ratio >= 2.0:
            spikes.append((kw, cnt, ratio))
    spikes.sort(key=lambda x: x[1], reverse=True)
    # Take top 3 per event
    for kw, cnt, ratio in spikes[:3]:
        # Find the spike date (day with max count in window)
        kw_mask = mask & df["kw_list"].apply(lambda x: kw in x)
        kw_dates = df.loc[kw_mask, "pub_date"]
        if len(kw_dates) == 0:
            continue
        spike_date = kw_dates.mode().iloc[0] if len(kw_dates) > 0 else kw_dates.min()
        days_before = int((event_date - spike_date).days)
        if 1 <= days_before <= 14:
            early_indicators.append({
                "keyword": kw,
                "spike_date": spike_date.strftime("%Y-%m-%d"),
                "event_date": event_date.strftime("%Y-%m-%d"),
                "days_before": days_before,
                "event": event_name,
            })
            print(f"  {kw}: spike={spike_date.strftime('%Y-%m-%d')} event={event_name} ({days_before}d before)")

# Sort by days_before, take top 20
early_indicators.sort(key=lambda x: x["days_before"])
early_indicators = early_indicators[:20]

# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------
output = {
    "lead_lag": lead_lag,
    "morning_afternoon": morning_afternoon,
    "topic_lifespan": topic_lifespan,
    "second_peaks": second_peaks,
    "dead_topics": dead_topics,
    "early_indicators": early_indicators,
    "meta": {
        "total_articles": int(len(df)),
        "date_range": [df["pub_date"].min().strftime("%Y-%m-%d"), df["pub_date"].max().strftime("%Y-%m-%d")],
        "dpa_articles": int(df["src_dpa"].sum()),
        "zeit_articles": int(df["src_zeit"].sum()),
        "avg_lag_days": avg_lag,
    },
}

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"\n=== Saved to {OUT} ===")
print(f"Keys: {list(output.keys())}")
print(f"lead_lag: {len(output['lead_lag'])} entries")
print(f"morning_afternoon: {len(output['morning_afternoon'])} entries")
print(f"topic_lifespan: {len(output['topic_lifespan'])} entries")
print(f"second_peaks: {len(output['second_peaks'])} entries")
print(f"dead_topics: {len(output['dead_topics'])} entries")
print(f"early_indicators: {len(output['early_indicators'])} entries")