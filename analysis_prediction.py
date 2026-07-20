"""
PREDICTION PATTERNS — Early indicators that predict major stories.
Produces prediction_data.json with six analytical sections.
"""
import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# Load and prep
# ----------------------------------------------------------------------
print('Loading data...')
df = pd.read_parquet('/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet')
print(f'Loaded {len(df)} articles')

# Clean date - use Published (already datetime64) directly
df['date'] = pd.to_datetime(df['Published'], errors='coerce', utc=True)
df = df.dropna(subset=['date']).copy()
df['date_only'] = df['date'].dt.normalize()  # keeps tz-aware datetime

# Parse keywords into lists
def parse_kw(s):
    if pd.isna(s):
        return []
    return [x.strip() for x in str(s).split(',') if x.strip()]

df['kw_list'] = df['KeyWords'].apply(parse_kw)
df['year'] = df['date'].dt.year.astype(int)
df['month'] = df['date'].dt.month.astype(int)

print(f'Date range: {df["date"].min()} to {df["date"].max()}')

# ----------------------------------------------------------------------
# Build daily keyword counts (exploded)
# ----------------------------------------------------------------------
print('Building daily keyword counts...')
df_exploded = df[['date', 'date_only', 'kw_list', 'Category', 'year', 'month']].explode('kw_list')
df_exploded = df_exploded.dropna(subset=['kw_list'])
df_exploded = df_exploded[df_exploded['kw_list'] != '']
df_exploded['kw_list'] = df_exploded['kw_list'].str.strip()
df_exploded = df_exploded[df_exploded['kw_list'] != 'News']

# Daily keyword counts
daily_kw = df_exploded.groupby(['date_only', 'kw_list']).size().reset_index(name='count')
daily_kw_pivot = daily_kw.pivot_table(index='date_only', columns='kw_list', values='count', fill_value=0)
daily_kw_pivot = daily_kw_pivot.sort_index()
print(f'Daily keyword pivot: {daily_kw_pivot.shape}')

# Get top keywords by total count
kw_totals = df_exploded['kw_list'].value_counts()
top_keywords = kw_totals.head(500).index.tolist()
print(f'Top 500 keywords computed, top: {list(kw_totals.head(10).index)}')

# Daily total article volume (for normalization)
daily_total = df.groupby('date_only').size().rename('total').sort_index()

# ----------------------------------------------------------------------
# 1. EARLY WARNING INDICATORS
# ----------------------------------------------------------------------
print('\n=== 1. Early Warning Indicators ===')

events = [
    ('Bundestagswahl 2025', '2025-02-23'),
    ('Trump election 2024', '2024-11-05'),
    ('Israel-Iran war 2025', '2025-06-13'),
    ('Magdeburg attack 2024', '2024-12-20'),
    ('DeepSeek release 2025', '2025-01-23'),
]

early_indicators = []

for event_name, event_date_str in events:
    event_date = pd.Timestamp(event_date_str, tz='UTC')
    # Look 14 days before (days -14 to -1, inclusive)
    window_start = event_date - timedelta(days=14)
    window_end = event_date - timedelta(days=1)
    # Baseline: 30 days before that (window_start-30 to window_start-1)
    baseline_start = window_start - timedelta(days=30)
    baseline_end = window_start - timedelta(days=1)

    window_mask = (daily_kw_pivot.index >= window_start) & (daily_kw_pivot.index <= window_end)
    baseline_mask = (daily_kw_pivot.index >= baseline_start) & (daily_kw_pivot.index <= baseline_end)

    window_data = daily_kw_pivot[window_mask]
    baseline_data = daily_kw_pivot[baseline_mask]

    window_sums = window_data.sum(axis=0)
    baseline_sums = baseline_data.sum(axis=0)

    # Find keywords that spiked in pre-event window
    for kw in top_keywords:
        if kw not in window_sums.index:
            continue
        w_count = int(window_sums.get(kw, 0))
        b_count = int(baseline_sums.get(kw, 0))
        if w_count >= 5 and b_count >= 2:
            ratio = w_count / max(b_count, 1)
            if ratio >= 3:
                # Find spike day (max day in window)
                window_series = window_data[kw]
                if window_series.max() > 0:
                    spike_date = window_series.idxmax()
                    days_before = (event_date - spike_date).days
                    early_indicators.append({
                        'event': event_name,
                        'event_date': event_date_str,
                        'keyword': kw,
                        'spike_date': str(spike_date.date()),
                        'days_before': int(days_before),
                        'count': w_count,
                        'baseline_count': b_count,
                        'ratio': round(float(ratio), 2),
                    })

# Sort by ratio descending, then take top 30 per event
from itertools import groupby
early_indicators.sort(key=lambda x: (x['event'], -x['ratio']))
ei_limited = []
for ev, group in groupby(early_indicators, key=lambda x: x['event']):
    g = list(group)
    ei_limited.extend(g[:30])
early_indicators = ei_limited
print(f'Found {len(early_indicators)} early indicators')
for x in early_indicators[:10]:
    print(f"  {x['event']}: {x['keyword']} ratio={x['ratio']} days_before={x['days_before']}")

# ----------------------------------------------------------------------
# 2. SEASONAL TEMPLATES
# ----------------------------------------------------------------------
print('\n=== 2. Seasonal Templates ===')

years_available = sorted(df['year'].unique())
print(f'Years available: {years_available}')

monthly_kw = df_exploded.groupby(['month', 'year', 'kw_list']).size().reset_index(name='count')

seasonal_templates = {}
for month in range(1, 13):
    month_data = monthly_kw[monthly_kw['month'] == month]
    year_keywords = {}
    for y in years_available:
        ydata = month_data[month_data['year'] == y]
        year_keywords[y] = set(ydata['kw_list'].tolist())

    # Intersection across all years that have data for this month
    active_years = [y for y in years_available if len(year_keywords[y]) > 0]
    common = set.intersection(*[year_keywords[y] for y in active_years]) if len(active_years) >= 2 else set()

    # Rank common keywords by total count
    common_totals = {}
    for kw in common:
        common_totals[kw] = int(month_data[month_data['kw_list'] == kw]['count'].sum())

    top10 = sorted(common_totals.items(), key=lambda x: -x[1])[:10]
    seasonal_templates[f'{month:02d}'] = [kw for kw, c in top10]

print(f'Seasonal templates for {len(seasonal_templates)} months')
for m, kws in list(seasonal_templates.items())[:3]:
    print(f'  Month {m}: {kws}')

# ----------------------------------------------------------------------
# 3. PRECURSOR PATTERNS (cross-correlation with lag)
# ----------------------------------------------------------------------
print('\n=== 3. Precursor Patterns ===')

# Use Pearson correlation between keyword series (shifted +lag) and category series
# lag = keyword spike leads category increase by 3-7 days
top60_kws = kw_totals.head(60).index.tolist()
top60_kws = [k for k in top60_kws if k in daily_kw_pivot.columns]

# Daily category counts
daily_cat = df.groupby(['date_only', 'Category']).size().reset_index(name='count')
daily_cat_pivot = daily_cat.pivot_table(index='date_only', columns='Category', values='count', fill_value=0).sort_index()

cat_clean = [c for c in df['Category'].value_counts().head(30).index if not c.startswith('Z+')]

precursors = []

# Precompute z-scored series for correlation
def zscore(s):
    if s.std() == 0:
        return s * 0
    return (s - s.mean()) / s.std()

for kw in top60_kws:
    kw_series = daily_kw_pivot[kw].reindex(daily_cat_pivot.index, fill_value=0)
    kw_z = zscore(kw_series)
    for cat in cat_clean[:15]:
        if cat not in daily_cat_pivot.columns:
            continue
        cat_series = daily_cat_pivot[cat]
        cat_z = zscore(cat_series)
        if cat_z.std() == 0:
            continue
        # Try lags 3-7
        best_lag = None
        best_corr = 0
        for lag in range(3, 8):
            shifted = kw_z.shift(lag)
            aligned = pd.concat([shifted, cat_z], axis=1).dropna()
            if len(aligned) < 30:
                continue
            corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
            if corr is not None and abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag
        if best_lag is not None and abs(best_corr) >= 0.2:
            precursors.append({
                'keyword_a': kw,
                'keyword_b': cat,
                'avg_lag_days': float(best_lag),
                'correlation': round(float(best_corr), 3),
            })

precursors.sort(key=lambda x: -abs(x['correlation']))
precursors = precursors[:30]
print(f'Found {len(precursors)} precursors')
for x in precursors[:5]:
    print(f"  {x['keyword_a']} -> {x['keyword_b']}: lag={x['avg_lag_days']}d corr={x['correlation']}")

# ----------------------------------------------------------------------
# 4. GROWTH VELOCITY
# ----------------------------------------------------------------------
print('\n=== 4. Growth Velocity ===')

# Weekly keyword counts
df_exploded['week'] = df_exploded['date'].dt.to_period('W').dt.start_time
weekly_kw = df_exploded.groupby(['week', 'kw_list']).size().reset_index(name='count')
weekly_kw_pivot = weekly_kw.pivot_table(index='week', columns='kw_list', values='count', fill_value=0).sort_index()

growth_velocity = []
for kw in kw_totals.head(500).index:
    if kw not in weekly_kw_pivot.columns:
        continue
    series = weekly_kw_pivot[kw]
    for i in range(1, len(series)):
        prev = series.iloc[i-1]
        curr = series.iloc[i]
        if prev >= 3 and curr >= 10:
            factor = curr / prev
            if factor >= 3:
                growth_velocity.append({
                    'keyword': kw,
                    'week': str(series.index[i].date()),
                    'count': int(curr),
                    'prev_count': int(prev),
                    'growth_factor': round(float(factor), 2),
                })

growth_velocity.sort(key=lambda x: -x['growth_factor'])
growth_velocity = growth_velocity[:30]
print(f'Found {len(growth_velocity)} growth velocity entries')
for x in growth_velocity[:5]:
    print(f"  {x['keyword']}: {x['growth_factor']}x ({x['prev_count']}->{x['count']}) week {x['week']}")

# ----------------------------------------------------------------------
# 5. PREDICTIVE KEYWORD PAIRS
# ----------------------------------------------------------------------
print('\n=== 5. Predictive Keyword Pairs ===')

top100_kws = kw_totals.head(100).index.tolist()
top100_kws = [k for k in top100_kws if k in daily_kw_pivot.columns]

# Compute spike threshold for each keyword (top 15% of active days)
spike_thresholds = {}
spike_dates = {}
for kw in top100_kws:
    s = daily_kw_pivot[kw]
    active = s[s > 0]
    if len(active) > 20:
        thr = np.percentile(active, 85)
        spike_thresholds[kw] = thr
        spike_dates[kw] = set(s[s >= thr].index)

# Build co-occurrence from articles to find candidate pairs
cooc = Counter()
top100_set = set(top100_kws)
for kws in df['KeyWords'].dropna():
    ks = list(set([x.strip() for x in str(kws).split(',') if x.strip() in top100_set]))
    for i in range(len(ks)):
        for j in range(len(ks)):
            if i != j:
                cooc[(ks[i], ks[j])] += 1

candidate_pairs = [p for p, c in cooc.most_common(10000) if c >= 10]
print(f'Testing {len(candidate_pairs)} candidate pairs...')

predictive_pairs = []
for kw_a, kw_b in candidate_pairs:
    if kw_a not in spike_dates or kw_b not in spike_dates:
        continue
    a_spikes = spike_dates[kw_a]
    b_spikes = spike_dates[kw_b]

    hits = 0
    lags = []
    total = len(a_spikes)
    for sp in a_spikes:
        for lag in range(3, 8):
            future = sp + timedelta(days=lag)
            if future in b_spikes:
                hits += 1
                lags.append(lag)
                break

    if total >= 3 and hits >= 2:
        confidence = hits / total
        avg_lag = np.mean(lags) if lags else 0
        if confidence >= 0.2:
            predictive_pairs.append({
                'from': kw_a,
                'to': kw_b,
                'lag_days': round(float(avg_lag), 1),
                'confidence': round(float(confidence), 3),
                'hits': hits,
                'total': total,
            })

predictive_pairs.sort(key=lambda x: -x['confidence'])
predictive_pairs = predictive_pairs[:30]
print(f'Found {len(predictive_pairs)} predictive pairs')
for x in predictive_pairs[:5]:
    print(f"  {x['from']} -> {x['to']}: lag={x['lag_days']}d conf={x['confidence']}")

# ----------------------------------------------------------------------
# 6. CATEGORY MOMENTUM
# ----------------------------------------------------------------------
print('\n=== 6. Category Momentum ===')

# Monthly category counts
monthly_cat = df.groupby([pd.Grouper(key='date', freq='ME'), 'Category']).size().reset_index(name='count')
monthly_cat_pivot = monthly_cat.pivot_table(index='date', columns='Category', values='count', fill_value=0).sort_index()

top15_cats = df['Category'].value_counts().head(15).index.tolist()

category_momentum = {}
for cat in top15_cats:
    if cat not in monthly_cat_pivot.columns:
        continue
    series = monthly_cat_pivot[cat]
    ma = series.rolling(window=3, min_periods=1).mean()

    current_avg = float(ma.iloc[-1]) if len(ma) > 0 else 0

    if len(series) >= 6:
        recent = series.iloc[-3:].mean()
        previous = series.iloc[-6:-3].mean()
        if previous > 0:
            momentum = (recent - previous) / previous
        else:
            momentum = 0
        if momentum > 0.1:
            trend = 'rising'
        elif momentum < -0.1:
            trend = 'falling'
        else:
            trend = 'stable'
    else:
        momentum = 0
        trend = 'insufficient_data'

    category_momentum[cat] = {
        'current_avg': round(current_avg, 1),
        'trend': trend,
        'momentum': round(float(momentum), 3),
    }

print(f'Category momentum for {len(category_momentum)} categories')

# ----------------------------------------------------------------------
# SAVE OUTPUT
# ----------------------------------------------------------------------
print('\n=== Saving output ===')

output = {
    'early_indicators': early_indicators,
    'seasonal_templates': seasonal_templates,
    'precursors': precursors,
    'growth_velocity': growth_velocity,
    'predictive_pairs': predictive_pairs,
    'category_momentum': category_momentum,
    '_metadata': {
        'total_articles': int(len(df)),
        'date_range': f"{str(df['date'].min().date())} to {str(df['date'].max().date())}",
        'total_unique_keywords': int(len(kw_totals)),
        'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
    },
}

import os
os.makedirs('/opt/data/ZeitScraper/dashboard', exist_ok=True)
with open('/opt/data/ZeitScraper/dashboard/prediction_data.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2, default=str)

print(f'Saved to /opt/data/ZeitScraper/dashboard/prediction_data.json')
print(f'\nSummary:')
print(f'  early_indicators: {len(early_indicators)}')
print(f'  seasonal_templates: {len(seasonal_templates)} months')
print(f'  precursors: {len(precursors)}')
print(f'  growth_velocity: {len(growth_velocity)}')
print(f'  predictive_pairs: {len(predictive_pairs)}')
print(f'  category_momentum: {len(category_momentum)} categories')
print('DONE')