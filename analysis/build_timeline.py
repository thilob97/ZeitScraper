import pandas as pd, numpy as np, json
from collections import Counter

df = pd.read_parquet('/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet')
df['Published_Date'] = pd.to_datetime(df['Published_Date'], errors='coerce')
df = df.dropna(subset=['Published_Date']).copy()

def clean_cat(c):
    if pd.isna(c): return 'Unbekannt'
    s = str(c)
    if 'Z+' in s:
        parts = [p.strip() for p in s.replace('\\n',' ').split(';') if p.strip()]
        s = parts[-1] if parts else s
    return s.strip()

df['Cat_clean'] = df['Category'].apply(clean_cat)

daily = df.groupby(df['Published_Date'].dt.date).size().reset_index(name='count')
daily['date'] = pd.to_datetime(daily['Published_Date'])
daily = daily.sort_values('date').reset_index(drop=True)
daily['dow'] = daily['date'].dt.day_name()
daily['roll_mean'] = daily['count'].rolling(7, min_periods=3).mean()
daily['roll_std'] = daily['count'].rolling(7, min_periods=3).std()
daily['z'] = (daily['count'] - daily['roll_mean']) / daily['roll_std'].replace(0, np.nan)

spikes = daily[daily['z'] > 1.5].copy()
spikes = spikes.sort_values('count', ascending=False).head(25)

# Detect publication gaps (missing days) - news droughts
full_range = pd.date_range(daily['date'].min(), daily['date'].max(), freq='D')
missing_dates = sorted(set(full_range.date) - set(daily['Published_Date']))
gaps = []
for md in missing_dates:
    # find surrounding published days
    prev_days = daily[daily['date'] < pd.Timestamp(md)].tail(2)
    next_days = daily[daily['date'] > pd.Timestamp(md)].head(2)
    gap_info = {'date': str(md), 'weekday': pd.Timestamp(md).day_name()}
    if len(prev_days):
        gap_info['prev_day_count'] = int(prev_days.iloc[-1]['count'])
        gap_info['prev_day_date'] = str(prev_days.iloc[-1]['date'].date())
    if len(next_days):
        gap_info['next_day_count'] = int(next_days.iloc[0]['count'])
        gap_info['next_day_date'] = str(next_days.iloc[0]['date'].date())
    gaps.append(gap_info)

out = {}
out['dataset_summary'] = {
    'total_articles': int(len(df)),
    'date_min': df['Published_Date'].min().strftime('%Y-%m-%d'),
    'date_max': df['Published_Date'].max().strftime('%Y-%m-%d'),
    'span_days': int((df['Published_Date'].max() - df['Published_Date'].min()).days),
    'mean_articles_per_day': round(float(daily['count'].mean()), 2),
    'median_articles_per_day': int(daily['count'].median()),
    'max_articles_day': {'date': str(daily.loc[daily['count'].idxmax(),'date'].date()), 'count': int(daily['count'].max())},
    'min_articles_day': {'date': str(daily.loc[daily['count'].idxmin(),'date'].date()), 'count': int(daily['count'].min())},
    'unique_categories': int(df['Cat_clean'].nunique()),
    'total_days': int(len(daily)),
}

monthly = df.groupby(df['Published_Date'].dt.to_period('M')).size().reset_index(name='count')
monthly['month'] = monthly['Published_Date'].astype(str)
out['monthly_trend'] = monthly[['month','count']].to_dict('records')

quarterly = df.groupby('Published_Quarter').size().reset_index(name='count')
out['quarterly_trend'] = quarterly.to_dict('records')

top_cats = df['Cat_clean'].value_counts().head(20)
out['top_categories_overall'] = [{'category': k, 'count': int(v)} for k,v in top_cats.items()]

yearly = df.groupby('Published_Year').size().reset_index(name='count')
out['yearly_trend'] = yearly.to_dict('records')

wd = df['Published_Weekday'].value_counts()
wd_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
wd_ordered = [{'weekday': w, 'count': int(wd.get(w,0))} for w in wd_order]
out['weekday_pattern'] = wd_ordered

hr = df.groupby('Published_Hour').size().reset_index(name='count')
hr['hour'] = hr['Published_Hour'].astype(int)
out['hour_pattern'] = hr[['hour','count']].to_dict('records')

events = []
for _, row in spikes.iterrows():
    d = row['date']
    day_df = df[df['Published_Date'].dt.date == d.date()]
    cats = day_df['Cat_clean'].value_counts().head(5)
    kw_all = []
    for k in day_df['KeyWords'].dropna():
        kw_all.extend([x.strip() for x in str(k).split(',') if x.strip()])
    kw_counter = Counter(kw_all)
    titles = day_df['Title'].dropna().head(8).tolist()
    events.append({
        'date': str(d.date()),
        'weekday': row['dow'],
        'count': int(row['count']),
        'z_score': round(float(row['z']), 2),
        'top_categories': [{'category': k, 'count': int(v)} for k,v in cats.items()],
        'top_keywords': [{'keyword': k, 'count': int(v)} for k,v in kw_counter.most_common(10)],
        'sample_titles': titles,
    })
out['notable_spike_events'] = events
out['publication_gaps'] = gaps

top_days = daily.nlargest(10, 'count')
top_days_events = []
for _, row in top_days.iterrows():
    d = row['date']
    day_df = df[df['Published_Date'].dt.date == d.date()]
    cats = day_df['Cat_clean'].value_counts().head(5)
    titles = day_df['Title'].dropna().head(6).tolist()
    top_days_events.append({
        'date': str(d.date()),
        'weekday': row['dow'],
        'count': int(row['count']),
        'top_categories': [{'category': k, 'count': int(v)} for k,v in cats.items()],
        'sample_titles': titles,
    })
out['top_10_busiest_days'] = top_days_events

with open('/opt/data/ZeitScraper/analysis/timeline_events.json','w',encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2, default=str)

print('Wrote /opt/data/ZeitScraper/analysis/timeline_events.json')
print('Keys:', list(out.keys()))
print('Events found:', len(events))
print('Gaps found:', len(gaps))
for g in gaps[:10]:
    print(f"  GAP {g}")
print('Top 5 spikes:')
for e in events[:5]:
    print(f"  {e['date']} ({e['weekday']}) n={e['count']} z={e['z_score']} - {[c['category'] for c in e['top_categories'][:2]]}")
print('Top 5 busiest days:')
for e in top_days_events[:5]:
    print(f"  {e['date']} ({e['weekday']}) n={e['count']}")
print('Monthly:')
for m in out['monthly_trend']:
    print(f"  {m['month']}: {m['count']}")