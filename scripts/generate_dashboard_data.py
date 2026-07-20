#!/usr/bin/env python3
"""Generate dashboard data JSON from consolidated parquet."""
import pandas as pd
import json, os, warnings
from collections import Counter
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(BASE, 'data', 'processed', 'articles_consolidated.parquet'))
print(f"Loaded {len(df)} articles")

# --- Clean categories ---
def clean_category(cat):
    if pd.isna(cat): return 'Unknown'
    c = str(cat).strip()
    if 'Z+ (abopflichtiger Inhalt)' in c:
        c = c.replace('Z+ (abopflichtiger Inhalt);', '').replace('\n','').strip()
        c = ' '.join(c.split())
    cat_map = {'Unfall': 'Unfälle', 'Bundesliga': 'Fußball-Bundesliga',
               '2. Bundesliga': '2. Fußball-Bundesliga', 'Verkehrsunfall': 'Verkehr'}
    return cat_map.get(c, c)

df['Category_Clean'] = df['Category'].apply(clean_category)

# --- Clean source ---
def clean_source(s):
    if pd.isna(s): return 'Unknown'
    s = str(s).strip()
    return s.split(',')[0].strip()

df['Source_Clean'] = df['Source'].apply(clean_source)

# === 1. TEMPORAL ===
daily = df.groupby(df['Published'].dt.date).size().reset_index(name='count')
daily.columns = ['date', 'count']
daily['date'] = daily['date'].astype(str)

monthly = df.groupby('Published_Month').size().reset_index(name='count')
monthly.columns = ['month', 'count']

hourly = df.groupby('Published_Hour').size().reset_index(name='count')
hourly.columns = ['hour', 'count']

weekday_order = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
weekday = df.groupby('Published_Weekday').size().reindex(weekday_order, fill_value=0).reset_index(name='count')
weekday.columns = ['weekday', 'count']

yearly = df.groupby('Published_Year').size().reset_index(name='count')
yearly.columns = ['year', 'count']
yearly['year'] = yearly['year'].astype(int)

# === 2. CATEGORIES ===
cat_counts = df['Category_Clean'].value_counts().head(30)
categories = [{"name": k, "count": int(v)} for k, v in cat_counts.items()]

top_cats = df['Category_Clean'].value_counts().head(15).index.tolist()
cat_trends = {}
for cat in top_cats:
    cat_df = df[df['Category_Clean'] == cat]
    monthly_cat = cat_df.groupby('Published_Month').size()
    cat_trends[cat] = {str(k): int(v) for k, v in monthly_cat.items()}

# === 3. AUTHORS ===
auth_counts = df['Author'].value_counts().head(30)
authors = [{"name": k, "count": int(v)} for k, v in auth_counts.items()]

# === 4. KEYWORDS ===
kw_counter = Counter()
for kw in df['KeyWords'].dropna():
    for part in str(kw).split(','):
        part = part.strip()
        if part and len(part) > 1:
            kw_counter[part] += 1
top_keywords = [{"name": k, "count": v} for k, v in kw_counter.most_common(50)]

# === 5. SOURCES ===
src_counts = df['Source_Clean'].value_counts().head(20)
sources = [{"name": k, "count": int(v)} for k, v in src_counts.items()]

dpa_regional = df[df['Source_Clean'].str.startswith('dpa')].groupby('Source_Clean').size().sort_values(ascending=False)
dpa_regions = [{"name": k, "count": int(v)} for k, v in dpa_regional.items()]

# === 6. PAYWALL ===
pw = df['Paywall'].value_counts()
paywall = {"free": int(pw.get('false', 0)), "paywalled": int(pw.get('true', 0))}

# === 7. CONTENT STATS ===
title_stats = {"avg": round(float(df['Title_Length'].mean()),1), "median": float(df['Title_Length'].median()),
               "max": int(df['Title_Length'].max()), "min": int(df['Title_Length'].min())}
desc_stats = {"avg": round(float(df['Desc_Length'].mean()),1), "median": float(df['Desc_Length'].median()),
              "max": int(df['Desc_Length'].max()), "min": int(df['Desc_Length'].min())}

# === 8. POLITICAL/SOCIETAL ===
political_topics = {
    "Ukraine-Krieg": ["ukrain","russland","krieg","putin","selensky","kyjiw","kyiv","donezk","charkiw"],
    "Naher Osten": ["israel","palestin","gaza","hamas","libanon","hezbollah"],
    "Klima/Energie": ["klima","energie","co2","erneuerbar","solar","windkraft","wasserstoff"],
    "Migration": ["migration","asyl","flüchtling","grenze","abschiebung"],
    "KI/Technologie": ["künstliche intelligenz","chatgpt","openai","deepmind","technologie"],
    "Wirtschaft": ["inflation","rezession","wirtschaft","börse","wirtschaftswachstum"],
    "Bundestagswahl": ["bundestagswahl","wahlkampf","merz","scholz","spd","cdu","grüne","afd","fdp","bsw"],
}
topic_counts = {}
topic_trends = {}
for topic, kws in political_topics.items():
    mask = df['Title'].str.lower().str.contains('|'.join(kws), na=False, regex=True) | \
           df['Description'].str.lower().str.contains('|'.join(kws), na=False, regex=True)
    tdf = df[mask]
    topic_counts[topic] = len(tdf)
    mt = tdf.groupby('Published_Month').size()
    topic_trends[topic] = {str(k): int(v) for k, v in mt.items()}

politicians = ["Scholz","Merz","Trump","Biden","Putin","Selensky","Merkel","Habeck","Söder","Lafontaine"]
politician_counts = {p: int(df['Title'].str.contains(p, case=False, na=False).sum() +
                           df['Description'].str.contains(p, case=False, na=False).sum()) for p in politicians}

# === 9. GEOGRAPHIC ===
states_list = ["Bayern","Berlin","Hamburg","Hessen","Niedersachsen","NRW","Nordrhein-Westfalen",
               "Sachsen","Baden-Württemberg","Rheinland-Pfalz","Schleswig-Holstein","Thüringen",
               "Mecklenburg-Vorpommern","Saarland","Brandenburg","Sachsen-Anhalt","Bremen"]
state_counts = {s: int(df['Title'].str.contains(s, case=False, na=False).sum()) for s in states_list}

countries_list = ["USA","Russland","Ukraine","China","Israel","Frankreich","Großbritannien","Türkei",
                  "Polen","Italien","Spanien","Japan","Indien","Brasilien"]
country_counts = {c: int(df['Title'].str.contains(c, case=False, na=False).sum()) for c in countries_list}

# === 10. SPORTS ===
sports_cats = [c for c in df['Category_Clean'].unique() if any(s in str(c).lower() for s in
              ['fußball','bundesliga','liga','champions','handball','basketball','tennis','sport','olympia'])]
sports_df = df[df['Category_Clean'].isin(sports_cats)]
sports_by_cat = sports_df.groupby('Category_Clean').size().sort_values(ascending=False)
sports_counts = [{"name": k, "count": int(v)} for k, v in sports_by_cat.head(15).items()]
sports_monthly = sports_df.groupby('Published_Month').size()
sports_trend = {str(k): int(v) for k, v in sports_monthly.items()}

# === ASSEMBLE ===
dashboard_data = {
    "meta": {
        "total_articles": len(df),
        "date_earliest": str(df['Published'].min()),
        "date_latest": str(df['Published'].max()),
        "source_files": int(df['_source_file'].nunique()),
        "generated": pd.Timestamp.now(tz='UTC').isoformat(),
    },
    "temporal": {
        "daily": daily.to_dict('records'),
        "monthly": monthly.to_dict('records'),
        "hourly": hourly.to_dict('records'),
        "weekday": weekday.to_dict('records'),
        "yearly": yearly.to_dict('records'),
    },
    "categories": {"top": categories, "trends": cat_trends},
    "authors": authors,
    "keywords": top_keywords,
    "sources": {"all": sources, "dpa_regional": dpa_regions},
    "paywall": paywall,
    "content_stats": {"title_length": title_stats, "desc_length": desc_stats},
    "political": {"topic_counts": topic_counts, "topic_trends": topic_trends, "politician_counts": politician_counts},
    "geographic": {"german_states": state_counts, "countries": country_counts},
    "sports": {"categories": sports_counts, "monthly_trend": sports_trend},
}

out_path = os.path.join(BASE, 'dashboard', 'data.json')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(dashboard_data, f, ensure_ascii=False, default=str)

print(f"Saved {os.path.getsize(out_path)/1024/1024:.1f} MB -> {out_path}")
print(f"Categories: {df['Category_Clean'].nunique()} | Authors: {df['Author'].nunique()} | Keywords: {len(kw_counter)}")
print(f"Political: {json.dumps(topic_counts)}")
print(f"Politicians: {json.dumps(politician_counts)}")