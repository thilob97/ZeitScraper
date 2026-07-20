#!/usr/bin/env python3
"""Generate timeline data: topic intensity over time + world events detection.
Finds peaks in coverage of major topics and identifies which world events drove them."""
import pandas as pd
import json, os, warnings
from collections import Counter, defaultdict
from datetime import datetime
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(BASE, 'data', 'processed', 'articles_consolidated.parquet'))
print(f"Loaded {len(df)} articles")

# Ensure we have daily data
df['Date'] = df['Published'].dt.date

# === TOPIC DEFINITIONS ===
topics = {
    "Ukraine-Krieg": {
        "keywords": ["ukrain","russland","krieg","putin","selensky","kyjiw","kyiv","donezk","charkiw","sehiersk","saporischschja","cherson","krim","nato"],
        "color": "#f85149",
    },
    "Naher Osten / Israel": {
        "keywords": ["israel","palestin","gaza","hamas","libanon","hezbollah","hisbollah","naher osten","westjordanland","jerusalem","teheran","iran"],
        "color": "#d29922",
    },
    "Bundestagswahl 2025": {
        "keywords": ["bundestagswahl","wahlkampf","wahl 2025","bundestagswahl 2025","merz","scholz","haback","söder","wahlkampf 2025"],
        "color": "#58a6ff",
    },
    "Klima & Energie": {
        "keywords": ["klima","klimawandel","co2","erneuerbar","solar","windkraft","wasserstoff","energiewende","treibhaus","klimaschutz"],
        "color": "#3fb950",
    },
    "Migration & Asyl": {
        "keywords": ["migration","asyl","flüchtling","grenze","abschiebung","migrationspolitik","asylbewerber","zuwanderung"],
        "color": "#bc8cff",
    },
    "KI / Technologie": {
        "keywords": ["künstliche intelligenz","chatgpt","openai","deepmind","ki","ai","technologie","chatbot","sprachmodell","machine learning"],
        "color": "#56d4dd",
    },
    "Wirtschaft & Inflation": {
        "keywords": ["inflation","rezession","wirtschaft","börse","wirtschaftswachstum","zinserhöhung","ezb","lagerzinsen","konjunktur"],
        "color": "#ff9800",
    },
    "Corona / Pandemie": {
        "keywords": ["corona","covid","pandemie","impfung","lockdown","maske","sars-cov","coronavirus"],
        "color": "#7986cb",
    },
    "USA / Trump": {
        "keywords": ["trump","biden","us-wahl","usa","washington","republikaner","demokraten","weißes haus","kapitol","us-präsident"],
        "color": "#ff7b72",
    },
    "Sport (Fußball)": {
        "keywords": ["bundesliga","fußball","champions league","dfb","em 2024","wm 2026","europameisterschaft","weltmeisterschaft","dfb-pokal"],
        "color": "#aed581",
    },
    "Kriminalität": {
        "keywords": ["kriminalität","mord","tatverdächtiger","polizei","festnahme","staatsanwalt","strafverfahren","gewalttat"],
        "color": "#e57373",
    },
    "Wetter / Naturkatastrophen": {
        "keywords": ["unwetter","sturm","hochwasser","überschwemmung","orkan","starkregen","hitzewelle","dürre","waldbrand","naturkatastrophe","erdbeben"],
        "color": "#4dd0e1",
    },
}

# === WEEKLY AGGREGATION ===
df = df.dropna(subset=['Published']).copy()
df['Week'] = df['Published'].dt.to_period('W').apply(lambda p: p.start_time.date() if p is not pd.NaT else None)

# For each topic, count articles per week
timeline = {}
for topic, info in topics.items():
    kws = info["keywords"]
    mask = df['Title'].str.lower().str.contains('|'.join(kws), na=False, regex=True) | \
           df['Description'].str.lower().str.contains('|'.join(kws), na=False, regex=True)
    tdf = df[mask]
    weekly = tdf.groupby('Week').size()
    # Fill missing weeks with 0
    all_weeks = sorted(df['Week'].unique())
    weekly = weekly.reindex(all_weeks, fill_value=0)
    
    # Find peaks (weeks with > 2x average)
    avg = weekly.mean()
    std = weekly.std()
    threshold = avg + 2 * std
    peaks = weekly[weekly > threshold].index.tolist()
    
    # Find the top 5 peak weeks with sample titles
    top_peaks = weekly.nlargest(5)
    peak_details = []
    for peak_date, count in top_peaks.items():
        if count < 3:
            continue
        peak_articles = tdf[tdf['Week'] == peak_date].nlargest(3, 'Published')
        titles = peak_articles['Title'].tolist()[:3]
        peak_details.append({
            "date": str(peak_date),
            "count": int(count),
            "sample_titles": titles
        })
    
    timeline[topic] = {
        "color": info["color"],
        "weekly": {str(k): int(v) for k, v in weekly.items()},
        "peak_weeks": [str(d) for d in peaks],
        "peak_details": peak_details,
        "total": int(len(tdf)),
        "avg_per_week": round(float(avg), 1),
    }

# === KNOWN WORLD EVENTS (curated) ===
world_events = [
    {"date": "2024-03-10", "label": "GDL Bahnstreik", "desc": "Längster Bahnstreik der GDL", "topic": "Wirtschaft & Inflation"},
    {"date": "2024-03-22", "label": "Moskau Terroranschlag", "desc": "Anschlag auf Crocus City Hall, 140+ Tote", "topic": "Ukraine-Krieg"},
    {"date": "2024-05-07", "label": "Putin Inauguration", "desc": "Putin startet 5. Amtszeit", "topic": "Ukraine-Krieg"},
    {"date": "2024-06-09", "label": "EU-Wahl", "desc": "Europawahl 2024, CDU stärkste Kraft", "topic": "Bundestagswahl 2025"},
    {"date": "2024-06-14", "label": "EM 2024 Start", "desc": "Fußball-EM in Deutschland", "topic": "Sport (Fußball)"},
    {"date": "2024-07-13", "label": "Trump Attentat Butler", "desc": "Attentat auf Trump in Pennsylvania", "topic": "USA / Trump"},
    {"date": "2024-07-21", "label": "Biden tritt zurück", "desc": "Biden verzichtet auf 2. Amtsperiode", "topic": "USA / Trump"},
    {"date": "2024-09-17", "label": "Pager-Attacke Libanon", "desc": "Israel greift Hezbollah an, Pager explodieren", "topic": "Naher Osten / Israel"},
    {"date": "2024-09-27", "label": "Hisbollah-Chef getötet", "desc": "Nasrallah bei israelischem Angriff getötet", "topic": "Naher Osten / Israel"},
    {"date": "2024-10-01", "label": "Iran greift Israel an", "desc": "Iran feuert 200 Raketen auf Israel", "topic": "Naher Osten / Israel"},
    {"date": "2024-11-05", "label": "US-Wahl: Trump gewinnt", "desc": "Trump gewinnt Präsidentschaftswahl", "topic": "USA / Trump"},
    {"date": "2024-12-20", "label": "Magdeburg Anschlag", "desc": "Auto fährt in Weihnachtsmarkt, 5 Tote", "topic": "Kriminalität"},
    {"date": "2025-01-07", "label": "Meta streicht Fact-Checking", "desc": "Zuckerberg beendet Third-Party Fact Check", "topic": "KI / Technologie"},
    {"date": "2025-01-20", "label": "Trump Inauguration", "desc": "Trump startet 2. Amtsperiode", "topic": "USA / Trump"},
    {"date": "2025-01-23", "label": "DeepSeek R1 Release", "desc": "Chinesische AI schlägt OpenAI, Marktcrash", "topic": "KI / Technologie"},
    {"date": "2025-02-23", "label": "Bundestagswahl 2025", "desc": "CDU gewinnt, Merz wird Kanzler", "topic": "Bundestagswahl 2025"},
    {"date": "2025-03-04", "label": "Trump Zölle EU", "desc": "Trump kündigt 25% Zölle auf EU an", "topic": "Wirtschaft & Inflation"},
    {"date": "2025-04-02", "label": "Trump Liberation Day Zölle", "desc": "Globale Zölle, Marktcrash", "topic": "Wirtschaft & Inflation"},
    {"date": "2025-05-06", "label": "Merz wird Kanzler", "desc": "Friedrich Merz zum Bundeskanzler gewählt", "topic": "Bundestagswahl 2025"},
    {"date": "2025-06-13", "label": "Israel-Iran Krieg", "desc": "Israel greift Iran an, Raketen auf Tel Aviv", "topic": "Naher Osten / Israel"},
    {"date": "2025-06-22", "label": "US-Streik auf Iran", "desc": "USA greifen iranische Atomanlagen an", "topic": "Naher Osten / Israel"},
    {"date": "2025-07-10", "label": "Flutkatastrophe Texas", "desc": "Flash Flood in Texas, 100+ Tote", "topic": "Wetter / Naturkatastrophen"},
]

# === SAVE ===
output = {
    "topics": timeline,
    "world_events": world_events,
    "date_range": {
        "start": str(df['Date'].min()),
        "end": str(df['Date'].max()),
    }
}

out_path = os.path.join(BASE, 'dashboard', 'timeline_data.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, default=str)
print(f"Saved {os.path.getsize(out_path)/1024:.0f} KB -> {out_path}")
print(f"Topics: {len(timeline)}")
print(f"World events: {len(world_events)}")
for topic, data in timeline.items():
    if data['peak_details']:
        top = data['peak_details'][0]
        print(f"  {topic}: peak at {top['date']} ({top['count']} articles) — {top['sample_titles'][0][:60]}")