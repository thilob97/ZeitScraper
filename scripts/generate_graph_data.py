#!/usr/bin/env python3
"""Generate graph/network data for the dashboard's vis.js network panel.
Produces graph_data.json with edge lists for different relationship types."""
import pandas as pd
import json, os, warnings
from collections import Counter, defaultdict
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(BASE, 'data', 'processed', 'articles_consolidated.parquet'))
print(f"Loaded {len(df)} articles")

# --- Clean ---
def clean_category(cat):
    if pd.isna(cat): return 'Unknown'
    c = str(cat).strip()
    if 'Z+ (abopflichtiger Inhalt)' in c:
        c = c.replace('Z+ (abopflichtiger Inhalt);', '').replace('\n','').strip()
        c = ' '.join(c.split())
    cat_map = {'Unfall':'Unfälle','Bundesliga':'Fußball-Bundesliga',
               '2. Bundesliga':'2. Fußball-Bundesliga','Verkehrsunfall':'Verkehr'}
    return cat_map.get(c, c)

df['Category_Clean'] = df['Category'].apply(clean_category)
df['Source_Clean'] = df['Source'].fillna('Unknown').apply(lambda s: str(s).strip().split(',')[0].strip())

# === 1. Category ↔ Keyword ===
cat_kw_edges = Counter()
for _, row in df.iterrows():
    cat = row['Category_Clean']
    kws = row['KeyWords']
    if pd.isna(kws): continue
    for kw in str(kws).split(','):
        kw = kw.strip()
        if kw and len(kw) > 2 and cat != 'Unknown':
            cat_kw_edges[(cat, kw)] += 1
cat_kw = [{"from": c, "to": k, "weight": w} for (c,k),w in cat_kw_edges.most_common(500)]

# === 2. Author → Category ===
auth_cat_edges = Counter()
for _, row in df.iterrows():
    auth = row['Author']
    cat = row['Category_Clean']
    if pd.isna(auth) or cat == 'Unknown': continue
    a = str(auth).strip()
    if len(a) < 2 or len(a) > 50: continue
    auth_cat_edges[(a, cat)] += 1
auth_cat = [{"from": a, "to": c, "weight": w} for (a,c),w in auth_cat_edges.most_common(500)]

# === 3. Source → Category ===
src_cat_edges = Counter()
for _, row in df.iterrows():
    src = row['Source_Clean']
    cat = row['Category_Clean']
    if src == 'Unknown' or cat == 'Unknown': continue
    src_cat_edges[(src, cat)] += 1
src_cat = [{"from": s, "to": c, "weight": w} for (s,c),w in src_cat_edges.most_common(500)]

# === 4. Keyword Co-Occurrence ===
kw_pairs = Counter()
for _, row in df.iterrows():
    kws = row['KeyWords']
    if pd.isna(kws): continue
    parts = [p.strip() for p in str(kws).split(',') if p.strip() and len(p.strip()) > 2]
    # Only take top 8 keywords per article to avoid explosion
    if len(parts) > 8:
        parts = parts[:8]
    for i in range(len(parts)):
        for j in range(i+1, len(parts)):
            pair = tuple(sorted([parts[i], parts[j]]))
            kw_pairs[pair] += 1
kw_cooccurrence = [{"from": a, "to": b, "weight": w} for (a,b),w in kw_pairs.most_common(500)]

# === Save ===
graph_data = {
    "category-keyword": cat_kw,
    "author-category": auth_cat,
    "source-category": src_cat,
    "keyword-keyword": kw_cooccurrence,
}
out_path = os.path.join(BASE, 'dashboard', 'graph_data.json')
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, ensure_ascii=False)
print(f"Saved graph_data.json ({os.path.getsize(out_path)/1024:.0f} KB)")
print(f"  category-keyword edges: {len(cat_kw)}")
print(f"  author-category edges: {len(auth_cat)}")
print(f"  source-category edges: {len(src_cat)}")
print(f"  keyword-keyword edges: {len(kw_cooccurrence)}")