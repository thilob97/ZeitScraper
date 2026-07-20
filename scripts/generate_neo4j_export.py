#!/usr/bin/env python3
"""Export Neo4j Cypher import script from the ZeitScraper data.
Generates a .cypher file that can be executed in Neo4j Browser or via cypher-shell.
Usage: .venv/bin/python scripts/generate_neo4j_export.py
Then in Neo4j: :source neo4j_import.cypher
"""
import pandas as pd
import json, os, warnings
from collections import Counter
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_parquet(os.path.join(BASE, 'data', 'processed', 'articles_consolidated.parquet'))

# Clean
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

# Sample top 5000 articles for Neo4j (full dataset would be too large for browser import)
SAMPLE_SIZE = 5000
top_articles = df.head(SAMPLE_SIZE)

lines = []
lines.append("// Neo4j Cypher Import Script for ZeitScraper Data")
lines.append(f"// Generated from {len(df)} articles (sampling top {SAMPLE_SIZE})")
lines.append("// Execute in Neo4j Browser with :source neo4j_import.cypher")
lines.append("")
lines.append("// === CONSTRAINTS ===")
lines.append("CREATE CONSTRAINT article_hash IF NOT EXISTS FOR (a:Article) REQUIRE a.hash IS UNIQUE;")
lines.append("CREATE CONSTRAINT category_name IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE;")
lines.append("CREATE CONSTRAINT author_name IF NOT EXISTS FOR (au:Author) REQUIRE au.name IS UNIQUE;")
lines.append("CREATE CONSTRAINT source_name IF NOT EXISTS FOR (s:Source) REQUIRE s.name IS UNIQUE;")
lines.append("CREATE CONSTRAINT keyword_name IF NOT EXISTS FOR (k:Keyword) REQUIRE k.name IS UNIQUE;")
lines.append("")
lines.append("// === ARTICLE NODES ===")
for _, row in top_articles.iterrows():
    title = str(row['Title']).replace("'", "\\'").replace('"', '\\"')[:200]
    hash_val = str(row['Hash']).strip()[:64]
    pub = str(row['Published'])[:10] if pd.notna(row['Published']) else ''
    cat = clean_category(row['Category']).replace("'", "\\'")[:100]
    pw = str(row.get('Paywall', 'false')).strip() == 'true'
    lines.append(f"MERGE (a:Article {{hash: '{hash_val}'}}) SET a.title = '{title}', a.published = '{pub}', a.paywall = {str(pw).lower()};")

lines.append("\n// === CATEGORY NODES + RELATIONSHIPS ===")
cat_set = set()
for _, row in top_articles.iterrows():
    cat = clean_category(row['Category']).replace("'", "\\'")[:100]
    if cat not in cat_set:
        lines.append(f"MERGE (c:Category {{name: '{cat}'}});")
        cat_set.add(cat)
    hash_val = str(row['Hash']).strip()[:64]
    lines.append(f"MATCH (a:Article {{hash: '{hash_val}'}}), (c:Category {{name: '{cat}'}}) MERGE (a)-[:BELONGS_TO]->(c);")

lines.append("\n// === AUTHOR NODES + RELATIONSHIPS ===")
auth_set = set()
for _, row in top_articles.iterrows():
    auth = str(row.get('Author','')).strip()
    if not auth or len(auth) < 2 or len(auth) > 50: continue
    auth_clean = auth.replace("'", "\\'")[:50]
    if auth_clean not in auth_set:
        lines.append(f"MERGE (au:Author {{name: '{auth_clean}'}});")
        auth_set.add(auth_clean)
    hash_val = str(row['Hash']).strip()[:64]
    lines.append(f"MATCH (a:Article {{hash: '{hash_val}'}}), (au:Author {{name: '{auth_clean}'}}) MERGE (au)-[:WROTE]->(a);")

lines.append("\n// === SOURCE NODES + RELATIONSHIPS ===")
src_set = set()
for _, row in top_articles.iterrows():
    src = str(row.get('Source_Clean','')).strip().replace("'", "\\'")[:80]
    if not src or src == 'Unknown': continue
    if src not in src_set:
        lines.append(f"MERGE (s:Source {{name: '{src}'}});")
        src_set.add(src)
    hash_val = str(row['Hash']).strip()[:64]
    lines.append(f"MATCH (a:Article {{hash: '{hash_val}'}}), (s:Source {{name: '{src}'}}) MERGE (s)-[:SOURCED]->(a);")

lines.append("\n// === KEYWORD NODES + RELATIONSHIPS ===")
kw_set = set()
for _, row in top_articles.iterrows():
    kws = row.get('KeyWords')
    if pd.isna(kws): continue
    hash_val = str(row['Hash']).strip()[:64]
    parts = [p.strip() for p in str(kws).split(',') if p.strip() and len(p.strip()) > 2][:5]
    for kw in parts:
        kw_clean = kw.replace("'", "\\'")[:50]
        if kw_clean not in kw_set:
            lines.append(f"MERGE (k:Keyword {{name: '{kw_clean}'}});")
            kw_set.add(kw_clean)
        lines.append(f"MATCH (a:Article {{hash: '{hash_val}'}}), (k:Keyword {{name: '{kw_clean}'}}) MERGE (a)-[:HAS_KEYWORD]->(k);")

lines.append("\n// === SAMPLE QUERIES ===")
lines.append("// Top categories: MATCH (c:Category)<-[:BELONGS_TO]-(a:Article) RETURN c.name, count(a) AS count ORDER BY count DESC LIMIT 20;")
lines.append("// Author network: MATCH (au:Author)-[:WROTE]->(a:Article)-[:BELONGS_TO]->(c:Category) RETURN au.name, c.name, count(a) AS articles ORDER BY articles DESC LIMIT 50;")
lines.append("// Keyword clusters: MATCH (a:Article)-[:HAS_KEYWORD]->(k:Keyword)<-[:HAS_KEYWORD]-(a2:Article) WHERE a <> a2 RETURN k.name, count(DISTINCT a2) AS co_mentions ORDER BY co_mentions DESC LIMIT 30;")
lines.append("// Source-Category: MATCH (s:Source)-[:SOURCED]->(a:Article)-[:BELONGS_TO]->(c:Category) RETURN s.name, c.name, count(a) AS count ORDER BY count DESC LIMIT 50;")

out_path = os.path.join(BASE, 'neo4j', 'neo4j_import.cypher')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print(f"Saved Neo4j Cypher script: {out_path} ({os.path.getsize(out_path)/1024:.0f} KB)")
print(f"Nodes: {SAMPLE_SIZE} Articles, {len(cat_set)} Categories, {len(auth_set)} Authors, {len(src_set)} Sources, {len(kw_set)} Keywords")