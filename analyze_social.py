#!/usr/bin/env python3
"""Analyze author collaborations and social networks in the ZeitScraper dataset."""
import pandas as pd
import numpy as np
import re
import json
from collections import Counter, defaultdict
from itertools import combinations
import os

DATA = '/opt/data/ZeitScraper/data/processed/articles_conolidated.parquet'
DATA = '/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet'
OUT = '/opt/data/ZeitScraper/dashboard/social_data.json'

# ---------- Load ----------
print("Loading data...")
df = pd.read_parquet(DATA)
print(f"Loaded {len(df)} articles")

# ---------- Clean Author column ----------
# Some author strings are contaminated with article subheadings (questions),
# photo credits, newlines, nbsp. Clean before splitting.
def clean_author_string(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return ''
    s = str(s)
    # Cut at first newline (contaminated text usually starts on new line)
    if '\n' in s:
        s = s.split('\n')[0]
    # Cut at first © (photo credits)
    if '©' in s:
        s = s.split('©')[0]
    # Cut at first '?' - FAQ-style questions are contamination
    if '?' in s:
        s = s[:s.index('?')]
    # Cut at numbered list patterns like "1. " which start article content
    m = re.search(r'\d+\.\s', s)
    if m:
        s = s[:m.start()]
    # Cut at common article-text markers (capitalized question words after a name)
    # Pattern: a lowercase letter followed by "Wie ", "Warum ", "Was ", "Wann ", "Wo ", "Wie kann", etc.
    for marker in ['Wie kann', 'Wie ist', 'Warum', 'Was ist', 'Wann ', 'Wo ', 'Wie ', 'Welche ', 'Hat ', 'Kann ', 'Soll ', 'Darf ', 'Kommt ', 'Gilt ', 'Braucht ', 'Wie viele', 'Wie groß', 'Wie oft', 'Wie schnell', 'Wie wahrscheinlich', 'In welchen', 'Um wie', 'Bei welchen', 'Seit wann', 'Ab wann']:
        idx = s.find(marker)
        if idx > 5:  # only cut if marker appears after some name content
            s = s[:idx]
    # Normalize whitespace and nbsp
    s = s.replace('\xa0', ' ').strip()
    # Strip trailing non-name chars
    s = s.rstrip(' ,;.')
    return s

# ---------- Split concatenated author names ----------
# Build a set of known single-author names from short clean strings for refinement.
# Primary method: split at lowercase→uppercase boundary, respecting prefixes/particles.

NAME_PREFIXES = {'dr.', 'prof.', 'dr.med.', 'dr.', 'prof.dr.'}
# lowercase particles that can appear inside a name (German nobility etc.)
PARTICLES = {'von', 'zu', 'de', 'van', 'der', 'den', 'vom', 'zur', 'le', 'la', 'du', 'di', 'da'}

# Regex: split position is a lowercase letter (incl. umlauts) immediately followed by uppercase
SPLIT_RE = re.compile(r'(?<=[a-zäöüß])(?=[A-ZÄÖÜ])')

def split_authors(s):
    """Split a concatenated author string into individual author names."""
    s = clean_author_string(s)
    if not s:
        return []
    # First, handle comma-separated (rare in Author col but possible)
    if ',' in s and len(s.split(',')) > 1 and all(len(p.strip()) > 2 for p in s.split(',')):
        parts = [p.strip() for p in s.split(',') if p.strip()]
    else:
        parts = SPLIT_RE.split(s)
    # Clean each part
    authors = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Normalize multiple spaces
        p = re.sub(r'\s+', ' ', p)
        # Skip pure article-text fragments (too long, contains lowercase sentence)
        # Real names are typically < 50 chars
        if len(p) > 60:
            # might be a name + text fragment; try to keep just leading name tokens
            tokens = p.split()
            keep = []
            for t in tokens:
                if any(c.islower() for c in t) and not t[0].isupper() and t.lower() not in PARTICLES and t.lower().rstrip('.') not in NAME_PREFIXES:
                    break
                keep.append(t)
            p = ' '.join(keep) if keep else p
            if len(p) > 60:
                continue
        # Skip fragments that are clearly article text (start lowercase, or contain sentence patterns)
        if p[0].islower() and len(p) > 5 and p.lower() not in {x for x in []}:
            # 3-letter codes are lowercase and valid
            if len(p) <= 4:
                authors.append(p)
                continue
            else:
                continue
        # Skip if it looks like a sentence fragment (has question mark remnant, etc.)
        if any(w in p.lower() for w in [' wie ', ' und ', ' oder ', ' aber ', ' dass ']):
            continue
        authors.append(p)
    return authors

# ---------- Parse source list ----------
def split_sources(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return []
    s = str(s).replace('\xa0', ' ')
    # Sources separated by commas
    parts = [p.strip() for p in s.split(',') if p.strip()]
    # Collapse multiple internal spaces
    parts = [re.sub(r'\s+', ' ', p) for p in parts]
    return parts

# ---------- Clean category ----------
def clean_category(s):
    if not s or (isinstance(s, float) and np.isnan(s)):
        return None
    s = str(s).replace('\xa0', ' ')
    # Strip "Z+ (abopflichtiger Inhalt);" prefixes
    s = re.sub(r'^Z\+\s*\(abopflichtiger Inhalt\)\s*[;,]?\s*', '', s)
    # Strip leading "Liveblog:" and whitespace
    s = re.sub(r'^Liveblog:\s*', '', s)
    # Normalize whitespace/newlines
    s = re.sub(r'\s+', ' ', s).strip()
    # Strip trailing semicolons
    s = s.rstrip('; ').strip()
    return s if s else None

# ---------- Process all articles ----------
print("Processing authors...")
author_lists = []  # list of (list_of_authors) per article
source_lists = []
categories = []
months = []
valid_idx = []

for i, row in enumerate(df.itertuples(index=False)):
    authors = split_authors(row.Author)
    sources = split_sources(row.Source)
    cat = clean_category(row.Category)
    # Published_Month
    m = getattr(row, 'Published_Month', None)
    if m is None or (isinstance(m, float) and np.isnan(m)):
        m = None
    else:
        m = str(m)
    author_lists.append(authors)
    source_lists.append(sources)
    categories.append(cat)
    months.append(m)

print(f"Processed {len(author_lists)} articles")
multi_author = sum(1 for a in author_lists if len(a) > 1)
print(f"Articles with multiple authors: {multi_author}")
any_author = sum(1 for a in author_lists if len(a) >= 1)
print(f"Articles with at least one author: {any_author}")

# ---------- Author frequency ----------
print("\nCounting author frequencies...")
author_freq = Counter()
for authors in author_lists:
    for a in authors:
        author_freq[a] += 1
print(f"Distinct authors: {len(author_freq)}")
print("Top 30 authors:")
for name, cnt in author_freq.most_common(30):
    print(f"  {cnt:5d}  {name}")

top30 = [a for a, _ in author_freq.most_common(30)]
top10 = top30[:10]

# ---------- Co-author edges ----------
print("\nBuilding co-author edges...")
coauth_pairs = Counter()
for authors in author_lists:
    if len(authors) < 2:
        continue
    unique = list(set(authors))  # dedupe within article
    if len(unique) < 2:
        continue
    for a, b in combinations(sorted(unique), 2):
        coauth_pairs[(a, b)] += 1

print(f"Total co-author pairs: {len(coauth_pairs)}")
top200_pairs = coauth_pairs.most_common(200)
coauthor_edges = [{"from": a, "to": b, "weight": w} for (a, b), w in top200_pairs]
print("Top 10 co-author pairs:")
for e in coauthor_edges[:10]:
    print(f"  {e['weight']:3d}  {e['from']}  <->  {e['to']}")

# ---------- Top-5 co-authors for each top-30 author ----------
print("\nTop-5 co-authors for each top-30 author...")
author_coauths = defaultdict(Counter)
for (a, b), w in coauth_pairs.items():
    author_coauths[a][b] += w
    author_coauths[b][a] += w
# (not output as separate key, but useful for clusters; will embed in clusters)

# ---------- Author clusters / teams ----------
# Use a simple community-detection-like approach via connected components on
# the top co-author graph (edges with weight >= 2), then rank clusters by size.
print("\nDetecting author clusters...")
# Build graph: nodes are authors with >=5 articles; edges with weight >= 3
NODE_MIN_ARTICLES = 5
EDGE_MIN_WEIGHT = 3
nodes = {a for a, c in author_freq.items() if c >= NODE_MIN_ARTICLES}
adj = defaultdict(set)
for (a, b), w in coauth_pairs.items():
    if w >= EDGE_MIN_WEIGHT and a in nodes and b in nodes:
        adj[a].add(b)
        adj[b].add(a)

# Connected components
visited = set()
clusters = []
for node in nodes:
    if node in visited:
        continue
    # BFS
    comp = []
    stack = [node]
    visited.add(node)
    while stack:
        n = stack.pop()
        comp.append(n)
        for nb in adj[n]:
            if nb not in visited:
                visited.add(nb)
                stack.append(nb)
    if len(comp) >= 3:  # clusters of 3+ authors
        clusters.append(comp)

# Rank clusters by total co-author weight inside them
def cluster_weight(comp):
    s = 0
    for a, b in combinations(sorted(comp), 2):
        s += coauth_pairs.get((a, b), 0)
    return s

clusters.sort(key=lambda c: cluster_weight(c), reverse=True)

# For each cluster, find common category
author_categories = defaultdict(Counter)
for i, authors in enumerate(author_lists):
    cat = categories[i]
    if cat:
        for a in authors:
            author_categories[a][cat] += 1

def cluster_common_category(comp):
    # aggregate categories across all authors in cluster
    combined = Counter()
    for a in comp:
        for cat, cnt in author_categories[a].items():
            combined[cat] += cnt
    if not combined:
        return None
    return combined.most_common(1)[0][0]

# Take top 30 clusters by weight
author_clusters = []
for comp in clusters[:30]:
    author_clusters.append({
        "authors": sorted(comp),
        "size": len(comp),
        "common_category": cluster_common_category(comp),
        "total_edge_weight": cluster_weight(comp),
    })
print(f"Clusters found (size>=3): {len(clusters)}")
for c in author_clusters[:5]:
    print(f"  size={c['size']} cat={c['common_category']} authors={c['authors'][:6]}...")

# ---------- Author-category specialization (top-30) ----------
print("\nAuthor-category specialization...")
author_specialization = {}
for a in top30:
    top_cats = [cat for cat, _ in author_categories[a].most_common(3)]
    author_specialization[a] = top_cats

# ---------- Author productivity timeline (top-10) ----------
print("\nAuthor productivity timeline (top-10)...")
author_productivity = {a: defaultdict(int) for a in top10}
for i, authors in enumerate(author_lists):
    m = months[i]
    if m is None:
        continue
    for a in authors:
        if a in author_productivity:
            author_productivity[a][m] += 1
# Convert to plain dict, sorted by month
author_productivity_out = {}
for a in top10:
    d = dict(author_productivity[a])
    author_productivity_out[a] = dict(sorted(d.items()))

# ---------- Source-author correlations ----------
print("\nSource-author correlations...")
source_author = Counter()
for i in range(len(author_lists)):
    authors = author_lists[i]
    sources = source_lists[i]
    if not authors or not sources:
        continue
    for s in sources:
        for a in authors:
            source_author[(s, a)] += 1
top_source_author = source_author.most_common(100)
source_author_out = [{"source": s, "author": a, "count": c} for (s, a), c in top_source_author]

# ---------- Source-source co-occurrence ----------
print("\nSource-source co-occurrence...")
source_pairs = Counter()
for sources in source_lists:
    if len(sources) < 2:
        continue
    unique = list(set(sources))
    if len(unique) < 2:
        continue
    for a, b in combinations(sorted(unique), 2):
        source_pairs[(a, b)] += 1
top_source_pairs = source_pairs.most_common(50)
source_cooccurrence = [{"from": a, "to": b, "weight": w} for (a, b), w in top_source_pairs]

# ---------- Assemble and save ----------
result = {
    "coauthor_edges": coauthor_edges,
    "author_clusters": author_clusters,
    "author_specialization": author_specialization,
    "author_productivity": author_productivity_out,
    "source_author": source_author_out,
    "source_cooccurrence": source_cooccurrence,
    # bonus metadata
    "_meta": {
        "total_articles": len(df),
        "multi_author_articles": multi_author,
        "distinct_authors": len(author_freq),
        "distinct_sources": len({s for srcs in source_lists for s in srcs}),
        "top30_authors": [{"name": a, "count": c} for a, c in author_freq.most_common(30)],
    },
}

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'w', encoding='utf-8') as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(f"\nSaved to {OUT}")
print(f"Size: {os.path.getsize(OUT)/1024:.1f} KB")
print("\nSummary:")
print(f"  coauthor_edges: {len(coauthor_edges)}")
print(f"  author_clusters: {len(author_clusters)}")
print(f"  author_specialization: {len(author_specialization)} authors")
print(f"  author_productivity: {len(author_productivity_out)} authors")
print(f"  source_author: {len(source_author_out)}")
print(f"  source_cooccurrence: {len(source_cooccurrence)}")