#!/usr/bin/env python3
"""Build topic correlation / thematic connection analysis for ZeitScraper dataset.

Approach to multi-category article assignment:
  - Each article has: first keyword = main section/topic (News, Politik, Sport, ...)
  - Category column = subtopic (after cleaning Z+ prefix)
  - Both are 'categories' of the article. An article therefore naturally has >=2 tags.
We treat (kw[0], cat_clean) as the two category tags co-occurring on an article,
and additionally every distinct pair of keywords within the KeyWords list as
co-occurring keywords. For #1 we use the union of {kw[0], cat_clean} and
optionally the cat as a co-occurring pair when both are non-null and distinct.

For #3 (topic correlation via title/description), we count articles where the
topic name appears as a case-insensitive substring of title+description.
"""
import json
import re
import sys
import os
from collections import Counter, defaultdict
from itertools import combinations

import pandas as pd

DATA = "/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet"
OUT = "/opt/data/ZeitScraper/dashboard/correlation_data.json"

PY = sys.executable
print(f"Using {PY} ({sys.version})")


def clean_cat(c):
    if pd.isna(c):
        return None
    c = re.sub(r"Z\+ \(abopflichtiger Inhalt\);\s*", "", c)
    c = re.sub(r"\\n", "", c)
    c = c.strip().strip('"').strip()
    c = re.sub(r"\s+", " ", c)
    return c if c else None


def kw_list(k):
    if pd.isna(k):
        return []
    return [p.strip() for p in k.split(",") if p.strip()]


def main():
    print("Loading dataset ...")
    df = pd.read_parquet(
        DATA,
        columns=[
            "Title", "Description", "Category", "KeyWords",
            "Source", "Published", "Published_Month",
        ],
    )
    print(f"Loaded {len(df):,} rows")

    # Clean categories & keywords
    df["cat_clean"] = df["Category"].apply(clean_cat)
    df["kw_list"] = df["KeyWords"].apply(kw_list)
    df["kw0"] = df["kw_list"].apply(lambda xs: xs[0] if xs else None)
    # Drop rows with no usable tags
    df = df[df["kw_list"].apply(len) > 0].reset_index(drop=True)
    print(f"After dropping kw-less rows: {len(df):,}")

    # Clean source - dpa regional pattern
    def clean_source(s):
        if pd.isna(s):
            return None
        # Normalise whitespace
        s = re.sub(r"\s+", " ", s).strip()
        # take first comma-separated chunk if it looks like a known agency
        # keep full source string but trim trailing initials
        # For regional dpa, extract dpa + region
        m = re.match(r"(dpa(?:\s+\S+)?)", s, re.IGNORECASE)
        if m:
            return m.group(1).strip()
        # First comma-separated part, stripped of trailing codes
        first = s.split(",")[0].strip()
        # remove trailing short codes (<=4 chars)
        first = re.sub(r"[\s,]+[a-z]{1,5}$", "", first, flags=re.IGNORECASE)
        return first if first else None
    df["src_clean"] = df["Source"].apply(clean_source)

    # ------------------------------------------------------------------ #
    # 0. Category rankings
    # ------------------------------------------------------------------ #
    # Primary topic = first keyword (News, Politik, Sport, Gesellschaft, ...)
    topic_counts = Counter(df["kw0"].dropna())
    # Secondary topic = cat_clean (Kriminalität, Bundesliga, ...)
    cat_counts = Counter(df["cat_clean"].dropna())
    # Combined "categories" for top-N lists:
    # use cat_clean primarily (richer topic signal), fall back to kw0
    # For #1 (category co-occurrence) we treat the pair (kw0, cat_clean)
    # as the co-occurring categories of an article.

    top20_cats = [c for c, _ in cat_counts.most_common(20)]
    top15_cats = [c for c, _ in cat_counts.most_common(15)]
    top10_cats = [c for c, _ in cat_counts.most_common(10)]
    print("Top 15 categories:", top15_cats)

    # ------------------------------------------------------------------ #
    # 1. Category co-occurrence edge list
    # ------------------------------------------------------------------ #
    print("Building category co-occurrence ...")
    co_edges = Counter()
    # Each article: pair (kw0, cat_clean) co-occurs
    for kw0, cat in zip(df["kw0"], df["cat_clean"]):
        if kw0 is None or cat is None:
            continue
        kw0 = str(kw0)
        cat = str(cat)
        if kw0 == cat:
            continue
        a, b = sorted((kw0, cat))
        co_edges[(a, b)] += 1
    # Additionally: for articles whose first keyword & cat differ but kw0 itself
    # is one of the main topics, we still have a real co-occurrence. Good.
    # Keep top 100
    category_cooccurrence = [
        {"from": a, "to": b, "weight": w}
        for (a, b), w in co_edges.most_common(100)
    ]
    print(f"  {len(co_edges)} unique category edges; kept top {len(category_cooccurrence)}")

    # ------------------------------------------------------------------ #
    # 2. Keyword-category correlation: top-10 keywords per top-20 category
    # ------------------------------------------------------------------ #
    print("Building keyword-category correlation ...")
    keyword_category = {}
    for cat in top20_cats:
        # articles whose cat_clean == cat
        sub = df[df["cat_clean"] == cat]
        kc = Counter()
        for kws in sub["kw_list"]:
            # exclude the topic itself and first-keyword noise
            for k in kws:
                if k == cat:
                    continue
                kc[k] += 1
        keyword_category[cat] = [k for k, _ in kc.most_common(10)]

    # ------------------------------------------------------------------ #
    # 3. Topic correlation matrix: top-15 cats, co-occur in title+description
    # ------------------------------------------------------------------ #
    print("Building topic correlation matrix (title+description) ...")
    # lowercase search text per category
    text = (df["Title"].fillna("") + " " + df["Description"].fillna("")).str.lower()
    # Build a presence matrix: for each top-15 cat, boolean mask
    presence = {}
    for cat in top15_cats:
        presence[cat] = text.str.contains(re.escape(cat.lower()), regex=True).to_numpy()

    topic_correlation = {}
    import numpy as np
    for c1 in top15_cats:
        topic_correlation[c1] = {}
        for c2 in top15_cats:
            if c1 == c2:
                topic_correlation[c1][c2] = int(presence[c1].sum())
            else:
                both = int(np.logical_and(presence[c1], presence[c2]).sum())
                topic_correlation[c1][c2] = both

    # ------------------------------------------------------------------ #
    # 4. Keyword clusters: top-100 keyword pairs by co-occurrence
    # ------------------------------------------------------------------ #
    print("Building keyword clusters (top 100 pairs) ...")
    kw_edges = Counter()
    # For speed, only consider first N keywords per article (limit explosion)
    MAX_KW_PER_ART = 15
    for kws in df["kw_list"]:
        # dedupe preserving order
        seen = set()
        uniq = []
        for k in kws:
            if k in seen:
                continue
            seen.add(k)
            uniq.append(k)
        if len(uniq) > MAX_KW_PER_ART:
            uniq = uniq[:MAX_KW_PER_ART]
        for a, b in combinations(sorted(uniq), 2):
            kw_edges[(a, b)] += 1
    keyword_clusters = [
        {"from": a, "to": b, "weight": w}
        for (a, b), w in kw_edges.most_common(100)
    ]
    print(f"  {len(kw_edges)} unique keyword edges; kept top {len(keyword_clusters)}")

    # ------------------------------------------------------------------ #
    # 5. Temporal correlation: monthly counts, Pearson, top-10 topics
    # ------------------------------------------------------------------ #
    print("Building temporal correlation ...")
    # use kw0 (main topic) for monthly counts
    df_t = df.dropna(subset=["kw0", "Published_Month"])
    months = sorted(df_t["Published_Month"].unique())
    # build series per top-10 cat (using cat_clean here since it's the
    # richer topic signal; fall back to kw0 if cat_clean missing)
    temporal = {}
    series = {}
    for cat in top10_cats:
        # articles tagged with this category
        mask = df_t["cat_clean"] == cat
        cnt = df_t.loc[mask].groupby("Published_Month").size()
        cnt = cnt.reindex(months, fill_value=0).to_numpy(dtype=float)
        series[cat] = cnt
    import numpy as np
    for c1 in top10_cats:
        temporal[c1] = {}
        for c2 in top10_cats:
            if c1 == c2:
                temporal[c1][c2] = 1.0
                continue
            x = series[c1]
            y = series[c2]
            if x.std() == 0 or y.std() == 0:
                temporal[c1][c2] = 0.0
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            if r != r:  # NaN
                r = 0.0
            temporal[c1][c2] = round(r, 4)

    # ------------------------------------------------------------------ #
    # 6. Source-topic affinity: top-15 sources x category counts
    # ------------------------------------------------------------------ #
    print("Building source-topic affinity ...")
    src_counts = Counter(df["src_clean"].dropna())
    top15_srcs = [s for s, _ in src_counts.most_common(15)]
    source_topic = {}
    for src in top15_srcs:
        sub = df[df["src_clean"] == src]
        cc = Counter()
        for cat in sub["cat_clean"].dropna():
            cc[cat] += 1
        # take top categories per source
        source_topic[src] = dict(cc.most_common(15))

    # ------------------------------------------------------------------ #
    # Save
    # ------------------------------------------------------------------ #
    out = {
        "category_cooccurrence": category_cooccurrence,
        "keyword_category": keyword_category,
        "topic_correlation": topic_correlation,
        "keyword_clusters": keyword_clusters,
        "temporal_correlation": temporal,
        "source_topic": source_topic,
    }
    print(f"Writing {OUT} ...")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Done. Size: {os.path.getsize(OUT):,} bytes")

    # quick sanity print
    print("\n--- summary ---")
    print("category_cooccurrence top 5:", category_cooccurrence[:5])
    print("keyword_category sample:", list(keyword_category.items())[:2])
    print("topic_correlation sample:", list(topic_correlation.items())[:1])
    print("keyword_clusters top 5:", keyword_clusters[:5])
    print("temporal_correlation sample:", list(temporal.items())[:1])
    print("source_topic sample:", list(source_topic.items())[:1])


if __name__ == "__main__":
    main()