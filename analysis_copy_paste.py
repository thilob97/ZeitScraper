#!/usr/bin/env python3
"""Copy-paste detector for dpa vs rewritten content in articles_consolidated.parquet."""
import json
import re
from collections import defaultdict
from itertools import combinations

import pandas as pd

PARQUET = "/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet"
OUT = "/opt/data/ZeitScraper/dashboard/copy_paste_data.json"

WORD_RE = re.compile(r"\b\w+\b", re.UNICODE)


def word_set(text):
    if not isinstance(text, str) or not text.strip():
        return set()
    return set(w.lower() for w in WORD_RE.findall(text))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def main():
    df = pd.read_parquet(PARQUET)
    df["Source"] = df["Source"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["Title"] = df["Title"].fillna("").astype(str)
    df["KeyWords"] = df["KeyWords"].fillna("").astype(str)
    df["Published_Date"] = pd.to_datetime(df["Published_Date"]).dt.date.astype(str)

    # ---- Identify source groups ----
    # dpa regional: starts with "dpa " (regional variant) or == "dpa" (national)
    dpa_regional_sources = sorted(
        s for s in df["Source"].unique() if s.startswith("dpa ") and s not in ("dpa Service",)
    )
    # exclude tiny noise sources
    dpa_regional_sources = [s for s in dpa_regional_sources if len(df[df["Source"] == s]) >= 1000]
    dpa_all = dpa_regional_sources + ["dpa"]

    # ZEIT-sourced (non-dpa)
    zeit_mask = (
        df["Source"].str.contains("ZEIT", case=False, na=False)
        & ~df["Source"].str.lower().str.contains("dpa", na=False)
    )
    zeit_df = df[zeit_mask].copy()
    dpa_mask = df["Source"].isin(dpa_all)
    dpa_df = df[dpa_mask].copy()

    # Precompute word sets
    dpa_df = dpa_df.copy()
    dpa_df["desc_words"] = dpa_df["Description"].map(word_set)
    dpa_df["title_words"] = dpa_df["Title"].map(word_set)
    zeit_df = zeit_df.copy()
    zeit_df["desc_words"] = zeit_df["Description"].map(word_set)
    zeit_df["title_words"] = zeit_df["Title"].map(word_set)

    # ---- 1 & 2: Near-duplicate detection within same source+day ----
    duplicate_groups = []
    verbatim_ratio = {}
    unique_ratio = {}
    keyword_overlap = {}

    for src in dpa_all:
        sub = dpa_df[dpa_df["Source"] == src]
        total = len(sub)
        if total == 0:
            continue

        has_dup_desc = [False] * total  # per-article flag
        same_kw_count = 0
        groups_found = []

        # group by date
        for date, grp in sub.groupby("Published_Date"):
            n = len(grp)
            if n < 2:
                continue
            idxs = grp.index.tolist()
            # Build per-row word sets & keyword strings
            desc_ws = [sub.at[i, "desc_words"] for i in idxs]
            kw_list = [sub.at[i, "KeyWords"] for i in idxs]
            titles = [sub.at[i, "Title"] for i in idxs]

            # ---- Description near-duplicates (>80% Jaccard) ----
            # union-find per group within this day
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            def union(a, b):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb

            for i, j in combinations(range(n), 2):
                if jaccard(desc_ws[i], desc_ws[j]) > 0.80:
                    union(i, j)
                    has_dup_desc[list(sub.index).index(idxs[i])] = True
                    has_dup_desc[list(sub.index).index(idxs[j])] = True

            # collect groups with >1 member
            comp = defaultdict(list)
            for k in range(n):
                comp[find(k)].append(k)
            for members in comp.values():
                if len(members) >= 2:
                    sample_idx = idxs[members[0]]
                    groups_found.append(
                        {
                            "source": src,
                            "date": str(date),
                            "group_size": len(members),
                            "sample_title": sub.at[sample_idx, "Title"],
                        }
                    )

            # ---- Keyword overlap: identical KeyWords within same source+day ----
            kw_counter = defaultdict(int)
            for kw in kw_list:
                if kw:
                    kw_counter[kw] += 1
            for kw, cnt in kw_counter.items():
                if cnt >= 2:
                    same_kw_count += cnt  # articles sharing that kw string

        # verbatim ratio: articles that have at least one near-duplicate desc within same day
        duplicates = sum(has_dup_desc)
        ratio = duplicates / total if total else 0.0
        verbatim_ratio[src] = {
            "total": int(total),
            "duplicates": int(duplicates),
            "ratio": round(ratio, 4),
        }

        # unique ratio: percentage of articles with NO near-duplicate
        unique_pct = round((1 - ratio) * 100, 2)
        unique_ratio[src] = unique_pct

        # keyword overlap
        keyword_overlap[src] = {
            "same_kw_count": int(same_kw_count),
            "total": int(total),
            "ratio": round(same_kw_count / total, 4) if total else 0.0,
        }

        # extend duplicate_groups
        groups_found.sort(key=lambda g: g["group_size"], reverse=True)
        duplicate_groups.extend(groups_found)

    # top 30 duplicate groups overall by size
    duplicate_groups.sort(key=lambda g: g["group_size"], reverse=True)
    duplicate_groups_top = duplicate_groups[:30]

    # ---- 3: Cross-source title overlap (>70%) same day, different sources ----
    cross_source_overlap = []
    # Use dpa_df + zeit_df combined; compare across sources within same day
    combined = pd.concat([dpa_df, zeit_df], ignore_index=True)
    # Only keep rows with non-empty title
    combined = combined[combined["title_words"].map(len) > 0].reset_index(drop=True)

    # To keep tractable: for each day, group by source, then compare titles across source-pairs.
    # Limit to reasonable days; cap results collection.
    CAP_RESULTS = 2000  # we'll keep top 30 at the end

    by_date = combined.groupby("Published_Date")
    n_days = len(by_date)
    processed = 0
    for date, grp in by_date:
        processed += 1
        if len(cross_source_overlap) >= CAP_RESULTS * 2:
            break
        # group by source
        src_groups = {s: g2 for s, g2 in grp.groupby("Source")}
        sources_on_day = list(src_groups.keys())
        if len(sources_on_day) < 2:
            continue
        for s1, s2 in combinations(sources_on_day, 2):
            g1 = src_groups[s1]
            g2 = src_groups[s2]
            # avoid O(n^2) blowups: cap pairs per day-pair
            if len(g1) > 50 or len(g2) > 50:
                continue
            for i in range(len(g1)):
                ti = g1.iloc[i]["title_words"]
                t1_text = g1.iloc[i]["Title"]
                for j in range(len(g2)):
                    tj = g2.iloc[j]["title_words"]
                    sim = jaccard(ti, tj)
                    if sim > 0.70:
                        cross_source_overlap.append(
                            {
                                "title1": t1_text,
                                "title2": g2.iloc[j]["Title"],
                                "source1": s1,
                                "source2": s2,
                                "date": str(date),
                                "similarity": round(sim, 4),
                            }
                        )
                        if len(cross_source_overlap) >= CAP_RESULTS * 2:
                            break
                if len(cross_source_overlap) >= CAP_RESULTS * 2:
                    break
            if len(cross_source_overlap) >= CAP_RESULTS * 2:
                break

    cross_source_overlap.sort(key=lambda x: x["similarity"], reverse=True)
    cross_source_overlap_top = cross_source_overlap[:30]

    # ---- 4: Description length comparison ----
    dpa_lengths = dpa_df["Desc_Length"].dropna().astype(int)
    zeit_lengths = zeit_df["Desc_Length"].dropna().astype(int)
    length_comparison = {
        "dpa": {
            "avg": round(float(dpa_lengths.mean()), 2) if len(dpa_lengths) else 0,
            "median": round(float(dpa_lengths.median()), 2) if len(dpa_lengths) else 0,
            "n": int(len(dpa_lengths)),
        },
        "zeit": {
            "avg": round(float(zeit_lengths.mean()), 2) if len(zeit_lengths) else 0,
            "median": round(float(zeit_lengths.median()), 2) if len(zeit_lengths) else 0,
            "n": int(len(zeit_lengths)),
        },
    }

    # ---- Assemble output ----
    output = {
        "duplicate_groups": duplicate_groups_top,
        "verbatim_ratio": verbatim_ratio,
        "cross_source_overlap": cross_source_overlap_top,
        "length_comparison": length_comparison,
        "keyword_overlap": keyword_overlap,
        "unique_ratio": unique_ratio,
        "meta": {
            "total_articles": int(len(df)),
            "dpa_articles": int(len(dpa_df)),
            "zeit_articles": int(len(zeit_df)),
            "dpa_sources_analyzed": dpa_all,
        },
    }

    import os
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"Written: {OUT}")
    print(f"duplicate_groups total found: {len(duplicate_groups)} (top 30 saved)")
    print(f"cross_source_overlap found: {len(cross_source_overlap)} (top 30 saved)")
    print("verbatim_ratio:")
    for k, v in verbatim_ratio.items():
        print(f"  {k}: {v}")
    print("unique_ratio:")
    for k, v in unique_ratio.items():
        print(f"  {k}: {v}%")
    print("length_comparison:", length_comparison)


if __name__ == "__main__":
    main()