#!/usr/bin/env python3
"""Copy-paste detector for dpa vs rewritten content - optimized."""
import json
import os
import re
import sys
import time
from collections import defaultdict
from itertools import combinations

import pandas as pd

PARQUET = "/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet"
OUT = "/opt/data/ZeitScraper/dashboard/copy_paste_data.json"

WORD_RE = re.compile(r"\w+", re.UNICODE)


def word_set(text):
    if not isinstance(text, str) or not text.strip():
        return frozenset()
    return frozenset(w.lower() for w in WORD_RE.findall(text))


def jaccard(a, b):
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    df = pd.read_parquet(PARQUET)
    log(f"Loaded {len(df)} articles")

    df["Source"] = df["Source"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["Title"] = df["Title"].fillna("").astype(str)
    df["KeyWords"] = df["KeyWords"].fillna("").astype(str)
    df["Published_Date"] = pd.to_datetime(df["Published_Date"]).dt.date.astype(str)

    dpa_regional = sorted(
        s
        for s in df["Source"].unique()
        if s.startswith("dpa ") and s not in ("dpa Service",) and len(df[df["Source"] == s]) >= 1000
    )
    dpa_all = dpa_regional + ["dpa"]

    zeit_mask = (
        df["Source"].str.contains("ZEIT", case=False, na=False)
        & ~df["Source"].str.lower().str.contains("dpa", na=False)
    )
    zeit_df = df[zeit_mask].copy()
    dpa_mask = df["Source"].isin(dpa_all)
    dpa_df = df[dpa_mask].copy()

    dpa_df["desc_words"] = dpa_df["Description"].map(word_set)
    dpa_df["title_words"] = dpa_df["Title"].map(word_set)
    zeit_df["desc_words"] = zeit_df["Description"].map(word_set)
    zeit_df["title_words"] = zeit_df["Title"].map(word_set)

    log(f"dpa: {len(dpa_df)} articles, ZEIT: {len(zeit_df)} articles")

    # ---- 1 & 2 & 5 & 6: per-source analysis ----
    duplicate_groups = []
    verbatim_ratio = {}
    unique_ratio = {}
    keyword_overlap = {}

    for src in dpa_all:
        t0 = time.time()
        sub = dpa_df[dpa_df["Source"] == src].copy()
        total = len(sub)
        if total == 0:
            continue

        # Precompute arrays for fast access
        sub = sub.reset_index(drop=True)
        desc_ws = sub["desc_words"].tolist()
        kw_list = sub["KeyWords"].tolist()
        titles = sub["Title"].tolist()
        dates = sub["Published_Date"].tolist()

        # Build date -> list of indices
        date_to_idxs = defaultdict(list)
        for idx, d in enumerate(dates):
            date_to_idxs[d].append(idx)

        has_dup = [False] * total
        same_kw_count = 0
        groups_found = []

        for date, idxs in date_to_idxs.items():
            n = len(idxs)
            if n < 2:
                continue

            # --- description near-dups (Jaccard > 0.80) ---
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

            local_descs = [desc_ws[i] for i in idxs]
            for i, j in combinations(range(n), 2):
                if jaccard(local_descs[i], local_descs[j]) > 0.80:
                    union(i, j)
                    has_dup[idxs[i]] = True
                    has_dup[idxs[j]] = True

            comp = defaultdict(list)
            for k in range(n):
                comp[find(k)].append(k)
            for members in comp.values():
                if len(members) >= 2:
                    groups_found.append(
                        {
                            "source": src,
                            "date": str(date),
                            "group_size": len(members),
                            "sample_title": titles[idxs[members[0]]],
                        }
                    )

            # --- keyword overlap: identical KeyWords ---
            kw_counter = defaultdict(int)
            for kw in [kw_list[i] for i in idxs]:
                if kw:
                    kw_counter[kw] += 1
            for kw, cnt in kw_counter.items():
                if cnt >= 2:
                    same_kw_count += cnt

        duplicates = sum(has_dup)
        ratio = duplicates / total if total else 0.0
        verbatim_ratio[src] = {"total": int(total), "duplicates": int(duplicates), "ratio": round(ratio, 4)}
        unique_ratio[src] = round((1 - ratio) * 100, 2)
        keyword_overlap[src] = {
            "same_kw_count": int(same_kw_count),
            "total": int(total),
            "ratio": round(same_kw_count / total, 4) if total else 0.0,
        }

        groups_found.sort(key=lambda g: g["group_size"], reverse=True)
        duplicate_groups.extend(groups_found)
        log(f"  {src}: {total} articles, {duplicates} dups ({ratio:.3f}), {len(groups_found)} groups in {time.time()-t0:.1f}s")

    duplicate_groups.sort(key=lambda g: g["group_size"], reverse=True)
    duplicate_groups_top = duplicate_groups[:30]
    log(f"Total duplicate groups: {len(duplicate_groups)}")

    # ---- 3: Cross-source title overlap ----
    log("Computing cross-source title overlap...")
    combined = pd.concat([dpa_df, zeit_df], ignore_index=True)
    combined = combined[combined["title_words"].map(len) > 0].reset_index(drop=True)
    log(f"Combined for cross-source: {len(combined)}")

    title_list = combined["title_words"].tolist()
    title_text = combined["Title"].tolist()
    source_list = combined["Source"].tolist()
    date_list = combined["Published_Date"].tolist()

    by_date = defaultdict(list)
    for idx, d in enumerate(date_list):
        by_date[d].append(idx)

    cross_source_overlap = []
    CAP = 5000
    days_processed = 0
    for date, idxs in by_date.items():
        days_processed += 1
        if len(cross_source_overlap) >= CAP:
            break
        # group by source
        src_to_local = defaultdict(list)
        for li, gi in enumerate(idxs):
            src_to_local[source_list[gi]].append(li)
        sources_on_day = list(src_to_local.keys())
        if len(sources_on_day) < 2:
            continue
        for s1, s2 in combinations(sources_on_day, 2):
            locals1 = src_to_local[s1]
            locals2 = src_to_local[s2]
            if len(locals1) > 40 or len(locals2) > 40:
                continue  # cap blowup
            for li1 in locals1:
                gi1 = idxs[li1]
                tw1 = title_list[gi1]
                t1 = title_text[gi1]
                for li2 in locals2:
                    gi2 = idxs[li2]
                    sim = jaccard(tw1, title_list[gi2])
                    if sim > 0.70:
                        cross_source_overlap.append(
                            {
                                "title1": t1,
                                "title2": title_text[gi2],
                                "source1": s1,
                                "source2": s2,
                                "date": str(date),
                                "similarity": round(sim, 4),
                            }
                        )
                        if len(cross_source_overlap) >= CAP:
                            break
                if len(cross_source_overlap) >= CAP:
                    break
            if len(cross_source_overlap) >= CAP:
                break
        if days_processed % 100 == 0:
            log(f"  cross-source: {days_processed} days, {len(cross_source_overlap)} overlaps")

    cross_source_overlap.sort(key=lambda x: x["similarity"], reverse=True)
    cross_source_overlap_top = cross_source_overlap[:30]
    log(f"Cross-source overlaps found: {len(cross_source_overlap)}")

    # ---- 4: length comparison ----
    dpa_lengths = dpa_df["Desc_Length"].dropna().astype(int)
    zeit_lengths = zeit_df["Desc_Length"].dropna().astype(int)
    length_comparison = {
        "dpa": {
            "avg": round(float(dpa_lengths.mean()), 2),
            "median": round(float(dpa_lengths.median()), 2),
            "n": int(len(dpa_lengths)),
        },
        "zeit": {
            "avg": round(float(zeit_lengths.mean()), 2),
            "median": round(float(zeit_lengths.median()), 2),
            "n": int(len(zeit_lengths)),
        },
    }

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
            "total_duplicate_groups": len(duplicate_groups),
            "total_cross_source_overlaps": len(cross_source_overlap),
        },
    }

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    log(f"Written: {OUT}")
    log("DONE")


if __name__ == "__main__":
    main()