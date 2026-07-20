#!/usr/bin/env python3
"""Information Gaps & Dark Patterns analysis for ZEIT articles."""
import json
import re
import warnings
from collections import Counter, defaultdict

import pandas as pd

warnings.filterwarnings("ignore")

INPUT = "/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet"
OUTPUT = "/opt/data/ZeitScraper/dashboard/gaps_data.json"


def is_paywalled(row):
    """A paywalled article: either Paywall flag True OR category contains Z+ marker."""
    if row.get("Paywall", False):
        return True
    cat = row.get("Category", "")
    if isinstance(cat, str) and "Z+" in cat:
        return True
    return False


def extract_keywords(kw_str):
    if not kw_str or pd.isna(kw_str) or not isinstance(kw_str, str):
        return []
    return [k.strip() for k in kw_str.split(",") if k.strip()]


def extract_category_clean(cat):
    """Strip Z+ prefix and whitespace from category."""
    if not cat or pd.isna(cat):
        return ""
    c = cat
    # Remove Z+ prefix if present
    c = re.sub(r"Z\+\s*\(abopflichtiger Inhalt\);?\s*\\n\s*\\n\s*", "", c)
    c = re.sub(r"Z\+\s*\(abopflichtiger Inhalt\);?\s*", "", c)
    c = c.strip()
    # collapse whitespace
    c = re.sub(r"\s+", " ", c)
    return c


def main():
    print("Loading data...")
    df = pd.read_parquet(INPUT)
    print(f"Loaded {len(df)} articles")

    # Precompute
    df["month"] = df["Published_Month"]
    df["year"] = df["Published_Year"].astype("Int64")
    df["kw_list"] = df["KeyWords"].apply(extract_keywords)
    df["cat_clean"] = df["Category"].apply(extract_category_clean)
    df["paywalled"] = df.apply(is_paywalled, axis=1)

    result = {}

    # ========================================================
    # 1. Disappeared topics: top-50 keywords from 2024 that vanish in 2025/2026
    # ========================================================
    print("1. Disappeared topics...")
    df_2024 = df[df["year"] == 2024]
    df_2025_26 = df[df["year"].isin([2025, 2026])]

    # Count keyword frequency in 2024
    kw_counter_2024 = Counter()
    for kws in df_2024["kw_list"]:
        kw_counter_2024.update(kws)

    # Use a larger pool: top-500 keywords from 2024 to find disappeared ones
    # (the strict top-50 are all generic and persist; broaden to find topics that actually vanished)
    top_pool_2024 = [k for k, _ in kw_counter_2024.most_common(500)]

    # Pre-compute keyword sets per article for 2025/26 for efficiency
    print("  Building keyword index for 2025/26...")
    kw_25_26 = Counter()
    for kws in df_2025_26["kw_list"]:
        kw_25_26.update(set(kws))

    disappeared = []
    for kw in top_pool_2024:
        count_25_26 = kw_25_26.get(kw, 0)
        if count_25_26 > 0:
            continue  # still appears in 2025/26

        # This keyword completely disappeared
        mask_all = df["kw_list"].apply(lambda x: kw in x)
        all_months = df[mask_all]["month"].dropna().sort_values()
        if len(all_months) == 0:
            continue
        last_month = str(all_months.iloc[-1])

        mask_2024 = df_2024["kw_list"].apply(lambda x: kw in x)
        monthly = df_2024[mask_2024].groupby("month").size()
        peak_count = int(monthly.max()) if len(monthly) > 0 else 0
        months_active = len(monthly)

        disappeared.append(
            {
                "keyword": kw,
                "last_month": last_month,
                "peak_count": peak_count,
                "months_active": months_active,
            }
        )

    # If top-500 doesn't yield 30, broaden to keywords with >=50 mentions in 2024
    if len(disappeared) < 30:
        print(f"  Only {len(disappeared)} from top-500; broadening to >=50 mentions pool...")
        broader_pool = [k for k, c in kw_counter_2024.most_common() if c >= 50]
        for kw in broader_pool:
            if any(d["keyword"] == kw for d in disappeared):
                continue
            if kw_25_26.get(kw, 0) > 0:
                continue
            mask_all = df["kw_list"].apply(lambda x: kw in x)
            all_months = df[mask_all]["month"].dropna().sort_values()
            if len(all_months) == 0:
                continue
            last_month = str(all_months.iloc[-1])
            mask_2024 = df_2024["kw_list"].apply(lambda x: kw in x)
            monthly = df_2024[mask_2024].groupby("month").size()
            peak_count = int(monthly.max()) if len(monthly) > 0 else 0
            months_active = len(monthly)
            disappeared.append(
                {
                    "keyword": kw,
                    "last_month": last_month,
                    "peak_count": peak_count,
                    "months_active": months_active,
                }
            )

    disappeared.sort(key=lambda x: x["peak_count"], reverse=True)
    result["disappeared_topics"] = disappeared[:30]
    print(f"  Found {len(disappeared)} disappeared topics (reporting top {min(30, len(disappeared))})")

    # ========================================================
    # 2. Keyword diversity per month
    # ========================================================
    print("2. Keyword diversity...")
    keyword_diversity = {}
    for month, group in df.groupby("month"):
        unique_kws = set()
        for kws in group["kw_list"]:
            unique_kws.update(kws)
        keyword_diversity[str(month)] = len(unique_kws)
    result["keyword_diversity"] = keyword_diversity

    # ========================================================
    # 3. Paywall analysis
    # ========================================================
    print("3. Paywall analysis...")
    # By category (use cleaned category)
    paywall_by_cat = {}
    cat_stats = defaultdict(lambda: {"total": 0, "paywalled": 0})
    for _, row in df.iterrows():
        cat = row["cat_clean"]
        if not cat:
            cat = "(uncategorized)"
        cat_stats[cat]["total"] += 1
        if row["paywalled"]:
            cat_stats[cat]["paywalled"] += 1

    for cat, stats in cat_stats.items():
        ratio = stats["paywalled"] / stats["total"] if stats["total"] > 0 else 0
        paywall_by_cat[cat] = {
            "total": stats["total"],
            "paywalled": stats["paywalled"],
            "ratio": round(ratio, 4),
        }

    # Sort by ratio descending, keep top 20 (with at least 50 articles to be meaningful)
    paywall_by_cat_sorted = dict(
        sorted(
            [(k, v) for k, v in paywall_by_cat.items() if v["total"] >= 50],
            key=lambda x: x[1]["ratio"],
            reverse=True,
        )[:20]
    )
    result["paywall_by_category"] = paywall_by_cat_sorted

    # Paywall trend by month
    paywall_trend = {}
    for month, group in df.groupby("month"):
        total = len(group)
        pw = int(group["paywalled"].sum())
        ratio = pw / total if total > 0 else 0
        paywall_trend[str(month)] = round(ratio, 4)
    result["paywall_trend"] = paywall_trend

    # ========================================================
    # 4. Copy-paste detection (dpa sources, same day, >80% word overlap)
    # ========================================================
    print("4. Copy-paste detection...")
    # Identify dpa variants
    dpa_mask = df["Source"].str.contains("dpa", case=False, na=False)
    dpa_df = df[dpa_mask].copy()
    print(f"  dpa articles: {len(dpa_df)}")

    # Group by Source + date
    dpa_df["date"] = pd.to_datetime(dpa_df["Published"]).dt.date.astype(str)
    groups = dpa_df.groupby(["Source", "date"])

    total_groups = 0
    duplicate_articles = 0
    by_source = defaultdict(int)

    for (source, date), group in groups:
        if len(group) < 2:
            continue
        total_groups += 1
        descs = group["Description"].fillna("").tolist()
        indices = group.index.tolist()
        # Tokenize all descriptions
        tokenized = []
        for d in descs:
            words = set(d.lower().split())
            tokenized.append(words)

        # Pairwise comparison
        seen_as_dup = set()
        for i in range(len(descs)):
            if i in seen_as_dup:
                continue
            for j in range(i + 1, len(descs)):
                if j in seen_as_dup:
                    continue
                wi = tokenized[i]
                wj = tokenized[j]
                if len(wi) == 0 and len(wj) == 0:
                    continue
                # word overlap ratio (Jaccard-like, but using overlap coefficient)
                overlap = len(wi & wj)
                min_size = min(len(wi), len(wj))
                if min_size == 0:
                    continue
                ratio = overlap / min_size
                if ratio > 0.80:
                    duplicate_articles += 1
                    seen_as_dup.add(j)
                    by_source[source] += 1

    total_dpa = len(dpa_df)
    duplicate_ratio = duplicate_articles / total_dpa if total_dpa > 0 else 0
    result["copy_paste_stats"] = {
        "total_groups": total_groups,
        "duplicate_articles": duplicate_articles,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "by_source": dict(by_source),
    }
    print(f"  Found {duplicate_articles} duplicate articles in {total_groups} groups")

    # ========================================================
    # 5. Ghost author detection
    # ========================================================
    print("5. Ghost author detection...")
    # 3-letter codes appear in Source field (lowercase, possibly with tabs/whitespace around them)
    # These are anonymous editorial codes — "ghost authors"
    code_pattern = re.compile(r"\b([a-z]{3})\b")

    # German words and common tokens to exclude from codes
    non_codes = {
        "dpa", "der", "die", "das", "und", "aft", "als", "bei", "dem", "den",
        "von", "mit", "vor", "auf", "aus", "durch", "ist", "war", "hat",
        "wie", "was", "wer", "wann", "wo", "wohin", "woher", "ohne",
        "mit", "nur", "nun", "aber", "oder", "sondern", "weder",
        "tok", "mit", "uri", "via",
    }

    # For each article, extract codes from Source
    code_counts = defaultdict(int)
    code_names = defaultdict(set)  # code -> set of full names seen alongside

    # Helper: parse concatenated names from Author field
    # Author field has names like "Maline HofmannEric Voigt" (no separator between names)
    def parse_names(author):
        if not author or not isinstance(author, str):
            return []
        # First try to split on boundaries where a lowercase letter is followed by uppercase
        # e.g. "HofmannEric" -> "Hofmann", "Eric"
        split_text = re.sub(r"([a-zäöüß])([A-ZÄÖÜ])", r"\1 \2", author)
        # Also handle "Dr." and "Prof." prefixes
        split_text = re.sub(r"(Dr|Prof)\.\s*", "", split_text)
        # Now extract name-like patterns: First Last (2 capitalized words)
        names = re.findall(r"[A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)+", split_text)
        # Filter out title-like phrases (questions, common non-name patterns)
        bad_start_patterns = ("Welche", "Welcher", "Welches", "Wie ", "Was ", "Wo ", "Warum",
                              "Sind ", "Haben ", "Küste", "Israels", "Deutschland",
                              "Designierte", "Neuer", "Großer", "Autoteile", "Importe",
                              "Handelspartnern", "Aluminium", "Bundeskanzleramt", "Kardinal")
        filtered = []
        for name in names:
            if name.startswith(bad_start_patterns):
                continue
            # Must have exactly 2-4 words, each at least 2 chars
            parts = name.split()
            if 2 <= len(parts) <= 4 and all(len(p) >= 2 for p in parts):
                filtered.append(name)
        return filtered

    for _, row in df.iterrows():
        source = row.get("Source", "")
        author = row.get("Author", "")
        if not isinstance(source, str):
            source = ""
        if not isinstance(author, str):
            author = ""

        # Find 3-letter codes in Source
        src_codes = set()
        for m in code_pattern.finditer(source):
            code = m.group(1)
            if code in non_codes:
                continue
            src_codes.add(code)

        # Extract full names from Author column (primary source of names)
        full_names = parse_names(author)

        for code in src_codes:
            code_counts[code] += 1
            for name in full_names:
                code_names[code].add(name)

    # Only keep codes that appear frequently enough (>= 10)
    ghost_authors = {}
    for code, count in sorted(code_counts.items(), key=lambda x: x[1], reverse=True):
        if count < 10:
            continue
        ghost_authors[code] = {
            "count": count,
            "possible_names": sorted(code_names[code])[:20],  # limit
        }
    result["ghost_authors"] = ghost_authors
    print(f"  Found {len(ghost_authors)} ghost author codes")

    # ========================================================
    # 6. SEO patterns
    # ========================================================
    print("6. SEO patterns...")
    seo_by_cat = defaultdict(lambda: {"optimized_count": 0, "total": 0, "stuffing_count": 0})

    for _, row in df.iterrows():
        cat = row["cat_clean"] or "(uncategorized)"
        title = row.get("Title", "") or ""
        kws = row.get("kw_list", [])

        seo_by_cat[cat]["total"] += 1

        if not kws or not title:
            continue

        title_lower = title.lower()
        # Count how many keywords appear in title
        kw_in_title = 0
        for kw in kws:
            if kw and kw.lower() in title_lower:
                kw_in_title += 1

        # Optimized: at least 1 keyword appears in title
        if kw_in_title >= 1:
            seo_by_cat[cat]["optimized_count"] += 1

        # Keyword stuffing: 4+ keywords from KeyWords list appear in title
        if kw_in_title >= 4:
            seo_by_cat[cat]["stuffing_count"] += 1

    seo_patterns = {}
    for cat, stats in seo_by_cat.items():
        ratio = stats["optimized_count"] / stats["total"] if stats["total"] > 0 else 0
        seo_patterns[cat] = {
            "optimized_count": stats["optimized_count"],
            "total": stats["total"],
            "ratio": round(ratio, 4),
            "stuffing_count": stats["stuffing_count"],
        }

    # Sort by total articles descending, top 15
    seo_patterns_sorted = dict(
        sorted(seo_patterns.items(), key=lambda x: x[1]["total"], reverse=True)[:15]
    )
    result["seo_patterns"] = seo_patterns_sorted

    # ========================================================
    # 7. Description emptiness
    # ========================================================
    print("7. Description emptiness...")
    empty_desc_by_source = defaultdict(int)
    empty_desc_by_cat = defaultdict(int)

    for _, row in df.iterrows():
        desc = row.get("Description", "")
        if not isinstance(desc, str):
            desc = ""
        if len(desc) < 20:
            source = row.get("Source", "") or ""
            if not isinstance(source, str):
                source = "(unknown)"
            else:
                source = source.strip() or "(unknown)"
            empty_desc_by_source[source] += 1
            cat = row["cat_clean"] or "(uncategorized)"
            empty_desc_by_cat[cat] += 1

    # Sort by count descending, top 30
    empty_by_source_sorted = dict(
        sorted(empty_desc_by_source.items(), key=lambda x: x[1], reverse=True)[:30]
    )
    empty_by_cat_sorted = dict(
        sorted(empty_desc_by_cat.items(), key=lambda x: x[1], reverse=True)[:30]
    )
    result["empty_descriptions"] = {
        "by_source": empty_by_source_sorted,
        "by_category": empty_by_cat_sorted,
    }

    # ========================================================
    # Save
    # ========================================================
    print(f"Saving to {OUTPUT}...")
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("Done!")
    print()
    print("Summary:")
    print(f"  Disappeared topics: {len(result['disappeared_topics'])}")
    print(f"  Keyword diversity months: {len(result['keyword_diversity'])}")
    print(f"  Paywall categories: {len(result['paywall_by_category'])}")
    print(f"  Paywall trend months: {len(result['paywall_trend'])}")
    print(f"  Copy-paste duplicates: {result['copy_paste_stats']['duplicate_articles']}")
    print(f"  Ghost author codes: {len(result['ghost_authors'])}")
    print(f"  SEO patterns categories: {len(result['seo_patterns'])}")
    print(f"  Empty descriptions by source: {len(result['empty_descriptions']['by_source'])}")


if __name__ == "__main__":
    main()