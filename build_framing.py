#!/usr/bin/env python3
"""Framing & narrative detection analysis for zeit.de articles.

Outputs framing_data.json with keys:
  politician_framing, narrative_shifts, sentiment_monthly,
  clickbait, title_desc_sentiment
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

DATA = Path("/opt/data/ZeitScraper/data/processed/articles_consolidated.parquet")
OUT = Path("/opt/data/ZeitScraper/dashboard/framing_data.json")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLITICIANS = ["Scholz", "Merz", "Trump", "Putin", "Selensky", "Habeck",
               "Söder", "AfD", "Lindner"]
# Note: "Selensky" matches "Selenskyj"; "Merz" appears twice in original list, dedup.

MIGRATION_TERMS = ["Flüchtlinge", "Migranten", "Zuwanderer",
                   "Asylbewerber", "Geflüchtete"]
KLIMA_TERMS = ["Klimawandel", "Klimakrise", "Klimaschutz", "Erderwärmung"]
KI_TERMS = ["KI", "Künstliche Intelligenz", "ChatGPT", "AI"]
TRACKED_TERMS = MIGRATION_TERMS + KLIMA_TERMS + KI_TERMS

POS_WORDS = ["Erfolg", "Gewinn", "Fortschritt", "positiv", "gut", "Sieg"]
NEG_WORDS = ["Krise", "Verlust", "Scheitern", "negativ", "Skandal", "Problem"]

CLICKBAIT_WORDS = ["Schock", "Skandal", "überraschend", "enthüllt",
                   "Sensation", "Wahnsinn", "ungenannt", "versteckt",
                   "geheim", "enthüllt:"]
CLICKBAIT_TITLE_PUNCT = ["?", "!"]

# German stopword + filler list (lightweight) to filter politician co-occurrence
STOPWORDS = set("""
der die das ein eine einer eines den dem des einer einen einem
und oder aber weil wenn als dass daß wie wo was wer wen wem
ist sind war waren sein gewesen wird werden worden worden
im in an am auf mit bei von zu zur zum für nach vor über unter
sich er sie es wir ihr ihr seid ich du man diese dieser dieses
nicht auch noch nur schon noch mehr sehr so doch ja nein
auch noch schon wieder gegen aus um durch
um zu beim zur zum
auf den die das
es ist war wird
hat haben hatte hatten gehabt
jetzt lesen hier finden sie informationen thema
alles alles weitere mehr alles weitere
unter weiter unten
hier hierzu dazu damit
nun etwa obwohl
""".split())

# Words that should always be filtered out of framing output (template residue,
# common verbs, deictic markers, generic newsroom fillers).
FRAMING_FILTER = set("""
jetzt lesen finden informationen thema hier sie ihre ihrer ihnen
ihnen wir unser uns ihr eure es ist war wird werden wurde wurden
hat haben hatte hatten kannst kann können könnte könnte
sollen will muss möchte will wollen wollte
sagt sagt sagte sagt sagte
auch noch schon mehr sehr nur auch nicht
eine einem einer den die das der dem des
und oder aber wenn als dass wie wo was wer
gegen über unter vor nach mit bei von zu aus um durch
auf im in an am zur zum beim
alles weitere mehr alles weitere unter weiter unten
warum wieso weshalb
hier hierzu dazu damit
""".split())

# Pre-compile word regex (German letters + umlauts + ß)
WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+(?:['’-][A-Za-zÄÖÜäöüß]+)?")
WHITESPACE = re.compile(r"\s+")

# ZEIT boilerplate template that wraps descriptions:
#   "Hier finden Sie Informationen zu dem Thema „X“. Lesen Sie jetzt „Y“."
# We strip it so politician co-occurrence reflects real article content,
# not the recurring newsroom template.
BOILERPLATE_RE = re.compile(
    r"Hier finden Sie Informationen zu dem Thema.*?Lesen Sie jetzt.*",
    re.IGNORECASE | re.DOTALL,
)


def strip_boilerplate(text):
    """Remove the recurring ZEIT description template."""
    if not text:
        return text
    return BOILERPLATE_RE.sub(" ", text)


def normalize_cat(c):
    if not isinstance(c, str):
        return "UNKNOWN"
    # strip Z+ paywall prefix and collapse whitespace
    c = re.sub(r"Z\+\s*\(abopflichtiger Inhalt\)[^;]*;\s*", "", c)
    c = WHITESPACE.sub(" ", c).strip()
    return c if c else "UNKNOWN"


def find_positions(text, name):
    """Case-insensitive substring positions of name in text."""
    if not text:
        return []
    return [m.start() for m in re.finditer(re.escape(name), text, re.IGNORECASE)]


def nearby_words(text, positions, window=50):
    """Yield lowercase words within `window` chars of any position in positions."""
    for pos in positions:
        start = max(0, pos - window)
        end = min(len(text), pos + window + len(text) - 0)  # safe bound
        end = min(len(text), pos + window)
        seg = text[start:end]
        for m in WORD_RE.finditer(seg):
            w = m.group()
            lw = w.lower()
            yield lw


def safe_word_boundary(term, text):
    """For single-char or short terms (KI, AI), require word boundaries."""
    # Whole-word match, case-sensitive enough; accept term as substring with boundaries
    return len(re.findall(r"\b" + re.escape(term) + r"\b", text))


def term_count(term, text):
    """Count occurrences of term in text. Multiword: substring. Short tokens: word-boundary."""
    if " " in term:
        return text.lower().count(term.lower())
    # For very short tokens (<=3 chars) use word boundary to avoid 'KI' inside 'Krise'
    if len(term) <= 3:
        return len(re.findall(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE))
    # longer single tokens: word-boundary match (catches umlauts)
    return len(re.findall(r"\b" + re.escape(term) + r"\b", text, re.IGNORECASE))


def sentiment_count(words, text):
    n = 0
    for w in words:
        if " " in w:
            n += text.lower().count(w.lower())
        else:
            n += len(re.findall(r"\b" + re.escape(w) + r"\b", text, re.IGNORECASE))
    return n


def polarity(pos, neg):
    if pos > neg:
        return "pos"
    if neg > pos:
        return "neg"
    return "neutral"


def main():
    print("Loading parquet...", flush=True)
    df = pd.read_parquet(DATA)
    # Ensure strings
    df["Title"] = df["Title"].fillna("").astype(str)
    df["Description"] = df["Description"].fillna("").astype(str)
    df["Description"] = df["Description"].map(strip_boilerplate)
    df["month"] = df["Published_Month"].astype(str)
    df["text"] = (df["Title"] + " " + df["Description"])
    df["cat_norm"] = df["Category"].map(normalize_cat)

    n = len(df)
    print(f"Loaded {n} articles", flush=True)

    out = {}

    # ----------------------------------------------------------------------
    # 1. Politician framing: word co-occurrence within 50 chars
    # ----------------------------------------------------------------------
    print("Politician framing...", flush=True)
    politician_framing = {}
    # collapse to single politician pass using raw text
    texts = df["text"].tolist()
    # Build per-politician exclusion set (own first/last name variants, plus
    # common first-name forms).
    NAME_EXTRAS = {
        "Scholz": {"olaf"},
        "Merz": {"friedrich"},
        "Trump": {"donald", "trumps"},
        "Putin": {"wladimir", "putins"},
        "Selensky": {"selenskyj", "wolodymyr"},
        "Habeck": {"robert", "habecks"},
        "Söder": {"markus", "söders"},
        "Lindner": {"christian", "lindners"},
    }
    for pol in POLITICIANS:
        counter = Counter()
        # handle Selensky -> also catch Selenskyj
        names = [pol]
        if pol == "Selensky":
            names.append("Selenskyj")
        pol_exclude = {pol.lower()}
        pol_exclude |= NAME_EXTRAS.get(pol, set())
        for text in texts:
            for nm in names:
                positions = find_positions(text, nm)
                if not positions:
                    continue
                for w in nearby_words(text, positions, window=50):
                    if w in STOPWORDS or w in FRAMING_FILTER:
                        continue
                    if len(w) < 4:
                        continue
                    if w in pol_exclude:
                        continue
                    counter[w] += 1
        politician_framing[pol] = dict(counter.most_common(20))
    out["politician_framing"] = politician_framing

    # ----------------------------------------------------------------------
    # 2. Narrative shifts: monthly frequency of tracked terms
    # ----------------------------------------------------------------------
    print("Narrative shifts...", flush=True)
    narrative_shifts = {t: defaultdict(int) for t in TRACKED_TERMS}
    # Group by month to reduce overhead
    months = df["month"].values
    texts_arr = df["text"].values
    for i in range(n):
        m = months[i]
        t = texts_arr[i]
        if not t:
            continue
        tl = t  # we use term_count which handles case
        for term in TRACKED_TERMS:
            c = term_count(term, t)
            if c:
                narrative_shifts[term][m] += c
    # convert defaultdicts to plain dict
    out["narrative_shifts"] = {t: dict(v) for t, v in narrative_shifts.items()}

    # ----------------------------------------------------------------------
    # 3. Sentiment proxy: per month positive/negative counts + ratio
    # ----------------------------------------------------------------------
    print("Sentiment monthly...", flush=True)
    sent_monthly = defaultdict(lambda: {"positive": 0, "negative": 0})
    for i in range(n):
        m = months[i]
        t = texts_arr[i]
        if not t:
            continue
        p = sentiment_count(POS_WORDS, t)
        ng = sentiment_count(NEG_WORDS, t)
        if p:
            sent_monthly[m]["positive"] += p
        if ng:
            sent_monthly[m]["negative"] += ng
    sentiment_monthly = {}
    for m, v in sent_monthly.items():
        pos = v["positive"]
        neg = v["negative"]
        ratio = round(pos / neg, 4) if neg > 0 else (float(pos) if pos > 0 else 0.0)
        sentiment_monthly[m] = {"positive": pos, "negative": neg, "ratio": ratio}
    out["sentiment_monthly"] = sentiment_monthly

    # ----------------------------------------------------------------------
    # 4. Clickbait detection
    # ----------------------------------------------------------------------
    print("Clickbait...", flush=True)
    clickbait_monthly = defaultdict(int)
    clickbait_cat = defaultdict(int)
    titles = df["Title"].values
    title_lens = df["Title_Length"].values
    cats = df["cat_norm"].values
    cb_word_re = re.compile(
        r"\b(" + "|".join(re.escape(w) for w in CLICKBAIT_WORDS) + r")",
        re.IGNORECASE,
    )
    for i in range(n):
        title = titles[i]
        tl_len = int(title_lens[i])
        is_cb = False
        if tl_len > 80:
            is_cb = True
        if "?" in title or "!" in title:
            is_cb = True
        if cb_word_re.search(title):
            is_cb = True
        if is_cb:
            clickbait_monthly[months[i]] += 1
            clickbait_cat[cats[i]] += 1
    # top categories
    top_cats = dict(sorted(clickbait_cat.items(), key=lambda kv: -kv[1])[:50])
    out["clickbait"] = {
        "monthly": dict(sorted(clickbait_monthly.items())),
        "by_category": top_cats,
        "total_clickbait": sum(clickbait_monthly.values()),
        "total_articles": int(n),
    }

    # ----------------------------------------------------------------------
    # 5. Title sentiment vs description sentiment
    # ----------------------------------------------------------------------
    print("Title vs desc sentiment...", flush=True)
    match = 0
    mismatch = 0
    descs = df["Description"].values
    for i in range(n):
        title = titles[i]
        desc = descs[i]
        tp = sentiment_count(POS_WORDS, title)
        tn = sentiment_count(NEG_WORDS, title)
        dp = sentiment_count(POS_WORDS, desc)
        dn = sentiment_count(NEG_WORDS, desc)
        tp_pol = polarity(tp, tn)
        dp_pol = polarity(dp, dn)
        if tp_pol == dp_pol:
            match += 1
        else:
            mismatch += 1
    out["title_desc_sentiment"] = {"match": match, "mismatch": mismatch}

    # ----------------------------------------------------------------------
    # Save
    # ----------------------------------------------------------------------
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"Saved {OUT} ({OUT.stat().st_size} bytes)", flush=True)

    # Summary print
    print("\n--- SUMMARY ---")
    print("Politicians:", list(politician_framing.keys()))
    for p, words in politician_framing.items():
        top = ", ".join(f"{w}:{c}" for w, c in list(words.items())[:5])
        print(f"  {p}: {top}")
    print("Tracked terms:", list(out["narrative_shifts"].keys()))
    print("Months (sentiment):", len(sentiment_monthly))
    print("Clickbait monthly months:", len(out["clickbait"]["monthly"]))
    print("Clickbait total:", out["clickbait"]["total_clickbait"], "/", n)
    print("Title/desc match:", match, "mismatch:", mismatch)


if __name__ == "__main__":
    main()