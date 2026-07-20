# ZEITScraper Analytics Dashboard

Interaktives Analytics-Dashboard für die `zeit.de`-Artikeldaten des ZeitScraper-Projects.

## Quick Start

```bash
# 1. Dependencies installieren (einmalig)
uv venv .venv
source .venv/bin/activate
uv pip install pandas pyarrow

# 2. Alle Daten generieren (consolidate + dashboard data + graph data)
.venv/bin/python scripts/daily_update.py

# 3. Dashboard starten
cd dashboard
python3 -m http.server 8099
# → http://localhost:8099
```

## Dashboard-Tabs

| Tab | Inhalt |
|-----|--------|
| **Overview** | KPIs, monatlicher Trend, Uhrzeit-Verteilung, Top Kategorien, Paywall |
| **Zeitlich** | Tägliche/Monatliche Artikel-Anzahl, Uhrzeiten, Wochentage, Jahres-Vergleich |
| **Kategorien** | Top 30 Kategorien, Kategorie-Trends über Zeit (Top 15) |
| **Autoren** | Top 30 Autoren mit Artikel-Anzahl |
| **Keywords** | Top 50 Keywords |
| **Quellen** | Nachrichtenagenturen, dpa-Regionalquellen |
| **Politik** | Themen-Analyse (Ukraine, Naher Osten, Klima, etc.), Politiker-Erwähnungen |
| **Regional** | Bundesländer & Länder in Artikel-Titeln |
| **Sport** | Sport-Kategorien, Sport-Berichterstattung über Zeit |
| **Graph-Netzwerk** | Interaktive vis.js Netzwerk-Visualisierung (4 Modi) |

## Graph-Netzwerk Modi

- **Kategorie ↔ Keyword**: Welche Keywords dominieren welche Kategorien?
- **Autor → Kategorie**: Worüber schreiben die Autoren?
- **Quelle → Kategorie**: Welche Quellen decken welche Themen ab?
- **Keyword Co-Occurrence**: Welche Keywords erscheinen gemeinsam?

## Tägliche Aktualisierung

```bash
# Nach dem Scrape-Lauf (neue CSVs in data/raw/articles/):
.venv/bin/python scripts/daily_update.py
```

Das Script:
1. `consolidate.py` — Lädt alle CSVs, dedupliziert nach Hash, speichert Parquet
2. `generate_dashboard_data.py` — Generiert `dashboard/data.json` (alle Charts)
3. `generate_graph_data.py` — Generiert `dashboard/graph_data.json` (Netzwerk)

## Neo4j Export

```bash
.venv/bin/python scripts/generate_neo4j_export.py
```

Generiert `neo4j/neo4j_import.cypher` — ausführbar in Neo4j Browser mit:
```
:source neo4j_import.cypher
```

Enthält: Article, Category, Author, Source, Keyword Nodes + Relationships.

## Daten

- **857 CSV-Dateien** (März 2024 – Juli 2026)
- **323.490 unique Artikel** (dedupliziert nach Hash)
- **Spalten**: Hash, Title, Link, Description, Category, KeyWords, Published, LastUpdated, Author, Source, Paywall

## Architektur

```
data/raw/articles/*.csv    →  857 tägliche Scraper-CSVs
data/processed/*.parquet   →  Konsolidiert + dedupliziert
dashboard/data.json        →  Chart-Daten (alle Tabs)
dashboard/graph_data.json  →  Netzwerk-Edge-Listen
dashboard/index.html       →  Self-contained Dashboard (Chart.js + vis.js)
neo4j/neo4j_import.cypher  →  Neo4j Cypher Import-Script
scripts/                   →  Python-Pipeline (consolidate, generate, update)
```