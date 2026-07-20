#!/usr/bin/env python3
"""Consolidate all ZeitScraper CSV files into a single parquet for analysis."""
import csv, os, glob, json, hashlib, warnings
from collections import defaultdict, Counter
from datetime import datetime

import pandas as pd
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'articles')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_all():
    files = sorted(glob.glob(os.path.join(DATA_DIR, '*.csv')))
    print(f"Found {len(files)} files")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, sep=';', encoding='utf-8')
            df.columns = [c.strip() for c in df.columns]
            for col in df.select_dtypes(include=['object', 'str']).columns:
                try:
                    df[col] = df[col].str.strip()
                except:
                    pass
            df['_source_file'] = os.path.basename(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
    combined = pd.concat(dfs, ignore_index=True)
    return combined

def main():
    df = load_all()
    print(f"Total rows: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    
    # Deduplicate by Hash (keep latest snapshot)
    if 'Hash' in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset=['Hash'], keep='last')
        print(f"After dedup by Hash: {len(df)} (removed {before - len(df)})")
    
    # Parse dates - use utc=True to handle mixed timezones
    if 'Published' in df.columns:
        df['Published'] = pd.to_datetime(df['Published'], errors='coerce', utc=True)
        df['Published_Date'] = df['Published'].dt.date
        df['Published_Hour'] = df['Published'].dt.hour
        df['Published_Weekday'] = df['Published'].dt.day_name()
        df['Published_Month'] = df['Published'].dt.to_period('M').astype(str)
        df['Published_Year'] = df['Published'].dt.year
        df['Published_Quarter'] = df['Published'].dt.to_period('Q').astype(str)
    
    if 'LastUpdated' in df.columns:
        df['LastUpdated'] = pd.to_datetime(df['LastUpdated'], errors='coerce', utc=True)
    
    # Title length
    if 'Title' in df.columns:
        df['Title_Length'] = df['Title'].astype(str).str.len()
    if 'Description' in df.columns:
        df['Desc_Length'] = df['Description'].astype(str).str.len()
    
    # Save processed
    out_parquet = os.path.join(OUTPUT_DIR, 'articles_consolidated.parquet')
    df.to_parquet(out_parquet, index=False)
    print(f"Saved parquet: {out_parquet}")
    
    # Save summary stats
    summary = {
        'total_articles': len(df),
        'date_range': {
            'earliest': str(df['Published'].min()) if 'Published' in df.columns else None,
            'latest': str(df['Published'].max()) if 'Published' in df.columns else None,
        },
        'columns': list(df.columns),
        'source_files': int(df['_source_file'].nunique()) if '_source_file' in df.columns else 0,
    }
    
    if 'Category' in df.columns:
        summary['top_categories'] = df['Category'].value_counts().head(40).to_dict()
    if 'Author' in df.columns:
        summary['top_authors'] = df['Author'].value_counts().head(40).to_dict()
    if 'Source' in df.columns:
        summary['sources'] = df['Source'].value_counts().head(20).to_dict()
    if 'Paywall' in df.columns:
        summary['paywall'] = df['Paywall'].value_counts().to_dict()
    if 'Published_Year' in df.columns:
        summary['by_year'] = df['Published_Year'].value_counts().sort_index().to_dict()
    if 'Published_Month' in df.columns:
        summary['by_month'] = df['Published_Month'].value_counts().sort_index().to_dict()
    
    with open(os.path.join(OUTPUT_DIR, 'summary_stats.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"Saved summary stats")
    
    # Print key stats
    print(f"\n=== KEY STATS ===")
    print(f"Total unique articles: {len(df)}")
    print(f"Date range: {summary['date_range']['earliest']} to {summary['date_range']['latest']}")
    if 'by_year' in summary:
        print(f"\nBy year: {json.dumps(summary['by_year'], indent=2)}")
    if 'top_categories' in summary:
        print(f"\nTop 20 categories: {json.dumps(dict(list(summary['top_categories'].items())[:20]), indent=2, ensure_ascii=False)}")
    if 'paywall' in summary:
        print(f"\nPaywall: {json.dumps(summary['paywall'], indent=2)}")

if __name__ == '__main__':
    main()