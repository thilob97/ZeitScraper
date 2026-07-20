#!/usr/bin/env python3
"""Daily update script: consolidate new CSVs, regenerate dashboard data + graph data.
Run this after the scraper has added new CSV files.
Usage: .venv/bin/python scripts/daily_update.py
"""
import subprocess, sys, os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VENV_PY = os.path.join(BASE, '.venv', 'bin', 'python')

scripts = [
    'scripts/consolidate.py',
    'scripts/generate_dashboard_data.py',
    'scripts/generate_graph_data.py',
]

for script in scripts:
    path = os.path.join(BASE, script)
    print(f"\n{'='*60}")
    print(f"Running {script}...")
    print(f"{'='*60}")
    result = subprocess.run([VENV_PY, path], capture_output=True, text=True, cwd=BASE)
    # Filter warnings
    for line in result.stdout.split('\n'):
        if 'Warning' not in line and 'select_dtypes' not in line:
            print(line)
    if result.returncode != 0:
        print(f"ERROR in {script}:")
        for line in result.stderr.split('\n'):
            if 'Warning' not in line and 'select_dtypes' not in line:
                print(line)
        sys.exit(1)

print(f"\n✅ Dashboard data updated successfully!")