#!/bin/bash
# daily_seed.sh — runs once per calendar day, on first laptop wake after midnight.
# Triggered every 30 min by LaunchAgent; guard file prevents duplicate runs.

GUARD="/tmp/enzyme_seed_$(date +%Y-%m-%d).lock"
LOG="/tmp/catalog_seed_daily.log"
ENZYME_DIR="/Users/chaitanya.lall/Desktop/Enzyme"
PYTHON="/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python"

# Already ran today — exit silently
if [ -f "$GUARD" ]; then
    exit 0
fi

# Mark as running for today
touch "$GUARD"

echo "=== $(date) — Starting daily seed ===" >> "$LOG"

# Run the seed
cd "$ENZYME_DIR" || exit 1
"$PYTHON" catalog_seed.py --skip-urls >> "$LOG" 2>&1
SEED_EXIT=$?

echo "=== $(date) — Seed exited ($SEED_EXIT) ===" >> "$LOG"

# Regenerate seen_ids.json so production has up-to-date Chai/Noel seen lists
"$PYTHON" - >> "$LOG" 2>&1 <<'PYEOF'
import os, json, warnings, re

CHAI_SEEN_FILE = os.path.expanduser("~/Documents/Chai Seen.numbers")
NOEL_SEEN_FILE = os.path.expanduser("~/Documents/Noel Seen.numbers")

def ids_from_numbers(path):
    try:
        import numbers_parser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc = numbers_parser.Document(path)
        rows = doc.sheets[0].tables[0].rows(values_only=True)
        ids = set()
        for row in rows:
            for cell in row:
                if isinstance(cell, str) and "imdb.com/title/" in cell:
                    m = re.search(r"tt\d+", cell)
                    if m:
                        ids.add(m.group(0))
        return sorted(ids)
    except Exception as e:
        print(f"Error reading {path}: {e}")
        return []

out = {"chai_seen": ids_from_numbers(CHAI_SEEN_FILE), "noel_seen": ids_from_numbers(NOEL_SEEN_FILE)}
with open(os.path.join(os.getcwd(), "data", "seen_ids.json"), "w") as f:
    json.dump(out, f)
print(f"seen_ids.json written: chai={len(out['chai_seen'])}, noel={len(out['noel_seen'])}")
PYEOF

# Git push — catalog parquet + seen IDs JSON
cd "$ENZYME_DIR" || exit 1
git add data/catalog_data.parquet data/seen_ids.json >> "$LOG" 2>&1
git commit -m "chore: daily catalog enrichment $(date +%Y-%m-%d)" >> "$LOG" 2>&1
git push origin main >> "$LOG" 2>&1
GIT_EXIT=$?

echo "=== $(date) — Git push exited ($GIT_EXIT) ===" >> "$LOG"
