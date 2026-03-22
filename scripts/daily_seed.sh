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

# Git push — only the catalog parquet, nothing else
cd "$ENZYME_DIR" || exit 1
git add data/catalog_data.parquet >> "$LOG" 2>&1
git commit -m "chore: daily catalog enrichment $(date +%Y-%m-%d)" >> "$LOG" 2>&1
git push origin main >> "$LOG" 2>&1
GIT_EXIT=$?

echo "=== $(date) — Git push exited ($GIT_EXIT) ===" >> "$LOG"
