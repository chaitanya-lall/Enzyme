#!/bin/bash
# render_seed.sh — runs on Render's cron service daily.
# Fetches all streaming services, enriches new titles, removes stale ones,
# then pushes the updated catalog back to GitHub → triggers API redeploy.

set -e

echo "=== $(date) — Starting Render seed ==="

python catalog_seed.py --skip-urls

echo "=== $(date) — Seed complete. Pushing to GitHub... ==="

git config user.name  "Enzyme Seed Bot"
git config user.email "seed@enzyme.app"
git add data/catalog_data.parquet

if git diff --staged --quiet; then
  echo "No catalog changes — nothing to push."
else
  git commit -m "chore: daily catalog enrichment $(date +%Y-%m-%d)"
  git push "https://x-access-token:${GITHUB_TOKEN}@github.com/chaitanya-lall/Enzyme.git" main
  echo "=== $(date) — Pushed. Render API will redeploy automatically. ==="
fi
