"""
One-time patch: add Writer column to both enriched CSVs.
OMDb already returns Writer in the same API call — we just never saved it.
Uses Chai's training key for Chai's CSV, Noel's key for Noel's CSV.
Safe to re-run: skips rows that already have a Writer value.
"""
import time
import requests
import pandas as pd
from config import (
    OMDB_API_KEY, OMDB_NOEL_KEY, OMDB_BASE_URL,
    OMDB_CSV, NOEL_OMDB_CSV,
)

SLEEP = 0.12   # ~8 req/sec — well within free-tier limits


def fetch_writer(imdb_id: str, api_key: str) -> str:
    try:
        resp = requests.get(
            OMDB_BASE_URL,
            params={"i": imdb_id, "apikey": api_key},
            timeout=10,
        )
        data = resp.json()
        if data.get("Response") == "True":
            return data.get("Writer", "N/A")
    except Exception as e:
        print(f"  ERROR {imdb_id}: {e}")
    return "N/A"


def patch_csv(csv_path: str, api_key: str, label: str):
    df = pd.read_csv(csv_path)
    print(f"\n[{label}] Loaded {len(df)} rows from {csv_path}")

    if "Writer" not in df.columns:
        df["Writer"] = None

    missing = df["Writer"].isna() | (df["Writer"].str.strip() == "") | (df["Writer"] == "N/A")
    pending = df[missing]
    print(f"  {len(pending)} rows need Writer fetched")

    for i, (idx, row) in enumerate(pending.iterrows()):
        writer = fetch_writer(row["Const"], api_key)
        df.at[idx, "Writer"] = writer

        if (i + 1) % 50 == 0:
            df.to_csv(csv_path, index=False)
            print(f"  {i+1}/{len(pending)} done (saved)…")

        time.sleep(SLEEP)

    df.to_csv(csv_path, index=False)
    print(f"  Done. Writer column saved → {csv_path}")
    print(f"  Sample: {df['Writer'].dropna().head(3).tolist()}")


if __name__ == "__main__":
    patch_csv(OMDB_CSV,      OMDB_API_KEY,  "Chai")
    patch_csv(NOEL_OMDB_CSV, OMDB_NOEL_KEY, "Noel")
    print("\nPatch complete.")
