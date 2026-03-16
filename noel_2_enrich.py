"""
Step 2 (Noel): Enrich each movie with OMDb metadata.
Resumable: skips already-fetched IMDb IDs by checking enriched_omdb.csv.
"""
import time
import requests
import pandas as pd
import os
from config import (
    OMDB_NOEL_KEY, OMDB_BASE_URL, OMDB_RATE_LIMIT_SLEEP,
    OMDB_DAILY_LIMIT, NOEL_RAW_CSV, NOEL_OMDB_CSV
)


def parse_runtime(val):
    try:
        return int(str(val).split()[0])
    except Exception:
        return None


def parse_money(val):
    try:
        return float(str(val).replace("$", "").replace(",", ""))
    except Exception:
        return None


def parse_votes(val):
    try:
        return int(str(val).replace(",", ""))
    except Exception:
        return None


def parse_rt(ratings_list):
    try:
        for r in ratings_list:
            if r.get("Source") == "Rotten Tomatoes":
                return float(r["Value"].replace("%", ""))
    except Exception:
        pass
    return None


def fetch_omdb(imdb_id):
    params = {"i": imdb_id, "plot": "full", "apikey": OMDB_NOEL_KEY}
    try:
        resp = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("Response") == "False":
            return None
        return data
    except Exception as e:
        print(f"  ERROR fetching {imdb_id}: {e}")
        return None


def parse_record(raw, user_rating, title, imdb_id):
    return {
        "Const": imdb_id,
        "Your Rating": user_rating,
        "Title": title,
        "Year": pd.to_numeric(raw.get("Year", "").replace("–", "").strip(), errors="coerce"),
        "Rated": raw.get("Rated", "N/A"),
        "Runtime": parse_runtime(raw.get("Runtime")),
        "Genre": raw.get("Genre", ""),
        "Director": raw.get("Director", ""),
        "Actors": raw.get("Actors", ""),
        "Plot": raw.get("Plot", ""),
        "Language": raw.get("Language", ""),
        "Country": raw.get("Country", ""),
        "Awards": raw.get("Awards", ""),
        "imdbRating": pd.to_numeric(raw.get("imdbRating"), errors="coerce"),
        "imdbVotes": parse_votes(raw.get("imdbVotes")),
        "Metascore": pd.to_numeric(raw.get("Metascore"), errors="coerce"),
        "BoxOffice": parse_money(raw.get("BoxOffice")),
        "RT_score": parse_rt(raw.get("Ratings", [])),
        "Type": raw.get("Type", ""),
    }


def enrich():
    raw_df = pd.read_csv(NOEL_RAW_CSV)
    print(f"Loaded {len(raw_df)} movies from {NOEL_RAW_CSV}")

    if os.path.exists(NOEL_OMDB_CSV):
        done_df = pd.read_csv(NOEL_OMDB_CSV)
        done_ids = set(done_df["Const"].tolist())
        print(f"Resuming: {len(done_ids)} already fetched, {len(raw_df) - len(done_ids)} remaining")
    else:
        done_df = pd.DataFrame()
        done_ids = set()

    pending = raw_df[~raw_df["Const"].isin(done_ids)].reset_index(drop=True)
    to_fetch = min(len(pending), OMDB_DAILY_LIMIT)
    print(f"Fetching up to {to_fetch} records...")

    new_records = []
    for i, row in pending.iterrows():
        if i >= to_fetch:
            print(f"\nDaily limit ({OMDB_DAILY_LIMIT}) reached. Re-run to continue.")
            break

        imdb_id = row["Const"]
        raw = fetch_omdb(imdb_id)
        if raw:
            record = parse_record(raw, row["Your Rating"], row["Title"], imdb_id)
            new_records.append(record)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{to_fetch} fetched...")
            _save(done_df, new_records)

        time.sleep(OMDB_RATE_LIMIT_SLEEP)

    _save(done_df, new_records)
    print(f"\nDone. Total enriched records saved to {NOEL_OMDB_CSV}")


def _save(done_df, new_records):
    if not new_records:
        return
    new_df = pd.DataFrame(new_records)
    combined = pd.concat([done_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Const"])
    combined.to_csv(NOEL_OMDB_CSV, index=False)


if __name__ == "__main__":
    enrich()
