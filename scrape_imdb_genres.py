"""
Scrape full genre lists from IMDb for all watched movies.
Uses JSON-LD structured data embedded in each title page — fast and reliable.
Saves incrementally to data/imdb_genres.csv so it can be resumed if interrupted.
"""
import json
import time
import random
import requests
import pandas as pd
from pathlib import Path

INPUT_CSV  = "data/all_consts.csv"
OUTPUT_CSV = "data/imdb_genres.csv"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_genres(const: str):
    """Fetch full genre list for one IMDb title via JSON-LD structured data."""
    url = f"https://www.imdb.com/title/{const}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        if r.status_code != 200:
            return None
        # IMDb embeds structured data in <script type="application/ld+json">
        start = r.text.find('application/ld+json')
        if start == -1:
            return None
        start = r.text.find('>', start) + 1
        end   = r.text.find('</script>', start)
        data  = json.loads(r.text[start:end])
        genres = data.get("genre", [])
        if isinstance(genres, str):
            genres = [genres]
        return genres
    except Exception as e:
        print(f"  ERROR {const}: {e}")
        return None


def main():
    all_movies = pd.read_csv(INPUT_CSV)
    total = len(all_movies)

    # Load existing results (checkpoint)
    out_path = Path(OUTPUT_CSV)
    if out_path.exists():
        done_df  = pd.read_csv(OUTPUT_CSV)
        done_set = set(done_df["Const"].tolist())
        results  = done_df.to_dict("records")
        print(f"Resuming — {len(done_set)}/{total} already done")
    else:
        done_set = set()
        results  = []

    remaining = all_movies[~all_movies["Const"].isin(done_set)]
    print(f"Fetching genres for {len(remaining)} movies…\n")

    for i, (_, row) in enumerate(remaining.iterrows(), 1):
        const = row["Const"]
        title = row["Title"]

        genres = fetch_genres(const)

        if genres is not None:
            results.append({"Const": const, "Title": title,
                            "imdb_genres": ", ".join(genres)})
            print(f"[{i}/{len(remaining)}] {title}: {', '.join(genres)}")
        else:
            results.append({"Const": const, "Title": title, "imdb_genres": ""})
            print(f"[{i}/{len(remaining)}] {title}: FAILED")

        # Save checkpoint every 25 movies
        if i % 25 == 0:
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
            print(f"  ── checkpoint saved ({len(results)} total) ──")

        # Polite delay: 0.6–1.2s between requests
        time.sleep(random.uniform(0.6, 1.2))

    # Final save
    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    failed = sum(1 for r in results if not r["imdb_genres"])
    print(f"\nDone. {len(results)} movies saved → {OUTPUT_CSV}")
    print(f"Failed/empty: {failed}")


if __name__ == "__main__":
    main()
