"""
Chunked RT enrichment worker.

Usage:
    python3 2b_enrich_rt_chunk.py --chunk K --of N

Reads data/enriched_rt.csv to find already-processed Consts (success or fail),
takes the K-th (0-indexed) chunk of N from the remaining movies,
and writes results to data/enriched_rt_chunk_K.csv.

After all N workers finish, run 2b_merge_rt_chunks.py to merge into enriched_rt.csv.
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import pandas as pd

from rt_enrichment import fetch_rt_data

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
OMDB_CSV  = os.path.join(DATA_DIR, "enriched_omdb.csv")
NOEL_CSV  = os.path.join(DATA_DIR, "noel", "enriched_omdb.csv")
RT_CSV    = os.path.join(DATA_DIR, "enriched_rt.csv")

CHECKPOINT_EVERY = 25


def _print(msg: str) -> None:
    print(msg, flush=True)


def load_all_movies() -> pd.DataFrame:
    dfs = []
    for path in (OMDB_CSV, NOEL_CSV):
        if os.path.exists(path):
            df = pd.read_csv(path, usecols=["Const", "Title", "Year"])
            dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["Year"] = (
        pd.to_numeric(
            combined["Year"].astype(str).str.split("-").str[0],
            errors="coerce",
        )
        .fillna(2000)
        .astype(int)
    )
    return combined.drop_duplicates(subset=["Const"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, required=True, help="0-indexed chunk number")
    parser.add_argument("--of",    type=int, required=True, help="total number of chunks")
    args = parser.parse_args()

    chunk_k = args.chunk
    n_chunks = args.of
    out_csv = os.path.join(DATA_DIR, f"enriched_rt_chunk_{chunk_k}.csv")

    movies = load_all_movies()

    # Already processed (success OR failure recorded in main file)
    already_done: set[str] = set()
    if os.path.exists(RT_CSV):
        done_df = pd.read_csv(RT_CSV)
        already_done = set(done_df["Const"].astype(str))

    # Also skip anything already saved in THIS chunk file (resume support)
    chunk_map: dict[str, dict] = {}
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        chunk_map = {str(r["Const"]): r for r in prev.to_dict("records")}
        already_done |= set(chunk_map.keys())

    # Remaining movies across all chunks
    remaining = movies[~movies["Const"].astype(str).isin(already_done)].reset_index(drop=True)

    # Assign this worker's slice
    my_movies = remaining.iloc[chunk_k::n_chunks].reset_index(drop=True)
    total = len(my_movies)

    _print(f"\n{'═'*55}")
    _print(f"  Chunk {chunk_k}/{n_chunks-1}: {total} movies to process")
    _print(f"{'═'*55}")

    successes = 0
    failures  = 0

    for i, (_, row) in enumerate(my_movies.iterrows(), 1):
        const = str(row["Const"])
        title = str(row["Title"])
        year  = int(row["Year"])

        rt = fetch_rt_data(title, year, sleep=True)

        entry = {
            "Const":     const,
            "Actors_RT": ", ".join(rt["cast_top5"]) if rt else "",
            "Studio":    (rt["studio"] or "") if rt else "",
        }
        chunk_map[const] = entry

        if rt:
            successes += 1
            preview = ", ".join(rt["cast_top5"][:2])
        else:
            failures += 1
            preview = "(no RT data)"

        _print(f"[{i}/{total}] {'✓' if rt else '✗'} {title} ({year}) → {preview}")

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(list(chunk_map.values())).to_csv(out_csv, index=False)
            fail_pct = 100 * failures / i
            _print(f"  ↳ checkpoint {i}/{total}  ✓{successes} ✗{failures} ({fail_pct:.0f}% fail)")

    # Final save
    pd.DataFrame(list(chunk_map.values())).to_csv(out_csv, index=False)
    fail_pct = 100 * failures / total if total else 0
    _print(f"\n{'═'*55}")
    _print(f"  Chunk {chunk_k} done: {successes}✓  {failures}✗  ({fail_pct:.0f}% fail)")
    _print(f"  Saved → {out_csv}")
    _print(f"{'═'*55}")


if __name__ == "__main__":
    main()
