"""
Retry pass for RT enrichment failures, chunked for parallel execution.

Usage:
    python3 2b_enrich_rt_retry.py --chunk K --of N

Reads data/enriched_rt.csv, finds all rows where Actors_RT is blank (failed),
takes chunk K of N from those failures, retries with RETRY_SLEEP delay,
and writes results to data/enriched_rt_retry_K.csv.

After all workers finish, run 2b_merge_rt_retry.py to merge back.
"""
from __future__ import annotations

import sys
import os
import time
import argparse
import pandas as pd

from rt_enrichment import fetch_rt_data

sys.stdout.reconfigure(line_buffering=True)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OMDB_CSV   = os.path.join(DATA_DIR, "enriched_omdb.csv")
NOEL_CSV   = os.path.join(DATA_DIR, "noel", "enriched_omdb.csv")
RT_CSV     = os.path.join(DATA_DIR, "enriched_rt.csv")
RETRY_SLEEP = 1.5
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
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--of",    type=int, required=True)
    args = parser.parse_args()

    chunk_k  = args.chunk
    n_chunks = args.of
    out_csv  = os.path.join(DATA_DIR, f"enriched_rt_retry_{chunk_k}.csv")

    movies = load_all_movies()

    # Find all previously-failed Consts
    rt_df = pd.read_csv(RT_CSV)
    failed_consts = set(
        rt_df[rt_df["Actors_RT"].fillna("") == ""]["Const"].astype(str)
    )

    # Resume support: skip anything this worker already handled
    chunk_map: dict[str, dict] = {}
    already_retried: set[str] = set()
    if os.path.exists(out_csv):
        prev = pd.read_csv(out_csv)
        chunk_map = {str(r["Const"]): r for r in prev.to_dict("records")}
        already_retried = set(chunk_map.keys())

    failed_movies = (
        movies[movies["Const"].astype(str).isin(failed_consts - already_retried)]
        .reset_index(drop=True)
    )

    # This worker's slice
    my_movies = failed_movies.iloc[chunk_k::n_chunks].reset_index(drop=True)
    total = len(my_movies)

    _print(f"\n{'═'*55}")
    _print(f"  Retry chunk {chunk_k}/{n_chunks-1}: {total} movies  (sleep={RETRY_SLEEP}s)")
    _print(f"{'═'*55}")

    successes = 0
    failures  = 0

    for i, (_, row) in enumerate(my_movies.iterrows(), 1):
        const = str(row["Const"])
        title = str(row["Title"])
        year  = int(row["Year"])

        rt = fetch_rt_data(title, year, sleep=False)
        time.sleep(RETRY_SLEEP)

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
            preview = "(still no RT data)"

        _print(f"[{i}/{total}] {'✓' if rt else '✗'} {title} ({year}) → {preview}")

        if i % CHECKPOINT_EVERY == 0:
            pd.DataFrame(list(chunk_map.values())).to_csv(out_csv, index=False)
            _print(f"  ↳ checkpoint {i}/{total}  ✓{successes} ✗{failures}")

    pd.DataFrame(list(chunk_map.values())).to_csv(out_csv, index=False)
    fail_pct = 100 * failures / total if total else 0
    _print(f"\n{'═'*55}")
    _print(f"  Retry chunk {chunk_k} done: {successes}✓  {failures}✗  ({fail_pct:.0f}% still failing)")
    _print(f"  Saved → {out_csv}")
    _print(f"{'═'*55}")


if __name__ == "__main__":
    main()
