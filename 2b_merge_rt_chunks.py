"""
Merge chunk files back into enriched_rt.csv after parallel workers finish.

Usage:  python3 2b_merge_rt_chunks.py --of N
"""
from __future__ import annotations

import argparse
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR  = os.path.join(BASE_DIR, "data")
RT_CSV    = os.path.join(DATA_DIR, "enriched_rt.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--of", type=int, required=True, help="number of chunks")
    args = parser.parse_args()

    dfs = []

    # Existing base file
    if os.path.exists(RT_CSV):
        dfs.append(pd.read_csv(RT_CSV))
        print(f"Base file:  {len(dfs[-1])} rows")

    # Chunk files
    for k in range(args.of):
        path = os.path.join(DATA_DIR, f"enriched_rt_chunk_{k}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            dfs.append(df)
            print(f"Chunk {k}:   {len(df)} rows  ({path})")
        else:
            print(f"Chunk {k}:   MISSING — {path}")

    combined = pd.concat(dfs, ignore_index=True)
    # Keep last entry per Const (chunks override blank entries from base)
    combined = combined.drop_duplicates(subset=["Const"], keep="last")

    combined.to_csv(RT_CSV, index=False)

    success = (combined["Actors_RT"].fillna("") != "").sum()
    fail    = len(combined) - success
    print(f"\nMerged → {RT_CSV}")
    print(f"Total:  {len(combined)}  ✓{success}  ✗{fail}  ({100*fail/len(combined):.1f}% fail)")

    # Clean up chunk files
    for k in range(args.of):
        path = os.path.join(DATA_DIR, f"enriched_rt_chunk_{k}.csv")
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed chunk file: {path}")


if __name__ == "__main__":
    main()
