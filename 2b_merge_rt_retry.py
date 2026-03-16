"""
Merge retry chunk files back into enriched_rt.csv.

Usage:  python3 2b_merge_rt_retry.py --of N
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
    parser.add_argument("--of", type=int, required=True)
    args = parser.parse_args()

    base_df = pd.read_csv(RT_CSV)
    base_map = {str(r["Const"]): r for r in base_df.to_dict("records")}
    print(f"Base file: {len(base_map)} rows  ({(base_df['Actors_RT'].fillna('')!='').sum()} with actors)")

    rescued = 0
    for k in range(args.of):
        path = os.path.join(DATA_DIR, f"enriched_rt_retry_{k}.csv")
        if not os.path.exists(path):
            print(f"Chunk {k}: MISSING — {path}")
            continue
        df = pd.read_csv(path)
        chunk_rescued = 0
        for r in df.to_dict("records"):
            const = str(r["Const"])
            if r.get("Actors_RT", ""):  # only overwrite if retry succeeded
                base_map[const] = r
                chunk_rescued += 1
        rescued += chunk_rescued
        print(f"Chunk {k}: {len(df)} retried, {chunk_rescued} rescued  ({path})")

    out = pd.DataFrame(list(base_map.values()))
    out.to_csv(RT_CSV, index=False)

    success = (out["Actors_RT"].fillna("") != "").sum()
    fail    = len(out) - success
    print(f"\nMerged → {RT_CSV}")
    print(f"Total: {len(out)}  ✓{success} (+{rescued} rescued)  ✗{fail}  ({100*fail/len(out):.1f}% still failing)")

    for k in range(args.of):
        path = os.path.join(DATA_DIR, f"enriched_rt_retry_{k}.csv")
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed: {path}")


if __name__ == "__main__":
    main()
