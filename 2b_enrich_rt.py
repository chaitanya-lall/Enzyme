"""
Step 2b: Batch-enrich training data with RT cast (top 5) + production company.

Reads:   data/enriched_omdb.csv          (Chai)
         data/noel/enriched_omdb.csv     (Noel, if present)
Writes:  data/enriched_rt.csv  — columns: Const, Actors_RT, Studio

Resumable: already-processed Const IDs are skipped on re-run.
Progress:  prints a summary every 5 minutes (successes, failures, rate).
Retry:     after the main pass, failed cases are retried once with a longer
           sleep (1.5s) — final remaining failures are logged for inspection.
"""
from __future__ import annotations

import sys
import os
import time
import pandas as pd

from rt_enrichment import fetch_rt_data

# Force unbuffered output so progress shows in real time
sys.stdout.reconfigure(line_buffering=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OMDB_CSV = os.path.join(DATA_DIR, "enriched_omdb.csv")
NOEL_CSV = os.path.join(DATA_DIR, "noel", "enriched_omdb.csv")
RT_CSV   = os.path.join(DATA_DIR, "enriched_rt.csv")

CHECKPOINT_EVERY = 50       # save every N processed rows
PROGRESS_EVERY   = 5 * 60  # print summary every N seconds
RETRY_SLEEP      = 1.5     # longer pause for retry pass


def _print(msg: str) -> None:
    """Print and immediately flush so output appears in real time."""
    print(msg, flush=True)


def load_all_movies() -> pd.DataFrame:
    """Combine Chai + Noel movies, deduplicated by Const."""
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


def _run_pass(movies: pd.DataFrame, results_map: dict,
              skip_consts: set, pass_label: str,
              sleep_override: float | None = None) -> tuple[int, int]:
    """
    Process all movies not in skip_consts.
    Updates results_map in-place: {const: {Const, Actors_RT, Studio}}.
    Returns (successes, failures) for this pass.
    """
    todo        = movies[~movies["Const"].astype(str).isin(skip_consts)]
    total_todo  = len(todo)
    successes   = 0
    failures    = 0
    last_progress = time.time()
    pass_start    = time.time()
    processed     = 0

    _print(f"\n{'═'*60}")
    _print(f"  {pass_label}: {total_todo} movies to process")
    _print(f"{'═'*60}")

    for _, row in todo.iterrows():
        const = str(row["Const"])
        title = str(row["Title"])
        year  = int(row["Year"])

        # Temporarily override RT_SLEEP if requested
        rt = fetch_rt_data(title, year,
                           sleep=(sleep_override is None))
        if sleep_override is not None and sleep_override > 0:
            time.sleep(sleep_override)

        entry = {
            "Const":     const,
            "Actors_RT": ", ".join(rt["cast_top5"]) if rt else "",
            "Studio":    (rt["studio"] or "")        if rt else "",
        }
        results_map[const] = entry
        processed += 1

        if rt:
            successes += 1
            preview = ", ".join(rt["cast_top5"][:2])
        else:
            failures += 1
            preview = "(no RT data)"

        done_total = len(results_map)
        _print(f"[{processed}/{total_todo}] {'✓' if rt else '✗'} {title} ({year}) → {preview}")

        # Checkpoint save every N rows
        if processed % CHECKPOINT_EVERY == 0:
            _save(results_map)
            _print(f"  ↳ Checkpoint saved ({done_total} total in file)")

        # 5-minute progress summary
        now = time.time()
        if now - last_progress >= PROGRESS_EVERY:
            elapsed    = now - pass_start
            rate       = processed / elapsed * 60 if elapsed > 0 else 0
            remaining  = total_todo - processed
            eta_min    = remaining / (rate if rate > 0 else 1)
            fail_pct   = 100 * failures / processed if processed > 0 else 0
            _print(
                f"\n{'─'*60}\n"
                f"  {pass_label} progress: {processed}/{total_todo} "
                f"({100*processed/total_todo:.1f}%)\n"
                f"  This pass:  {successes} ✓  {failures} ✗  "
                f"(fail rate {fail_pct:.1f}%)\n"
                f"  Speed:      {rate:.1f} movies/min\n"
                f"  ETA:        {eta_min:.0f} min\n"
                f"{'─'*60}\n"
            )
            last_progress = now

    return successes, failures


def _save(results_map: dict) -> None:
    pd.DataFrame(list(results_map.values())).to_csv(RT_CSV, index=False)


def main() -> None:
    movies = load_all_movies()
    total  = len(movies)
    _print(f"Total unique movies: {total}")

    # ── Load existing results (for resume) ───────────────────────────────────
    if os.path.exists(RT_CSV):
        done_df     = pd.read_csv(RT_CSV)
        results_map = {
            str(r["Const"]): r
            for r in done_df.to_dict("records")
        }
        # Split existing into successes and failures
        already_done    = {c for c, r in results_map.items() if str(r.get("Actors_RT", "")) != ""}
        already_failed  = {c for c, r in results_map.items() if str(r.get("Actors_RT", "")) == ""}
        _print(f"Resuming: {len(already_done)} ✓  {len(already_failed)} previous failures to retry")
        # Only skip the ones that already succeeded
        skip_for_main = already_done
    else:
        results_map   = {}
        skip_for_main = set()

    # ── Main pass ────────────────────────────────────────────────────────────
    s1, f1 = _run_pass(movies, results_map, skip_for_main, "Main pass")
    _save(results_map)

    # ── Retry pass (failed cases only, longer sleep) ─────────────────────────
    failed_consts = {c for c, r in results_map.items() if str(r.get("Actors_RT", "")) == ""}
    if failed_consts:
        _print(f"\n{len(failed_consts)} failures — retrying with {RETRY_SLEEP}s delay…")
        # Build a sub-dataframe of only the failed movies
        failed_movies = movies[movies["Const"].astype(str).isin(failed_consts)].copy()
        s2, f2 = _run_pass(failed_movies, results_map, set(),
                           "Retry pass", sleep_override=RETRY_SLEEP)
        _save(results_map)
    else:
        s2, f2 = 0, 0

    # ── Final report ─────────────────────────────────────────────────────────
    out_df        = pd.DataFrame(list(results_map.values()))
    total_success = (out_df["Actors_RT"].fillna("") != "").sum()
    total_fail    = total - total_success

    _print(f"\n{'='*60}")
    _print(f"  Enrichment complete")
    _print(f"  Successful: {total_success}/{total} ({100*total_success/total:.1f}%)")
    _print(f"  Failed:     {total_fail}/{total}    ({100*total_fail/total:.1f}%)")
    if total_fail > 0:
        still_failed = out_df[out_df["Actors_RT"].fillna("") == ""]["Const"].tolist()
        _print(f"\n  Still failed after retry ({len(still_failed)} movies):")
        for c in still_failed[:30]:
            row = movies[movies["Const"].astype(str) == str(c)]
            t = row["Title"].values[0] if len(row) else c
            _print(f"    ✗ {t} ({c})")
        if len(still_failed) > 30:
            _print(f"    … and {len(still_failed) - 30} more")
    _print(f"\n  Saved → {RT_CSV}")
    _print(f"{'='*60}")


if __name__ == "__main__":
    main()
