#!/usr/bin/env python3
"""
Batch pre-compute ML pipeline for all catalog movies.
Saves results to data/ml_cache.json — loaded by api.py at startup
so every detail page is instant (no on-demand ML pipeline).

Usage:
    python3 precompute_ml.py                  # full run with narratives
    python3 precompute_ml.py --no-narrative   # skip Groq narratives (faster)
    python3 precompute_ml.py --limit 10       # test on 10 movies first

Resumes automatically from last checkpoint if interrupted.
ETA: ~5-8 hours for all 3209 movies (with narratives).
     ~3-4 hours without narratives.
"""
import sys
import os
import json
import math
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

CACHE_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'ml_cache.json')
CHECKPOINT_EVERY = 5   # save to disk every N movies


# ── Helpers (mirrors api.py) ──────────────────────────────────────────────────

def _tags_to_drivers(feature_tags):
    return [
        {"label": t["label"], "impact": "pos" if t["direction"] == "+" else "neg"}
        for t in (feature_tags or [])
    ]


def _similar_to_match(similar):
    if not similar:
        return None
    sim    = similar.get("similarity", 0) or 0
    rating = similar.get("rating", 0) or 0
    if isinstance(sim,    float) and math.isnan(sim):    sim    = 0.0
    if isinstance(rating, float) and math.isnan(rating): rating = 0.0
    return {
        "title":    similar.get("title", "Unknown"),
        "score":    round(float(rating), 1),
        "matchPct": round(float(sim) * 100),
    }


def _generate_narrative(result, person):
    try:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            return None
        from groq import Groq
        from ui_components import build_why_prompt
        rec    = result.get("rec", {})
        prompt = build_why_prompt(
            rec,
            pred_score = result.get("pred_score", 5.0),
            match_pct  = result.get("match_pct",  50.0),
            top_pos    = result.get("top_pos",    []),
            top_neg    = result.get("top_neg",    []),
            similar    = result.get("similar") or {"title": "Unknown", "rating": 5.0},
            vibe       = result.get("vibe",       50.0),
            person     = person,
        )
        client   = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model    = "llama-3.1-8b-instant",
            messages = [{"role": "user", "content": prompt}],
            max_tokens = 350,
            stream   = False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"  [narrative:{person}] {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-narrative', action='store_true',
                        help='Skip Groq narrative generation (faster, uses fewer API calls)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Only process first N movies (useful for testing)')
    args = parser.parse_args()

    # Load catalog
    df = pd.read_parquet(os.path.join(os.path.dirname(__file__), 'data', 'catalog_data.parquet'))
    df = df[df['type'] == 'movie'].copy()
    movies = [(row['imdb_id'], str(row['title'])) for _, row in df.iterrows()]

    # Load existing cache (enables resume)
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        print(f"Resuming: {len(cache)} already done")
    else:
        cache = {}

    todo = [(iid, title) for iid, title in movies if iid not in cache]
    if args.limit:
        todo = todo[:args.limit]

    print(f"Total catalog: {len(movies)}  |  Done: {len(cache)}  |  To process: {len(todo)}")
    print(f"Narratives: {'OFF' if args.no_narrative else 'ON (requires GROQ_API_KEY)'}")
    print("Press Ctrl+C at any time — progress is saved every 5 movies.\n")

    from predict import predict_movie
    from predict_noel import predict_movie_noel

    done   = 0
    errors = 0
    start  = time.time()

    for i, (imdb_id, title) in enumerate(todo):
        try:
            print(f"[{i+1}/{len(todo)}] {title} ({imdb_id}) ...", end=' ', flush=True)
            t0 = time.time()

            chai_result = predict_movie(title, imdb_id=imdb_id)
            if chai_result is None:
                print("SKIP — OMDb not found")
                cache[imdb_id] = None
                errors += 1
                continue

            rec        = chai_result["rec"]
            tags_dict  = chai_result.get("tags_dict")
            pg_ratings = chai_result.get("pg_ratings")

            noel_result = predict_movie_noel(rec, tags_dict=tags_dict, pg_ratings=pg_ratings)

            chai_narrative = None if args.no_narrative else _generate_narrative(chai_result, "Chai")
            noel_narrative = None if args.no_narrative else _generate_narrative(noel_result, "Noel")

            cache[imdb_id] = {
                "chai_drivers":       _tags_to_drivers(chai_result.get("tags", [])),
                "chai_closest_match": _similar_to_match(chai_result.get("similar")),
                "chai_narrative":     chai_narrative,
                "noel_drivers":       _tags_to_drivers(noel_result.get("tags", [])),
                "noel_closest_match": _similar_to_match(noel_result.get("similar")),
                "noel_narrative":     noel_narrative,
            }

            done += 1
            print(f"✓ ({time.time() - t0:.1f}s)")

        except KeyboardInterrupt:
            print("\nInterrupted — saving checkpoint...")
            break
        except Exception as e:
            print(f"ERROR: {e}")
            cache[imdb_id] = None
            errors += 1

        # Checkpoint
        if (i + 1) % CHECKPOINT_EVERY == 0:
            _save(cache)
            elapsed   = time.time() - start
            rate      = done / elapsed if elapsed > 0 else 0.01
            remaining = len(todo) - (i + 1)
            eta_mins  = remaining / rate / 60
            print(f"  ↳ Checkpoint saved ({done} done, {errors} errors, ETA ≈ {eta_mins:.0f}m remaining)\n")

    _save(cache)
    elapsed = time.time() - start
    print(f"\nFinished: {done} processed, {errors} errors, {elapsed/60:.1f} min total")
    print(f"Cache saved → {CACHE_PATH}")
    print("Commit data/ml_cache.json and push to GitHub — Render will pick it up on next deploy.")


def _save(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, 'w') as f:
        json.dump(cache, f)


if __name__ == '__main__':
    main()
