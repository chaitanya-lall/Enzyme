"""
Enzyme REST API — serves catalog data from parquet + parents guide to the React website.
Run: python3 api.py
Port: 5001
"""
from flask import Flask, jsonify, request
from flask_cors import CORS
import requests as http_requests
import pandas as pd
import json
import os
import sys
import threading

# Ensure Enzyme's own modules (predict.py, config.py, etc.) are importable
_ENZYME_DIR = os.path.dirname(os.path.abspath(__file__))
if _ENZYME_DIR not in sys.path:
    sys.path.insert(0, _ENZYME_DIR)

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Load data at startup ──────────────────────────────────────────────────────
df = pd.read_parquet(os.path.join(BASE_DIR, 'data', 'catalog_data.parquet'))
pg_df = pd.read_csv(os.path.join(BASE_DIR, 'data', 'parents_guide.csv'))
seen = json.load(open(os.path.join(BASE_DIR, 'data', 'seen_ids.json')))

# Movies only
df = df[df['type'] == 'movie'].copy()

chai_seen = set(seen.get('chai_seen', []))
noel_seen = set(seen.get('noel_seen', []))

# In-process ML pipeline result cache + in-flight tracker
# Pre-populated from data/ml_cache.json if it exists (built by precompute_ml.py)
_ML_CACHE_PATH = os.path.join(BASE_DIR, 'data', 'ml_cache.json')
if os.path.exists(_ML_CACHE_PATH):
    with open(_ML_CACHE_PATH) as _f:
        _ml_cache: dict = json.load(_f)
    print(f"ML cache loaded: {len(_ml_cache)} pre-computed movies")
else:
    _ml_cache: dict = {}
_ml_running: set = set()

# Parents guide keyed by imdb_id
pg_lookup = {row['Const']: row for _, row in pg_df.iterrows()}

PG_ORDINAL = {'None': 0, 'Mild': 1, 'Moderate': 2, 'Severe': 3}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _safe(v, default=None):
    try:
        if pd.isna(v):
            return default
    except Exception:
        pass
    return v


def _format_runtime(minutes):
    if _safe(minutes) is None:
        return 'N/A'
    m = int(minutes)
    h, m2 = divmod(m, 60)
    return f"{h}h {m2}m" if h > 0 else f"{m2}m"


def _format_awards(row):
    try:
        ow = int(row['oscar_win'])  if pd.notna(row['oscar_win'])  else 0
        on = int(row['oscar_nom'])  if pd.notna(row['oscar_nom'])  else 0
        aw = int(row['award_wins']) if pd.notna(row['award_wins']) else 0
        an = int(row['award_noms']) if pd.notna(row['award_noms']) else 0
    except Exception:
        return None
    if ow:
        noms = f" / {on} nom{'s' if on != 1 else ''}" if on else ''
        return f"{ow} Oscar{'s' if ow != 1 else ''}{noms}"
    if on:
        return f"{on} Oscar nom{'s' if on != 1 else ''}"
    if aw:
        noms = f" / {an} nom{'s' if an != 1 else ''}" if an else ''
        return f"{aw} win{'s' if aw != 1 else ''}{noms}"
    if an:
        return f"{an} nomination{'s' if an != 1 else ''}"
    return None


def _confidence(score):
    if score is None:
        return 'Mixed Signals'
    if score >= 90: return 'Top Pick'
    if score >= 80: return 'Strong Contender'
    if score >= 70: return 'Likely Enjoy'
    return 'Mixed Signals'


def _row_to_card(row):
    genre_str = row['genre'] if pd.notna(row['genre']) else ''
    genres = [g.strip() for g in genre_str.split(',')
              if g.strip() and g.strip() != 'N/A']

    chai_pct = _safe(row['chai_pct'])
    noel_pct = _safe(row['noel_pct'])
    imdb    = _safe(row['imdb_score'])

    return {
        'id':        row['imdb_id'],
        'title':     row['title'],
        'year':      int(row['year']) if pd.notna(row['year']) else None,
        'genre':     genres,
        'runtime':   _format_runtime(row['runtime']),
        'rating':    _safe(row['rated'], 'N/A'),
        'poster':    _safe(row['poster_url'], ''),
        'chaiScore': round(chai_pct) if chai_pct is not None else None,
        'noelScore': round(noel_pct) if noel_pct is not None else None,
        'imdbScore': round(float(imdb), 1) if imdb is not None else None,
        'service':   row['service'],
        'chaiSeen':  row['imdb_id'] in chai_seen,
        'noelSeen':  row['imdb_id'] in noel_seen,
    }


def _row_to_detail(row):
    card = _row_to_card(row)
    iid  = row['imdb_id']

    pg_row   = pg_lookup.get(iid)
    def pg_level(field):
        if pg_row is None:
            return 0
        v = pg_row[field]
        return PG_ORDINAL.get(str(v) if pd.notna(v) else 'None', 0)

    actors_str = _safe(row['actors'], '')
    cast = [a.strip() for a in actors_str.split(',')
            if a.strip() and a.strip() != 'N/A']

    rt   = _safe(row['rt_score'])
    meta = _safe(row['metascore'])

    card.update({
        'synopsis':   _safe(row['plot'], ''),
        'director':   _safe(row['director'], 'N/A'),
        'cast':       cast,
        'rtScore':    int(rt)   if rt   is not None else None,
        'metaScore':  int(meta) if meta is not None else None,
        'awards':     _format_awards(row),
        'parentalTags': {
            'sex':        pg_level('sex_nudity'),
            'violence':   pg_level('violence_gore'),
            'profanity':  pg_level('profanity'),
            'drugs':      pg_level('alcohol_drugs'),
            'frightening':pg_level('intensity'),
        },
        'chai': {
            'confidence':   _confidence(card['chaiScore']),
            'narrative':    None,
            'drivers':      [],
            'closestMatch': None,
        },
        'noel': {
            'confidence':   _confidence(card['noelScore']),
            'narrative':    None,
            'drivers':      [],
            'closestMatch': None,
        },
    })
    return card


# Pre-build card list once at startup
print("Building catalog cards…")
_cards = []
for _, _row in df.iterrows():
    try:
        _cards.append(_row_to_card(_row))
    except Exception as e:
        pass
print(f"Catalog ready: {len(_cards)} movies")


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/catalog')
def catalog():
    return jsonify(_cards)


def _generate_narrative(result, person):
    """Call Groq (non-streaming) to generate a Why narrative. Returns string or None."""
    try:
        groq_key = os.environ.get("GROQ_API_KEY", "")
        if not groq_key:
            return None
        from groq import Groq
        from ui_components import build_why_prompt
        rec       = result.get("rec", {})
        prompt    = build_why_prompt(
            rec,
            pred_score=result.get("pred_score", 5.0),
            match_pct=result.get("match_pct", 50.0),
            top_pos=result.get("top_pos", []),
            top_neg=result.get("top_neg", []),
            similar=result.get("similar") or {"title": "Unknown", "rating": 5.0},
            vibe=result.get("vibe", 50.0),
            person=person,
        )
        client   = Groq(api_key=groq_key)
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Narrative] Error for {person}: {e}")
        return None


def _run_ml_pipeline(imdb_id, title):
    """
    Run Chai + Noel ML pipeline for a single movie.
    Returns dict with chai_tags, chai_similar, noel_tags, noel_similar, or None on error.
    Results are cached in _ml_cache.
    """
    if imdb_id in _ml_cache:
        return _ml_cache[imdb_id]

    try:
        from predict import predict_movie
        from predict_noel import predict_movie_noel

        print(f"[ML] Running pipeline for {imdb_id} ({title})…")
        chai_result = predict_movie(title, imdb_id=imdb_id)
        if chai_result is None:
            _ml_cache[imdb_id] = None
            return None

        rec       = chai_result["rec"]
        tags_dict = chai_result.get("tags_dict")
        pg_ratings = chai_result.get("pg_ratings")

        noel_result = predict_movie_noel(rec, tags_dict=tags_dict, pg_ratings=pg_ratings)

        def _tags_to_drivers(feature_tags):
            return [
                {"label": t["label"], "impact": "pos" if t["direction"] == "+" else "neg"}
                for t in (feature_tags or [])
            ]

        def _similar_to_match(similar):
            import math
            if not similar:
                return None
            sim = similar.get("similarity", 0) or 0
            if isinstance(sim, float) and math.isnan(sim):
                sim = 0.0
            rating = similar.get("rating", 0) or 0
            if isinstance(rating, float) and math.isnan(rating):
                rating = 0.0
            return {
                "title":    similar.get("title", "Unknown"),
                "score":    round(float(rating), 1),
                "matchPct": round(float(sim) * 100),
            }

        chai_narrative = _generate_narrative(chai_result, "Chai")
        noel_narrative = _generate_narrative(noel_result, "Noel")

        result = {
            "chai_drivers":       _tags_to_drivers(chai_result.get("tags", [])),
            "chai_closest_match": _similar_to_match(chai_result.get("similar")),
            "chai_narrative":     chai_narrative,
            "noel_drivers":       _tags_to_drivers(noel_result.get("tags", [])),
            "noel_closest_match": _similar_to_match(noel_result.get("similar")),
            "noel_narrative":     noel_narrative,
        }
        _ml_cache[imdb_id] = result
        print(f"[ML] Done: {len(result['chai_drivers'])} chai drivers, {len(result['noel_drivers'])} noel drivers")
        return result

    except Exception as e:
        print(f"[ML] Pipeline error for {imdb_id}: {e}")
        _ml_cache[imdb_id] = None
        return None


@app.route('/api/movie/<imdb_id>')
def movie_detail(imdb_id):
    """Returns basic parquet data immediately — no ML pipeline."""
    rows = df[df['imdb_id'] == imdb_id]
    if rows.empty:
        return jsonify({'error': 'Not found'}), 404
    return jsonify(_row_to_detail(rows.iloc[0]))


def _run_ml_pipeline_bg(imdb_id, title):
    """Background thread wrapper — runs ML and stores result in cache."""
    _run_ml_pipeline(imdb_id, title)
    _ml_running.discard(imdb_id)


@app.route('/api/movie/<imdb_id>/ml')
def movie_ml(imdb_id):
    """
    Async ML endpoint. Call after the basic detail loads.
    Returns {"status": "done", ...drivers/match...} or {"status": "running"}.
    Kicks off a background thread on first call.
    """
    if imdb_id in _ml_cache:
        result = _ml_cache[imdb_id]
        if result is None:
            return jsonify({'status': 'error'})
        return jsonify({'status': 'done', **result})

    if imdb_id not in _ml_running:
        rows = df[df['imdb_id'] == imdb_id]
        if rows.empty:
            return jsonify({'status': 'error'})
        title = str(rows.iloc[0]['title'])
        _ml_running.add(imdb_id)
        t = threading.Thread(target=_run_ml_pipeline_bg, args=(imdb_id, title), daemon=True)
        t.start()

    return jsonify({'status': 'running'})


@app.route('/api/search')
def search():
    """Search OMDb for any movie by title. Returns up to 10 results, enriched with catalog scores if available."""
    q = request.args.get('q', '').strip()
    if not q:
        return jsonify([])

    key = os.environ.get('OMDB_WEBSITE_KEY', '')
    if not key:
        return jsonify({'error': 'OMDB_WEBSITE_KEY not configured'}), 500

    data = None
    try:
        resp = http_requests.get('https://www.omdbapi.com/', params={'s': q, 'type': 'movie', 'apikey': key}, timeout=5)
        result = resp.json()
        if result.get('Response') == 'True':
            data = result.get('Search', [])
    except Exception:
        pass

    if data is None:
        return jsonify([])

    # Build imdb_id → catalog row lookup for score enrichment
    catalog_lookup = {row['imdb_id']: row for _, row in df.iterrows()}

    results = []
    for item in data[:10]:
        iid = item.get('imdbID', '')
        row = catalog_lookup.get(iid)
        chai_pct = _safe(row['chai_pct']) if row is not None else None
        noel_pct = _safe(row['noel_pct']) if row is not None else None
        results.append({
            'id':        iid,
            'title':     item.get('Title', ''),
            'year':      item.get('Year', ''),
            'poster':    item.get('Poster', '') if item.get('Poster') != 'N/A' else '',
            'chaiScore': round(chai_pct) if chai_pct is not None else None,
            'noelScore': round(noel_pct) if noel_pct is not None else None,
            'chaiSeen':  iid in chai_seen,
            'noelSeen':  iid in noel_seen,
        })

    return jsonify(results)


@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'movies': len(_cards)})


if __name__ == '__main__':
    app.run(port=5001, debug=False)
