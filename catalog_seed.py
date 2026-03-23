"""
catalog_seed.py  —  Populate data/catalog_data.parquet with Netflix & Max content.

Usage:
    python catalog_seed.py            # Full pull of Netflix + Max US titles
    python catalog_seed.py --limit 50 # Cap at N titles per service (for testing)
    python catalog_seed.py --skip-urls # Skip per-title streaming URL fetch (saves quota)

Requirements:
    Add to .streamlit/secrets.toml:
        WATCHMODE_API_KEY = "your_key_here"

What it does:
    1. Fetches Netflix + Max title lists from Watchmode API (US region)
    2. Fetches OMDb metadata for each title (rotating across API keys)
    3. Batch-encodes all plot text with SentenceTransformer
    4. Vectorized ML scoring for both Chai and Noel
    5. Saves to data/catalog_data.parquet
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from config import (
    OMDB_API_KEY, OMDB_API_KEY_2, OMDB_NOEL_KEY, OMDB_APP_KEY,
    OMDB_API_KEY_5, OMDB_API_KEY_6,
    OMDB_BASE_URL, DATA_DIR, PG_ORDINAL, PG_COLS,
)
from predict import (
    _load_all, parse_omdb, _build_enriched_text,
    NUMERIC_COLS,
)
from predict_noel import _load_all as _load_all_noel
from tag_features import ALL_TAG_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

CATALOG_PATH = os.path.join(DATA_DIR, "catalog_data.parquet")
WATCHMODE_BASE = "https://api.watchmode.com/v1"

# OMDB_API_KEY_6 is reserved exclusively for the live website — never used in seeding.
# Seed uses all other keys; when they're exhausted, partial progress is saved and
# the script exits cleanly so tomorrow's re-run continues from where it left off.
_OMDB_KEYS = [k for k in [
    OMDB_API_KEY_5,
    OMDB_APP_KEY, OMDB_API_KEY, OMDB_API_KEY_2, OMDB_NOEL_KEY,
] if k]
_omdb_key_idx = 0


def _next_omdb_key() -> str:
    global _omdb_key_idx
    key = _OMDB_KEYS[_omdb_key_idx % len(_OMDB_KEYS)]
    _omdb_key_idx += 1
    return key


def _load_secrets() -> dict:
    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return {}
    try:
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                return {}
        with open(secrets_path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _get_watchmode_keys() -> list[str]:
    """Return all configured Watchmode API keys (up to 3)."""
    keys = []
    secrets = _load_secrets()
    for env_key, secret_key in [
        ("WATCHMODE_API_KEY",   "WATCHMODE_API_KEY"),
        ("WATCHMODE_API_KEY_2", "WATCHMODE_API_KEY_2"),
        ("WATCHMODE_API_KEY_3", "WATCHMODE_API_KEY_3"),
    ]:
        val = os.environ.get(env_key) or secrets.get(secret_key, "")
        if val:
            keys.append(val)
    return keys


_watchmode_key_idx = 0


def _next_watchmode_key(keys: list[str]) -> str:
    global _watchmode_key_idx
    key = keys[_watchmode_key_idx % len(keys)]
    _watchmode_key_idx += 1
    return key


def _get_watchmode_key() -> str:
    """Return the primary Watchmode key (for backward compat)."""
    keys = _get_watchmode_keys()
    return keys[0] if keys else ""


# ── Watchmode helpers ─────────────────────────────────────────────────────────

def get_source_ids(api_key: str) -> dict[str, int]:
    """Fetch Watchmode source list and return {service_key: id} for all major US streamers."""
    url = f"{WATCHMODE_BASE}/sources/"
    resp = requests.get(url, params={"apiKey": api_key}, timeout=20)
    resp.raise_for_status()
    sources = resp.json()
    result: dict[str, int] = {}
    for s in sources:
        name = (s.get("name") or "").lower().strip()
        regions = s.get("regions") or []
        if "US" not in regions:
            continue
        if "netflix" in name:
            result.setdefault("netflix", s["id"])
        elif name in ("max", "hbo max") or ("max" in name and "starz" not in name
                                            and "showmax" not in name
                                            and "amazon" not in name
                                            and "paramount" not in name):
            result.setdefault("max", s["id"])
        # Prime Video excluded for now
        elif "disney" in name and "plus" in name:
            result.setdefault("disney", s["id"])
        elif "disney+" in name:
            result.setdefault("disney", s["id"])
        elif name in ("hulu",) or ("hulu" in name and "live" not in name):
            result.setdefault("hulu", s["id"])
        elif "appletv+" in name.replace(" ", "") or ("apple" in name and "tv" in name and "+" in name):
            result.setdefault("apple", s["id"])
        elif "peacock" in name:
            result.setdefault("peacock", s["id"])
        elif "paramount" in name and ("plus" in name or "+" in name):
            result.setdefault("paramount", s["id"])
    return result


def fetch_watchmode_titles(api_key: str, source_id: int,
                           service_name: str, limit: int = 0) -> list[dict]:
    """Fetch title list for one streaming source (US region)."""
    titles: list[dict] = []
    page = 1
    while True:
        params = {
            "apiKey": api_key,
            "source_ids": source_id,
            "regions": "US",
            "types": "movie,tv_movie,tv_series,tv_miniseries",
            "page": page,
            "limit": 250,
        }
        try:
            resp = requests.get(f"{WATCHMODE_BASE}/list-titles/", params=params, timeout=25)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error(f"Watchmode error on page {page} for {service_name}: {e}")
            break

        page_titles = data.get("titles") or []
        for t in page_titles:
            titles.append({
                "watchmode_id": t.get("id"),
                "imdb_id":      t.get("imdb_id") or "",
                "title":        t.get("title", ""),
                "year":         t.get("year"),
                "type":         t.get("type", "movie"),
                "service":      service_name,
            })

        total_pages = data.get("total_pages", 1)
        log.info(f"  {service_name}: page {page}/{total_pages} — {len(titles)} titles so far")

        if page >= total_pages:
            break
        if limit and len(titles) >= limit:
            break
        page += 1
        time.sleep(0.4)

    if limit:
        titles = titles[:limit]
    return titles


def fetch_watchmode_streaming_url(api_key: str, watchmode_id: int,
                                   service_source_id: int) -> str:
    """Fetch the direct streaming URL for one title on one service."""
    try:
        resp = requests.get(
            f"{WATCHMODE_BASE}/title/{watchmode_id}/sources/",
            params={"apiKey": api_key}, timeout=15,
        )
        resp.raise_for_status()
        for s in resp.json():
            if s.get("source_id") == service_source_id and s.get("region") == "US":
                return s.get("web_url") or s.get("android_url") or ""
        for s in resp.json():
            if s.get("region") == "US" and s.get("web_url"):
                return s["web_url"]
    except Exception:
        pass
    return ""


# ── OMDb helpers ──────────────────────────────────────────────────────────────

_omdb_exhausted: set[str] = set()  # keys that returned 401 today


def fetch_omdb_metadata(imdb_id: str) -> dict | None:
    """Try each OMDb key in rotation; skip keys that have returned 401."""
    available = [k for k in _OMDB_KEYS if k not in _omdb_exhausted]
    if not available:
        return None
    for key in available:
        try:
            resp = requests.get(
                OMDB_BASE_URL,
                params={"i": imdb_id, "plot": "full", "apikey": key},
                timeout=12,
            )
            if resp.status_code == 401:
                log.warning(f"OMDb key {key[:6]}… exhausted (401) — removing from rotation")
                _omdb_exhausted.add(key)
                continue
            resp.raise_for_status()
            data = resp.json()
            return data if data.get("Response") != "False" else None
        except Exception as e:
            log.warning(f"OMDb error for {imdb_id}: {e}")
            return None
    return None


# ── Vectorized feature building ───────────────────────────────────────────────

def _build_features_catalog(rec: dict, artifacts: dict,
                              embedding: np.ndarray,
                              tags_encoded: dict | None = None,
                              pg_ratings: dict | None = None) -> pd.DataFrame:
    """
    Build one feature row for the catalog.
    Mirrors _build_single_features from predict.py but uses a pre-computed
    embedding (for batching). Tags and PG ratings are passed in from cached
    CSVs so scores match the Search tab's prediction logic exactly.
    Falls back to zeros/(-1) for movies with no cached data.
    """
    scaler        = artifacts["scaler"]
    mlb           = artifacts["mlb"]
    feature_names = artifacts["feature_names"]

    # 1. Numerics — impute missing with scaler means
    num_arr = np.array([[
        float(rec.get(col)) if rec.get(col) is not None else np.nan
        for col in NUMERIC_COLS
    ]], dtype=float)
    for i in range(len(NUMERIC_COLS)):
        if np.isnan(num_arr[0, i]):
            num_arr[0, i] = scaler.mean_[i]
    num_feat = pd.DataFrame(scaler.transform(num_arr), columns=NUMERIC_COLS)

    # 2. Director / actor / studio mean encoding
    overall_avg    = artifacts.get("overall_avg", 6.5)
    director_stats = artifacts.get("director_stats", {})
    actor_stats    = artifacts.get("actor_stats",    {})
    studio_stats   = artifacts.get("studio_stats",   {})
    director1 = str(rec.get("Director") or "").split(",")[0].strip()
    actor1    = str(rec.get("Actors")   or "").split(",")[0].strip()
    studio    = str(rec.get("Production") or "").strip() or "Unknown"
    if studio == "N/A":
        studio = "Unknown"
    d = director_stats.get(director1, {})
    a = actor_stats.get(actor1, {})
    s = studio_stats.get(studio, {})
    dir_act_feat = pd.DataFrame([{
        "director_film_count": float(d.get("count", 0)),
        "director_avg_rating": float(d.get("avg",   overall_avg)),
        "actor1_film_count":   float(a.get("count", 0)),
        "actor1_avg_rating":   float(a.get("avg",   overall_avg)),
        "studio_film_count":   float(s.get("count", 0)),
        "studio_avg_rating":   float(s.get("avg",   overall_avg)),
    }])

    # 3. Decade one-hot
    try:
        yr = int(str(rec.get("Year") or "2000").split(".")[0][:4])
    except Exception:
        yr = 2000
    decade = (yr // 10) * 10
    decade_feat = pd.DataFrame([{
        "decade_pre1970s": int(decade < 1970),
        "decade_1970s":    int(decade == 1970),
        "decade_1980s":    int(decade == 1980),
        "decade_1990s":    int(decade == 1990),
        "decade_2000s":    int(decade == 2000),
        "decade_2010s":    int(decade == 2010),
        "decade_2020s":    int(decade >= 2020),
    }])

    # 4. Content rating one-hot
    _rated_map = {
        "R": "R", "TV-MA": "R", "18+": "R",
        "PG-13": "PG13", "TV-14": "PG13", "16+": "PG13",
        "PG": "PG", "TV-PG": "PG",
        "G": "G", "TV-G": "G",
    }
    rb = _rated_map.get(str(rec.get("Rated", "N/A")), "NR")
    rated_feat = pd.DataFrame([{
        "rated_G": int(rb == "G"), "rated_PG": int(rb == "PG"),
        "rated_PG13": int(rb == "PG13"), "rated_R": int(rb == "R"),
        "rated_NR": int(rb == "NR"),
    }])

    # 5. Language binary
    lang = (rec.get("Language") or "").split(",")[0].strip()
    lang_feat = pd.DataFrame([{
        "lang_english": int(lang == "English"),
        "lang_hindi":   int(lang == "Hindi"),
        "lang_other":   int(lang not in ("English", "Hindi")),
    }])

    # 6. Country binary
    country = rec.get("Country") or ""
    country_feat = pd.DataFrame([{
        "country_us":    int("United States" in country),
        "country_uk":    int("United Kingdom" in country),
        "country_india": int("India" in country),
        "country_other": int(not any(c in country for c in
                                     ("United States", "United Kingdom", "India"))),
    }])

    # 7. Genre encoding
    genres = [g.strip() for g in (rec.get("Genre") or "").split(",") if g.strip()]
    try:
        genre_arr = mlb.transform([genres])
    except Exception:
        genre_arr = np.zeros((1, len(mlb.classes_)))
    genre_feat = pd.DataFrame(genre_arr, columns=[f"genre_{c}" for c in mlb.classes_])

    # 8. Tags — use cached values if available, else zeros (same as Search tab)
    if tags_encoded:
        tag_row = {col: int(tags_encoded.get(col, 0)) for col in ALL_TAG_COLS}
    else:
        tag_row = {col: 0 for col in ALL_TAG_COLS}
    tag_feat = pd.DataFrame([tag_row])

    # 9a. Parents Guide — use cached values if available, else -1 (same as Search tab)
    if pg_ratings:
        pg_row = {f"pg_{k}": PG_ORDINAL.get(v, -1) if v else -1
                  for k, v in pg_ratings.items()}
    else:
        pg_row = {col: -1 for col in PG_COLS}
    pg_feat = pd.DataFrame([pg_row])

    # 9. Pre-computed embedding
    plot_feat = pd.DataFrame(
        embedding.reshape(1, -1),
        columns=[f"plot_{i}" for i in range(len(embedding))],
    )

    row = pd.concat([num_feat, dir_act_feat, decade_feat, rated_feat,
                     lang_feat, country_feat, genre_feat, tag_feat,
                     pg_feat, plot_feat], axis=1)
    return row.reindex(columns=feature_names, fill_value=0.0)


def score_catalog_batch(recs: list[dict],
                         chai_arts: dict,
                         noel_arts: dict) -> tuple[list[float], list[float]]:
    """
    Score all recs vectorized using the same feature pipeline as the Search tab.
    LLM tags and Parents Guide data are loaded from the cached CSVs
    (data/movie_tags.csv and data/parents_guide.csv) so predictions are
    identical to what predict_movie() would return for the same movie.
    Falls back to zeros/-1 for movies not yet in the cache.
    """
    from tag_features import encode_tags, ALL_TAG_COLS

    # Load LLM tag cache keyed by IMDb ID
    tags_lookup: dict[str, dict] = {}
    _tags_csv = os.path.join(DATA_DIR, "movie_tags.csv")
    if os.path.exists(_tags_csv):
        _tags_df = pd.read_csv(_tags_csv)
        _tag_cols = [c for c in _tags_df.columns if c.startswith("tag_")]
        for _, row in _tags_df.iterrows():
            iid = str(row.get("Const", "")).strip()
            if iid:
                tags_lookup[iid] = {c: int(row[c]) for c in _tag_cols if c in row}
        log.info(f"Loaded LLM tags for {len(tags_lookup)} movies from cache")

    # Load Parents Guide cache keyed by IMDb ID
    pg_lookup: dict[str, dict] = {}
    _pg_csv = os.path.join(DATA_DIR, "parents_guide.csv")
    if os.path.exists(_pg_csv):
        _pg_df = pd.read_csv(_pg_csv)
        for _, row in _pg_df.iterrows():
            iid = str(row.get("Const", "")).strip()
            if iid:
                pg_lookup[iid] = {
                    "sex_nudity":    row.get("sex_nudity"),
                    "violence_gore": row.get("violence_gore"),
                    "profanity":     row.get("profanity"),
                    "alcohol_drugs": row.get("alcohol_drugs"),
                    "intensity":     row.get("intensity"),
                }
        log.info(f"Loaded Parents Guide for {len(pg_lookup)} movies from cache")

    log.info(f"Batch-encoding {len(recs)} plots with SentenceTransformer…")
    nlp = chai_arts["nlp"]
    texts = [_build_enriched_text(r) for r in recs]
    embeddings = nlp.encode(texts, batch_size=64, show_progress_bar=True)

    log.info("Building Chai feature matrix…")
    chai_rows = [
        _build_features_catalog(
            r, chai_arts, embeddings[i],
            tags_encoded=tags_lookup.get(r.get("imdb_id", ""), None),
            pg_ratings=pg_lookup.get(r.get("imdb_id", ""), None),
        )
        for i, r in enumerate(recs)
    ]
    chai_matrix = pd.concat(chai_rows, ignore_index=True)
    chai_preds = np.clip(chai_arts["model"].predict(chai_matrix), 1, 10)
    chai_pcts  = [round(float(p) / 10 * 100, 1) for p in chai_preds]

    log.info("Building Noel feature matrix…")
    noel_rows = [
        _build_features_catalog(
            r, noel_arts, embeddings[i],
            tags_encoded=tags_lookup.get(r.get("imdb_id", ""), None),
            pg_ratings=pg_lookup.get(r.get("imdb_id", ""), None),
        )
        for i, r in enumerate(recs)
    ]
    noel_matrix = pd.concat(noel_rows, ignore_index=True)
    noel_preds  = np.clip(noel_arts["model"].predict(noel_matrix), 1, 10)
    noel_pcts   = [round(float(p) / 10 * 100, 1) for p in noel_preds]

    tagged   = sum(1 for r in recs if r.get("imdb_id", "") in tags_lookup)
    pg_found = sum(1 for r in recs if r.get("imdb_id", "") in pg_lookup)
    log.info(f"Tags applied: {tagged}/{len(recs)} | PG applied: {pg_found}/{len(recs)}")

    return chai_pcts, noel_pcts


# ── Main entry point ──────────────────────────────────────────────────────────

def run_seed(limit_per_service: int = 0,
             skip_streaming_urls: bool = False) -> pd.DataFrame | None:
    """Full pipeline: fetch → score → save parquet. Returns the DataFrame."""

    api_key = _get_watchmode_key()
    if not api_key:
        log.error(
            "WATCHMODE_API_KEY not found.\n"
            "Add it to .streamlit/secrets.toml:\n"
            "  WATCHMODE_API_KEY = 'your_key_here'\n"
            "Get a free key at: https://api.watchmode.com/"
        )
        return None

    # 1. Resolve source IDs
    log.info("Fetching Watchmode source IDs…")
    source_ids = get_source_ids(api_key)
    if not source_ids:
        log.error("Could not find Netflix/Max source IDs. Check your Watchmode API key.")
        return None
    log.info(f"Sources: {source_ids}")

    # 2. Fetch title lists
    all_titles: list[dict] = []
    seen_imdb: set[str] = set()
    for service, src_id in source_ids.items():
        log.info(f"Fetching {service.title()} titles (US)…")
        titles = fetch_watchmode_titles(api_key, src_id, service,
                                        limit=limit_per_service)
        for t in titles:
            iid = t.get("imdb_id", "")
            if iid:
                if iid not in seen_imdb:
                    seen_imdb.add(iid)
                    all_titles.append(t)
            else:
                all_titles.append(t)
        log.info(f"  {service.title()}: {len(titles)} unique titles")

    log.info(f"Total unique titles to score: {len(all_titles)}")

    # 3. Load existing catalog to reuse already-fetched OMDb data
    existing_omdb: dict[str, dict] = {}
    if os.path.exists(CATALOG_PATH):
        try:
            existing_df = pd.read_parquet(CATALOG_PATH)
            for _, row in existing_df.iterrows():
                iid = row.get("imdb_id", "")
                if iid:
                    existing_omdb[iid] = {
                        "Title":      row.get("title", ""),
                        "Year":       str(row.get("year", "")),
                        "Type":       "series" if row.get("type") == "tv" else "movie",
                        "imdbRating": row.get("imdb_score"),
                        "Metascore":  row.get("metascore"),
                        "Poster":     row.get("poster_url", ""),
                        "Plot":       row.get("plot", ""),
                        "Director":   row.get("director", ""),
                        "Actors":     row.get("actors", ""),
                        "Genre":      row.get("genre", ""),
                        "Runtime":    str(row.get("runtime", "")),
                        "Rated":      row.get("rated", ""),
                        "Language":   row.get("language", ""),
                        "Country":    row.get("country", ""),
                    }
            log.info(f"Loaded {len(existing_omdb)} existing catalog entries (skipping OMDb re-fetch)")
        except Exception as e:
            log.warning(f"Could not load existing catalog: {e}")

    # 4. Fetch OMDb metadata only for NEW titles not in existing catalog
    log.info("Fetching OMDb metadata for new titles…")
    titles_with_imdb = [t for t in all_titles if t.get("imdb_id")]
    titles_needing_omdb = [t for t in titles_with_imdb if t["imdb_id"] not in existing_omdb]
    omdb_results: dict[str, dict] = dict(existing_omdb)  # seed with existing

    log.info(f"  {len(existing_omdb)} reused from cache, {len(titles_needing_omdb)} need fresh OMDb fetch")

    _quota_exhausted = False

    def _fetch_omdb(t: dict):
        raw = fetch_omdb_metadata(t["imdb_id"])
        return t["imdb_id"], raw

    # Sequential fetch so we can detect quota exhaustion mid-run and stop cleanly
    newly_fetched = 0
    for i, t in enumerate(titles_needing_omdb):
        # All seed keys exhausted — save what we have and exit
        available = [k for k in _OMDB_KEYS if k not in _omdb_exhausted]
        if not available:
            log.warning(
                f"All OMDb seed keys exhausted after {newly_fetched} new fetches "
                f"({len(omdb_results)} total in catalog). "
                f"Re-run tomorrow to continue enriching the remaining "
                f"{len(titles_needing_omdb) - i} titles."
            )
            _quota_exhausted = True
            break
        time.sleep(0.25)
        iid, raw = _fetch_omdb(t)
        if raw:
            omdb_results[iid] = raw
            newly_fetched += 1
        if (i + 1) % 200 == 0:
            log.info(f"  OMDb: {i+1}/{len(titles_needing_omdb)} new — "
                     f"{len(omdb_results)} total, {len(_omdb_exhausted)} keys exhausted")

    log.info(f"OMDb metadata for {len(omdb_results)} titles "
             f"({len(existing_omdb)} cached, {newly_fetched} newly fetched)")

    # 4a. Drop titles no longer on any service
    removed_ids = set(omdb_results.keys()) - seen_imdb
    if removed_ids:
        log.info(f"Removing {len(removed_ids)} titles no longer on any service")
        for iid in removed_ids:
            del omdb_results[iid]

    # 4. Parse into enriched records
    service_map:     dict[str, str] = {}
    watchmode_id_map: dict[str, int] = {}
    for t in all_titles:
        iid = t.get("imdb_id", "")
        if iid:
            service_map[iid]      = t["service"]
            watchmode_id_map[iid] = t["watchmode_id"]

    enriched: list[dict] = []
    for iid, raw in omdb_results.items():
        rec = parse_omdb(raw)
        rec["imdb_id"]      = iid
        rec["imdbID"]       = iid
        rec["service"]      = service_map.get(iid, "unknown")
        rec["watchmode_id"] = watchmode_id_map.get(iid)
        enriched.append(rec)

    if not enriched:
        log.error("No titles enriched — check API keys.")
        return None

    log.info(f"Loading ML models…")
    chai_arts = _load_all()
    noel_arts = _load_all_noel()

    # 5. Vectorized ML scoring
    chai_pcts, noel_pcts = score_catalog_batch(enriched, chai_arts, noel_arts)

    # 6. Build catalog DataFrame
    rows = []
    for i, rec in enumerate(enriched):
        imdb_score = None
        try:
            v = rec.get("imdbRating")
            if v is not None and str(v) not in ("nan", "None", ""):
                imdb_score = float(v)
        except Exception:
            pass

        year = None
        try:
            y = rec.get("Year")
            if y is not None:
                year = int(float(str(y).split(".")[0][:4]))
        except Exception:
            pass

        content_type = "movie"
        raw_type = str(rec.get("Type") or "").lower()
        if "series" in raw_type or "episode" in raw_type or "mini" in raw_type:
            content_type = "tv"

        rows.append({
            "imdb_id":       rec["imdb_id"],
            "watchmode_id":  rec.get("watchmode_id"),
            "title":         rec.get("Title") or "",
            "year":          year,
            "type":          content_type,
            "service":       rec.get("service") or "",
            "streaming_url": "",
            "imdb_score":    imdb_score,
            "chai_pct":      chai_pcts[i],
            "noel_pct":      noel_pcts[i],
            "poster_url":    rec.get("Poster") or "",
            "plot":          rec.get("Plot")   or "",
            "director":      rec.get("Director") or "",
            "actors":        rec.get("Actors")   or "",
            "genre":         rec.get("Genre")    or "",
            "runtime":       rec.get("Runtime"),
            "rated":         rec.get("Rated")    or "",
            "metascore":     rec.get("Metascore"),
            "language":      rec.get("Language") or "",
            "country":       rec.get("Country")  or "",
            "last_updated":  datetime.now().isoformat(),
        })

    df = pd.DataFrame(rows)

    # 7. Optional: fetch streaming URLs (1 Watchmode request per title)
    if not skip_streaming_urls:
        log.info("Fetching streaming URLs…")
        for service, src_id in source_ids.items():
            mask = df["service"] == service
            wm_ids = df.loc[mask & df["watchmode_id"].notna(), "watchmode_id"].tolist()
            for wm_id in wm_ids:
                url = fetch_watchmode_streaming_url(api_key, int(wm_id), src_id)
                if url:
                    df.loc[df["watchmode_id"] == wm_id, "streaming_url"] = url
                time.sleep(0.25)

    # 8. Save
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_parquet(CATALOG_PATH, index=False)
    log.info(f"Saved {len(df)} titles → {CATALOG_PATH}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enzyme catalog seeder")
    parser.add_argument("--limit", type=int, default=0,
                        help="Max titles per service (0 = all)")
    parser.add_argument("--skip-urls", action="store_true",
                        help="Skip streaming URL fetches (saves Watchmode quota)")
    args = parser.parse_args()
    run_seed(limit_per_service=args.limit, skip_streaming_urls=args.skip_urls)
