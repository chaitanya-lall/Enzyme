"""
Inference engine for the PMTPE model.
Handles: OMDb fetch → feature engineering → prediction → SHAP → similarity lookup.
All artifact loading is lazy and cached for Streamlit performance.
"""
from __future__ import annotations

import os
import re
import time
import concurrent.futures
import numpy as np
import pandas as pd
import joblib
import requests
import shap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from config import (
    OMDB_APP_KEY, OMDB_BASE_URL, OMDB_RATE_LIMIT_SLEEP,
    MODELS_DIR, DATA_DIR, SENTENCE_TRANSFORMER_MODEL,
    PG_ORDINAL, PG_COLS,
)
from tag_features import call_groq_tagger, encode_tags, ALL_TAG_COLS, tag_display_label
from parents_guide import scrape_imdb_parents_guide, cache_pg_rating
from rt_enrichment import fetch_rt_data

# Artifact paths
MODEL_PATH            = os.path.join(MODELS_DIR, "pmtpe_model.pkl")
SCALER_PATH           = os.path.join(MODELS_DIR, "scaler.pkl")
MLB_PATH              = os.path.join(MODELS_DIR, "mlb_genres.pkl")
FEATURE_NAMES_PATH    = os.path.join(MODELS_DIR, "feature_names.pkl")
TRAIN_META_PATH       = os.path.join(DATA_DIR,   "train_meta.pkl")
TRAIN_EMB_PATH        = os.path.join(MODELS_DIR, "train_plot_embeddings.npy")
DIRECTOR_STATS_PATH   = os.path.join(MODELS_DIR, "director_stats.pkl")
ACTOR_STATS_PATH      = os.path.join(MODELS_DIR, "actor_stats.pkl")
STUDIO_STATS_PATH     = os.path.join(MODELS_DIR, "studio_stats.pkl")
OVERALL_AVG_PATH      = os.path.join(MODELS_DIR, "overall_avg.pkl")

RT_AUD_PATH   = os.path.join(DATA_DIR, "rt_audience_scores.json")
AWARDS_PATH   = os.path.join(DATA_DIR, "movie_awards.json")

_rt_aud_cache: dict | None = None
_awards_cache: dict | None = None


def _get_extra_data() -> tuple[dict, dict]:
    import json
    global _rt_aud_cache, _awards_cache
    if _rt_aud_cache is None:
        try:
            with open(RT_AUD_PATH) as f:
                _rt_aud_cache = json.load(f)
        except FileNotFoundError:
            _rt_aud_cache = {}
    if _awards_cache is None:
        try:
            with open(AWARDS_PATH) as f:
                _awards_cache = json.load(f)
        except FileNotFoundError:
            _awards_cache = {}
    return _rt_aud_cache, _awards_cache


NUMERIC_COLS = [
    "imdbRating", "Metascore", "BoxOffice", "imdbVotes", "Runtime", "RT_score",
    "award_wins", "award_noms", "oscar_win", "oscar_nom",
]

DECADE_COLS  = [
    "decade_pre1970s", "decade_1970s", "decade_1980s", "decade_1990s",
    "decade_2000s", "decade_2010s", "decade_2020s",
]
RATED_COLS   = ["rated_G", "rated_PG", "rated_PG13", "rated_R", "rated_NR"]
LANG_COLS    = ["lang_english", "lang_hindi", "lang_other"]
COUNTRY_COLS = ["country_us", "country_uk", "country_india", "country_other"]
DIRECTOR_ACTOR_COLS = [
    "director_film_count", "director_avg_rating",
    "actor1_film_count",   "actor1_avg_rating",
    "studio_film_count",   "studio_avg_rating",
]

# Human-readable labels for feature names shown in the UI
FEATURE_LABELS = {
    "imdbRating":      "IMDb Rating",
    "imdbVotes":       "IMDb Popularity",
    "Metascore":       "Metacritic Score",
    "RT_score":        "RT Critic",
    "BoxOffice":       "Box Office",
    "Runtime":         "Runtime",
    "award_wins":      "Awards Won",
    "award_noms":      "Award Nominations",
    "oscar_win":       "Oscar Winner",
    "oscar_nom":       "Oscar Nominated",
    # decade
    "decade_pre1970s": "Pre-1970s Film",
    "decade_1970s":    "1970s Film",
    "decade_1980s":    "1980s Film",
    "decade_1990s":    "1990s Film",
    "decade_2000s":    "2000s Film",
    "decade_2010s":    "2010s Film",
    "decade_2020s":    "2020s Film",
    # rated
    "rated_G":         "Rated G",
    "rated_PG":        "Rated PG",
    "rated_PG13":      "Rated PG-13",
    "rated_R":         "Rated R",
    "rated_NR":        "Not Rated",
    # language
    "lang_english":    "English Language",
    "lang_hindi":      "Hindi Language",
    "lang_other":      "Foreign Language",
    # country
    "country_us":      "US Production",
    "country_uk":      "UK Production",
    "country_india":   "Indian Production",
    "country_other":   "International Production",
    # director / actor / studio mean encoding
    "director_film_count": "Director: Films Seen",
    "director_avg_rating": "Director: Avg Rating",
    "actor1_film_count":   "Actor: Films Seen",
    "actor1_avg_rating":   "Actor: Avg Rating",
    "studio_film_count":   "Studio: Films Seen",
    "studio_avg_rating":   "Studio: Avg Rating",
    # parents guide
    "pg_sex_nudity":    "PG: Sex & Nudity",
    "pg_violence_gore": "PG: Violence & Gore",
    "pg_profanity":     "PG: Profanity",
    "pg_alcohol_drugs": "PG: Alcohol & Drugs",
    "pg_intensity":     "PG: Intensity",
}

# ─── Artifact loading ─────────────────────────────────────────────────────────

_cache = {}

def _load_all():
    if _cache:
        return _cache

    _cache["model"]         = joblib.load(MODEL_PATH)
    _cache["scaler"]        = joblib.load(SCALER_PATH)
    _cache["mlb"]           = joblib.load(MLB_PATH)
    _cache["feature_names"] = joblib.load(FEATURE_NAMES_PATH)
    _cache["nlp"]           = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    _cache["train_meta"]    = pd.read_pickle(TRAIN_META_PATH)
    _cache["train_emb"]     = np.load(TRAIN_EMB_PATH)
    _cache["director_stats"] = joblib.load(DIRECTOR_STATS_PATH) if os.path.exists(DIRECTOR_STATS_PATH) else {}
    _cache["actor_stats"]    = joblib.load(ACTOR_STATS_PATH)    if os.path.exists(ACTOR_STATS_PATH)    else {}
    _cache["studio_stats"]   = joblib.load(STUDIO_STATS_PATH)   if os.path.exists(STUDIO_STATS_PATH)   else {}
    _cache["overall_avg"]    = joblib.load(OVERALL_AVG_PATH)    if os.path.exists(OVERALL_AVG_PATH)    else 6.5

    meta = _cache["train_meta"]
    emb  = _cache["train_emb"]

    # Centroid of highly-rated movies (≥ 8/10) for vibe match
    mask = meta["Your Rating"] >= 8
    _cache["high_rated_centroid"] = (
        emb[mask.values].mean(axis=0, keepdims=True) if mask.sum() > 0
        else emb.mean(axis=0, keepdims=True)
    )

    # SHAP explainer
    _cache["explainer"] = shap.TreeExplainer(_cache["model"])

    return _cache


# ─── OMDb helpers ─────────────────────────────────────────────────────────────

def _parse_runtime(val):
    try:
        return int(str(val).split()[0])
    except Exception:
        return None

def _parse_money(val):
    try:
        return float(str(val).replace("$", "").replace(",", ""))
    except Exception:
        return None

def _parse_votes(val):
    try:
        return int(str(val).replace(",", ""))
    except Exception:
        return None

def _parse_rt(ratings_list):
    try:
        for r in ratings_list:
            if r.get("Source") == "Rotten Tomatoes":
                return float(r["Value"].replace("%", ""))
    except Exception:
        pass
    return None


def _normalize_title(t: str) -> str:
    """
    Normalize a movie title for fuzzy OMDb lookup:
    - Remove mid-word apostrophes/curly-quotes  ("it's" → "its", "schindler's" → "schindlers")
    - Replace hyphens, commas, colons, exclamations, periods, semicolons → space
    - Collapse extra spaces
    """
    t = re.sub(r"(?<=[A-Za-z])[''`](?=[A-Za-z])", "", t)   # mid-word apostrophes
    t = re.sub(r"[\-,!?\.;:]+", " ", t)                      # other punctuation → space
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_variants(query: str) -> list[str]:
    """
    Return alternate search strings derived from *query* so OMDb finds
    titles whose punctuation the user omitted or changed.
    Includes:
      1. Punctuation-stripped form (always tried)
      2. Word-split forms for run-together words like "spiderman" → "spider man"
    """
    vs: set[str] = set()
    norm = _normalize_title(query)
    if norm != query:
        vs.add(norm)
    # For each long all-alpha word try inserting a space at several positions
    # so "spiderman" → "spider man", "ironman" → "iron man", etc.
    base = norm if norm else query
    for word in base.split():
        if len(word) > 5 and word.isalpha():
            for pos in range(max(3, len(word) - 4), min(len(word) - 2, len(word))):
                candidate = base.replace(word, word[:pos] + " " + word[pos:], 1)
                if candidate != base:
                    vs.add(candidate)
    return list(vs)


def search_omdb(title: str) -> list[dict]:
    """Search OMDb for multiple matches, tolerating missing/extra punctuation and run-together words."""

    def _search(t):
        try:
            resp = requests.get(OMDB_BASE_URL, params={"s": t, "type": "movie", "apikey": OMDB_APP_KEY}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data.get("Search", []) if data.get("Response") != "False" else []
        except Exception:
            return []

    def _merge(results: list, extra: list, seen: set) -> list:
        for r in extra:
            if r.get("imdbID") and r["imdbID"] not in seen:
                seen.add(r["imdbID"])
                results.append(r)
        return results

    # Primary search — deduplicate OMDb's own response first
    seen: set[str] = set()
    results: list = []
    for r in _search(title):
        if r.get("imdbID") and r["imdbID"] not in seen:
            seen.add(r["imdbID"])
            results.append(r)

    # Run variant searches in parallel and merge new hits
    variants = _title_variants(title)
    if variants:
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(variants), 6)) as ex:
            for hits in ex.map(_search, variants):
                _merge(results, hits, seen)

    return [{"title": r["Title"], "year": r.get("Year", ""), "imdbID": r["imdbID"]}
            for r in results if r.get("imdbID")]


def fetch_by_imdb_id(imdb_id: str) -> dict | None:
    """Fetch a movie from OMDb by IMDb ID."""
    try:
        resp = requests.get(OMDB_BASE_URL, params={"i": imdb_id, "plot": "full", "apikey": OMDB_APP_KEY}, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data if data.get("Response") != "False" else None
    except Exception as e:
        print(f"OMDb error: {e}")
        return None


def fetch_by_title(title: str) -> dict | None:
    """Search OMDb by movie title. Returns raw OMDb JSON or None."""
    def _omdb_get(t):
        try:
            resp = requests.get(OMDB_BASE_URL, params={"t": t, "plot": "full", "apikey": OMDB_APP_KEY}, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            return data if data.get("Response") != "False" else None
        except Exception as e:
            print(f"OMDb error: {e}")
            return None

    # 1. Try exact title
    result = _omdb_get(title)
    if result:
        return result

    # 2. Normalize punctuation (remove mid-word apostrophes, replace hyphens/commas/etc. with space)
    normalized = _normalize_title(title)
    if normalized != title:
        result = _omdb_get(normalized)
        if result:
            return result

    print(f"OMDb: could not find '{title}'")
    return None



def _parse_awards(awards_str) -> dict:
    """Parse OMDb Awards string → numeric signals."""
    s = str(awards_str) if awards_str else ""
    oscar_win = 1 if re.search(r'\bwon\b.*\boscar', s, re.IGNORECASE) else 0
    oscar_nom = 1 if (re.search(r'\bnominat', s, re.IGNORECASE) and
                      re.search(r'\boscar', s, re.IGNORECASE)) else 0
    wins = sum(int(x) for x in re.findall(r'(\d+)\s+win', s, re.IGNORECASE))
    noms = sum(int(x) for x in re.findall(r'(\d+)\s+nomination', s, re.IGNORECASE))
    return {
        "award_wins": float(wins),
        "award_noms": float(noms),
        "oscar_win":  float(oscar_win),
        "oscar_nom":  float(oscar_nom),
    }


def _build_enriched_text(rec: dict) -> str:
    """Combine Director + Cast + Writer + Plot into one text for embedding.
    Uses Actors_RT (up to 5) when available, falls back to Actors (OMDb).
    """
    parts = []
    director = str(rec.get("Director", "") or "").strip()
    actors   = str(rec.get("Actors_RT", "") or rec.get("Actors", "") or "").strip()
    writer   = str(rec.get("Writer",   "") or "").strip()
    plot     = str(rec.get("Plot",     "") or "").strip()
    if director and director not in ("N/A", "Unknown"):
        parts.append(f"Director: {director}.")
    if actors and actors not in ("N/A", "Unknown"):
        parts.append(f"Cast: {actors}.")
    if writer and writer not in ("N/A", "Unknown"):
        parts.append(f"Writer: {writer}.")
    if plot and plot != "N/A":
        parts.append(plot)
    elif not parts:
        parts.append(rec.get("Title", ""))
    return " ".join(parts)


def parse_omdb(raw: dict) -> dict:
    """Flatten raw OMDb JSON into a flat feature dict."""
    awards_str = raw.get("Awards", "")
    award_feats = _parse_awards(awards_str)
    rec = {
        "Title":      raw.get("Title", "Unknown"),
        "Year":       pd.to_numeric(raw.get("Year", "").split("–")[0].strip(), errors="coerce"),
        "Rated":      raw.get("Rated", "N/A"),
        "Runtime":    _parse_runtime(raw.get("Runtime")),
        "Genre":      raw.get("Genre", ""),
        "Director":   raw.get("Director", "N/A"),
        "Actors":     raw.get("Actors", "N/A"),
        "Writer":     raw.get("Writer", "N/A"),
        "Plot":       raw.get("Plot", ""),
        "Language":   raw.get("Language", ""),
        "Country":    raw.get("Country", ""),
        "Awards":     awards_str,
        "imdbRating": pd.to_numeric(raw.get("imdbRating"), errors="coerce"),
        "imdbVotes":  _parse_votes(raw.get("imdbVotes")),
        "Metascore":  pd.to_numeric(raw.get("Metascore"), errors="coerce"),
        "BoxOffice":  _parse_money(raw.get("BoxOffice")),
        "RT_score":   _parse_rt(raw.get("Ratings", [])),
        "Poster":     raw.get("Poster", ""),
        "Type":       raw.get("Type", ""),
        "imdbID":     raw.get("imdbID", ""),
        "Production": raw.get("Production", ""),
    }
    rec.update(award_feats)
    return rec


# ─── Feature engineering (single movie) ──────────────────────────────────────

def _build_single_features(rec: dict, artifacts: dict,
                            tags_encoded: dict | None = None,
                            pg_ratings: dict | None = None) -> tuple:
    """Build the full feature vector for one movie, aligned to training columns."""
    scaler        = artifacts["scaler"]
    mlb           = artifacts["mlb"]
    nlp           = artifacts["nlp"]
    feature_names = artifacts["feature_names"]

    # 1. Numerics
    num_vals = {}
    for col in NUMERIC_COLS:
        v = rec.get(col)
        num_vals[col] = float(v) if v is not None else np.nan
    num_df = pd.DataFrame([num_vals])
    num_arr = num_df.values.astype(float)
    col_means = scaler.mean_
    for i in range(len(NUMERIC_COLS)):
        if np.isnan(num_arr[0, i]):
            num_arr[0, i] = col_means[i]
    num_scaled = scaler.transform(num_arr)
    num_feat = pd.DataFrame(num_scaled, columns=NUMERIC_COLS)

    # 2. Director / actor / studio mean encoding (look up in training stats)
    director_stats = artifacts.get("director_stats", {})
    actor_stats    = artifacts.get("actor_stats",    {})
    studio_stats   = artifacts.get("studio_stats",   {})
    overall_avg    = artifacts.get("overall_avg",    6.5)

    director1 = str(rec.get("Director", "") or "").split(",")[0].strip()
    # Prefer RT actor1 (top-billed from RT), fall back to OMDb actor1
    actor1    = (
        str(rec.get("Actors_RT", "") or "").split(",")[0].strip()
        or str(rec.get("Actors", "") or "").split(",")[0].strip()
    )
    studio    = str(rec.get("Studio", "") or "").strip()
    if not studio or studio in ("N/A",):
        studio = "Unknown"

    d_info = director_stats.get(director1, {})
    a_info = actor_stats.get(actor1, {})
    s_info = studio_stats.get(studio, {})
    dir_act_row = {
        "director_film_count": float(d_info.get("count", 0)),
        "director_avg_rating": float(d_info.get("avg",   overall_avg)),
        "actor1_film_count":   float(a_info.get("count", 0)),
        "actor1_avg_rating":   float(a_info.get("avg",   overall_avg)),
        "studio_film_count":   float(s_info.get("count", 0)),
        "studio_avg_rating":   float(s_info.get("avg",   overall_avg)),
    }
    dir_act_feat = pd.DataFrame([dir_act_row])

    # 3. Decade one-hot (replaces raw Year)
    try:
        year_raw = rec.get("Year")
        yr = int(str(int(float(year_raw)))[:4]) if year_raw is not None else 2000
    except Exception:
        yr = 2000
    decade = (yr // 10) * 10
    decade_row = {
        "decade_pre1970s": int(decade < 1970),
        "decade_1970s":    int(decade == 1970),
        "decade_1980s":    int(decade == 1980),
        "decade_1990s":    int(decade == 1990),
        "decade_2000s":    int(decade == 2000),
        "decade_2010s":    int(decade == 2010),
        "decade_2020s":    int(decade >= 2020),
    }
    decade_feat = pd.DataFrame([decade_row])

    # 3. Content rating one-hot
    _rated_map = {
        "R": "R", "TV-MA": "R", "18+": "R",
        "PG-13": "PG13", "TV-14": "PG13", "16+": "PG13",
        "PG": "PG", "TV-PG": "PG",
        "G": "G", "TV-G": "G",
    }
    rated_bucket = _rated_map.get(str(rec.get("Rated", "N/A")), "NR")
    rated_row = {
        "rated_G":    int(rated_bucket == "G"),
        "rated_PG":   int(rated_bucket == "PG"),
        "rated_PG13": int(rated_bucket == "PG13"),
        "rated_R":    int(rated_bucket == "R"),
        "rated_NR":   int(rated_bucket == "NR"),
    }
    rated_feat = pd.DataFrame([rated_row])

    # 4. Language binary
    primary_lang = (rec.get("Language") or "").split(",")[0].strip()
    lang_row = {
        "lang_english": int(primary_lang == "English"),
        "lang_hindi":   int(primary_lang == "Hindi"),
        "lang_other":   int(primary_lang not in ("English", "Hindi")),
    }
    lang_feat = pd.DataFrame([lang_row])

    # 5. Country binary
    country_str = rec.get("Country") or ""
    country_row = {
        "country_us":    int("United States" in country_str),
        "country_uk":    int("United Kingdom" in country_str),
        "country_india": int("India" in country_str),
        "country_other": int(not any(c in country_str for c in
                                     ("United States", "United Kingdom", "India"))),
    }
    country_feat = pd.DataFrame([country_row])

    # 6. Genre encoding
    genres = [g.strip() for g in rec.get("Genre", "").split(",") if g.strip()]
    genre_arr = mlb.transform([genres])
    genre_feat = pd.DataFrame(genre_arr, columns=[f"genre_{c}" for c in mlb.classes_])

    # 7. LLM categorical tag features
    tag_row = {col: int(tags_encoded.get(col, 0)) if tags_encoded else 0
               for col in ALL_TAG_COLS}
    tag_feat = pd.DataFrame([tag_row])

    # 8. Parents Guide features
    pg_row = {}
    for col in PG_COLS:
        raw_key = col.replace("pg_", "")  # e.g. pg_sex_nudity → sex_nudity
        val = (pg_ratings or {}).get(raw_key)
        pg_row[col] = PG_ORDINAL.get(val, -1) if val else -1
    pg_feat = pd.DataFrame([pg_row])

    # 9. Enriched text embedding (Director + Cast + Writer + Plot)
    enriched_text = _build_enriched_text(rec)
    embedding = nlp.encode([enriched_text])
    plot_feat = pd.DataFrame(embedding, columns=[f"plot_{i}" for i in range(embedding.shape[1])])

    # 10. Combine & align columns
    row = pd.concat([num_feat, dir_act_feat, decade_feat, rated_feat, lang_feat, country_feat,
                     genre_feat, tag_feat, pg_feat, plot_feat], axis=1)
    row = row.reindex(columns=feature_names, fill_value=0.0)

    return row, embedding[0]


# ─── Similarity helpers ───────────────────────────────────────────────────────

def find_similar_movie_combined(
    embedding: np.ndarray,
    tags_dict: dict,
    imdb_rating: float,
    artifacts: dict,
    threshold: float = 0.0,
    exclude_const: str | None = None,
) -> dict | None:
    """
    Find the most similar movie using a weighted combination:
      60% plot cosine similarity + 30% tag Jaccard + 10% IMDb proximity.
    Returns {title, const, rating, similarity} or None if best score < threshold.
    """
    from tag_features import ALL_TAG_COLS, encode_tags

    train_emb  = artifacts["train_emb"]
    train_meta = artifacts["train_meta"]

    # --- Plot similarity [0,1] ---
    plot_sims = cosine_similarity(embedding.reshape(1, -1), train_emb)[0]
    plot_sims = np.clip(plot_sims, 0, 1)

    # --- Tag Jaccard similarity [0,1] ---
    avail_tags = [c for c in ALL_TAG_COLS if c in train_meta.columns]
    if avail_tags and tags_dict:
        encoded = encode_tags(tags_dict)
        q = np.array([encoded.get(c, 0) for c in avail_tags], dtype=float)
        train_tags = train_meta[avail_tags].values.astype(float)
        intersection = train_tags.dot(q)
        union = train_tags.sum(axis=1) + q.sum() - intersection
        tag_sims = np.where(union > 0, intersection / union, 0.0)
    else:
        tag_sims = np.zeros(len(train_meta))

    # --- IMDb proximity [0,1] ---
    train_imdb = pd.to_numeric(
        train_meta["imdbRating"], errors="coerce"
    ).fillna(5.0).values
    imdb_sims = np.clip(
        1.0 - np.abs(train_imdb - float(imdb_rating or 5.0)) / 10.0, 0, 1
    )

    # --- Combined score (adaptive: drop tag weight if no tag signal) ---
    has_tags = avail_tags and tags_dict and q.sum() > 0
    if has_tags:
        combined = 0.60 * plot_sims + 0.30 * tag_sims + 0.10 * imdb_sims
    else:
        combined = 0.85 * plot_sims + 0.15 * imdb_sims

    # Exclude self (if the movie is already in training set)
    if exclude_const:
        mask = train_meta["Const"].values == exclude_const
        combined[mask] = -1.0

    best_idx = int(np.argmax(combined))
    best_sim = float(combined[best_idx])

    if best_sim < threshold:
        return None

    row = train_meta.iloc[best_idx]
    return {
        "title":      str(row.get("Title", "Unknown")),
        "const":      str(row.get("Const", "")),
        "rating":     float(row.get("Your Rating", 0)),
        "similarity": best_sim,
    }


def find_similar_movie(
    embedding: np.ndarray,
    artifacts: dict,
    tags_dict: dict | None = None,
    imdb_rating: float = 5.0,
) -> dict:
    """Backward-compat wrapper — returns best match unconditionally (used for Why prompt)."""
    result = find_similar_movie_combined(
        embedding, tags_dict or {}, imdb_rating, artifacts, threshold=0.0
    )
    return result or {"title": "Unknown", "const": "", "rating": 5.0, "similarity": 0.0}


def compute_vibe_match(embedding: np.ndarray, artifacts: dict) -> float:
    """Cosine similarity between this movie and the centroid of your top-rated films."""
    centroid = artifacts["high_rated_centroid"]
    sim = cosine_similarity(embedding.reshape(1, -1), centroid)[0][0]
    # Map [-1,1] cosine sim → [0,100%], clamped
    pct = float(np.clip((sim + 1) / 2 * 100, 0, 100))
    return round(pct, 1)


# ─── SHAP interpretation ──────────────────────────────────────────────────────

def get_shap_contributions(row: pd.DataFrame, artifacts: dict, rec: dict = None) -> list[dict]:
    """Return list of {feature, label, value, shap} sorted by |shap| desc."""
    explainer = artifacts["explainer"]
    shap_vals = explainer.shap_values(row)[0]
    feature_names = artifacts["feature_names"]

    rec = rec or {}
    director1 = str(rec.get("Director", "") or "").split(",")[0].strip() or "Director"
    actor1 = (
        str(rec.get("Actors_RT", "") or "").split(",")[0].strip()
        or str(rec.get("Actors", "") or "").split(",")[0].strip()
        or "Actor"
    )
    studio = str(rec.get("Studio", "") or "").strip() or "Studio"
    if studio in ("N/A",):
        studio = "Studio"

    _named_labels = {
        "director_film_count": f"Director:{director1}: {{val:.0f}} Films Seen",
        "director_avg_rating": f"Director:{director1}: Avg Rating {{val:.1f}}",
        "actor1_film_count":   f"Actor:{actor1}: {{val:.0f}} Films Seen",
        "actor1_avg_rating":   f"Actor:{actor1}: Avg Rating {{val:.1f}}",
        "studio_film_count":   f"Studio:{studio}: {{val:.0f}} Films Seen",
        "studio_avg_rating":   f"Studio:{studio}: Avg Rating {{val:.1f}}",
    }

    contributions = []
    for feat, sv, fv in zip(feature_names, shap_vals, row.values[0]):
        if feat.startswith("plot_"):
            label = "Plot Themes"
        elif feat.startswith("genre_"):
            label = f"Genre: {feat.replace('genre_', '')}"
        elif feat.startswith("tag_"):
            label = tag_display_label(feat) or feat
        elif feat in _named_labels:
            label = _named_labels[feat].format(val=fv)
        else:
            label = FEATURE_LABELS.get(feat, feat)

        contributions.append({
            "feature": feat,
            "label":   label,
            "value":   float(fv),
            "shap":    float(sv),
        })

    # Sort by |shap| descending
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return contributions


def format_feature_tags(contributions: list[dict], rec: dict) -> list[dict]:
    """Build human-readable feature impact tags for the UI badges."""
    tags = []
    seen_labels = set()

    # Pre-build count lookup to suppress avg_rating when count == 0
    _counts = {c["feature"]: c["value"] for c in contributions
               if c["feature"] in ("director_film_count", "actor1_film_count", "studio_film_count")}

    for c in contributions:
        label = c["label"]

        # De-duplicate "Plot Themes" (many plot_X features)
        if label == "Plot Themes":
            if "Plot Themes" in seen_labels:
                continue
            seen_labels.add("Plot Themes")
            direction = "+" if c["shap"] > 0 else "-"
            tags.append({"label": "Plot Themes", "direction": direction, "shap": c["shap"]})
            continue

        if label in seen_labels:
            continue

        # For one-hot features, only show if the film actually has that attribute (value == 1).
        # Covers genre_, tag_, decade_, rated_, lang_, country_ — avoids showing e.g.
        # "2020s Film" or "Rated PG-13" for films that are NOT those things.
        _ONE_HOT_PREFIXES = ("genre_", "tag_", "decade_", "rated_", "lang_", "country_")
        if any(c["feature"].startswith(p) for p in _ONE_HOT_PREFIXES) and c["value"] == 0:
            continue

        seen_labels.add(label)

        direction = "+" if c["shap"] > 0 else "-"

        def _is_missing(v):
            """True if v is None, NaN, empty string, or 'N/A'."""
            if v is None:
                return True
            if isinstance(v, float) and pd.isna(v):
                return True
            if str(v).strip() in ("", "N/A", "nan", "None"):
                return True
            return False

        # Add context value for numeric/genre tags
        if c["feature"] == "imdbRating":
            raw_val = rec.get("imdbRating")
            if _is_missing(raw_val):
                continue
            display = f"IMDb: {raw_val}"
        elif c["feature"] == "Runtime":
            raw_val = rec.get("Runtime")
            if _is_missing(raw_val):
                continue
            display = f"Runtime: {raw_val}min"
        elif c["feature"] == "Year":
            raw_val = rec.get("Year")
            if _is_missing(raw_val):
                continue
            display = f"Year: {int(float(raw_val))}"
        elif c["feature"] == "Metascore":
            raw_val = rec.get("Metascore")
            if _is_missing(raw_val):
                continue
            display = f"Metacritic: {int(float(raw_val))}"
        elif c["feature"] == "RT_score":
            raw_val = rec.get("RT_score")
            if _is_missing(raw_val):
                continue
            display = f"RT Critic: {raw_val}%"
        elif c["feature"] == "award_wins":
            raw_val = rec.get("award_wins", 0)
            if _is_missing(raw_val) or int(float(raw_val)) == 0:
                continue
            display = f"Awards: {int(float(raw_val))} wins"
        elif c["feature"] == "award_noms":
            raw_val = rec.get("award_noms", 0)
            if _is_missing(raw_val) or int(float(raw_val)) == 0:
                continue
            display = f"Nominations: {int(float(raw_val))}"
        elif c["feature"] == "oscar_win":
            display = "Oscar Winner" if rec.get("oscar_win", 0) else "No Oscar Win"
        elif c["feature"] == "oscar_nom":
            display = "Oscar Nominated" if rec.get("oscar_nom", 0) else "No Oscar Nom"
        elif c["feature"] == "director_avg_rating":
            if _counts.get("director_film_count", 0) == 0:
                continue
            d1 = str(rec.get("Director", "") or "").split(",")[0].strip()
            display = f"{d1}: Avg Rating {c['value']:.1f}" if d1 and d1 not in ("N/A", "Unknown") else "Director: Avg Rating"
        elif c["feature"] == "director_film_count":
            if c["value"] == 0:
                continue
            d1 = str(rec.get("Director", "") or "").split(",")[0].strip()
            display = f"{d1}: {int(c['value'])} Films Seen" if d1 and d1 not in ("N/A", "Unknown") else "Director: Films Seen"
        elif c["feature"] == "actor1_avg_rating":
            if _counts.get("actor1_film_count", 0) == 0:
                continue
            a1 = (
                str(rec.get("Actors_RT", "") or "").split(",")[0].strip()
                or str(rec.get("Actors", "") or "").split(",")[0].strip()
            )
            display = f"{a1}: Avg Rating {c['value']:.1f}" if a1 and a1 not in ("N/A", "Unknown") else "Actor: Avg Rating"
        elif c["feature"] == "actor1_film_count":
            if c["value"] == 0:
                continue
            a1 = (
                str(rec.get("Actors_RT", "") or "").split(",")[0].strip()
                or str(rec.get("Actors", "") or "").split(",")[0].strip()
            )
            display = f"{a1}: {int(c['value'])} Films Seen" if a1 and a1 not in ("N/A", "Unknown") else "Actor: Films Seen"
        elif c["feature"] == "studio_avg_rating":
            s = str(rec.get("Studio", "") or "").strip()
            if not s or s in ("N/A", "Unknown") or _counts.get("studio_film_count", 0) == 0:
                continue
            display = f"Studio: {s}: Avg Rating {c['value']:.1f}"
        elif c["feature"] == "studio_film_count":
            s = str(rec.get("Studio", "") or "").strip()
            if not s or s in ("N/A", "Unknown") or c["value"] == 0:
                continue
            display = f"Studio: {s}: {int(c['value'])} Films Seen"
        else:
            display = label

        tags.append({"label": display, "direction": direction, "shap": c["shap"]})

        if len(tags) >= 8:
            break

    return tags


# ─── Main prediction entry point ─────────────────────────────────────────────

def predict_movie(title: str, imdb_id: str | None = None) -> dict | None:
    """
    Full pipeline for a single movie title.
    Returns a rich dict with all UI data, or None if not found.
    """
    # 1. Fetch from OMDb
    raw = fetch_by_imdb_id(imdb_id) if imdb_id else fetch_by_title(title)
    if raw is None:
        return None

    rec = parse_omdb(raw)
    if not rec.get("Plot") or rec["Plot"] == "N/A":
        rec["Plot"] = rec.get("Title", title)

    # 1a. Attach RT audience score + IMDb-scraped awards for UI display
    imdb_id = rec.get("imdbID", "")
    if imdb_id:
        rt_aud, awards = _get_extra_data()
        rec["rt_audience"] = rt_aud.get(imdb_id)
        aw = awards.get(imdb_id) or {}
        rec["oscar_wins"] = aw.get("oscar_wins", 0)
        rec["oscar_noms"] = aw.get("oscar_noms", 0)
        rec["gg_wins"]    = aw.get("gg_wins", 0)
        rec["gg_noms"]    = aw.get("gg_noms", 0)

    # 1b. RT enrichment: top-5 cast + studio (best-effort, ~0.5s)
    try:
        rt_year = int(str(rec.get("Year") or 2000).split(".")[0])
        rt_data = fetch_rt_data(rec["Title"], rt_year, sleep=True)
    except Exception:
        rt_data = None
    if rt_data:
        rec["Actors_RT"] = ", ".join(rt_data["cast_top5"])
        rec["Studio"]    = rt_data["studio"] or rec.get("Production", "") or ""
    else:
        rec["Actors_RT"] = ""
        rec["Studio"]    = rec.get("Production", "") or ""

    # 2. LLM tag inference
    tags_dict    = call_groq_tagger(rec)
    tags_encoded = encode_tags(tags_dict)

    # 3. Scrape IMDb Parents Guide and cache to CSV for future lookups
    pg_ratings = scrape_imdb_parents_guide(rec.get("imdbID", ""))
    if pg_ratings:
        cache_pg_rating(rec.get("imdbID", ""), rec.get("Title", ""), pg_ratings)

    # 4. Load artifacts
    artifacts = _load_all()

    # 5. Build features
    row, embedding = _build_single_features(rec, artifacts, tags_encoded, pg_ratings)

    # 6. Predict
    pred_raw = float(artifacts["model"].predict(row)[0])
    pred_score = float(np.clip(pred_raw, 1, 10))
    match_pct  = round(pred_score / 10 * 100, 1)

    # 6. SHAP
    contributions = get_shap_contributions(row, artifacts, rec=rec)
    feature_tags = format_feature_tags(contributions, rec)

    # 7. Similarity (used for Why prompt — unconstrained best match)
    similar = find_similar_movie(
        embedding, artifacts,
        tags_dict=tags_dict,
        imdb_rating=float(rec.get("imdbRating") or 5.0),
    )
    vibe = compute_vibe_match(embedding, artifacts)

    # 8. Top SHAP for narrative prompt (top 4 pos + top 4 neg, skipping duplicates)
    top_pos = [c for c in contributions if c["shap"] > 0][:4]
    top_neg = [c for c in contributions if c["shap"] < 0][:4]

    return {
        "rec":           rec,
        "pred_score":    pred_score,
        "match_pct":     match_pct,
        "contributions": contributions,
        "tags":          feature_tags,
        "tags_dict":     tags_dict,      # {category: value} for UI display
        "similar":       similar,
        "vibe":          vibe,
        "top_pos":       top_pos,
        "top_neg":       top_neg,
        "embedding":     embedding,
        "pg_ratings":    pg_ratings,   # {sex_nudity: ..., ...} or None
    }
