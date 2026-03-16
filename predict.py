"""
Inference engine for the PMTPE model.
Handles: OMDb fetch → feature engineering → prediction → SHAP → similarity lookup.
All artifact loading is lazy and cached for Streamlit performance.
"""
from __future__ import annotations

import os
import time
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
)
from tag_features import call_groq_tagger, encode_tags, ALL_TAG_COLS, tag_display_label

# Artifact paths
MODEL_PATH          = os.path.join(MODELS_DIR, "pmtpe_model.pkl")
SCALER_PATH         = os.path.join(MODELS_DIR, "scaler.pkl")
MLB_PATH            = os.path.join(MODELS_DIR, "mlb_genres.pkl")
FEATURE_NAMES_PATH  = os.path.join(MODELS_DIR, "feature_names.pkl")
TRAIN_META_PATH     = os.path.join(DATA_DIR,   "train_meta.pkl")
TRAIN_EMB_PATH      = os.path.join(MODELS_DIR, "train_plot_embeddings.npy")

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

# Human-readable labels for feature names shown in the UI
FEATURE_LABELS = {
    "imdbRating":      "IMDb Rating",
    "imdbVotes":       "IMDb Popularity",
    "Metascore":       "Metacritic Score",
    "RT_score":        "Rotten Tomatoes",
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


def fetch_by_title(title: str) -> dict | None:
    """Search OMDb by movie title. Returns raw OMDb JSON or None."""
    params = {"t": title, "plot": "full", "apikey": OMDB_APP_KEY}
    try:
        resp = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("Response") == "False":
            err = data.get("Error", "Not found")
            print(f"OMDb: {err}")
            return None
        return data
    except Exception as e:
        print(f"OMDb error: {e}")
        return None


import re as _re

def _parse_awards(awards_str) -> dict:
    """Parse OMDb Awards string → numeric signals."""
    s = str(awards_str) if awards_str else ""
    oscar_win = 1 if _re.search(r'\bwon\b.*\boscar', s, _re.IGNORECASE) else 0
    oscar_nom = 1 if (_re.search(r'\bnominat', s, _re.IGNORECASE) and
                      _re.search(r'\boscar', s, _re.IGNORECASE)) else 0
    wins = sum(int(x) for x in _re.findall(r'(\d+)\s+win', s, _re.IGNORECASE))
    noms = sum(int(x) for x in _re.findall(r'(\d+)\s+nomination', s, _re.IGNORECASE))
    return {
        "award_wins": float(wins),
        "award_noms": float(noms),
        "oscar_win":  float(oscar_win),
        "oscar_nom":  float(oscar_nom),
    }


def _build_enriched_text(rec: dict) -> str:
    """Combine Director + Cast + Writer + Plot into one text for embedding."""
    parts = []
    director = str(rec.get("Director", "") or "").strip()
    actors   = str(rec.get("Actors",   "") or "").strip()
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
    }
    rec.update(award_feats)
    return rec


# ─── Feature engineering (single movie) ──────────────────────────────────────

def _build_single_features(rec: dict, artifacts: dict,
                            tags_encoded: dict | None = None) -> tuple:
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

    # 2. Decade one-hot (replaces raw Year)
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

    # 8. Enriched text embedding (Director + Cast + Writer + Plot)
    enriched_text = _build_enriched_text(rec)
    embedding = nlp.encode([enriched_text])
    plot_feat = pd.DataFrame(embedding, columns=[f"plot_{i}" for i in range(embedding.shape[1])])

    # 9. Combine & align columns
    row = pd.concat([num_feat, decade_feat, rated_feat, lang_feat, country_feat,
                     genre_feat, tag_feat, plot_feat], axis=1)
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

def get_shap_contributions(row: pd.DataFrame, artifacts: dict) -> list[dict]:
    """Return list of {feature, label, value, shap} sorted by |shap| desc."""
    explainer = artifacts["explainer"]
    shap_vals = explainer.shap_values(row)[0]
    feature_names = artifacts["feature_names"]

    contributions = []
    for feat, sv, fv in zip(feature_names, shap_vals, row.values[0]):
        if feat.startswith("plot_"):
            label = "Plot Themes"
        elif feat.startswith("genre_"):
            label = f"Genre: {feat.replace('genre_', '')}"
        elif feat.startswith("tag_"):
            label = tag_display_label(feat) or feat
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

        # For one-hot features (genre/tag), only show if the film actually has it (value == 1)
        if (c["feature"].startswith("genre_") or c["feature"].startswith("tag_")) and c["value"] == 0:
            continue

        seen_labels.add(label)

        direction = "+" if c["shap"] > 0 else "-"

        # Add context value for numeric/genre tags
        if c["feature"] == "imdbRating":
            raw_val = rec.get("imdbRating")
            display = f"IMDb: {raw_val}" if raw_val else "IMDb Rating"
        elif c["feature"] == "Runtime":
            raw_val = rec.get("Runtime")
            display = f"Runtime: {raw_val}min" if raw_val else "Runtime"
        elif c["feature"] == "Year":
            raw_val = rec.get("Year")
            display = f"Year: {int(raw_val)}" if raw_val else "Year"
        elif c["feature"] == "Metascore":
            raw_val = rec.get("Metascore")
            if raw_val is None or (isinstance(raw_val, float) and pd.isna(raw_val)):
                continue  # no Metacritic score — skip this driver
            display = f"Metacritic: {int(raw_val)}"
        elif c["feature"] == "RT_score":
            raw_val = rec.get("RT_score")
            display = f"RT: {raw_val}%" if raw_val else "RT Score"
        elif c["feature"] == "award_wins":
            raw_val = rec.get("award_wins", 0)
            display = f"Awards: {int(raw_val)} wins" if raw_val else "Award Wins"
        elif c["feature"] == "award_noms":
            raw_val = rec.get("award_noms", 0)
            display = f"Nominations: {int(raw_val)}" if raw_val else "Nominations"
        elif c["feature"] == "oscar_win":
            display = "Oscar Winner" if rec.get("oscar_win", 0) else "No Oscar Win"
        elif c["feature"] == "oscar_nom":
            display = "Oscar Nominated" if rec.get("oscar_nom", 0) else "No Oscar Nom"
        else:
            display = label

        tags.append({"label": display, "direction": direction, "shap": c["shap"]})

        if len(tags) >= 8:
            break

    return tags


# ─── Main prediction entry point ─────────────────────────────────────────────

def predict_movie(title: str) -> dict | None:
    """
    Full pipeline for a single movie title.
    Returns a rich dict with all UI data, or None if not found.
    """
    # 1. Fetch from OMDb
    raw = fetch_by_title(title)
    if raw is None:
        return None

    rec = parse_omdb(raw)
    if not rec.get("Plot") or rec["Plot"] == "N/A":
        rec["Plot"] = rec.get("Title", title)

    # 2. LLM tag inference
    tags_dict    = call_groq_tagger(rec)
    tags_encoded = encode_tags(tags_dict)

    # 3. Load artifacts
    artifacts = _load_all()

    # 4. Build features
    row, embedding = _build_single_features(rec, artifacts, tags_encoded)

    # 5. Predict
    pred_raw = float(artifacts["model"].predict(row)[0])
    pred_score = float(np.clip(pred_raw, 1, 10))
    match_pct  = round(pred_score / 10 * 100, 1)

    # 6. SHAP
    contributions = get_shap_contributions(row, artifacts)
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
    }
