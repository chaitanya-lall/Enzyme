"""
Inference engine for Noel's Movie-Meter model.
Mirrors predict.py exactly but loads Noel's artifacts.
OMDb fetching is delegated to predict.py (shared).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
import shap
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
from config import (
    NOEL_MODEL_PATH, NOEL_SCALER_PATH, NOEL_MLB_PATH,
    NOEL_FEATURE_NAMES_PATH, NOEL_TRAIN_META_PATH, NOEL_TRAIN_EMB_PATH,
    NOEL_MODELS_DIR, SENTENCE_TRANSFORMER_MODEL, PG_ORDINAL, PG_COLS,
)
from predict import parse_omdb, fetch_by_title, FEATURE_LABELS, NUMERIC_COLS, _build_enriched_text

NOEL_DIRECTOR_STATS_PATH = os.path.join(NOEL_MODELS_DIR, "director_stats.pkl")
NOEL_ACTOR_STATS_PATH    = os.path.join(NOEL_MODELS_DIR, "actor_stats.pkl")
NOEL_STUDIO_STATS_PATH   = os.path.join(NOEL_MODELS_DIR, "studio_stats.pkl")
NOEL_OVERALL_AVG_PATH    = os.path.join(NOEL_MODELS_DIR, "overall_avg.pkl")
from tag_features import call_groq_tagger, encode_tags, ALL_TAG_COLS, tag_display_label

_cache: dict = {}


def _load_all():
    if _cache:
        return _cache

    _cache["model"]         = joblib.load(NOEL_MODEL_PATH)
    _cache["scaler"]        = joblib.load(NOEL_SCALER_PATH)
    _cache["mlb"]           = joblib.load(NOEL_MLB_PATH)
    _cache["feature_names"] = joblib.load(NOEL_FEATURE_NAMES_PATH)
    _cache["nlp"]           = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    _cache["train_meta"]    = pd.read_pickle(NOEL_TRAIN_META_PATH)
    _cache["train_emb"]     = np.load(NOEL_TRAIN_EMB_PATH)
    _cache["director_stats"] = joblib.load(NOEL_DIRECTOR_STATS_PATH) if os.path.exists(NOEL_DIRECTOR_STATS_PATH) else {}
    _cache["actor_stats"]    = joblib.load(NOEL_ACTOR_STATS_PATH)    if os.path.exists(NOEL_ACTOR_STATS_PATH)    else {}
    _cache["studio_stats"]   = joblib.load(NOEL_STUDIO_STATS_PATH)   if os.path.exists(NOEL_STUDIO_STATS_PATH)   else {}
    _cache["overall_avg"]    = joblib.load(NOEL_OVERALL_AVG_PATH)    if os.path.exists(NOEL_OVERALL_AVG_PATH)    else 6.5

    meta = _cache["train_meta"]
    emb  = _cache["train_emb"]

    mask = meta["Your Rating"] >= 8
    _cache["high_rated_centroid"] = (
        emb[mask.values].mean(axis=0, keepdims=True) if mask.sum() > 0
        else emb.mean(axis=0, keepdims=True)
    )

    _cache["explainer"] = shap.TreeExplainer(_cache["model"])
    return _cache


def _build_single_features(rec: dict, artifacts: dict,
                            tags_encoded: dict | None = None,
                            pg_ratings: dict | None = None):
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

    # 2. Director / actor / studio mean encoding
    director_stats = artifacts.get("director_stats", {})
    actor_stats    = artifacts.get("actor_stats",    {})
    studio_stats   = artifacts.get("studio_stats",   {})
    overall_avg    = artifacts.get("overall_avg",    6.5)

    director1 = str(rec.get("Director", "") or "").split(",")[0].strip()
    actor1    = (
        str(rec.get("Actors_RT", "") or "").split(",")[0].strip()
        or str(rec.get("Actors", "") or "").split(",")[0].strip()
    )
    studio = str(rec.get("Studio", "") or "").strip()
    if not studio or studio in ("N/A",):
        studio = "Unknown"

    d_info = director_stats.get(director1, {})
    a_info = actor_stats.get(actor1, {})
    s_info = studio_stats.get(studio, {})
    dir_act_feat = pd.DataFrame([{
        "director_film_count": float(d_info.get("count", 0)),
        "director_avg_rating": float(d_info.get("avg",   overall_avg)),
        "actor1_film_count":   float(a_info.get("count", 0)),
        "actor1_avg_rating":   float(a_info.get("avg",   overall_avg)),
        "studio_film_count":   float(s_info.get("count", 0)),
        "studio_avg_rating":   float(s_info.get("avg",   overall_avg)),
    }])

    # 3. Decade one-hot
    try:
        year_raw = rec.get("Year")
        yr = int(str(int(float(year_raw)))[:4]) if year_raw is not None else 2000
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
    rated_bucket = _rated_map.get(str(rec.get("Rated", "N/A")), "NR")
    rated_feat = pd.DataFrame([{
        "rated_G":    int(rated_bucket == "G"),
        "rated_PG":   int(rated_bucket == "PG"),
        "rated_PG13": int(rated_bucket == "PG13"),
        "rated_R":    int(rated_bucket == "R"),
        "rated_NR":   int(rated_bucket == "NR"),
    }])

    # 5. Language binary
    primary_lang = (rec.get("Language") or "").split(",")[0].strip()
    lang_feat = pd.DataFrame([{
        "lang_english": int(primary_lang == "English"),
        "lang_hindi":   int(primary_lang == "Hindi"),
        "lang_other":   int(primary_lang not in ("English", "Hindi")),
    }])

    # 6. Country binary
    country_str = rec.get("Country") or ""
    country_feat = pd.DataFrame([{
        "country_us":    int("United States" in country_str),
        "country_uk":    int("United Kingdom" in country_str),
        "country_india": int("India" in country_str),
        "country_other": int(not any(c in country_str for c in
                                     ("United States", "United Kingdom", "India"))),
    }])

    # 7. Genre encoding
    genres = [g.strip() for g in rec.get("Genre", "").split(",") if g.strip()]
    genre_arr = mlb.transform([genres])
    genre_feat = pd.DataFrame(genre_arr, columns=[f"genre_{c}" for c in mlb.classes_])

    # 8. LLM categorical tag features
    tag_row = {col: int(tags_encoded.get(col, 0)) if tags_encoded else 0
               for col in ALL_TAG_COLS}
    tag_feat = pd.DataFrame([tag_row])

    # 9. Parents Guide features
    pg_row = {}
    for col in PG_COLS:
        raw_key = col.replace("pg_", "")
        val = (pg_ratings or {}).get(raw_key)
        pg_row[col] = PG_ORDINAL.get(val, -1) if val else -1
    pg_feat = pd.DataFrame([pg_row])

    # 10. Enriched text embedding (Director + Cast + Writer + Plot)
    enriched_text = _build_enriched_text(rec)
    embedding = nlp.encode([enriched_text])
    plot_feat = pd.DataFrame(embedding, columns=[f"plot_{i}" for i in range(embedding.shape[1])])

    row = pd.concat([num_feat, dir_act_feat, decade_feat, rated_feat, lang_feat, country_feat,
                     genre_feat, tag_feat, pg_feat, plot_feat], axis=1)
    row = row.reindex(columns=feature_names, fill_value=0.0)
    return row, embedding[0]


def find_similar_movie(embedding: np.ndarray, artifacts: dict) -> dict:
    train_emb  = artifacts["train_emb"]
    train_meta = artifacts["train_meta"]
    sims = cosine_similarity(embedding.reshape(1, -1), train_emb)[0]
    best_idx = int(np.argmax(sims))
    row = train_meta.iloc[best_idx]
    return {
        "title":      row.get("Title", "Unknown"),
        "rating":     float(row.get("Your Rating", 0)),
        "similarity": float(sims[best_idx]),
    }


def compute_vibe_match(embedding: np.ndarray, artifacts: dict) -> float:
    centroid = artifacts["high_rated_centroid"]
    sim = cosine_similarity(embedding.reshape(1, -1), centroid)[0][0]
    return round(float(np.clip((sim + 1) / 2 * 100, 0, 100)), 1)


def get_shap_contributions(row: pd.DataFrame, artifacts: dict, rec: dict = None) -> list[dict]:
    shap_vals = artifacts["explainer"].shap_values(row)[0]
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
        contributions.append({"feature": feat, "label": label, "value": float(fv), "shap": float(sv)})
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return contributions


def format_feature_tags(contributions: list[dict], rec: dict) -> list[dict]:
    tags = []
    seen_labels = set()

    # Pre-build count lookup to suppress avg_rating when count == 0
    _counts = {c["feature"]: c["value"] for c in contributions
               if c["feature"] in ("director_film_count", "actor1_film_count", "studio_film_count")}

    for c in contributions:
        label = c["label"]
        if label == "Plot Themes":
            if "Plot Themes" in seen_labels:
                continue
            seen_labels.add("Plot Themes")
            tags.append({"label": "Plot Themes", "direction": "+" if c["shap"] > 0 else "-", "shap": c["shap"]})
            continue
        if label in seen_labels:
            continue

        # For one-hot features, only show if the film actually has that attribute (value == 1)
        _ONE_HOT_PREFIXES = ("genre_", "tag_", "decade_", "rated_", "lang_", "country_")
        if any(c["feature"].startswith(p) for p in _ONE_HOT_PREFIXES) and c["value"] == 0:
            continue

        seen_labels.add(label)
        direction = "+" if c["shap"] > 0 else "-"

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
                continue
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


def predict_movie_noel(rec: dict, tags_dict: dict | None = None,
                       pg_ratings: dict | None = None) -> dict:
    """
    Run Noel's model on an already-fetched & parsed OMDb record.
    Pass the result of predict.parse_omdb() directly.
    tags_dict: {category: value} from call_groq_tagger (already called by predict_movie).
    pg_ratings: {sex_nudity: ..., ...} from scrape_imdb_parents_guide (already called by predict_movie).
    """
    if not rec.get("Plot") or rec["Plot"] == "N/A":
        rec["Plot"] = rec.get("Title", "")

    tags_encoded = encode_tags(tags_dict) if tags_dict else None

    artifacts = _load_all()
    row, embedding = _build_single_features(rec, artifacts, tags_encoded, pg_ratings)

    pred_raw   = float(artifacts["model"].predict(row)[0])
    pred_score = float(np.clip(pred_raw, 1, 10))
    match_pct  = round(pred_score / 10 * 100, 1)

    contributions = get_shap_contributions(row, artifacts, rec=rec)
    feature_tags  = format_feature_tags(contributions, rec)
    similar       = find_similar_movie(embedding, artifacts)
    vibe          = compute_vibe_match(embedding, artifacts)

    top_pos = [c for c in contributions if c["shap"] > 0][:4]
    top_neg = [c for c in contributions if c["shap"] < 0][:4]

    return {
        "pred_score":    pred_score,
        "match_pct":     match_pct,
        "contributions": contributions,
        "tags":          feature_tags,
        "similar":       similar,
        "vibe":          vibe,
        "top_pos":       top_pos,
        "top_neg":       top_neg,
        "embedding":     embedding,
    }
