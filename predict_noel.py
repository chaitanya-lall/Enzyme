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

from config import (
    NOEL_MODEL_PATH, NOEL_SCALER_PATH, NOEL_MLB_PATH,
    NOEL_FEATURE_NAMES_PATH, NOEL_TRAIN_META_PATH, NOEL_TRAIN_EMB_PATH,
    SENTENCE_TRANSFORMER_MODEL,
)
from predict import parse_omdb, fetch_by_title, FEATURE_LABELS, NUMERIC_COLS, _build_enriched_text
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
                            tags_encoded: dict | None = None):
    scaler        = artifacts["scaler"]
    mlb           = artifacts["mlb"]
    nlp           = artifacts["nlp"]
    feature_names = artifacts["feature_names"]

    # 1. Numerics (includes award features parsed in parse_omdb)
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

    # 2. Genre encoding
    genres = [g.strip() for g in rec.get("Genre", "").split(",") if g.strip()]
    genre_arr = mlb.transform([genres])
    genre_feat = pd.DataFrame(genre_arr, columns=[f"genre_{c}" for c in mlb.classes_])

    # 3. LLM categorical tag features
    tag_row = {col: int(tags_encoded.get(col, 0)) if tags_encoded else 0
               for col in ALL_TAG_COLS}
    tag_feat = pd.DataFrame([tag_row])

    # 4. Enriched text embedding (Director + Cast + Writer + Plot)
    enriched_text = _build_enriched_text(rec)
    embedding = nlp.encode([enriched_text])
    plot_feat = pd.DataFrame(embedding, columns=[f"plot_{i}" for i in range(embedding.shape[1])])

    row = pd.concat([num_feat, genre_feat, tag_feat, plot_feat], axis=1)
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


def get_shap_contributions(row: pd.DataFrame, artifacts: dict) -> list[dict]:
    shap_vals = artifacts["explainer"].shap_values(row)[0]
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
        contributions.append({"feature": feat, "label": label, "value": float(fv), "shap": float(sv)})
    contributions.sort(key=lambda x: abs(x["shap"]), reverse=True)
    return contributions


def format_feature_tags(contributions: list[dict], rec: dict) -> list[dict]:
    tags = []
    seen_labels = set()
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

        # For one-hot features (genre/tag), only show if the film actually has it (value == 1)
        if (c["feature"].startswith("genre_") or c["feature"].startswith("tag_")) and c["value"] == 0:
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
            display = f"Metacritic: {raw_val}" if raw_val else "Metacritic"
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


def predict_movie_noel(rec: dict, tags_dict: dict | None = None) -> dict:
    """
    Run Noel's model on an already-fetched & parsed OMDb record.
    Pass the result of predict.parse_omdb() directly.
    tags_dict: {category: value} from call_groq_tagger (already called by predict_movie).
    """
    if not rec.get("Plot") or rec["Plot"] == "N/A":
        rec["Plot"] = rec.get("Title", "")

    tags_encoded = encode_tags(tags_dict) if tags_dict else None

    artifacts = _load_all()
    row, embedding = _build_single_features(rec, artifacts, tags_encoded)

    pred_raw   = float(artifacts["model"].predict(row)[0])
    pred_score = float(np.clip(pred_raw, 1, 10))
    match_pct  = round(pred_score / 10 * 100, 1)

    contributions = get_shap_contributions(row, artifacts)
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
