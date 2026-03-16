"""
Step 3 (Noel): Feature engineering, Gradient Boosting training, and SHAP explainability.
Input:  data/noel/enriched_omdb.csv  (must have Writer column — run patch_add_writer.py first)
Output: models/noel/*, outputs/noel/*

Features:
  - Numerics (10): imdbRating, Metascore, BoxOffice, imdbVotes, Runtime,
                   RT_score, award_wins, award_noms, oscar_win, oscar_nom
  - Director/actor/studio mean encoding (6)
  - Decade one-hot (replaces raw Year): decade_pre1970s … decade_2020s
  - Content rating one-hot: rated_G, rated_PG, rated_PG13, rated_R, rated_NR
  - Language binary: lang_english, lang_hindi, lang_other
  - Country binary:  country_us, country_uk, country_india, country_other
  - Genre one-hot  (~25 tags)
  - LLM thematic tags (111 columns)
  - Parents Guide ordinal (5 columns)
  - Enriched text embedding (384-dim): "Director: X. Cast: Y. Writer: Z. {Plot}"
"""
import re
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

from sentence_transformers import SentenceTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from config import (
    NOEL_OMDB_CSV, RT_CSV,
    NOEL_MLB_PATH, NOEL_SCALER_PATH, NOEL_FEATURE_NAMES_PATH,
    NOEL_SHAP_PLOT_PATH, NOEL_EVAL_PATH, NOEL_SHAP_VALUES_PATH,
    NOEL_MODEL_PATH, NOEL_TRAIN_META_PATH, NOEL_TRAIN_EMB_PATH,
    TRAIN_TEST_SPLIT, RANDOM_STATE, SENTENCE_TRANSFORMER_MODEL,
    NOEL_MODELS_DIR, NOEL_OUTPUTS_DIR, TAG_CSV,
    PARENTS_GUIDE_CSV, PG_ORDINAL, PG_COLS,
)
from tag_features import ALL_TAG_COLS

NUMERIC_COLS = [
    "imdbRating", "Metascore", "BoxOffice", "imdbVotes", "Runtime", "RT_score",
    "award_wins", "award_noms", "oscar_win", "oscar_nom",
]

DECADE_COLS = [
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

NOEL_DIRECTOR_STATS_PATH = os.path.join(NOEL_MODELS_DIR, "director_stats.pkl")
NOEL_ACTOR_STATS_PATH    = os.path.join(NOEL_MODELS_DIR, "actor_stats.pkl")
NOEL_STUDIO_STATS_PATH   = os.path.join(NOEL_MODELS_DIR, "studio_stats.pkl")
NOEL_OVERALL_AVG_PATH    = os.path.join(NOEL_MODELS_DIR, "overall_avg.pkl")


# ─── Awards parsing ───────────────────────────────────────────────────────────

def parse_awards(awards_str) -> dict:
    s = str(awards_str) if pd.notna(awards_str) else ""
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


def add_award_features(df: pd.DataFrame) -> pd.DataFrame:
    award_df = pd.DataFrame(list(df["Awards"].apply(parse_awards)))
    return pd.concat([df.reset_index(drop=True), award_df.reset_index(drop=True)], axis=1)


# ─── Text enrichment ──────────────────────────────────────────────────────────

def build_enriched_text(row) -> str:
    """Combine crew + plot. Uses Actors_RT (up to 5) when available."""
    parts = []
    director = str(row.get("Director", "") or "").strip()
    actors   = str(row.get("Actors_RT", "") or row.get("Actors", "") or "").strip()
    writer   = str(row.get("Writer",   "") or "").strip()
    plot     = str(row.get("Plot",     "") or "").strip()

    if director and director not in ("N/A", "Unknown"):
        parts.append(f"Director: {director}.")
    if actors and actors not in ("N/A", "Unknown"):
        parts.append(f"Cast: {actors}.")
    if writer and writer not in ("N/A", "Unknown"):
        parts.append(f"Writer: {writer}.")
    if plot and plot != "N/A":
        parts.append(plot)
    elif not parts:
        parts.append(row.get("Title", ""))
    return " ".join(parts)


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_and_clean(path):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")

    df = df.dropna(subset=["Your Rating", "Plot"])
    df = df[df["Plot"].str.strip() != "N/A"]
    print(f"After dropping missing ratings/plots: {len(df)} rows")

    base_numeric = ["Year", "imdbRating", "Metascore", "BoxOffice", "imdbVotes", "Runtime", "RT_score"]
    for col in base_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in base_numeric:
        df[col] = df[col].fillna(df[col].median())

    for col in ["Genre", "Rated", "Director", "Actors", "Writer", "Language", "Country"]:
        df[col] = df[col].fillna("Unknown")

    df = add_award_features(df)
    for col in ["award_wins", "award_noms", "oscar_win", "oscar_nom"]:
        df[col] = df[col].fillna(0.0)

    # Merge RT enrichment (top-5 cast + studio) — optional, graceful fallback
    try:
        rt_df = pd.read_csv(RT_CSV)[["Const", "Actors_RT", "Studio"]]
        df = df.merge(rt_df, on="Const", how="left")
        df["Actors_RT"] = df["Actors_RT"].fillna("")
        df["Studio"]    = df["Studio"].fillna("")
        rt_count = (df["Actors_RT"] != "").sum()
        print(f"RT enrichment merged: {rt_count}/{len(df)} movies have RT actor data")
    except FileNotFoundError:
        print("Warning: enriched_rt.csv not found — run 2b_enrich_rt.py first. "
              "Falling back to OMDb actors; no studio feature.")
        df["Actors_RT"] = ""
        df["Studio"]    = ""
    except Exception as e:
        print(f"Warning: RT merge failed ({e}) — falling back to OMDb actors")
        df["Actors_RT"] = ""
        df["Studio"]    = ""

    # Merge LLM categorical tags (if available)
    try:
        tags_df = pd.read_csv(TAG_CSV)
        before = len(df)
        df = df.merge(tags_df[["Const"] + ALL_TAG_COLS], on="Const", how="left")
        df[ALL_TAG_COLS] = df[ALL_TAG_COLS].fillna(0).astype(int)
        tagged = (df[ALL_TAG_COLS].sum(axis=1) > 0).sum()
        print(f"Tag features merged: {tagged}/{before} movies have tags")
    except FileNotFoundError:
        print("Warning: movie_tags.csv not found — tag features will be zero (run tag_movies.py first)")
        for col in ALL_TAG_COLS:
            df[col] = 0

    # Merge Parents Guide features (if available)
    try:
        pg_df = pd.read_csv(PARENTS_GUIDE_CSV, keep_default_na=False, na_values=[""])[
            ["Const", "sex_nudity", "violence_gore", "profanity", "alcohol_drugs", "intensity"]
        ]
        pg_df = pg_df.rename(columns={
            "sex_nudity": "pg_sex_nudity", "violence_gore": "pg_violence_gore",
            "profanity": "pg_profanity", "alcohol_drugs": "pg_alcohol_drugs",
            "intensity": "pg_intensity",
        })
        for col in PG_COLS:
            pg_df[col] = pg_df[col].map(PG_ORDINAL)
        df = df.merge(pg_df[["Const"] + PG_COLS], on="Const", how="left")
        df[PG_COLS] = df[PG_COLS].fillna(-1).astype(int)
        pg_count = (df[PG_COLS] >= 0).all(axis=1).sum()
        print(f"Parents Guide features merged: {pg_count}/{len(df)} movies have PG ratings")
    except FileNotFoundError:
        print("Warning: parents_guide.csv not found — PG features will be -1 (run parents_guide.py first)")
        for col in PG_COLS:
            df[col] = -1

    return df.reset_index(drop=True)


# ─── Feature engineering ──────────────────────────────────────────────────────

def _clean_year(y):
    try:
        s = str(int(y))
        return int(s[:4]) if len(s) >= 4 else int(s)
    except Exception:
        return 2000


def encode_decade(df):
    years   = df["Year"].apply(_clean_year)
    decades = (years // 10) * 10
    result = {
        "decade_pre1970s": (decades < 1970).astype(int),
        "decade_1970s":    (decades == 1970).astype(int),
        "decade_1980s":    (decades == 1980).astype(int),
        "decade_1990s":    (decades == 1990).astype(int),
        "decade_2000s":    (decades == 2000).astype(int),
        "decade_2010s":    (decades == 2010).astype(int),
        "decade_2020s":    (decades >= 2020).astype(int),
    }
    return pd.DataFrame(result, index=df.index)


def encode_rated(df):
    mapping = {
        "R": "R", "TV-MA": "R", "18+": "R",
        "PG-13": "PG13", "TV-14": "PG13", "16+": "PG13",
        "PG": "PG", "TV-PG": "PG",
        "G": "G", "TV-G": "G",
    }
    bucket = df["Rated"].map(mapping).fillna("NR")
    result = {
        "rated_G":    (bucket == "G").astype(int),
        "rated_PG":   (bucket == "PG").astype(int),
        "rated_PG13": (bucket == "PG13").astype(int),
        "rated_R":    (bucket == "R").astype(int),
        "rated_NR":   (bucket == "NR").astype(int),
    }
    return pd.DataFrame(result, index=df.index)


def encode_language(df):
    primary = df["Language"].str.split(",").str[0].str.strip()
    result = {
        "lang_english": (primary == "English").astype(int),
        "lang_hindi":   (primary == "Hindi").astype(int),
        "lang_other":   (~primary.isin(["English", "Hindi"])).astype(int),
    }
    return pd.DataFrame(result, index=df.index)


def encode_country(df):
    country = df["Country"].fillna("")
    result = {
        "country_us":    country.str.contains("United States", case=False, na=False).astype(int),
        "country_uk":    country.str.contains("United Kingdom", case=False, na=False).astype(int),
        "country_india": country.str.contains("India", case=False, na=False).astype(int),
        "country_other": (~country.str.contains(
            "United States|United Kingdom|India", case=False, na=False)).astype(int),
    }
    return pd.DataFrame(result, index=df.index)


def encode_director_actor(df):
    """
    Leave-one-out mean encoding for primary director, top-billed actor, and studio.
    Actor source priority: Actors_RT (RT, top-billed) > Actors (OMDb fallback).
    Returns (feature_df, director_stats, actor_stats, studio_stats, overall_avg).
    """
    df = df.copy()
    overall_avg = float(df["Your Rating"].mean())

    df["_dir1"] = df["Director"].str.split(",").str[0].str.strip()
    df["_act1"] = df.apply(
        lambda r: (
            str(r.get("Actors_RT", "") or "").split(",")[0].strip()
            or str(r.get("Actors", "") or "").split(",")[0].strip()
        ),
        axis=1,
    )
    df["_studio"] = df["Studio"].fillna("").str.strip()
    df["_studio"] = df["_studio"].where(df["_studio"] != "", "Unknown")

    def _loo(col):
        s   = df.groupby(col)["Your Rating"].transform("sum")
        cnt = df.groupby(col)["Your Rating"].transform("count")
        loo_cnt = cnt - 1
        loo_avg = (s - df["Your Rating"]) / loo_cnt.replace(0, float("nan"))
        loo_avg = loo_avg.fillna(overall_avg)
        return loo_cnt, loo_avg

    dir_loo_cnt, dir_loo_avg = _loo("_dir1")
    act_loo_cnt, act_loo_avg = _loo("_act1")
    stu_loo_cnt, stu_loo_avg = _loo("_studio")

    feat_df = pd.DataFrame({
        "director_film_count": dir_loo_cnt.values.astype(float),
        "director_avg_rating": dir_loo_avg.values.astype(float),
        "actor1_film_count":   act_loo_cnt.values.astype(float),
        "actor1_avg_rating":   act_loo_avg.values.astype(float),
        "studio_film_count":   stu_loo_cnt.values.astype(float),
        "studio_avg_rating":   stu_loo_avg.values.astype(float),
    }, index=df.index)

    def _stats(col):
        return (
            df.groupby(col)["Your Rating"]
            .agg(count="count", avg="mean")
            .to_dict("index")
        )

    dir_stats    = _stats("_dir1")
    act_stats    = _stats("_act1")
    studio_stats = _stats("_studio")

    print(f"  Unique directors: {len(dir_stats)}  |  actors: {len(act_stats)}  "
          f"|  studios: {len(studio_stats)}")

    return feat_df, dir_stats, act_stats, studio_stats, overall_avg


def encode_genres(df):
    df["Genre_list"] = df["Genre"].apply(
        lambda x: [g.strip() for g in x.split(",")] if isinstance(x, str) and x != "Unknown" else []
    )
    mlb = MultiLabelBinarizer()
    genre_arr = mlb.fit_transform(df["Genre_list"])
    genre_df = pd.DataFrame(
        genre_arr,
        columns=[f"genre_{c}" for c in mlb.classes_],
        index=df.index,
    )
    return genre_df, mlb


def embed_enriched_text(df):
    print(f"Loading sentence-transformer '{SENTENCE_TRANSFORMER_MODEL}'...")
    nlp_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    texts = df.apply(build_enriched_text, axis=1).tolist()
    print(f"Encoding {len(texts)} enriched texts (Director+Cast+Writer+Plot)…")
    embeddings = nlp_model.encode(texts, show_progress_bar=True, batch_size=64)
    plot_df = pd.DataFrame(
        embeddings,
        columns=[f"plot_{i}" for i in range(embeddings.shape[1])],
        index=df.index,
    )
    return plot_df, embeddings


def build_features(df):
    print("\n--- Feature Engineering ---")
    scaler = StandardScaler()
    num_arr = scaler.fit_transform(df[NUMERIC_COLS])
    num_df = pd.DataFrame(num_arr, columns=NUMERIC_COLS, index=df.index)

    dir_act_df, dir_stats, act_stats, studio_stats, overall_avg = encode_director_actor(df)
    print(f"Dir/actor/studio features: {len(DIRECTOR_ACTOR_COLS)} columns")

    decade_df  = encode_decade(df)
    rated_df   = encode_rated(df)
    lang_df    = encode_language(df)
    country_df = encode_country(df)
    print(f"Categorical features: {len(DECADE_COLS)} decade + {len(RATED_COLS)} rated "
          f"+ {len(LANG_COLS)} language + {len(COUNTRY_COLS)} country")

    genre_df, mlb = encode_genres(df)
    print(f"Genres encoded: {len(mlb.classes_)} unique genre tags")

    tag_df = df[ALL_TAG_COLS].reset_index(drop=True)
    print(f"Tag features: {len(ALL_TAG_COLS)} columns")

    pg_df = df[PG_COLS].reset_index(drop=True)
    print(f"Parents Guide features: {len(PG_COLS)} columns")

    plot_df, raw_embeddings = embed_enriched_text(df)

    X = pd.concat([num_df, dir_act_df, decade_df, rated_df, lang_df, country_df,
                   genre_df, tag_df, pg_df, plot_df], axis=1)
    y = df["Your Rating"].astype(float)
    return X, y, scaler, mlb, raw_embeddings, dir_stats, act_stats, studio_stats, overall_avg


# ─── Training ─────────────────────────────────────────────────────────────────

def train(X, y):
    print("\n--- Training ---")
    y_bins = pd.cut(y, bins=[0, 3, 5, 7, 8, 10], labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_bins
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = HistGradientBoostingRegressor(
        max_iter=500,
        learning_rate=0.03,
        max_depth=5,
        l2_regularization=0.5,
        min_samples_leaf=12,
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("Training complete.")
    return model, X_train, X_test, y_train, y_test


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test):
    preds = np.clip(model.predict(X_test), 1, 10)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2   = r2_score(y_test, preds)

    lines = [
        "=== Noel's Movie-Meter Evaluation (Hold-out Test Set) ===",
        f"Test set size : {len(y_test)} movies",
        f"MAE           : {mae:.4f}  (target < 1.0)",
        f"RMSE          : {rmse:.4f}",
        f"R²            : {r2:.4f}",
        "",
        "Sample predictions vs actuals:",
    ]
    sample = pd.DataFrame({
        "Actual":    y_test.values[:10],
        "Predicted": preds[:10].round(2),
    })
    lines.append(sample.to_string(index=False))
    report = "\n".join(lines)
    print("\n" + report)

    with open(NOEL_EVAL_PATH, "w") as f:
        f.write(report)
    print(f"\nEvaluation saved → {NOEL_EVAL_PATH}")
    return preds


# ─── SHAP ─────────────────────────────────────────────────────────────────────

def explain(model, X_test):
    print("\n--- SHAP Explainability ---")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    np.save(NOEL_SHAP_VALUES_PATH, shap_values)

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=X_test.columns)
    print("\nTop 15 most impactful features (mean |SHAP|):")
    print(feature_importance.nlargest(15).to_string())

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, max_display=20, show=False, plot_size=(10, 8))
    plt.tight_layout()
    plt.savefig(NOEL_SHAP_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"SHAP summary plot → {NOEL_SHAP_PLOT_PATH}")
    return explainer, shap_values


# ─── Save artifacts ───────────────────────────────────────────────────────────

def save_artifacts(model, scaler, mlb, X, df, raw_embeddings,
                   dir_stats, act_stats, studio_stats, overall_avg):
    os.makedirs(NOEL_MODELS_DIR, exist_ok=True)
    os.makedirs(NOEL_OUTPUTS_DIR, exist_ok=True)

    joblib.dump(model,           NOEL_MODEL_PATH)
    joblib.dump(scaler,          NOEL_SCALER_PATH)
    joblib.dump(mlb,             NOEL_MLB_PATH)
    joblib.dump(list(X.columns), NOEL_FEATURE_NAMES_PATH)
    joblib.dump(dir_stats,       NOEL_DIRECTOR_STATS_PATH)
    joblib.dump(act_stats,       NOEL_ACTOR_STATS_PATH)
    joblib.dump(studio_stats,    NOEL_STUDIO_STATS_PATH)
    joblib.dump(overall_avg,     NOEL_OVERALL_AVG_PATH)

    meta_cols = (
        ["Const", "Title", "Your Rating", "Genre", "Director",
         "Actors", "Writer", "Awards", "Year", "Runtime", "imdbRating", "Plot"]
        + ALL_TAG_COLS
    )
    meta_cols = [c for c in meta_cols if c in df.columns]
    df[meta_cols].reset_index(drop=True).to_pickle(NOEL_TRAIN_META_PATH)
    np.save(NOEL_TRAIN_EMB_PATH, raw_embeddings)

    print(f"\nArtifacts saved:")
    for label, path in [
        ("Model",            NOEL_MODEL_PATH),
        ("Scaler",           NOEL_SCALER_PATH),
        ("Genre MLB",        NOEL_MLB_PATH),
        ("Feature list",     NOEL_FEATURE_NAMES_PATH),
        ("Train meta",       NOEL_TRAIN_META_PATH),
        ("Train embeddings", NOEL_TRAIN_EMB_PATH),
        ("Director stats",   NOEL_DIRECTOR_STATS_PATH),
        ("Actor stats",      NOEL_ACTOR_STATS_PATH),
        ("Studio stats",     NOEL_STUDIO_STATS_PATH),
    ]:
        print(f"  {label:<18} → {path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df = load_and_clean(NOEL_OMDB_CSV)
    X, y, scaler, mlb, raw_embeddings, dir_stats, act_stats, studio_stats, overall_avg = build_features(df)
    model, X_train, X_test, y_train, y_test = train(X, y)
    evaluate(model, X_test, y_test)
    explain(model, X_test)
    save_artifacts(model, scaler, mlb, X, df, raw_embeddings,
                   dir_stats, act_stats, studio_stats, overall_avg)
    print("\n=== Noel's Model Training Complete ===")
