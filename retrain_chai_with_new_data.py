"""
Fetch missing Chai OMDb data, retrain model, and deploy only if metrics improve.
Current baseline: MAE=0.6541, RMSE=0.8251, R²=0.3117 (998 movies)
"""
import os, shutil, time, re
import requests
import numpy as np
import pandas as pd
import joblib

from config import (
    OMDB_API_KEY, OMDB_API_KEY_2, OMDB_BASE_URL, OMDB_RATE_LIMIT_SLEEP,
    OMDB_CSV, RAW_CSV, MLB_PATH, SCALER_PATH, FEATURE_NAMES_PATH,
    SHAP_PLOT_PATH, EVAL_PATH, SHAP_VALUES_PATH,
    TRAIN_TEST_SPLIT, RANDOM_STATE,
    SENTENCE_TRANSFORMER_MODEL, MODELS_DIR, DATA_DIR, TAG_CSV,
)

MODEL_PATH = os.path.join(MODELS_DIR, "pmtpe_model.pkl")
TRAIN_META_PATH = os.path.join(DATA_DIR, "train_meta.pkl")
TRAIN_EMBEDDINGS_PATH = os.path.join(MODELS_DIR, "train_plot_embeddings.npy")

BASELINE_MAE = 0.6541

BACKUP_SUFFIX = ".backup_before_retrain"

# ─── OMDb fetch helpers (copied from 2_enrich.py) ────────────────────────────

def parse_runtime(val):
    try: return int(str(val).split()[0])
    except: return None

def parse_money(val):
    try: return float(str(val).replace("$", "").replace(",", ""))
    except: return None

def parse_votes(val):
    try: return int(str(val).replace(",", ""))
    except: return None

def parse_rt(ratings_list):
    try:
        for r in ratings_list:
            if r.get("Source") == "Rotten Tomatoes":
                return float(r["Value"].replace("%", ""))
    except: pass
    return None

def fetch_omdb(imdb_id, api_key):
    params = {"i": imdb_id, "plot": "full", "apikey": api_key}
    try:
        resp = requests.get(OMDB_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data.get("Response") == "False":
            return None
        return data
    except Exception as e:
        print(f"  ERROR fetching {imdb_id}: {e}")
        return None

def parse_record(raw, user_rating, title, imdb_id):
    return {
        "Const": imdb_id, "Your Rating": user_rating, "Title": title,
        "Year": pd.to_numeric(raw.get("Year", "").replace("–", "").strip(), errors="coerce"),
        "Rated": raw.get("Rated", "N/A"),
        "Runtime": parse_runtime(raw.get("Runtime")),
        "Genre": raw.get("Genre", ""),
        "Director": raw.get("Director", ""),
        "Actors": raw.get("Actors", ""),
        "Plot": raw.get("Plot", ""),
        "Language": raw.get("Language", ""),
        "Country": raw.get("Country", ""),
        "Awards": raw.get("Awards", ""),
        "imdbRating": pd.to_numeric(raw.get("imdbRating"), errors="coerce"),
        "imdbVotes": parse_votes(raw.get("imdbVotes")),
        "Metascore": pd.to_numeric(raw.get("Metascore"), errors="coerce"),
        "BoxOffice": parse_money(raw.get("BoxOffice")),
        "RT_score": parse_rt(raw.get("Ratings", [])),
        "Type": raw.get("Type", ""),
        "Writer": raw.get("Writer", ""),
    }

# ─── Step 1: Fetch missing movies ─────────────────────────────────────────────

def fetch_missing():
    raw_df = pd.read_csv(RAW_CSV)
    enriched_df = pd.read_csv(OMDB_CSV)
    done_ids = set(enriched_df["Const"].tolist())
    pending = raw_df[~raw_df["Const"].isin(done_ids)].reset_index(drop=True)

    print(f"\n=== STEP 1: Fetch Missing OMDb Data ===")
    print(f"Raw: {len(raw_df)}, Already enriched: {len(enriched_df)}, Missing: {len(pending)}")

    if len(pending) == 0:
        print("Nothing to fetch — already fully enriched.")
        return enriched_df

    new_records = []
    failed = []
    api_keys = [OMDB_API_KEY, OMDB_API_KEY_2]

    for i, row in pending.iterrows():
        imdb_id = row["Const"]
        raw = None
        for key in api_keys:
            raw = fetch_omdb(imdb_id, key)
            if raw:
                break
        if raw:
            record = parse_record(raw, row["Your Rating"], row["Title"], imdb_id)
            new_records.append(record)
        else:
            failed.append(imdb_id)

        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(pending)} fetched ({len(new_records)} success, {len(failed)} failed)...")
            # Incremental save
            if new_records:
                new_df = pd.DataFrame(new_records)
                combined = pd.concat([enriched_df, new_df], ignore_index=True).drop_duplicates(subset=["Const"])
                combined.to_csv(OMDB_CSV, index=False)

        time.sleep(OMDB_RATE_LIMIT_SLEEP)

    # Final save
    if new_records:
        new_df = pd.DataFrame(new_records)
        combined = pd.concat([enriched_df, new_df], ignore_index=True).drop_duplicates(subset=["Const"])
        combined.to_csv(OMDB_CSV, index=False)
        print(f"\nSaved {len(combined)} total records to {OMDB_CSV}")
        if failed:
            print(f"Still unfetchable ({len(failed)} movies): {failed[:10]}{'...' if len(failed) > 10 else ''}")
        return combined
    else:
        print("No new records fetched.")
        return enriched_df

# ─── Step 2: Retrain ──────────────────────────────────────────────────────────

def retrain():
    # Import training functions from 3_train.py
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from three_train import load_and_clean, build_features, train, evaluate_metrics, explain, save_artifacts

    print(f"\n=== STEP 2: Retrain Model ===")
    df = load_and_clean(OMDB_CSV)
    X, y, scaler, mlb, raw_embeddings = build_features(df)
    model, X_train, X_test, y_train, y_test = train(X, y)
    mae, rmse, r2, preds = evaluate_metrics(model, X_test, y_test)
    return model, scaler, mlb, X, df, raw_embeddings, X_test, y_test, mae, rmse, r2

# ─── Step 3: Compare & deploy ─────────────────────────────────────────────────

def backup_artifacts():
    paths = [MODEL_PATH, SCALER_PATH, MLB_PATH, FEATURE_NAMES_PATH,
             TRAIN_META_PATH, TRAIN_EMBEDDINGS_PATH, EVAL_PATH]
    for p in paths:
        if os.path.exists(p):
            shutil.copy2(p, p + BACKUP_SUFFIX)
    print("Backed up existing artifacts.")

def restore_artifacts():
    paths = [MODEL_PATH, SCALER_PATH, MLB_PATH, FEATURE_NAMES_PATH,
             TRAIN_META_PATH, TRAIN_EMBEDDINGS_PATH, EVAL_PATH]
    for p in paths:
        backup = p + BACKUP_SUFFIX
        if os.path.exists(backup):
            shutil.move(backup, p)
    print("Restored original artifacts.")

def cleanup_backups():
    paths = [MODEL_PATH, SCALER_PATH, MLB_PATH, FEATURE_NAMES_PATH,
             TRAIN_META_PATH, TRAIN_EMBEDDINGS_PATH, EVAL_PATH]
    for p in paths:
        backup = p + BACKUP_SUFFIX
        if os.path.exists(backup):
            os.remove(backup)

# ─── Inline training logic (avoiding import issues) ───────────────────────────

def run_training():
    """Run full training inline, return metrics without saving."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap as shap_lib
    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from tag_features import ALL_TAG_COLS

    NUMERIC_COLS = [
        "Year", "imdbRating", "Metascore", "BoxOffice", "imdbVotes", "Runtime", "RT_score",
        "award_wins", "award_noms", "oscar_win", "oscar_nom",
    ]

    def parse_awards(awards_str):
        s = str(awards_str) if pd.notna(awards_str) else ""
        oscar_win = 1 if re.search(r'\bwon\b.*\boscar', s, re.IGNORECASE) else 0
        oscar_nom = 1 if re.search(r'\bnominat', s, re.IGNORECASE) and re.search(r'\boscar', s, re.IGNORECASE) else 0
        wins = sum(int(x) for x in re.findall(r'(\d+)\s+win', s, re.IGNORECASE))
        noms = sum(int(x) for x in re.findall(r'(\d+)\s+nomination', s, re.IGNORECASE))
        return {"award_wins": float(wins), "award_noms": float(noms),
                "oscar_win": float(oscar_win), "oscar_nom": float(oscar_nom)}

    # Load & clean
    df = pd.read_csv(OMDB_CSV)
    print(f"Loaded {len(df)} rows")
    df = df.dropna(subset=["Your Rating", "Plot"])
    df = df[df["Plot"].str.strip() != "N/A"]
    print(f"After cleaning: {len(df)} rows")

    base_numeric = ["Year", "imdbRating", "Metascore", "BoxOffice", "imdbVotes", "Runtime", "RT_score"]
    for col in base_numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(pd.to_numeric(df[col], errors="coerce").median())
    for col in ["Genre", "Rated", "Director", "Actors", "Writer", "Language", "Country"]:
        df[col] = df[col].fillna("Unknown")

    award_rows = df["Awards"].apply(parse_awards)
    award_df = pd.DataFrame(list(award_rows))
    df = pd.concat([df.reset_index(drop=True), award_df.reset_index(drop=True)], axis=1)
    for col in ["award_wins", "award_noms", "oscar_win", "oscar_nom"]:
        df[col] = df[col].fillna(0.0)

    try:
        tags_df = pd.read_csv(TAG_CSV)
        df = df.merge(tags_df[["Const"] + ALL_TAG_COLS], on="Const", how="left")
        df[ALL_TAG_COLS] = df[ALL_TAG_COLS].fillna(0).astype(int)
        tagged = (df[ALL_TAG_COLS].sum(axis=1) > 0).sum()
        print(f"Tags merged: {tagged}/{len(df)} movies have tags")
    except FileNotFoundError:
        for col in ALL_TAG_COLS:
            df[col] = 0

    df = df.reset_index(drop=True)

    # Features
    scaler = StandardScaler()
    num_arr = scaler.fit_transform(df[NUMERIC_COLS])
    num_df = pd.DataFrame(num_arr, columns=NUMERIC_COLS, index=df.index)

    df["Genre_list"] = df["Genre"].apply(
        lambda x: [g.strip() for g in x.split(",")] if isinstance(x, str) and x != "Unknown" else []
    )
    mlb = MultiLabelBinarizer()
    genre_arr = mlb.fit_transform(df["Genre_list"])
    genre_df = pd.DataFrame(genre_arr, columns=[f"genre_{c}" for c in mlb.classes_], index=df.index)
    print(f"Genres: {len(mlb.classes_)} unique")

    tag_df = df[ALL_TAG_COLS].reset_index(drop=True)

    nlp_model = SentenceTransformer(SENTENCE_TRANSFORMER_MODEL)
    texts = df.apply(lambda row: " ".join(filter(None, [
        f"Director: {str(row.get('Director','') or '').strip()}." if str(row.get('Director','') or '').strip() not in ('', 'N/A', 'Unknown') else '',
        f"Cast: {str(row.get('Actors','') or '').strip()}." if str(row.get('Actors','') or '').strip() not in ('', 'N/A', 'Unknown') else '',
        f"Writer: {str(row.get('Writer','') or '').strip()}." if str(row.get('Writer','') or '').strip() not in ('', 'N/A', 'Unknown') else '',
        str(row.get('Plot','') or '').strip() if str(row.get('Plot','') or '').strip() not in ('', 'N/A') else row.get('Title',''),
    ])), axis=1).tolist()
    print(f"Encoding {len(texts)} texts...")
    embeddings = nlp_model.encode(texts, show_progress_bar=True, batch_size=64)
    plot_df = pd.DataFrame(embeddings, columns=[f"plot_{i}" for i in range(embeddings.shape[1])], index=df.index)

    X = pd.concat([num_df, genre_df, tag_df, plot_df], axis=1)
    y = df["Your Rating"].astype(float)

    # Train
    y_bins = pd.cut(y, bins=[0, 3, 5, 7, 8, 10], labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_SPLIT, random_state=RANDOM_STATE, stratify=y_bins
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    model = HistGradientBoostingRegressor(
        max_iter=500, learning_rate=0.03, max_depth=5,
        l2_regularization=0.5, min_samples_leaf=12, random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("Training complete.")

    preds = np.clip(model.predict(X_test), 1, 10)
    mae  = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2   = r2_score(y_test, preds)

    print(f"\nNew metrics → MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
    print(f"Baseline    → MAE: {BASELINE_MAE:.4f}")

    return model, scaler, mlb, X, df, embeddings, X_test, y_test, mae, rmse, r2, preds

def save_all_artifacts(model, scaler, mlb, X, df, raw_embeddings, X_test, y_test, mae, rmse, r2, preds):
    from tag_features import ALL_TAG_COLS

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(mlb, MLB_PATH)
    joblib.dump(list(X.columns), FEATURE_NAMES_PATH)

    meta_cols = ["Const", "Title", "Your Rating", "Genre", "Director",
                 "Actors", "Writer", "Awards", "Year", "Runtime", "imdbRating", "Plot"] + ALL_TAG_COLS
    meta_cols = [c for c in meta_cols if c in df.columns]
    df[meta_cols].reset_index(drop=True).to_pickle(TRAIN_META_PATH)
    np.save(TRAIN_EMBEDDINGS_PATH, raw_embeddings)

    # Write evaluation
    lines = [
        "=== PMTPE Evaluation (Hold-out Test Set) ===",
        f"Test set size : {len(y_test)} movies",
        f"MAE           : {mae:.4f}  (target < 1.0)",
        f"RMSE          : {rmse:.4f}",
        f"R²            : {r2:.4f}",
        "",
        "Sample predictions vs actuals:",
    ]
    import pandas as pd
    sample = pd.DataFrame({
        "Actual":    y_test.values[:10],
        "Predicted": preds[:10].round(2),
        "Match %":   (preds[:10] / 10 * 100).round(1),
    })
    lines.append(sample.to_string(index=False))
    with open(EVAL_PATH, "w") as f:
        f.write("\n".join(lines))

    print(f"\nAll artifacts saved to {MODELS_DIR}")

# ─── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Step 1: Fetch missing
    fetch_missing()

    # Step 2: Retrain
    result = run_training()
    model, scaler, mlb, X, df, raw_embeddings, X_test, y_test, mae, rmse, r2, preds = result

    # Step 3: Compare & deploy
    print(f"\n=== STEP 3: Deploy Decision ===")
    print(f"Baseline MAE : {BASELINE_MAE:.4f}")
    print(f"New MAE      : {mae:.4f}")

    if mae < BASELINE_MAE:
        improvement = BASELINE_MAE - mae
        print(f"✅ Improved by {improvement:.4f} — deploying new model.")
        backup_artifacts()
        save_all_artifacts(model, scaler, mlb, X, df, raw_embeddings, X_test, y_test, mae, rmse, r2, preds)
        cleanup_backups()
        print("\n=== New Chai model deployed. ===")
    else:
        degradation = mae - BASELINE_MAE
        print(f"❌ No improvement (degraded by {degradation:.4f}) — keeping original model.")
        print("Original model artifacts unchanged.")
