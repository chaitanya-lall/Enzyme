# Enzyme — Claude Instructions

This file governs how Claude should work in this project. Read it at the start of every session.

---

## What this project is

**Enzyme** is a personalized movie and show recommender. It predicts how *you* would rate a film — not based on critics, but on your own watch history. It supports two users side by side: Chai (💖) and Noel (👔), each with their own trained ML model.

**Live app:** `/Users/chaitanya.lall/Desktop/Enzyme/`

---

## CRITICAL: The dead project — never touch it

There is an old, abandoned version of this app at:
```
/Users/chaitanya.lall/Desktop/ai_projects/Movie_Recommender/
```
**Never start it. Never modify it. Never reference it.** It is branded "The Personalized Critic" and is not Enzyme. If asked to open or run the app, always use the Enzyme directory above.

---

## How to run the app

**Always kill any existing process first** (stale processes load old code):
```bash
lsof -ti:8501 | xargs kill -9 2>/dev/null
cd /Users/chaitanya.lall/Desktop/Enzyme
python3 -m streamlit run app.py --server.port 8501 --server.headless true
```

The app runs on **port 8501**.

---

## File map

| File | What it does |
|---|---|
| `app.py` | Streamlit UI — nav bar, Search tab, Recommend tab, all rendering |
| `predict.py` | Chai's inference: OMDb fetch → feature engineering → XGBoost → SHAP → similarity |
| `predict_noel.py` | Noel's inference (same structure, separate model artifacts) |
| `config.py` | API keys (via env/Streamlit secrets), file paths, hyperparameters, TAG_TAXONOMY |
| `tag_features.py` | Groq-powered LLM tagger — assigns films across 7 tag dimensions |
| `parents_guide.py` | Scrapes IMDb Parents Guide (sex/nudity, violence, profanity, etc.) |
| `rt_enrichment.py` | Fetches Rotten Tomatoes audience scores |
| `catalog_sync.py` | Background thread that syncs the streaming catalog |
| `catalog_seed.py` | One-time script to build `data/catalog_data.parquet` via Watchmode API |

**Data pipeline scripts (run manually to retrain):**

| Script | Purpose |
|---|---|
| `1_extract.py` | Export IMDb ratings from Numbers → raw CSV |
| `2_enrich.py` | Fetch OMDb metadata for Chai's films |
| `3_train.py` | Train Chai's XGBoost model |
| `noel_1_extract.py` | Same for Noel |
| `noel_2_enrich.py` | Same for Noel |
| `noel_3_train.py` | Same for Noel |

**One-off/patch scripts at root** (`retrain_and_deploy.py`, `retrain_chai_with_new_data.py`, `patch_add_writer.py`) — check with user before running.

---

## Model artifacts

**Chai's models:** `models/` inside `/Desktop/Enzyme/`
- `pmtpe_model.pkl`, `scaler.pkl`, `mlb_genres.pkl`, `feature_names.pkl`
- `train_plot_embeddings.npy`, `director_stats.pkl`, `actor_stats.pkl`, `studio_stats.pkl`

**Noel's models:** `models/noel/` inside `/Desktop/Enzyme/`
- `noel_model.pkl`, `scaler.pkl`, `mlb_genres.pkl`, `feature_names.pkl`
- `train_plot_embeddings.npy`, `director_stats.pkl`, `actor_stats.pkl`, `studio_stats.pkl`

---

## App structure (tabs and key features)

The app has two top-level tabs via a custom nav bar (not Streamlit's native tabs — state lives in `st.session_state["active_tab"]`):

1. **Recommend tab (`🍿 Recommend`)** — `render_recommend_tab()` in `app.py`
   - Loads `data/catalog_data.parquet` (built by `catalog_seed.py`)
   - Filter bar: 7 streaming services (Netflix, Max, Prime, Hulu, Apple TV+, Peacock, Paramount+), watch status, genre, Chai score, Noel score, sort
   - Mobile: full-screen overlay filter drawer (`_mobile_filters_panel()`)
   - Cards rendered by `_render_catalog_card()`

2. **Search tab (`🔍 Search`)** — `_render_search_tab()` in `app.py`
   - `streamlit-searchbox` autocomplete → OMDb lookup by title
   - After selection: Chai score + Noel score rendered side by side via `render_meter_column()`
   - Shows: SVG donut gauge, score verdict, SHAP key drivers (positive/negative tags), AI narrative (streamed via Groq), Film Profile tags, closest match from watch history, Parents Guide

---

## Regression testing — MANDATORY

**After every code change**, before declaring done, verify these features still work:

### Search tab
- [ ] Searchbox autocomplete returns results
- [ ] Selecting a film shows both Chai 💖 and Noel 👔 scores side by side
- [ ] Donut gauge renders (not blank)
- [ ] SHAP key drivers (green positive tags, red negative tags) render
- [ ] AI narrative streams
- [ ] Closest match box appears
- [ ] Film Profile tags appear

### Recommend tab
- [ ] Catalog grid loads (cards appear)
- [ ] Filter bar pills are visible and clickable
- [ ] Filtering by service (e.g., Netflix) updates the grid
- [ ] Chai score / Noel score filter sliders work
- [ ] "Load More" loads additional cards

### Nav bar
- [ ] "🍿 Recommend" and "🔍 Search" buttons both render
- [ ] Active button is highlighted blue; inactive is grey
- [ ] Switching tabs works and doesn't lose scroll state

### Mobile (resize to ≤768px)
- [ ] Cards display in 2-column grid
- [ ] Filters button opens full-screen overlay
- [ ] Apply button in overlay closes it and applies filters

**For CSS changes specifically:** Check all three of the above nav/recommend/search sections. CSS cascade errors in `app.py` have previously broken unrelated components. The nth-child selectors in the global stylesheet affect nearly every layout element.

---

## Known fragile areas (history of repeated bugs)

- **Nav bar highlighting** — driven by dynamic CSS injected in `main()` using f-strings with `active_tab` state. Changes to column counts or layout break it silently.
- **Card padding/clipping** — the global `.stHorizontalBlock > .stColumn:nth-child(2/3)` rule applies padding everywhere. Any new horizontal layout gets padding it doesn't want. Always add a counter-rule scoped to your new component's key (e.g., `.st-key-my-thing`).
- **Mobile filter state** — filter state is duplicated: desktop keys (`f_svc_netflix`) and mobile keys (`f_mob_svc_netflix`) are OR'd together. If you add a new filter, add it in both places.
- **streamlit-searchbox** — pinned to `==0.1.23` due to protobuf conflict. Do not upgrade without testing.
- **Groq narrative** — uses streaming. If you change the prompt, always verify the stream actually renders (not just that the function returns).
- **Catalog sync** — runs in a background thread. Don't add blocking calls to `catalog_sync.py` — it will hang the UI.

---

## CSS architecture

All styles are a single `st.markdown("""<style>...</style>""")` block at the top of `app.py` (~line 39–484), plus a small dynamic block injected in `main()` for nav button colors.

**Key CSS classes/selectors to know:**
- `.tag`, `.tag-pos`, `.tag-neg` — SHAP driver pills
- `.narrative-box` — AI explanation text
- `.section-head` — section labels
- `.anchor-box` — closest match display
- `.stat-chip` — metadata chips (year, runtime, etc.)
- `.catalog-card` — catalog grid cards
- `.filter-sep` — pipe separators in filter bar
- `st-key-filter-bar` — scopes filter bar overrides to avoid polluting global layout

---

## API keys

All keys are injected via environment variables or Streamlit secrets (`.streamlit/secrets.toml` — gitignored). Never hardcode keys. Use `config._secret("KEY_NAME")`.

**Keys in use:**
- `OMDB_API_KEY`, `OMDB_API_KEY_2`, `OMDB_NOEL_KEY`, `OMDB_APP_KEY`, `OMDB_API_KEY_5`, `OMDB_API_KEY_6`
  - `OMDB_API_KEY_6` is reserved exclusively for the live website search. Do not use it in pipeline/seed scripts.
- `WATCHMODE_API_KEY` — streaming catalog (Watchmode free tier)
- `GROQ_API_KEY` — LLM narrative generation
- `GEMINI_API_KEY` — LLM movie tagging

---

## Local data files (not in git)

These live on the local machine and are gitignored:
- `/Users/chaitanya.lall/Documents/Chai IMDb rankings.numbers` — Chai's ratings source
- `/Users/chaitanya.lall/Documents/Chai Seen.numbers` — Chai's seen list
- `/Users/chaitanya.lall/Documents/Noel's Ratings.numbers` — Noel's ratings source
- `/Users/chaitanya.lall/Documents/Noel Seen.numbers` — Noel's seen list
- `data/enriched_raw.csv`, `data/enriched_omdb.csv` — training data (gitignored)
- `.streamlit/secrets.toml` — all API keys (gitignored)

---

## Retraining workflow

When Chai adds new ratings and wants to retrain:
```bash
python3 1_extract.py          # Numbers → data/enriched_raw.csv
python3 2_enrich.py           # OMDb enrich → data/enriched_omdb.csv
python3 3_train.py            # Train → models/pmtpe_model.pkl + artifacts
```
For Noel: same with `noel_` prefix scripts, outputs go to `data/noel/` and `models/noel/`.

After retraining, restart the app (kill + rerun) to load new model artifacts.

---

## What NOT to do

- Do not start the app without killing port 8501 first
- Do not run scripts from `/Desktop/ai_projects/Movie_Recommender/`
- Do not commit `config.py` with real API keys (use `config.example.py` as the template)
- Do not upgrade `streamlit-searchbox` without testing the search autocomplete
- Do not make CSS changes without regression testing the nav bar, catalog cards, and search results
- Do not add blocking I/O to `catalog_sync.py`
- Do not use `OMDB_API_KEY_6` in any pipeline script — it is reserved for the website
