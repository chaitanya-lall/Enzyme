# ENZYME — Movies, broken down for you.

Enzyme is a personalized movie recommender that predicts how *you* would rate any film — not based on critic scores or crowd averages, but on your own watch history and taste.

Search a movie title. Get a match percentage, a plain-English explanation of why it fits (or doesn't), and the closest film you've already rated.

---

## What it does

- **Predicts your personal rating** for any movie using an XGBoost model trained on your IMDb ratings
- **Explains the prediction** with an AI-generated narrative (powered by Groq / LLaMA) and SHAP feature attribution
- **Tags each film** across 7 dimensions: Social Context, Pacing, Cognitive Load, Social Value, Vibes, Narrative Resolve, and Tension Profile
- **Shows a closest match** from your watch history — the most similar film you've already seen and rated
- **Supports two users** side by side (Chai and Noel), each with their own trained model
- **Parents Guide** — displays content ratings (sex/nudity, violence, profanity, etc.) sourced from IMDb

---

## How it works

### Data pipeline

| Script | What it does |
|---|---|
| `1_extract.py` | Exports your IMDb ratings from a Numbers spreadsheet into a raw CSV |
| `2_enrich.py` | Fetches full movie metadata (plot, cast, runtime, Rotten Tomatoes score, etc.) from OMDb API |
| `3_train.py` | Trains an XGBoost regression model on your enriched ratings |

### At prediction time (`predict.py`)

1. Looks up the searched movie on OMDb
2. Generates a plot embedding using `all-MiniLM-L6-v2` (sentence transformers)
3. Tags the film across 7 dimensions using Gemini
4. Runs the XGBoost model to produce a predicted score
5. Converts the score to a 0–100% match using a fitted sigmoid
6. Computes SHAP values to identify the top positive and negative drivers
7. Finds the most similar film in your watch history (plot 60% + tags 30% + IMDb 10%)
8. Streams a personalised narrative via Groq (LLaMA 3.1)

### Tech stack

- **Streamlit** — UI
- **XGBoost** — personal rating prediction
- **SHAP** — feature explainability
- **Sentence Transformers** (`all-MiniLM-L6-v2`) — plot embeddings
- **OMDb API** — movie metadata
- **Groq / LLaMA 3.1** — AI narrative generation
- **Gemini** — movie tagging

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/chaitanya-lall/Enzyme.git
cd Enzyme
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

```bash
cp config.example.py config.py
```

Then open `config.py` and fill in:
- `OMDB_API_KEY` — free at [omdbapi.com](https://www.omdbapi.com/apikey.aspx)
- `GROQ_API_KEY` — free at [console.groq.com](https://console.groq.com)
- `GEMINI_API_KEY` — free at [aistudio.google.com](https://aistudio.google.com)
- `NUMBERS_FILE` — path to your IMDb ratings Numbers file

### 4. Run the data pipeline

```bash
python 1_extract.py
python 2_enrich.py
python 3_train.py
```

### 5. Launch the app

```bash
streamlit run app.py
```

---

## Project structure

```
Enzyme/
├── app.py                  # Streamlit UI
├── predict.py              # Prediction logic (Chai)
├── predict_noel.py         # Prediction logic (Noel)
├── 1_extract.py            # Step 1: extract ratings
├── 2_enrich.py             # Step 2: enrich with OMDb
├── 3_train.py              # Step 3: train model
├── config.example.py       # Config template (copy to config.py)
├── tag_movies.py           # Tag films using Gemini
├── tag_features.py         # Tag feature engineering
├── retrain_and_deploy.py   # Retrain and update model
├── requirements.txt
└── assets/
    └── logo.png
```

Data, models, and outputs are excluded from version control (too large / user-specific). Run the pipeline to generate them locally.
