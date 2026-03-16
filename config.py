import os

# OMDb API — reads from environment (Streamlit Cloud secrets) with local fallback
OMDB_API_KEY      = os.environ.get("OMDB_API_KEY",   "")
OMDB_API_KEY_2    = os.environ.get("OMDB_API_KEY_2", "")
OMDB_NOEL_KEY     = os.environ.get("OMDB_NOEL_KEY",  "")
OMDB_APP_KEY      = os.environ.get("OMDB_APP_KEY",   "")

# AI narrative keys
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY      = os.environ.get("GROQ_API_KEY",   "")
OMDB_BASE_URL = "https://www.omdbapi.com/"
OMDB_RATE_LIMIT_SLEEP = 0.25  # seconds between requests (free tier safe)
OMDB_DAILY_LIMIT = 1000        # free tier cap

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

NUMBERS_FILE = "/Users/chaitanya.lall/Documents/Chai IMDb rankings.numbers"
RAW_CSV = os.path.join(DATA_DIR, "enriched_raw.csv")
OMDB_CSV = os.path.join(DATA_DIR, "enriched_omdb.csv")

# Noel's paths
NOEL_NUMBERS_FILE = "/Users/chaitanya.lall/Documents/Noel's Ratings.numbers"
NOEL_DATA_DIR    = os.path.join(BASE_DIR, "data",    "noel")
NOEL_MODELS_DIR  = os.path.join(BASE_DIR, "models",  "noel")
NOEL_OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs", "noel")

NOEL_RAW_CSV          = os.path.join(NOEL_DATA_DIR,   "enriched_raw.csv")
NOEL_OMDB_CSV         = os.path.join(NOEL_DATA_DIR,   "enriched_omdb.csv")
NOEL_TRAIN_META_PATH  = os.path.join(NOEL_DATA_DIR,   "train_meta.pkl")

NOEL_MODEL_PATH        = os.path.join(NOEL_MODELS_DIR, "noel_model.pkl")
NOEL_MLB_PATH          = os.path.join(NOEL_MODELS_DIR, "mlb_genres.pkl")
NOEL_SCALER_PATH       = os.path.join(NOEL_MODELS_DIR, "scaler.pkl")
NOEL_FEATURE_NAMES_PATH= os.path.join(NOEL_MODELS_DIR, "feature_names.pkl")
NOEL_TRAIN_EMB_PATH    = os.path.join(NOEL_MODELS_DIR, "train_plot_embeddings.npy")

NOEL_SHAP_PLOT_PATH    = os.path.join(NOEL_OUTPUTS_DIR, "shap_summary.png")
NOEL_EVAL_PATH         = os.path.join(NOEL_OUTPUTS_DIR, "evaluation.txt")
NOEL_SHAP_VALUES_PATH  = os.path.join(NOEL_OUTPUTS_DIR, "shap_values.npy")

# Model artifacts
MODEL_PATH = os.path.join(MODELS_DIR, "pmtpe_model.json")
MLB_PATH = os.path.join(MODELS_DIR, "mlb_genres.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_NAMES_PATH = os.path.join(MODELS_DIR, "feature_names.pkl")

# Outputs
SHAP_PLOT_PATH = os.path.join(OUTPUTS_DIR, "shap_summary.png")
EVAL_PATH = os.path.join(OUTPUTS_DIR, "evaluation.txt")
SHAP_VALUES_PATH = os.path.join(OUTPUTS_DIR, "shap_values.npy")

# Model hyperparameters
XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
TRAIN_TEST_SPLIT = 0.2
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 20

# NLP
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"

# LLM Categorical Tag Taxonomy (up to 2 tags per category per movie)
# Derived from movies_tagged_final.json — 1463 movies, 111 one-hot columns
TAG_TAXONOMY = {
    "social_context": [
        "Mainstream Blockbuster", "Award Season Contender", "Legacy Franchise",
        "Family Mainstay", "Regional Cinema", "Direct-to-Streaming",
        "Indie Darling", "Date Night Pick", "Niche/Cult Audience",
        "Cult Classic", "Documentary/Edu-tainment", "Ensemble/Group Watch",
        "Teen/YA Audience", "Holiday/Seasonal", "Documentary Reality",
        "Film Festival Niche", "Girls' Night Pick", "B-Movie/Grindhouse",
        "New Wave/Avant-Garde",
    ],
    "pacing": [
        "Standard Narrative", "Fast-Paced/Punchy", "Slow Burn/Atmospheric",
        "Light/Breezy", "Breakneck/Relentless", "Steady/Procedural",
        "Rising/Crescendo", "Episodic/Non-Linear", "Tight/Economical",
        "Staccato/Erratic", "Meditative/Languid", "Hyper-Active/Frenetic",
        "Fragmented/Chaotic", "Graceful/Flowing", "Meandering/Picaresque",
    ],
    "cognitive_load": [
        "Light/Popcorn", "Moderate/Intellectual", "High/Demanding",
        "Intuitive/Emotional", "Socially Challenging", "Stress-Inducing/Tense",
        "Dense/Philosophical", "Puzzle-Box/Mystery", "Educational/Informative",
        "Complex/Mind-Bender", "Cryptic/Abstract", "Meta/Self-Referential",
        "Symphonic/Visual",
    ],
    "social_value": [
        "High Prestige", "Universal/Classic", "Mainstream Popular",
        "Nostalgia Trip", "Guilty Pleasure", "Water-Cooler Event",
        "Iconic/Legendary", "Niche Interest", "Feel-Good Hit",
        "Hidden Gem", "Cult Favourite", "Safe Crowd-Pleaser",
        "Divisive/Controversial", "Culturally Specific", "Political/Activist",
        "Legacy Franchise", "Trend-Setter", "Ephemeral/Dated",
    ],
    "vibes": [
        "Warm/Nostalgic", "Vibrant/Whimsical", "Grandiose/Epic",
        "Gritty/Raw", "Gritty/Neon", "Naturalistic/Earth-Toned",
        "Uplifting/Inspirational", "Dark/Gothic", "Cerebral/Minimalist",
        "Bleak/Desaturated", "Chaotic/Absurdist", "Intimate/Raw",
        "Surreal/Dreamlike", "Indie/Quirky", "Sleek/Stylish",
        "Nostalgic/Retro", "Pastel/Soft", "Psychedelic/Acid-Trip",
        "Abrasive/Industrial", "Technicolor/Vintage", "Symphonic/Visual",
    ],
    "narrative_resolve": [
        "Feel-Good/Triumphant", "Ambiguous/Open-Ended", "Hopeful/Redemptive",
        "Bittersweet/Poignant", "Cynical/Grim", "Tragic/Devastating",
        "Justice-Oriented", "Clean Resolution", "Ongoing/No Resolution",
        "Cathartic Release", "Absurdist/Satirical", "Cliffhanger",
    ],
    "tension_profile": [
        "Escalating Tension", "Low-Stakes/Gentle", "Simmering/Understated",
        "Sustained Tension", "Psychological/Creeping", "No Real Tension",
        "Explosive/Visceral", "Intermittent/Burst", "High-Stakes/Urgent",
        "Comic Tension", "Campy/Performative", "Constant/Anxiety-Inducing",
        "Intellectual/Competitive",
    ],
}
TAG_CSV           = os.path.join(DATA_DIR, "movie_tags.csv")
PARENTS_GUIDE_CSV = os.path.join(DATA_DIR, "parents_guide.csv")

# Parents Guide ordinal encoding (for ML features)
PG_ORDINAL = {"None": 0, "Mild": 1, "Moderate": 2, "Severe": 3}
PG_COLS    = ["pg_sex_nudity", "pg_violence_gore", "pg_profanity", "pg_alcohol_drugs", "pg_intensity"]
