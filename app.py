"""
Enzyme — Streamlit UI
Run with:  streamlit run app.py
"""
from __future__ import annotations

import base64
import os
from groq import Groq
import streamlit as st
import requests
from PIL import Image
from io import BytesIO

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import streamlit.components.v1 as _components
from streamlit_searchbox import st_searchbox
from predict import predict_movie, _load_all, find_similar_movie_combined, search_omdb
from predict_noel import predict_movie_noel, _load_all as _load_all_noel
from config import TAG_TAXONOMY, PARENTS_GUIDE_CSV, CATALOG_PATH, CHAI_SEEN_FILE, NOEL_SEEN_FILE
from tag_features import tag_col_name
from catalog_sync import start_background_sync, get_sync_status, catalog_age_days

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Personalized Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #0a0b0f; color: #f0f0f0; }
header[data-testid="stHeader"] { display: none !important; }
#MainMenu { display: none !important; }
footer { display: none !important; }
.block-container { 
  padding-top: 1rem !important; 
  padding-left: 2rem !important; 
  padding-right: 2rem !important; 
  max-width: 1400px !important;
}

/* White search bar */
div[data-testid="stTextInput"] > div > div > input {
  background-color: #ffffff !important;
  color: #1a1a2a !important;
  border-radius: 10px !important;
  caret-color: #1a1a2a !important;
  font-size: 1rem !important;
}
div[data-testid="stTextInput"] > div > div {
  background-color: #ffffff !important;
  border-radius: 10px !important;
  border-color: transparent !important;
}
div[data-testid="stTextInput"] label { display: none !important; }

/* Rounded corners on the searchbox iframe itself */
iframe[title="streamlit_searchbox.searchbox"] {
  border-radius: 10px !important;
  overflow: hidden !important;
}

/* Tighten vertical spacing around the search bar */
.stElementContainer:has(iframe[title="streamlit_searchbox.searchbox"]) {
  margin-top: -16px !important;
  margin-bottom: 0 !important;
}

/* Row container — gap between the three cards */
div[data-testid="stHorizontalBlock"] {
  gap: 0.75rem !important;
  align-items: flex-start !important;
  background: transparent !important;
}

/* Match columns (2nd and 3rd) — padding inside the card */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
  padding: 1.4rem !important;
}
/* Override for catalog grid — strip padding from ALL positions */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2):has(.catalog-card),
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3):has(.catalog-card) {
  padding: 0 !important;
}

/* Movie column (1st) — overflow visible so Film Profile tags aren't clipped */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) {
  overflow: visible !important;
  padding: 0 !important;
}

/* Expander — no box, no border, fully transparent */
[data-testid="stExpander"] {
  background: transparent !important;
  border: none !important;
}
[data-testid="stExpander"] > details {
  background: transparent !important;
  border: none !important;
  border-radius: 0 !important;
}
[data-testid="stExpander"] > details > summary {
  background: transparent !important;
  color: #e2e8f0 !important;
  font-size: 0.86rem !important;
  font-weight: 600 !important;
  font-family: inherit !important;
  letter-spacing: normal !important;
  text-transform: none !important;
  border-radius: 0 !important;
  padding-left: 0 !important;
}
[data-testid="stExpander"] > details[open] > summary {
  background: transparent !important;
  color: #e2e8f0 !important;
  border-radius: 0 !important;
}
[data-testid="stExpander"] > details > summary:hover {
  background: transparent !important;
  color: #e2e8f0 !important;
}
[data-testid="stExpander"] > details > summary svg {
  fill: #6b7280 !important;
}
[data-testid="stExpanderDetails"] {
  background: transparent !important;
  border-radius: 0 !important;
  padding-left: 0 !important;
}

/* Tag pills */
.tag {
  display: inline-block; padding: 0.25rem 0.65rem; border-radius: 6px;
  font-size: 0.78rem; font-weight: 600; margin: 0.2rem 0.15rem;
}
.tag-pos { background: rgba(80,200,120,0.15); color: #50c878; border: 1px solid #50c87840; }
.tag-neg { background: rgba(255,80,80,0.12); color: #ff6464; border: 1px solid #ff646440; }

/* Narrative box */
.narrative-box {
  font-size: 0.86rem; line-height: 1.7; color: #9ca3af;
  margin: 0.5rem 0 0.9rem 0;
}

/* Section headers */
.section-head {
  font-size: 0.86rem; font-weight: 600; color: #e2e8f0;
  margin: 1.1rem 0 0.35rem 0;
}

/* Anchor/closest match box */
.anchor-box {
  background: #1a1f2e; border-radius: 8px; padding: 0.7rem 1rem;
  font-size: 0.88rem; color: #9ca3af; border: 1px solid #2d3748;
}
.anchor-box span { color: #e2e8f0; font-weight: 600; }

/* Stat chips */
.stat-chip {
  display: inline-block; background: #1e2536; border-radius: 6px;
  padding: 0.2rem 0.55rem; font-size: 0.78rem; color: #9ca3af;
  margin: 0.2rem 0.15rem; border: 1px solid #2d3748;
}

@media (max-width: 768px) {
  .block-container {
    padding-top: 0.5rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    padding-bottom: 0 !important;
  }
  /* Remove bottom white bar */
  .stApp, [data-testid="stAppViewBlockContainer"] {
    padding-bottom: 0 !important;
    margin-bottom: 0 !important;
  }
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
    padding: 0.7rem !important;
  }
  /* Equal padding around poster — strip default margins from the markdown container */
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) [data-testid="stMarkdownContainer"],
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(1) .stMarkdownContainer {
    margin: 0 !important;
    padding: 0 !important;
  }
}


/* ── Nav bar ─────────────────────────────────────── */
/* Nav columns — :has(.nav-btn) targets the nav horizontal block specifically */
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"]:nth-child(2),
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"]:nth-child(3) {
  background: transparent !important;
  border-radius: 0 !important;
  overflow: visible !important;
  padding: 0 0.2rem !important;
}
/* Nav buttons — pill shape */
div[data-testid="stHorizontalBlock"]:has(.nav-btn) button {
  border-radius: 20px !important;
  font-weight: 700 !important;
  letter-spacing: 0.08em !important;
  font-size: 0.82rem !important;
  text-transform: uppercase !important;
  padding: 0.45rem 1.1rem !important;
  border: none !important;
  height: 2.4rem !important;
  margin-top: 0.6rem !important;
}

/* ── Catalog cards ───────────────────────────────── */
/* Reset column padding/overflow for columns that contain a catalog card */
div[data-testid="stColumn"]:has(.catalog-card) {
  padding: 0 !important;
  overflow: hidden !important;
  border-radius: 13px !important;
}
.catalog-card {
  background: #13161f;
  border-radius: 13px;
  overflow: hidden;
}
.catalog-card-body {
  padding: 0.6rem 0.7rem 0.5rem 0.7rem;
}
.catalog-card-title {
  font-size: 0.82rem; font-weight: 700; color: #e2e8f0;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  margin-bottom: 0.1rem;
}
.catalog-card-year { font-size: 0.72rem; color: #6b7280; margin-bottom: 0.3rem; }
.catalog-card-scores { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-bottom: 0.1rem; }
/* Details button inside a card column */
div[data-testid="stColumn"]:has(.catalog-card) .stButton > button {
  width: 100%;
  background: #1a1f2e !important;
  color: #6b7280 !important;
  border: none !important;
  border-top: 1px solid #1f2937 !important;
  border-radius: 0 !important;
  font-size: 0.72rem !important;
  padding: 0.25rem 0.5rem !important;
  margin-top: 0 !important;
  letter-spacing: 0.04em !important;
}
div[data-testid="stColumn"]:has(.catalog-card) .stButton > button:hover {
  background: #252d42 !important;
  color: #e2e8f0 !important;
}
/* ── Filter bar ──────────────────────────────────── */
.st-key-filter-bar {
  background: transparent;
  padding: 0 0 1.2rem 0;
  margin-bottom: 0;
  border: none;
}
/* Vertically center ALL columns in the filter row */
.st-key-filter-bar div[data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 0.3rem !important;
}
.st-key-filter-bar [data-testid="stColumn"] {
  background: transparent !important;
  border-radius: 0 !important;
  overflow: visible !important;
  padding: 0 !important;
}
/* Strip all internal spacing so nothing pushes elements out of alignment */
.st-key-filter-bar [data-testid="stElementContainer"],
.st-key-filter-bar [data-testid="stVerticalBlock"] {
  gap: 0 !important;
  padding: 0 !important;
  margin: 0 !important;
}
/* Pill buttons — all popover triggers AND the Clear all button */
.st-key-filter-bar button {
  border-radius: 20px !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  height: 2.2rem !important;
  white-space: nowrap !important;
}
/* Popover trigger pills — identical fixed width, outlined style */
.st-key-filter-bar [data-testid="stPopover"] {
  width: 100% !important;
}
.st-key-filter-bar [data-testid="stPopover"] button {
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  width: 100% !important;
  padding: 0 0.9rem !important;
  text-align: center !important;
  justify-content: center !important;
}
/* Force black text on the <p> tag where Streamlit actually renders button labels */
.st-key-filter-bar [data-testid="stPopover"] button p,
.st-key-filter-bar [data-testid="stPopover"] button div,
.st-key-filter-bar [data-testid="stPopover"] button span {
  color: #000000 !important;
}
.st-key-filter-bar [data-testid="stPopover"] button:hover {
  border-color: #4f8ef7 !important;
}
/* Fix: nth-child(2/3) global padding rule has higher specificity — counter it for filter bar */
.st-key-filter-bar div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
.st-key-filter-bar div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
  padding-left: 0 !important;
  padding-right: 0 !important;
}
/* Clear all — borderless text */
.st-key-filter-bar [data-testid="stBaseButton-secondary"]:not([data-testid="stPopover"] button) {
  background: transparent !important;
  border: none !important;
  color: #6b7280 !important;
  font-weight: 400 !important;
  padding: 0 0.3rem !important;
}
.st-key-filter-bar [data-testid="stBaseButton-secondary"]:not([data-testid="stPopover"] button):hover {
  color: #111827 !important;
}
/* Sort selectbox — pill style */
.st-key-filter-bar [data-testid="stSelectbox"] > div > div {
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  border-radius: 20px !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  min-height: 2.2rem !important;
  color: #000000 !important;
  padding: 0 0.5rem 0 0.9rem !important;
}
/* Sliders inside popovers */
[data-testid="stPopoverBody"] [data-testid="stSlider"] {
  padding: 0.5rem 0.2rem !important;
  min-width: 220px;
}
/* Multiselect inside popovers */
[data-testid="stPopoverBody"] [data-testid="stMultiSelect"] > div > div {
  background: #131825 !important;
  border: 1px solid #252f42 !important;
  border-radius: 8px !important;
  font-size: 0.84rem !important;
  min-width: 200px;
}
[data-testid="stPopoverBody"] [data-testid="stMultiSelect"] [data-testid="stMultiSelectOption"] {
  font-size: 0.84rem !important;
}
/* Separator pipe */
.filter-sep {
  color: #374151;
  font-size: 1.2rem;
  line-height: 2.2rem;
  text-align: center;
  user-select: none;
}
/* Modal column backgrounds */
[data-testid="stModal"] div[data-testid="stColumn"] {
  background: transparent !important;
  border-radius: 0 !important;
  padding: 0 !important;
}

</style>
""", unsafe_allow_html=True)


# ─── Color helpers ────────────────────────────────────────────────────────────

def score_color(pct: float) -> str:
    if pct >= 90:   return "#FFD700"
    elif pct >= 70: return "#4f8ef7"
    elif pct >= 50: return "#708090"
    else:           return "#555555"

def score_verdict(pct: float) -> tuple[str, str]:
    if pct >= 90:   return "Must Watch",          "#1a3a1a"
    elif pct >= 70: return "Strong Contender",     "#1a2a4e"
    elif pct >= 50: return "Coin Toss",            "#1e1e2e"
    else:           return "Probably Not For You", "#2a1a1a"


# ─── SVG donut ring ───────────────────────────────────────────────────────────

def build_gauge(pct: float) -> str:
    """Return an HTML string with a full-circle SVG donut ring matching the design."""
    color = score_color(pct)
    # Circle geometry
    r = 54          # radius of ring centre-line
    cx = cy = 80    # centre of 160×160 viewBox
    stroke_w = 12   # ring thickness
    circumference = 2 * 3.14159265 * r
    filled = circumference * (pct / 100)
    gap    = circumference - filled
    # Format display: show one decimal if not a whole number
    display = f"{pct:.1f}%" if pct % 1 != 0 else f"{int(pct)}%"
    return f"""
<div style="display:flex; flex-direction:column; align-items:center;
            justify-content:center; padding:16px 0 8px 0;">
  <svg viewBox="0 0 160 160" width="190" height="190"
       style="overflow:visible;">
    <!-- Track ring -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
            fill="none" stroke="#1e2533" stroke-width="{stroke_w}"
            stroke-linecap="round"/>
    <!-- Filled arc — starts at 12 o'clock (rotate -90 deg) -->
    <circle cx="{cx}" cy="{cy}" r="{r}"
            fill="none" stroke="{color}" stroke-width="{stroke_w}"
            stroke-linecap="round"
            stroke-dasharray="{filled:.2f} {gap:.2f}"
            transform="rotate(-90 {cx} {cy})"/>
    <!-- Centre label -->
    <text x="{cx}" y="{cy + 7}" text-anchor="middle"
          font-size="26" font-weight="800"
          font-family="'Arial Black', Arial, sans-serif"
          fill="{color}">{display}</text>
  </svg>
</div>
"""


# ─── "Why" narrative ──────────────────────────────────────────────────────────

import pandas as pd

def _preference_stats(train_meta, rec: dict) -> dict:
    """Compute concrete preference stats from a person's rating history."""
    import numpy as np
    stats = {}
    total = len(train_meta)
    stats["total_rated"] = total

    liked = train_meta[train_meta["Your Rating"] >= 7]
    avg_overall = float(train_meta["Your Rating"].mean())
    stats["avg_overall"] = round(avg_overall, 1)

    # Runtime preference (from liked films that have a runtime)
    if "Runtime" in train_meta.columns:
        rt_liked = pd.to_numeric(liked["Runtime"], errors="coerce").dropna()
        if len(rt_liked) > 5:
            stats["avg_runtime_liked"] = round(float(rt_liked.mean()))

    # Genre stats
    if rec.get("Genre") and "Genre" in train_meta.columns:
        main_genre = rec["Genre"].split(",")[0].strip()
        genre_mask = train_meta["Genre"].str.contains(main_genre, case=False, na=False)
        genre_rows = train_meta[genre_mask]
        if len(genre_rows) >= 3:
            stats["genre_name"] = main_genre
            stats["genre_count"] = len(genre_rows)
            stats["genre_avg"] = round(float(genre_rows["Your Rating"].mean()), 1)

    # Director stat
    director = rec.get("Director", "")
    if director and director != "N/A" and "Director" in train_meta.columns:
        dir_rows = train_meta[train_meta["Director"].str.contains(director.split(",")[0], case=False, na=False)]
        if len(dir_rows) >= 2:
            stats["director_count"] = len(dir_rows)
            stats["director_avg"] = round(float(dir_rows["Your Rating"].mean()), 1)

    # Tag preference stats — find the tag whose avg most deviates from overall avg
    tags_dict = rec.get("tags_dict", {})
    overall_avg = stats.get("avg_overall", 5.0)
    best_tag_key = None
    best_tag_delta = 0.0
    for cat, val_or_list in tags_dict.items():
        vals = val_or_list if isinstance(val_or_list, list) else [val_or_list]
        for val in vals:
            col = tag_col_name(cat, val)
            if col in train_meta.columns:
                tagged_rows = train_meta[train_meta[col] == 1]
                if len(tagged_rows) >= 5:
                    avg = round(float(tagged_rows["Your Rating"].mean()), 1)
                    delta = abs(avg - overall_avg)
                    if delta > best_tag_delta:
                        best_tag_delta = delta
                        best_tag_key = (cat, val, avg, len(tagged_rows))
    if best_tag_key:
        cat, val, avg, count = best_tag_key
        stats["top_tag"] = {"category": cat, "value": val, "avg": avg, "count": count}

    return stats


_PERSON_PRONOUNS = {
    "Chai": ("he", "him", "his"),
    "Noel": ("she", "her", "her"),
}

def build_why_prompt(rec: dict, pred_score: float, match_pct: float,
                     top_pos: list, top_neg: list, similar: dict,
                     vibe: float, person: str, train_meta=None) -> str:

    stats = _preference_stats(train_meta, rec) if train_meta is not None else {}

    # Build a plain-English preference context block
    pref_lines = []
    pref_lines.append(f"{person} has rated {stats.get('total_rated', '?')} films, averaging {stats.get('avg_overall', '?')}/10.")

    if "avg_runtime_liked" in stats and rec.get("Runtime"):
        rt_movie = int(rec["Runtime"])
        rt_avg = stats["avg_runtime_liked"]
        diff = rt_movie - rt_avg
        if diff > 20:
            pref_lines.append(f"At {rt_movie} min, this is {diff} min longer than {person}'s typical liked film ({rt_avg} min avg).")
        elif diff < -20:
            pref_lines.append(f"At {rt_movie} min, this is shorter than {person}'s typical liked film ({rt_avg} min avg) — generally a positive.")
        else:
            pref_lines.append(f"At {rt_movie} min, this fits comfortably within {person}'s usual runtime range (~{rt_avg} min).")

    if "genre_name" in stats:
        pref_lines.append(
            f"{person} has rated {stats['genre_count']} {stats['genre_name']} films, "
            f"averaging {stats['genre_avg']}/10 for that genre."
        )

    if "director_count" in stats:
        pref_lines.append(
            f"{person} has seen {stats['director_count']} films by {rec['Director'].split(',')[0]}, "
            f"averaging {stats['director_avg']}/10."
        )

    if "top_tag" in stats:
        t = stats["top_tag"]
        cat_label = t["category"].replace("_", " ").title()
        direction = "above" if t["avg"] > stats.get("avg_overall", 5.0) else "below"
        pref_lines.append(
            f"Films tagged {cat_label}: '{t['value']}' average {t['avg']}/10 for {person} "
            f"({t['count']} films) — {direction} their overall average."
        )

    pref_block = "\n".join(f"- {l}" for l in pref_lines)

    # Top SHAP drivers (readable labels only, skip plot_ noise)
    pos_drivers = [c for c in top_pos if not c["feature"].startswith("plot_")][:3]
    neg_drivers = [c for c in top_neg if not c["feature"].startswith("plot_")][:3]
    pos_lines = ", ".join(c["label"] for c in pos_drivers) if pos_drivers else "Plot themes"
    neg_lines = ", ".join(c["label"] for c in neg_drivers) if neg_drivers else "none"

    # Build a summary of the film's tags for the prompt
    tags_dict = rec.get("tags_dict", {})
    tag_lines = []
    cat_label_map = {
        "social_context":    "Social Context",
        "pacing":            "Pacing",
        "cognitive_load":    "Cognitive Load",
        "social_value":      "Social Value",
        "vibes":             "Vibes",
        "narrative_resolve": "Narrative Resolve",
        "tension_profile":   "Tension Profile",
    }
    for cat, label in cat_label_map.items():
        val_or_list = tags_dict.get(cat)
        if val_or_list:
            vals = val_or_list if isinstance(val_or_list, list) else [val_or_list]
            tag_lines.append(f"- {label}: {', '.join(vals)}")
    tag_block = "\n".join(tag_lines) if tag_lines else "- (no tags)"

    title   = rec['Title']
    year    = rec.get('Year', '?')
    genre   = rec.get('Genre', '?')
    director = rec.get('Director', '?').split(',')[0].strip()
    plot    = (rec.get('Plot', '') or '')[:300]
    imdb    = rec.get('imdbRating', '?')
    rt      = rec.get('RT_score', '?')

    # Build a weighted feature list for the prompt (positive drivers first, negatives after)
    feature_parts = []
    for c in pos_drivers:
        feature_parts.append(f"{c['label']} (+{c['shap']:.2f})")
    for c in neg_drivers:
        feature_parts.append(f"{c['label']} ({c['shap']:.2f})")
    feature_list = ", ".join(feature_parts) if feature_parts else "Plot themes"

    subj, obj, poss = _PERSON_PRONOUNS.get(person, ("they", "them", "their"))

    return f"""Write a short explanation of what {person} will likely experience watching "{title}" ({year}).

Data: top drivers = {feature_list}; genre = {genre}; director = {director}; IMDb = {imdb}; synopsis = {plot[:200]}; most similar rated film = "{similar['title']}" ({similar['rating']}/10).

Rules:
- {person} has NOT seen this film. {person}'s pronouns are {subj}/{obj}/{poss} — use them consistently.
- Distinguish facts from opinions: use plain "will" for objective facts about the film ("the film will immerse {person} in..."), use "may" or "might" for {person}'s subjective reaction ("{person} may find the pacing slow").
- Do NOT use "likely" for factual statements about the film. Reserve hedging language ("may", "might") for {person}'s opinions only.
- Do NOT reference the predicted score or any number rating.
- Do NOT open with the movie title or "{person} will score/rate".
- Do NOT reveal endings, twists, or how the story resolves.
- NEVER reference the model, scoring mechanics, or phrases like "will influence the score".
- Watch grammar on introductory clauses — "As a crime drama, {person}..." wrongly implies {person} is the crime drama. Always make the subject of the sentence match the opening clause ("As a crime drama, the film will..." or "For a crime drama fan, {person} will...").
- A negative SHAP on a high critic score means {person} tends to rate critically acclaimed films slightly below the hype — frame it as a taste note about {person}, not a flaw in the film.
- {match_pct}% match — let the tone reflect this. Only flag a negative if it's genuinely significant.
- Short, simple sentences. One idea per sentence. No sub-clauses.
- 2-4 sentences. Cut anything that doesn't help {person} decide whether to watch."""


def stream_why_narrative(rec: dict, pred_score: float, match_pct: float,
                         top_pos: list, top_neg: list, similar: dict,
                         vibe: float, person: str, train_meta=None):
    """Stream a Groq-generated 'Why' narrative."""
    groq_key = st.secrets.get("GROQ_API_KEY", "") or os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        yield "*(Set GROQ_API_KEY in config.py to enable AI explanations.)*"
        return

    prompt = build_why_prompt(rec, pred_score, match_pct, top_pos, top_neg,
                               similar, vibe, person, train_meta)
    try:
        client = Groq(api_key=groq_key)
        stream = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=350,
            stream=True,
        )
        for chunk in stream:
            text = chunk.choices[0].delta.content
            if text:
                yield text
    except Exception as e:
        yield f"*(Could not generate explanation: {e})*"


# ─── Poster loader ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_poster(url: str):
    if not url or url == "N/A":
        return None
    try:
        resp = requests.get(url, timeout=8)
        return Image.open(BytesIO(resp.content))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_poster_b64(url: str):
    """Load poster and return as base64 JPEG string for inline HTML embedding."""
    if not url or url == "N/A":
        return None
    try:
        resp = requests.get(url, timeout=8)
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


# ─── Closest-match logic (Part A / Part B) ───────────────────────────────────

# Raw combined score ceiling is ~0.65 for near-identical films.
# 0.52 raw ≈ 80% on the normalised display scale (÷ 0.65 × 100).
SIMILARITY_THRESHOLD = 0.52
_SIM_DISPLAY_CEIL    = 0.65   # denominator for % shown to user

def get_closest_matches(
    embedding: np.ndarray,
    tags_dict: dict,
    imdb_rating: float,
    movie_const: str,
    chai_artifacts: dict,
    noel_artifacts: dict,
) -> tuple:
    """
    Part A — Chai's anchor:
      Find Chai's highest-similarity film (plot 60% + tags 30% + IMDb 10%).
      If Chai has watched this exact film it will score ~100% and show first.
      Only shows if combined score >= threshold.

    Part B — Noel's anchor:
      If Noel has watched Chai's anchor → use same film, show Noel's rating.
      Otherwise find Noel's own highest-similarity film.

    Returns (chai_anchor, noel_anchor), each is
      {title, const, rating, similarity} or None.
    Note: exclude_const is intentionally NOT passed — if the user has already
    watched this film it should appear as the top match (100% similarity).
    """
    # Part A: Chai — no exclusion, exact matches are valid and expected
    chai_anchor = find_similar_movie_combined(
        embedding, tags_dict, imdb_rating,
        chai_artifacts, SIMILARITY_THRESHOLD, exclude_const=None,
    )

    # Part B: Noel
    noel_anchor = None
    if chai_anchor is not None:
        noel_meta = noel_artifacts["train_meta"]
        noel_row  = noel_meta[noel_meta["Const"] == chai_anchor["const"]]
        if len(noel_row) > 0:
            noel_anchor = {
                "title":      chai_anchor["title"],
                "const":      chai_anchor["const"],
                "rating":     float(noel_row["Your Rating"].values[0]),
                "similarity": chai_anchor["similarity"],
            }

    # Noel hasn't watched Chai's anchor — find Noel's own best match
    if noel_anchor is None:
        noel_anchor = find_similar_movie_combined(
            embedding, tags_dict, imdb_rating,
            noel_artifacts, SIMILARITY_THRESHOLD, exclude_const=None,
        )

    return chai_anchor, noel_anchor


# ─── Artifacts preload ────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading Chai's model…")
def preload_artifacts():
    return _load_all()

@st.cache_resource(show_spinner="Loading Noel's model…")
def preload_artifacts_noel():
    return _load_all_noel()


@st.cache_data(show_spinner=False)
def load_parents_guide() -> dict[str, dict[str, str]]:
    """Load parents_guide.csv → {imdb_id: {sex_nudity: ..., ...}}. Returns {} if missing."""
    import pandas as pd
    import os
    if not os.path.exists(PARENTS_GUIDE_CSV):
        return {}
    df = pd.read_csv(PARENTS_GUIDE_CSV, keep_default_na=False, na_values=[""])
    result = {}
    for _, row in df.iterrows():
        const = str(row.get("Const", "")).strip()
        if not const:
            continue
        result[const] = {
            "sex_nudity":   str(row.get("sex_nudity",   "") or ""),
            "violence_gore":str(row.get("violence_gore","") or ""),
            "profanity":    str(row.get("profanity",    "") or ""),
            "alcohol_drugs":str(row.get("alcohol_drugs","") or ""),
            "intensity":    str(row.get("intensity",    "") or ""),
        }
    return result


# ─── Meter column renderer ────────────────────────────────────────────────────

def render_meter_column(rec: dict, meter_result: dict, person: str,
                        anchor: dict | None = None,
                        train_meta=None,
                        cached_narrative: str | None = None):
    """Render a Match column (Figma card style). Returns the narrative text."""
    pct        = meter_result["match_pct"]
    tags       = meter_result["tags"]
    similar    = meter_result["similar"]
    vibe       = meter_result["vibe"]
    pred_score = meter_result["pred_score"]
    top_pos    = meter_result["top_pos"]
    top_neg    = meter_result["top_neg"]
    verdict, verdict_bg = score_verdict(pct)

    # ── Header
    st.markdown(
        f"<div style='font-size:1.15rem; font-weight:700; color:#e2e8f0; "
        f"margin-bottom:0.25rem;'>{person}'s Match</div>",
        unsafe_allow_html=True,
    )

    # ── Donut ring
    st.markdown(build_gauge(pct), unsafe_allow_html=True)

    # ── Verdict badge — full-width rectangle (Figma style)
    st.markdown(
        f"<div style='background:{verdict_bg}; border-radius:8px; padding:0.55rem 1rem; "
        f"text-align:center; font-size:0.82rem; font-weight:700; letter-spacing:0.12em; "
        f"text-transform:uppercase; color:#ffffff; margin:0.25rem 0 1rem 0;'>"
        f"{verdict}</div>",
        unsafe_allow_html=True,
    )

    # ── Thin divider
    st.markdown("<hr style='border:none; border-top:1px solid #1f2937; margin:0 0 0.75rem 0;'/>",
                unsafe_allow_html=True)

    # ── Why This Score
    st.markdown("<div class='section-head'>Why This Score?</div>", unsafe_allow_html=True)
    narrative_container = st.empty()
    if cached_narrative:
        full_text = cached_narrative
        narrative_container.markdown(
            f"<div class='narrative-box'>{full_text}</div>",
            unsafe_allow_html=True,
        )
    else:
        full_text = ""
        for chunk in stream_why_narrative(rec, pred_score, pct, top_pos, top_neg,
                                           similar, vibe, person, train_meta):
            full_text += chunk
            narrative_container.markdown(
                f"<div class='narrative-box'>{full_text}</div>",
                unsafe_allow_html=True,
            )
    # ── Key Drivers
    with st.expander("Key Drivers", expanded=False):
        tags_html = "".join(
            f"<span class='tag {'tag-pos' if t['direction'] == '+' else 'tag-neg'}'>"
            f"{t['direction']} {t['label']}</span>"
            for t in tags
        )
        st.markdown(tags_html, unsafe_allow_html=True)

    # ── Closest Match (always visible)
    st.markdown("<div class='section-head'>🔗 Closest Match In Their History</div>",
                unsafe_allow_html=True)
    import math
    if anchor and anchor.get("similarity") is not None and not math.isnan(float(anchor["similarity"])):
        sim_pct   = min(100, int(float(anchor["similarity"]) / _SIM_DISPLAY_CEIL * 100))
        sim_color = score_color(anchor["rating"] * 10)
        st.markdown(
            f"<div style='font-size:0.86rem; color:#9ca3af; padding:0.2rem 0;'>"
            f"{sim_pct}% similar to <span style='color:#9ca3af; font-weight:600;'>'{anchor['title']}'</span>"
            f" — {person} rated it "
            f"<span style='color:{sim_color}; font-weight:700;'>{anchor['rating']:.0f}/10</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='font-size:0.86rem; color:#9ca3af; font-style:italic; padding:0.2rem 0;'>"
            f"{person} hasn't watched a film that's at least 80% similar "
            f"in plot, feel, and theme."
            f"</div>",
            unsafe_allow_html=True,
        )
    return full_text


# ─── Catalog UI helpers ───────────────────────────────────────────────────────

def _render_movie_analysis(imdb_id: str, pfx: str = "_cached") -> None:
    """Fetch and render the full 3-column movie analysis.
    Shared between the Search tab (pfx='_cached') and the bottom sheet (pfx='_modal').
    """
    _cache_hit = st.session_state.get(f"{pfx}_imdb") == imdb_id
    if not _cache_hit:
        with st.spinner("Analyzing…"):
            result = predict_movie("", imdb_id=imdb_id)

        if result is None:
            st.error("Could not load this title. The daily API limit may have been reached.")
            return

        rec = result["rec"]
        tags_dict = result.get("tags_dict", {})
        rec["tags_dict"] = tags_dict

        noel_result = predict_movie_noel(rec, tags_dict=tags_dict,
                                         pg_ratings=result.get("pg_ratings"))
        chai_result = {
            "pred_score": result["pred_score"],
            "match_pct":  result["match_pct"],
            "tags":       result["tags"],
            "similar":    result["similar"],
            "vibe":       result["vibe"],
            "top_pos":    result["top_pos"],
            "top_neg":    result["top_neg"],
        }

        chai_artifacts = preload_artifacts()
        noel_artifacts = preload_artifacts_noel()
        chai_anchor, noel_anchor = get_closest_matches(
            embedding      = result["embedding"],
            tags_dict      = tags_dict,
            imdb_rating    = float(rec.get("imdbRating") or 5.0),
            movie_const    = None,
            chai_artifacts = chai_artifacts,
            noel_artifacts = noel_artifacts,
        )

        st.session_state[f"{pfx}_imdb"]           = imdb_id
        st.session_state[f"{pfx}_rec"]            = rec
        st.session_state[f"{pfx}_tags_dict"]      = tags_dict
        st.session_state[f"{pfx}_pg_ratings"]     = result.get("pg_ratings")
        st.session_state[f"{pfx}_chai_result"]    = chai_result
        st.session_state[f"{pfx}_noel_result"]    = noel_result
        st.session_state[f"{pfx}_chai_anchor"]    = chai_anchor
        st.session_state[f"{pfx}_noel_anchor"]    = noel_anchor
        st.session_state[f"{pfx}_chai_artifacts"] = chai_artifacts
        st.session_state[f"{pfx}_noel_artifacts"] = noel_artifacts
    else:
        rec            = st.session_state[f"{pfx}_rec"]
        tags_dict      = st.session_state[f"{pfx}_tags_dict"]
        chai_result    = st.session_state[f"{pfx}_chai_result"]
        noel_result    = st.session_state[f"{pfx}_noel_result"]
        chai_anchor    = st.session_state[f"{pfx}_chai_anchor"]
        noel_anchor    = st.session_state[f"{pfx}_noel_anchor"]
        chai_artifacts = st.session_state[f"{pfx}_chai_artifacts"]
        noel_artifacts = st.session_state[f"{pfx}_noel_artifacts"]

    # ── Three columns: Movie Details | Chai's Movie-Meter | Noel's Movie-Meter
    col_movie, col_chai, col_noel = st.columns([2.5, 3.75, 3.75], gap="large")

    with col_movie:
        # Build poster HTML
        poster_b64 = load_poster_b64(rec.get("Poster", ""))
        if poster_b64:
            poster_html = (
                f"<div style='width:100%; overflow:hidden; border-radius:13px 13px 0 0;'>"
                f"<img src='data:image/jpeg;base64,{poster_b64}' "
                f"style='width:100%; height:auto; display:block;'/>"
                f"</div>"
            )
        else:
            poster_html = (
                "<div style='width:100%; height:260px; background:#1a1f2e; "
                "border-radius:13px 13px 0 0; display:flex; align-items:center; "
                "justify-content:center; font-size:3rem;'>🎬</div>"
            )

        # Chips
        chips = []
        if rec.get("Year"):    chips.append(str(int(rec["Year"])))
        if rec.get("Rated") and rec["Rated"] != "N/A": chips.append(rec["Rated"])
        if rec.get("Runtime"): chips.append(f"{rec['Runtime']} min")
        chips_html = " ".join(
            f"<span style='display:inline-block; background:#1e2536; border-radius:6px; "
            f"padding:0.2rem 0.55rem; font-size:0.78rem; color:#9ca3af; "
            f"margin:0 0.2rem 0.2rem 0; border:1px solid #2d3748;'>{c}</span>"
            for c in chips
        )

        # Award badges
        award_parts = []
        def _safe_int(v):
            try:
                import math
                f = float(v)
                return 0 if math.isnan(f) else int(f)
            except (TypeError, ValueError):
                return 0
        oscar_wins = _safe_int(rec.get("oscar_wins", 0))
        oscar_noms = _safe_int(rec.get("oscar_noms", 0))
        gg_wins    = _safe_int(rec.get("gg_wins", 0))
        gg_noms    = _safe_int(rec.get("gg_noms", 0))
        if oscar_wins > 0:
            label = f"{'🏆' * min(int(oscar_wins), 3)} {int(oscar_wins)} Oscar{'s' if oscar_wins != 1 else ''}"
            if oscar_noms > oscar_wins:
                label += f" / {int(oscar_noms)} Nomination{'s' if oscar_noms != 1 else ''}"
            award_parts.append(
                f"<span style='display:inline-block; background:rgba(255,215,0,0.12); "
                f"border:1px solid rgba(255,215,0,0.35); border-radius:6px; "
                f"padding:0.22rem 0.6rem; font-size:0.78rem; color:#FFD700; "
                f"margin:0.25rem 0.2rem 0 0;'>{label}</span>"
            )
        elif oscar_noms > 0:
            award_parts.append(
                f"<span style='display:inline-block; background:rgba(255,215,0,0.07); "
                f"border:1px solid rgba(255,215,0,0.2); border-radius:6px; "
                f"padding:0.22rem 0.6rem; font-size:0.78rem; color:#b8a000; "
                f"margin:0.25rem 0.2rem 0 0;'>"
                f"🎬 {int(oscar_noms)} Oscar Nomination{'s' if oscar_noms != 1 else ''}</span>"
            )
        else:
            award_parts.append(
                f"<span style='display:inline-block; background:rgba(255,215,0,0.04); "
                f"border:1px solid rgba(255,215,0,0.12); border-radius:6px; "
                f"padding:0.22rem 0.6rem; font-size:0.78rem; color:#6b5e00; "
                f"margin:0.25rem 0.2rem 0 0;'>No Oscar Nominations</span>"
            )
        if gg_wins > 0:
            label = f"✨ {int(gg_wins)} Golden Globe{'s' if gg_wins != 1 else ''}"
            if gg_noms > gg_wins:
                label += f" / {int(gg_noms)} Nomination{'s' if gg_noms != 1 else ''}"
            award_parts.append(
                f"<span style='display:inline-block; background:rgba(192,192,255,0.1); "
                f"border:1px solid rgba(192,192,255,0.25); border-radius:6px; "
                f"padding:0.22rem 0.6rem; font-size:0.78rem; color:#a0a8ff; "
                f"margin:0.25rem 0.2rem 0 0;'>{label}</span>"
            )
        elif gg_noms > 0:
            award_parts.append(
                f"<span style='display:inline-block; background:rgba(192,192,255,0.06); "
                f"border:1px solid rgba(192,192,255,0.15); border-radius:6px; "
                f"padding:0.22rem 0.6rem; font-size:0.78rem; color:#7077c0; "
                f"margin:0.25rem 0.2rem 0 0;'>"
                f"🌐 {int(gg_noms)} Golden Globe Nomination{'s' if gg_noms != 1 else ''}</span>"
            )
        else:
            award_parts.append(
                f"<span style='display:inline-block; background:rgba(192,192,255,0.04); "
                f"border:1px solid rgba(192,192,255,0.12); border-radius:6px; "
                f"padding:0.22rem 0.6rem; font-size:0.78rem; color:#3d3f70; "
                f"margin:0.25rem 0.2rem 0 0;'>No Golden Globe Nominations</span>"
            )
        award_html = (
            f"<div style='font-size:0.86rem; font-weight:600; color:#e2e8f0; "
            f"margin:1rem 0 0.4rem 0;'>Awards</div>"
            f"{''.join(award_parts)}"
        )

        # Director / Cast
        dir_html = ""
        if rec.get("Director") and rec["Director"] != "N/A":
            dir_html = (
                f"<p style='margin:0.75rem 0 0.2rem 0; font-size:0.86rem; color:#9ca3af;'>"
                f"<span style='color:#e2e8f0; font-weight:600;'>Director</span><br>"
                f"{rec['Director']}</p>"
            )
        cast_html = ""
        if rec.get("Actors") and rec["Actors"] != "N/A":
            actors = rec["Actors"]
            cast_html = (
                f"<p style='margin:0.35rem 0; font-size:0.86rem; color:#9ca3af;'>"
                f"<span style='color:#e2e8f0; font-weight:600;'>Cast</span><br>"
                f"{actors}</p>"
            )

        # Scores
        def _safe_num(v):
            try:
                f = float(v)
                import math
                return None if math.isnan(f) else f
            except (TypeError, ValueError):
                return None

        crit_items = []
        if _safe_num(rec.get("imdbRating")) is not None:
            crit_items.append(("IMDb", f"{rec['imdbRating']}/10"))
        if _safe_num(rec.get("RT_score")) is not None:
            crit_items.append(("RT Critic", f"{int(float(rec['RT_score']))}%"))
        if _safe_num(rec.get("rt_audience")) is not None:
            crit_items.append(("RT Audience", f"{int(float(rec['rt_audience']))}%"))
        if _safe_num(rec.get("Metascore")) is not None:
            crit_items.append(("Metacritic", f"{int(float(rec['Metascore']))}/100"))
        critics_html = ""
        if crit_items:
            crit_cols = "".join(
                f"<div style='flex:1;'>"
                f"<div style='font-size:0.75rem; color:#6b7280; text-transform:uppercase; "
                f"letter-spacing:0.08em; margin-bottom:0.1rem;'>{lbl}</div>"
                f"<div style='font-size:0.86rem; color:#9ca3af;'>{val}</div>"
                f"</div>"
                for lbl, val in crit_items
            )
            critics_html = (
                f"<div style='font-size:0.86rem; font-weight:600; color:#e2e8f0; "
                f"margin:1rem 0 0.4rem 0;'>Scores</div>"
                f"<div style='display:flex; gap:0.5rem; flex-wrap:wrap;'>{crit_cols}</div>"
            )

        # Film Profile tags
        profile_html = ""
        if tags_dict:
            CATEGORY_LABELS = {
                "social_context":    ("🎟️", "Social Context"),
                "pacing":            ("⏱️", "Pacing"),
                "cognitive_load":    ("🧠", "Cognitive Load"),
                "social_value":      ("💬", "Social Value"),
                "vibes":             ("🎨", "Vibes"),
                "narrative_resolve": ("🏁", "Narrative Resolve"),
                "tension_profile":   ("😬", "Tension Profile"),
            }
            pills = ""
            for cat, (emoji, label) in CATEGORY_LABELS.items():
                val_or_list = tags_dict.get(cat)
                if val_or_list:
                    vals = val_or_list if isinstance(val_or_list, list) else [val_or_list]
                    for v in vals:
                        pills += (
                            f"<span title='{label}' style='display:inline-block; "
                            f"padding:0.2rem 0.55rem; border-radius:6px; font-size:0.86rem; "
                            f"background:rgba(30,37,54,0.8); color:#9ca3af; "
                            f"border:1px solid #2d3748; margin:0.2rem 0.15rem; "
                            f"cursor:default;'>{v}</span>"
                        )
            if pills:
                profile_html = (
                    f"<div style='font-size:0.86rem; font-weight:600; color:#e2e8f0; "
                    f"margin:1rem 0 0.4rem 0;'>Film Profile</div>"
                    f"<div style='display:flex; flex-wrap:wrap; gap:0.2rem 0;'>{pills}</div>"
                )

        # Parents Guide — prefer live scraped data, fall back to CSV cache
        pg_html = ""
        imdb_id_rec = rec.get("imdbID", "")
        _live_pg = st.session_state.get(f"{pfx}_pg_ratings")
        pg_data = load_parents_guide()
        _csv_pg = pg_data.get(imdb_id_rec) if imdb_id_rec else None
        _pg_rec = _live_pg or _csv_pg
        if _pg_rec:
            PG_CATEGORIES = [
                ("sex_nudity",    "Sex & Nudity"),
                ("violence_gore", "Violence & Gore"),
                ("profanity",     "Profanity"),
                ("alcohol_drugs", "Alcohol & Drugs"),
                ("intensity",     "Intensity"),
            ]
            PG_COLORS = {
                "None":     ("#1a2a1a", "#6ee78a"),
                "Mild":     ("#3a3010", "#facc15"),
                "Moderate": ("#3a1f00", "#fb923c"),
                "Severe":   ("#3a0a0a", "#f87171"),
            }
            rows = ""
            pg_rec = _pg_rec
            for col_key, cat_label in PG_CATEGORIES:
                raw_rating = pg_rec.get(col_key, "")
                rating = raw_rating if raw_rating in ("Mild", "Moderate", "Severe") else "None"
                bg, fg = PG_COLORS[rating]
                rows += (
                    f"<div style='display:flex; align-items:center; justify-content:space-between; "
                    f"padding:0.2rem 0;'>"
                    f"<span style='font-size:0.8rem; color:#9ca3af;'>{cat_label}</span>"
                    f"<span style='font-size:0.75rem; font-weight:600; padding:0.1rem 0.5rem; "
                    f"border-radius:4px; background:{bg}; color:{fg};'>{rating}</span>"
                    f"</div>"
                )
            if rows:
                pg_html = (
                    f"<div style='font-size:0.86rem; font-weight:600; color:#e2e8f0; "
                    f"margin:1rem 0 0.4rem 0;'>Parents Guide</div>"
                    f"<div style='display:flex; flex-direction:column; gap:0.05rem;'>{rows}</div>"
                )

        _body = "".join([
            f"<div style='margin-bottom:0.25rem;'>{chips_html}</div>",
            dir_html,
            cast_html,
            critics_html,
            award_html,
            pg_html,
            profile_html,
        ])
        movie_card = f"""
<div style='background:#13161f; border-radius:13px; overflow:visible; margin-bottom:0;'>
  {poster_html}
  <div style='padding:1rem 1.2rem 1.4rem 1.2rem;'>
    <h2 style='font-size:1.5rem; font-weight:800; color:#ffffff; margin:0 0 0.5rem 0; line-height:1.2;'>{rec['Title']}</h2>
{_body}
  </div>
</div>
"""
        st.markdown(movie_card, unsafe_allow_html=True)

    with col_chai:
        chai_narrative = render_meter_column(
            rec, chai_result, "Chai", anchor=chai_anchor,
            train_meta=chai_artifacts["train_meta"],
            cached_narrative=st.session_state.get(f"{pfx}_chai_narrative") if _cache_hit else None,
        )
        if not _cache_hit:
            st.session_state[f"{pfx}_chai_narrative"] = chai_narrative

    with col_noel:
        noel_narrative = render_meter_column(
            rec, noel_result, "Noel", anchor=noel_anchor,
            train_meta=noel_artifacts["train_meta"],
            cached_narrative=st.session_state.get(f"{pfx}_noel_narrative") if _cache_hit else None,
        )
        if not _cache_hit:
            st.session_state[f"{pfx}_noel_narrative"] = noel_narrative


@st.dialog(" ", width="large")
def catalog_movie_detail(imdb_id: str) -> None:
    """Bottom sheet — full movie analysis identical to the Search tab view."""
    st.markdown("""<style>
/* ── Bottom sheet: override Streamlit dialog → full-width, slides from bottom ── */
[data-testid="stDialog"] {
  position: fixed !important;
  bottom: 0 !important; left: 0 !important; right: 0 !important; top: auto !important;
  width: 100vw !important; max-width: 100vw !important;
  height: 80vh !important; max-height: 80vh !important;
  border-radius: 20px 20px 0 0 !important;
  margin: 0 !important;
  background: #0a0b0f !important;
  overflow-y: auto !important; overflow-x: hidden !important;
  animation: sheetUp 0.35s cubic-bezier(0.32, 0.72, 0, 1) !important;
}
@keyframes sheetUp {
  from { transform: translateY(100%); }
  to   { transform: translateY(0); }
}
/* Inner white dialog box → match app background */
[data-testid="stDialog"] > div > div {
  background: #0a0b0f !important;
  border: none !important;
  box-shadow: 0 -4px 24px rgba(0,0,0,0.6) !important;
  width: 100% !important;
  max-width: 100% !important;
  padding: 0 !important;
}
/* Depth-3 content containers — halve Streamlit's default 24px padding */
[data-testid="stDialog"] > div > div > div {
  padding-left: 12px !important;
  padding-right: 12px !important;
  padding-top: 12px !important;
  width: 100% !important;
  max-width: 100% !important;
}
/* Dark column boxes inside the sheet (mirrors Search tab styling) */
[data-testid="stDialog"] [data-testid="stColumn"] {
  background: #13161f !important;
  border-radius: 13px !important;
  overflow: hidden !important;
}
/* Restore nth-child padding for the analysis columns inside dialog */
[data-testid="stDialog"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(2),
[data-testid="stDialog"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"]:nth-child(3) {
  padding: 1.4rem !important;
}
</style>""", unsafe_allow_html=True)

    _render_movie_analysis(imdb_id, pfx="_modal")


def _render_catalog_card(item) -> None:
    """Render one catalog card (HTML + a Details button)."""
    poster_url = str(item.get("poster_url") or "")
    title      = str(item.get("title") or "Unknown")
    year       = item.get("year")
    imdb       = item.get("imdb_score")
    chai_pct   = float(item.get("chai_pct") or 0)
    noel_pct   = float(item.get("noel_pct") or 0)
    service    = str(item.get("service") or "").lower()

    def _sc(pct: float) -> str:
        if pct >= 90:   return "#FFD700"
        elif pct >= 70: return "#4f8ef7"
        elif pct >= 50: return "#708090"
        else:           return "#555555"

    # padding-bottom trick: enforces 2:3 ratio in all browsers regardless of image natural size
    _poster_wrap = "position:relative; width:100%; padding-bottom:150%; overflow:hidden;"
    _poster_inner = "position:absolute; top:0; left:0; width:100%; height:100%; object-fit:cover; display:block;"
    if poster_url and poster_url not in ("N/A", "nan", ""):
        poster_html = (
            f"<div style='{_poster_wrap}'>"
            f"<img src='{poster_url}' loading='lazy' style='{_poster_inner}'/>"
            f"</div>"
        )
    else:
        poster_html = (
            f"<div style='{_poster_wrap} background:#1a1f2e;'>"
            f"<div style='position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);font-size:2rem;'>🎬</div>"
            f"</div>"
        )

    svc_colors = {"netflix": "#E50914", "max": "#0056FF"}
    svc_color  = svc_colors.get(service, "#444")
    svc_badge  = (
        f"<span style='position:absolute; top:6px; right:6px; background:{svc_color}; "
        f"color:white; font-size:0.6rem; font-weight:800; padding:0.12rem 0.35rem; "
        f"border-radius:3px; letter-spacing:0.04em;'>{service.title()}</span>"
    ) if service else ""

    imdb_badge = (
        f"<span style='position:absolute; top:6px; left:6px; background:rgba(0,0,0,0.72); "
        f"color:#f5c518; font-size:0.68rem; font-weight:700; padding:0.12rem 0.35rem; "
        f"border-radius:3px;'>⭐ {imdb:.1f}</span>"
    ) if imdb else ""

    year_str   = str(int(year)) if year else ""
    title_esc  = title.replace("'", "&#39;").replace('"', "&quot;")

    card_html = f"""
<div class='catalog-card'>
  <div style='position:relative;'>
    {poster_html}
    {svc_badge}{imdb_badge}
  </div>
  <div class='catalog-card-body'>
    <div class='catalog-card-title' title='{title_esc}'>{title}</div>
    <div class='catalog-card-year'>{year_str}</div>
    <div class='catalog-card-scores'>
      <span style='font-size:0.75rem; font-weight:700; color:{_sc(chai_pct)};'>Chai {chai_pct:.0f}%</span>
      <span style='font-size:0.75rem; font-weight:700; color:{_sc(noel_pct)};'>Noel {noel_pct:.0f}%</span>
    </div>
  </div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)
    btn_key = f"cat_{item.get('imdb_id', '')}_{abs(hash(title)) % 999999}"
    if st.button("Details", key=btn_key, use_container_width=True):
        imdb_id = item.get("imdb_id", "")
        if imdb_id:
            catalog_movie_detail(imdb_id)


@st.cache_data(show_spinner=False, ttl=300)
def _ids_from_seen_numbers(path: str) -> set:
    """Extract IMDb IDs (tt...) from a *Seen.numbers file's URL column."""
    import re
    import warnings
    try:
        import numbers_parser
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            doc = numbers_parser.Document(path)
        rows = doc.sheets[0].tables[0].rows(values_only=True)
        ids = set()
        for row in rows:
            for cell in row:
                if isinstance(cell, str) and "imdb.com/title/" in cell:
                    m = re.search(r"tt\d+", cell)
                    if m:
                        ids.add(m.group(0))
        return ids
    except Exception:
        return set()


def _load_catalog() -> "pd.DataFrame":
    import pandas as pd
    df = pd.read_parquet(CATALOG_PATH)
    # Annotate which movies each user has seen (from their Seen.numbers files)
    chai_ids = _ids_from_seen_numbers(CHAI_SEEN_FILE)
    noel_ids = _ids_from_seen_numbers(NOEL_SEEN_FILE)
    df["chai_seen"] = df["imdb_id"].isin(chai_ids)
    df["noel_seen"] = df["imdb_id"].isin(noel_ids)
    return df


def render_recommend_tab() -> None:
    """Render the 🍿 Recommend tab content."""
    import os
    import pandas as pd

    sync_status = get_sync_status()

    # Catalog not yet built
    if not os.path.exists(CATALOG_PATH):
        if sync_status["running"]:
            st.info(
                "⏳ Building your catalog for the first time — this may take a few minutes. "
                "Come back soon!",
                icon="🔄",
            )
        else:
            st.markdown(
                """
<div style='text-align:center; padding:3rem 1rem;'>
  <div style='font-size:3.5rem; margin-bottom:1rem;'>🍿</div>
  <h3 style='color:#e2e8f0; margin-bottom:0.5rem;'>Catalog Not Yet Built</h3>
  <p style='color:#6b7280; max-width:520px; margin:0 auto 1.5rem auto; line-height:1.7;'>
    Run the one-time seed script to pull Netflix &amp; Max content and pre-score everything.
    You'll need a <strong style='color:#e2e8f0;'>Watchmode API key</strong>
    (free at <code>api.watchmode.com</code>) in <code>.streamlit/secrets.toml</code>.
  </p>
  <code style='display:inline-block; background:#1e2536; padding:0.45rem 1rem;
               border-radius:8px; font-size:0.86rem; color:#4f8ef7;
               border:1px solid #2d3748;'>python catalog_seed.py</code>
</div>
""",
                unsafe_allow_html=True,
            )
        return

    df = _load_catalog()

    # ── Filter bar ────────────────────────────────────────────────────────────
    _svc_netflix      = st.session_state.get("f_svc_netflix",  False)
    _svc_max          = st.session_state.get("f_svc_max",      False)
    _svc_disney       = st.session_state.get("f_svc_disney",   False)
    _svc_hulu         = st.session_state.get("f_svc_hulu",     False)
    _svc_apple        = st.session_state.get("f_svc_apple",    False)
    _svc_peacock      = st.session_state.get("f_svc_peacock",  False)
    _svc_paramount    = st.session_state.get("f_svc_paramount",False)
    _type_movies      = st.session_state.get("f_type_movies",  False)
    _type_tv          = st.session_state.get("f_type_tv",      False)
    _w_chai_seen      = st.session_state.get("f_w_chai_seen",      False)
    _w_chai_not_seen  = st.session_state.get("f_w_chai_not_seen",  False)
    _w_noel_seen      = st.session_state.get("f_w_noel_seen",      False)
    _w_noel_not_seen  = st.session_state.get("f_w_noel_not_seen",  False)
    _imdb_val         = float(st.session_state.get("f_imdb", 0.0))
    _yr_val           = st.session_state.get("f_yr", (1950, 2026))

    # Map display name → catalog service key
    _SVC_MAP = {
        "Netflix":    "netflix",
        "Max":        "max",
        "Disney+":    "disney",
        "Hulu":       "hulu",
        "Apple TV+":  "apple",
        "Peacock":    "peacock",
        "Paramount+": "paramount",
    }
    services      = [k for k, v in [
        ("Netflix",    _svc_netflix),
        ("Max",        _svc_max),
        ("Disney+",    _svc_disney),
        ("Hulu",       _svc_hulu),
        ("Apple TV+",  _svc_apple),
        ("Peacock",    _svc_peacock),
        ("Paramount+", _svc_paramount),
    ] if v]
    content_types = [t for t, v in [("Movies", _type_movies), ("TV Shows", _type_tv)] if v]
    _watch_sel    = [w for w, v in [
        ("Chai Seen",     _w_chai_seen),
        ("Chai Not Seen", _w_chai_not_seen),
        ("Noel Seen",     _w_noel_seen),
        ("Noel Not Seen", _w_noel_not_seen),
    ] if v]

    _svc_lbl = (
        f"{len(services)} Services" if len(services) > 1
        else (services[0] if services else "All Services")
    )
    _both_types = _type_movies and _type_tv
    _type_lbl = (
        "Movies & TV" if (not _type_movies and not _type_tv) or _both_types
        else ("Movies" if _type_movies else "TV Shows")
    )
    _watch_lbl = (
        f"{len(_watch_sel)} Statuses" if len(_watch_sel) > 1
        else (_watch_sel[0] if _watch_sel else "Watch Status")
    )
    _imdb_lbl = f"IMDb ≥ {_imdb_val:.1f}" if _imdb_val > 0 else "IMDb Score"
    _yr_lbl   = f"{_yr_val[0]}–{_yr_val[1]}" if _yr_val != (1950, 2026) else "Release Year"

    _any_active = bool(services or content_types or _watch_sel or _imdb_val > 0 or _yr_val != (1950, 2026))

    # Active pill highlights — columns in order: svc(1) type(2) watch(3) imdb(4) yr(5)
    _pill_css = ""
    for _i, _active in enumerate([bool(services), bool(content_types), bool(_watch_sel),
                                   _imdb_val > 0, _yr_val != (1950, 2026)], 1):
        if _active:
            _pill_css += (
                f".st-key-filter-bar [data-testid='stColumn']:nth-child({_i})"
                f" [data-testid='stPopover'] button {{"
                f"background:#4f8ef7!important;border-color:#4f8ef7!important;"
                f"color:#ffffff!important;}}"
            )
    if _pill_css:
        st.markdown(f"<style>{_pill_css}</style>", unsafe_allow_html=True)

    with st.container(key="filter-bar"):
        fc_svc, fc_type, fc_watch, fc_imdb, fc_yr, fc_sep, fc_clr, fc_srt = st.columns(
            [1.6, 1.6, 1.6, 1.6, 1.6, 0.12, 0.9, 2.2], vertical_alignment="center"
        )
        with fc_svc:
            with st.popover(_svc_lbl, use_container_width=True):
                st.checkbox("Netflix",  key="f_svc_netflix")
                st.checkbox("Max",      key="f_svc_max")
                st.checkbox("Disney+", key="f_svc_disney")
                st.checkbox("Hulu",        key="f_svc_hulu")
                st.checkbox("Apple TV+",   key="f_svc_apple")
                st.checkbox("Peacock",     key="f_svc_peacock")
                st.checkbox("Paramount+",  key="f_svc_paramount")
        with fc_type:
            with st.popover(_type_lbl, use_container_width=True):
                st.checkbox("Movies",   key="f_type_movies")
                st.checkbox("TV Shows", key="f_type_tv")
        with fc_watch:
            with st.popover(_watch_lbl, use_container_width=True):
                st.checkbox("Chai Seen",     key="f_w_chai_seen")
                st.checkbox("Chai Not Seen", key="f_w_chai_not_seen")
                st.checkbox("Noel Seen",     key="f_w_noel_seen")
                st.checkbox("Noel Not Seen", key="f_w_noel_not_seen")
        with fc_imdb:
            with st.popover(_imdb_lbl, use_container_width=True):
                min_imdb = st.slider(
                    "Min IMDb", 0.0, 10.0, value=0.0, step=0.5,
                    format="%.1f", key="f_imdb",
                )
        with fc_yr:
            with st.popover(_yr_lbl, use_container_width=True):
                year_range = st.slider(
                    "Year", 1900, 2026, value=(1950, 2026), key="f_yr",
                )
        with fc_sep:
            st.markdown("<div class='filter-sep'>|</div>", unsafe_allow_html=True)
        with fc_clr:
            if st.button("Clear all", key="f_clear", disabled=not _any_active):
                for _k in ["f_svc_netflix", "f_svc_max", "f_svc_disney",
                            "f_svc_hulu", "f_svc_apple", "f_svc_peacock", "f_svc_paramount",
                            "f_type_movies", "f_type_tv",
                            "f_w_chai_seen", "f_w_chai_not_seen", "f_w_noel_seen", "f_w_noel_not_seen",
                            "f_imdb", "f_yr"]:
                    st.session_state.pop(_k, None)
                st.rerun()
        with fc_srt:
            sort_by = st.selectbox(
                "Sort",
                ["Compatibility", "Chai Score", "Noel Score", "IMDb Score", "Newest First"],
                format_func=lambda x: f"Sort: {x}",
                label_visibility="collapsed", key="f_sort",
            )

    # ── Apply filters ─────────────────────────────────────────────────────────
    dff = df.copy()
    if services:
        dff = dff[dff["service"].isin([_SVC_MAP.get(s, s.lower()) for s in services])]
    if content_types:
        _type_map = {"Movies": "movie", "TV Shows": "tv"}
        dff = dff[dff["type"].isin([_type_map[t] for t in content_types])]
    if _watch_sel:
        import numpy as np
        _masks = []
        for _ws in _watch_sel:
            if _ws == "Chai Seen":
                _masks.append(dff["chai_seen"].astype(bool))
            elif _ws == "Chai Not Seen":
                _masks.append(~dff["chai_seen"].astype(bool))
            elif _ws == "Noel Seen":
                _masks.append(dff["noel_seen"].astype(bool))
            elif _ws == "Noel Not Seen":
                _masks.append(~dff["noel_seen"].astype(bool))
        if _masks:
            _combined = _masks[0]
            for _m in _masks[1:]:
                _combined = _combined & _m
            dff = dff[_combined]
    if min_imdb > 0:
        dff = dff[dff["imdb_score"].notna() & (dff["imdb_score"] >= min_imdb)]
    year_from, year_to = year_range
    if "year" in dff.columns:
        dff = dff[dff["year"].notna() & (dff["year"] >= year_from) & (dff["year"] <= year_to)]

    # ── Sort ──────────────────────────────────────────────────────────────────
    if sort_by == "Chai Score":
        dff = dff.sort_values("chai_pct", ascending=False)
    elif sort_by == "Noel Score":
        dff = dff.sort_values("noel_pct", ascending=False)
    elif sort_by == "Compatibility":
        dff = dff.assign(_compat=(dff["chai_pct"] + dff["noel_pct"]) / 2).sort_values("_compat", ascending=False)
    elif sort_by == "IMDb Score":
        dff = dff.sort_values("imdb_score", ascending=False, na_position="last")
    elif sort_by == "Newest First":
        dff = dff.sort_values("year", ascending=False, na_position="last")

    # ── Grid (paginated — 16 items per page) ──────────────────────────────────
    # Reset page when filters change (use a hash of filter state as key)
    _filter_key = f"{services}|{content_types}|{_watch_sel}|{min_imdb}|{year_range}|{sort_by}"
    if st.session_state.get("_catalog_filter_key") != _filter_key:
        st.session_state["_catalog_filter_key"] = _filter_key
        st.session_state["_catalog_visible"] = 8

    visible = st.session_state.get("_catalog_visible", 8)

    if len(dff) == 0:
        st.markdown(
            "<div style='text-align:center; padding:2.5rem; color:#6b7280;'>"
            "No matches found — try adjusting your filters!</div>",
            unsafe_allow_html=True,
        )
    else:
        N = 4
        dff_reset = dff.reset_index(drop=True).iloc[:visible]
        for row_start in range(0, len(dff_reset), N):
            chunk = dff_reset.iloc[row_start: row_start + N]
            cols  = st.columns(N)
            for j, (_, item) in enumerate(chunk.iterrows()):
                with cols[j]:
                    _render_catalog_card(item)
            st.markdown("<div style='margin-bottom:0.5rem;'></div>",
                        unsafe_allow_html=True)

        if visible < len(dff):
            remaining = len(dff) - visible
            _, load_col, _ = st.columns([2, 1, 2])
            with load_col:
                if st.button(
                    f"Load more  (+{min(16, remaining)})",
                    key="catalog_load_more",
                    use_container_width=True,
                ):
                    st.session_state["_catalog_visible"] = visible + 16
                    st.rerun()

    # ── Footer ────────────────────────────────────────────────────────────────
    age = catalog_age_days()
    if age is not None:
        age_str = f"{int(age)} day{'s' if age >= 2 else ''} ago" if age >= 1 else "today"
    else:
        age_str = "unknown"
    syncing_str = " · 🔄 Syncing in background…" if sync_status["running"] else ""
    st.markdown(
        f"<div style='text-align:center; font-size:0.72rem; color:#374151; "
        f"padding:1.5rem 0 0.5rem 0;'>"
        f"Last Sync: {age_str} · {len(df):,} Titles Available{syncing_str}</div>",
        unsafe_allow_html=True,
    )


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    preload_artifacts()
    preload_artifacts_noel()

    # ── Background catalog sync ────────────────────────────────────────────────
    if start_background_sync():
        st.session_state["_sync_started_this_session"] = True

    sync_status = get_sync_status()
    if sync_status["finished"] and sync_status.get("new_count", 0) > 0:
        if not st.session_state.get("_sync_toast_shown"):
            st.toast(
                f"🔄 Catalog refreshed — {sync_status['new_count']:,} titles updated!",
                icon="🍿",
            )
            st.session_state["_sync_toast_shown"] = True
            _load_catalog.clear()   # invalidate the 5-min cache

    _LOGO_B64_HDR = open(
        os.path.join(os.path.dirname(__file__), "assets", "logo_b64.txt")
    ).read().strip()
    st.session_state["_logo_b64"] = _LOGO_B64_HDR

    # ── Nav bar ───────────────────────────────────────────────────────────────
    if "active_tab" not in st.session_state:
        st.session_state["active_tab"] = "recommend"

    active = st.session_state["active_tab"]
    rec_active  = active == "recommend"
    srch_active = active == "search"

    # Dynamic CSS: button colours only (searchbox now lives inside _render_search_tab)
    st.markdown(f"""<style>
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"]:nth-last-child(2) button {{
  background: {"#4f8ef7" if rec_active  else "#1e2536"} !important;
  color:      {"#ffffff" if rec_active  else "#9ca3af"} !important;
}}
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"]:last-child button {{
  background: {"#4f8ef7" if srch_active else "#1e2536"} !important;
  color:      {"#ffffff" if srch_active else "#9ca3af"} !important;
}}
</style>""", unsafe_allow_html=True)

    col_logo, col_rec, col_srch = st.columns([6.5, 1.6, 1.2])

    with col_logo:
        # .nav-btn marker identifies this horizontal block for CSS :has() targeting
        st.markdown(
            f"""<div class='nav-btn' style='display:flex; align-items:center; gap:0.65rem;
                            padding:0.4rem 0 0.5rem 0;'>
  <img src='data:image/png;base64,{_LOGO_B64_HDR}'
       style='height:42px; width:auto; opacity:0.95;'/>
  <div>
    <div style='font-size:1.89rem; font-weight:900; letter-spacing:0.30em;
                text-transform:uppercase; color:#ffffff; line-height:1;'>ENZYME</div>
    <div style='font-size:0.88rem; color:#6b7280; letter-spacing:0.10em;
                font-weight:400; margin-top:2px;'>Movies and Shows, broken down for you.</div>
  </div>
</div>""",
            unsafe_allow_html=True,
        )

    with col_rec:
        if st.button("🍿  Recommend", key="nav_btn_rec", use_container_width=True):
            if st.session_state.get("active_tab") != "recommend":
                st.session_state["active_tab"] = "recommend"
                st.rerun()

    with col_srch:
        if st.button("🔍  Search", key="nav_btn_srch", use_container_width=True):
            if st.session_state.get("active_tab") != "search":
                st.session_state["active_tab"] = "search"
                st.rerun()

    st.markdown(
        "<hr style='border:none; border-top:1px solid #1f2937; margin:0 0 1rem 0;'/>",
        unsafe_allow_html=True,
    )

    # ── Content: search tab (includes its own searchbox) OR recommend grid
    if active == "search":
        _render_search_tab()
    else:
        render_recommend_tab()


def _render_search_tab():
    """Renders the search bar and movie prediction results."""
    # ── Searchbox ─────────────────────────────────────────────────────────────
    _components.html("""
<script>
(function() {
  function injectFont(iframe) {
    try {
      var doc = iframe.contentDocument || iframe.contentWindow.document;
      if (!doc || !doc.head || doc._fontInjected) return;
      doc._fontInjected = true;
      var link = doc.createElement('link');
      link.rel = 'stylesheet';
      link.href = 'https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap';
      doc.head.appendChild(link);
      var s = doc.createElement('style');
      s.textContent = '*, input, div, span { font-family: \\'Source Sans Pro\\', sans-serif !important; } [class*="-control"] { border-radius: 10px !important; }';
      doc.head.appendChild(s);
    } catch(e) {}
  }
  function scan() {
    var iframe = parent.document.querySelector('iframe[title="streamlit_searchbox.searchbox"]');
    if (!iframe) return;
    if (iframe.contentDocument && iframe.contentDocument.head) {
      injectFont(iframe);
    } else {
      iframe.addEventListener('load', function() { injectFont(iframe); });
    }
  }
  new MutationObserver(scan).observe(parent.document.body, { childList: true, subtree: true });
  scan();
})();
</script>
""", height=0)

    def _movie_search(q: str):
        if not q or len(q) < 2:
            return []
        matches = search_omdb(q)
        return [(f"{m['title']} ({m['year']})", m["imdbID"]) for m in matches]

    _FONT = "'Source Sans Pro', sans-serif"
    selected = st_searchbox(
        _movie_search,
        placeholder="🔍  Search a movie title…",
        key="movie_searchbox",
        clear_on_submit=False,
        style_absolute=True,
        style_overrides={
            "searchbox": {
                "optionEmpty": "hidden",
                "input":       {"fontFamily": _FONT, "fontSize": "1rem"},
                "placeholder": {"fontFamily": _FONT, "fontSize": "1rem"},
                "singleValue": {"fontFamily": _FONT, "fontSize": "1rem"},
                "control":     {"fontFamily": _FONT, "borderRadius": "10px"},
                "menuList":    {"fontFamily": _FONT, "fontSize": "1rem"},
                "option":      {"fontFamily": _FONT, "fontSize": "1rem"},
            }
        },
    )
    st.markdown(
        "<div style='border-bottom:1px solid #1f2937; margin:-1rem 0 1.4rem 0;'></div>",
        unsafe_allow_html=True,
    )
    if selected:
        st.session_state["last_selected_imdb"] = selected

    # ── Results ───────────────────────────────────────────────────────────────
    selected_imdb_id = st.session_state.get("last_selected_imdb")

    if not selected_imdb_id:
        return

    # Dark column boxes for results — only injected when on search tab.
    # Nav columns stay transparent via :has(.nav-btn) override.
    st.markdown("""<style>
div[data-testid="stColumn"] {
  background: #13161f !important;
  border-radius: 13px !important;
  overflow: hidden !important;
}
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"],
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"]:nth-child(2),
div[data-testid="stHorizontalBlock"]:has(.nav-btn) > div[data-testid="stColumn"]:nth-child(3) {
  background: transparent !important;
  border-radius: 0 !important;
  overflow: visible !important;
  padding: 0 0.2rem !important;
}
</style>""", unsafe_allow_html=True)

    _render_movie_analysis(selected_imdb_id, "_cached")


if __name__ == "__main__":
    main()
