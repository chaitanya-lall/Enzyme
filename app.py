"""
The Personalized Critic — Streamlit UI
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

from predict import predict_movie, _load_all, find_similar_movie_combined
from predict_noel import predict_movie_noel, _load_all as _load_all_noel
from config import GROQ_API_KEY, TAG_TAXONOMY, PARENTS_GUIDE_CSV
from tag_features import tag_col_name

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

/* Search button */
div[data-testid="stButton"] > button {
  background-color: #4f8ef7 !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 10px !important;
  font-size: 0.95rem !important;
  font-weight: 600 !important;
  height: 42px !important;
  width: 100% !important;
  cursor: pointer !important;
  margin-top: 0 !important;
}
div[data-testid="stButton"] > button:hover {
  background-color: #3a7ae0 !important;
}

/* Row container — gap between the three cards */
div[data-testid="stHorizontalBlock"] {
  gap: 0.75rem !important;
  align-items: flex-start !important;
  background: transparent !important;
}

/* All three column cards — same dark box (correct selector: stColumn) */
div[data-testid="stColumn"] {
  background: #13161f !important;
  border-radius: 13px !important;
  overflow: hidden !important;
}

/* Match columns (2nd and 3rd) — padding inside the card */
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(2),
div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:nth-child(3) {
  padding: 1.4rem !important;
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

    return f"""Write a short explanation of what {person} will likely experience watching "{title}" ({year}).

Data: top drivers = {feature_list}; genre = {genre}; director = {director}; IMDb = {imdb}; synopsis = {plot[:200]}; most similar rated film = "{similar['title']}" ({similar['rating']}/10).

Rules:
- {person} has NOT seen this film.
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
    if not GROQ_API_KEY:
        yield "*(Set GROQ_API_KEY in config.py to enable AI explanations.)*"
        return

    prompt = build_why_prompt(rec, pred_score, match_pct, top_pos, top_neg,
                               similar, vibe, person, train_meta)
    try:
        client = Groq(api_key=GROQ_API_KEY)
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
    df = pd.read_csv(PARENTS_GUIDE_CSV)
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
                        train_meta=None):
    """Render a Match column (Figma card style)."""
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


# ─── Main app ─────────────────────────────────────────────────────────────────

def main():
    preload_artifacts()
    preload_artifacts_noel()

    _LOGO_B64_HDR = open(
        os.path.join(os.path.dirname(__file__), "assets", "logo_b64.txt")
    ).read().strip()

    st.markdown(
        f"""<div style='display:flex; align-items:center; gap:0.65rem; 
                        padding:0.5rem 0 0.75rem 0;'>
  <img src='data:image/png;base64,{_LOGO_B64_HDR}'
       style='height:42px; width:auto; opacity:0.95;'/>
  <div>
    <div style='font-size:1.89rem; font-weight:900; letter-spacing:0.30em;
                text-transform:uppercase; color:#ffffff; line-height:1;'>ENZYME</div>
    <div style='font-size:0.88rem; color:#6b7280; letter-spacing:0.10em;
                font-weight:400; margin-top:2px;'>Movies, broken down for you.</div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )

    col_input, col_btn = st.columns([5, 1])
    with col_input:
        query = st.text_input(
            label="search",
            label_visibility="collapsed",
            placeholder="🔍  Search a movie title…",
            key="movie_search",
        )
    with col_btn:
        st.markdown("<div style='height:27px'></div>", unsafe_allow_html=True)
        st.button("Search", key="search_btn", use_container_width=True)

    st.markdown(
        "<div style='border-bottom:1px solid #1f2937; margin:0.4rem 0 1.4rem 0;'></div>",
        unsafe_allow_html=True,
    )

    if not query:
        return

    with st.spinner(f'Analyzing "{query}"…'):
        result = predict_movie(query)

    if result is None:
        st.error(
            f'Could not find "{query}" on OMDb. '
            "Check the spelling or the daily limit may have been reached."
        )
        return

    rec = result["rec"]
    tags_dict = result.get("tags_dict", {})

    # Attach tags_dict to rec so _preference_stats() can access them
    rec["tags_dict"] = tags_dict

    # Noel's prediction reuses same OMDb record and same tags — no extra API calls
    noel_result = predict_movie_noel(rec, tags_dict=tags_dict)

    chai_result = {
        "pred_score": result["pred_score"],
        "match_pct":  result["match_pct"],
        "tags":       result["tags"],
        "similar":    result["similar"],
        "vibe":       result["vibe"],
        "top_pos":    result["top_pos"],
        "top_neg":    result["top_neg"],
    }

    # Find each person's closest match (>=80% combined similarity)
    chai_artifacts = preload_artifacts()
    noel_artifacts = preload_artifacts_noel()
    chai_anchor, noel_anchor = get_closest_matches(
        embedding    = result["embedding"],
        tags_dict    = tags_dict,
        imdb_rating  = float(rec.get("imdbRating") or 5.0),
        movie_const  = None,
        chai_artifacts = chai_artifacts,
        noel_artifacts = noel_artifacts,
    )

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
            actors = ", ".join(rec["Actors"].split(", ")[:3])
            cast_html = (
                f"<p style='margin:0.35rem 0; font-size:0.86rem; color:#9ca3af;'>"
                f"<span style='color:#e2e8f0; font-weight:600;'>Cast</span><br>"
                f"{actors}</p>"
            )

        # Critics
        crit_items = []
        if rec.get("imdbRating"):
            crit_items.append(("IMDb", f"{rec['imdbRating']}/10"))
        if rec.get("RT_score"):
            crit_items.append(("RT Critic", f"{int(rec['RT_score'])}%"))
        if rec.get("Metascore") and str(rec["Metascore"]) != "nan":
            crit_items.append(("Metacritic", f"{int(rec['Metascore'])}/100"))
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
                f"margin:1rem 0 0.4rem 0;'>Critics &amp; Audience</div>"
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

        # Parents Guide badges
        pg_html = ""
        pg_data = load_parents_guide()
        imdb_id = rec.get("imdbID", "")
        if imdb_id and imdb_id in pg_data:
            PG_CATEGORIES = [
                ("sex_nudity",    "Sex & Nudity"),
                ("violence_gore", "Violence & Gore"),
                ("profanity",     "Profanity"),
                ("alcohol_drugs", "Alcohol & Drugs"),
                ("intensity",     "Intensity"),
            ]
            PG_COLORS = {
                "None":     ("#1a3a1a", "#4ade80"),
                "Mild":     ("#3a3010", "#facc15"),
                "Moderate": ("#3a1f00", "#fb923c"),
                "Severe":   ("#3a0a0a", "#f87171"),
            }
            rows = ""
            pg_rec = pg_data[imdb_id]
            for col_key, cat_label in PG_CATEGORIES:
                rating = pg_rec.get(col_key, "")
                if rating not in PG_COLORS:
                    continue
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

        # Assemble full card
        movie_card = f"""
<div style='background:#13161f; border-radius:13px; overflow:visible;
            margin-bottom:0;'>
  {poster_html}
  <div style='padding:1rem 1.2rem 1.4rem 1.2rem;'>
    <h2 style='font-size:1.5rem; font-weight:800; color:#ffffff; 
               margin:0 0 0.5rem 0; line-height:1.2;'>{rec['Title']}</h2>
    <div style='margin-bottom:0.25rem;'>{chips_html}</div>
    {dir_html}
    {cast_html}
    {critics_html}
    {profile_html}
    {pg_html}
  </div>
</div>
"""
        st.markdown(movie_card, unsafe_allow_html=True)

    with col_chai:
        render_meter_column(rec, chai_result, "Chai", anchor=chai_anchor,
                            train_meta=chai_artifacts["train_meta"])

    with col_noel:
        render_meter_column(rec, noel_result, "Noel", anchor=noel_anchor,
                            train_meta=noel_artifacts["train_meta"])


if __name__ == "__main__":
    main()
