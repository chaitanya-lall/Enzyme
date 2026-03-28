"""
ui_components.py — Reusable UI components and helper functions for the Enzyme app.

Exports:
  score_color, score_verdict, build_gauge
  _preference_stats, build_why_prompt, stream_why_narrative
  load_poster, load_poster_b64
  SIMILARITY_THRESHOLD, _SIM_DISPLAY_CEIL, get_closest_matches
  preload_artifacts, preload_artifacts_noel
  load_parents_guide
  render_meter_column
  _render_movie_analysis
  catalog_movie_detail  (@st.dialog)
  _mobile_filters_panel
  _render_catalog_card
  _ids_from_seen_numbers
  _load_catalog
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import warnings

import numpy as np
import pandas as pd
import requests
import streamlit as st
from groq import Groq
from io import BytesIO
from PIL import Image

from predict import predict_movie, _load_all, find_similar_movie_combined
from predict_noel import predict_movie_noel, _load_all as _load_all_noel
from config import PARENTS_GUIDE_CSV
from tag_features import tag_col_name
from catalog_sync import CATALOG_PATH

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHAI_SEEN_FILE = os.path.join(os.path.expanduser("~"), "Documents", "Chai Seen.numbers")
NOEL_SEEN_FILE = os.path.join(os.path.expanduser("~"), "Documents", "Noel Seen.numbers")


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

def _preference_stats(train_meta, rec: dict) -> dict:
    """Compute concrete preference stats from a person's rating history."""
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
  will-change: transform !important;
  animation: sheetUp 0.35s cubic-bezier(0.32, 0.72, 0, 1) !important;
}
@keyframes sheetUp {
  from { transform: translateY(100%); }
  to   { transform: translateY(0); }
}
/* d1: Streamlit's backdrop wrapper — make transparent + block layout so d2 starts at dialog top */
[data-testid="stDialog"] > div {
  background: transparent !important;
  padding: 0 !important;
  display: block !important;
}
/* d2: inner dialog box — flush with outer shell, no visual separation */
[data-testid="stDialog"] > div > div {
  background: #0a0b0f !important;
  border: none !important;
  box-shadow: none !important;
  border-radius: 0 !important;
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 !important;
  padding: 0 !important;
}
/* d3 children: uniform 6px padding; first-child is the CSS-injection div — collapse it to zero */
[data-testid="stDialog"] > div > div > div {
  padding: 6px !important;
  width: 100% !important;
  max-width: 100% !important;
  box-sizing: border-box !important;
}
[data-testid="stDialog"] > div > div > div:first-child {
  padding: 0 !important;
  height: 0 !important;
  overflow: hidden !important;
}
/* Pull content up to absorb Streamlit's ~36px internal stVerticalBlock gap before the columns */
[data-testid="stDialog"] > div > div > div:not(:first-child) {
  margin-top: -36px !important;
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


def _mobile_filters_panel():
    """Mobile-only inline filter+sort panel (uses f_mob_* keys to avoid ID conflicts).
    Rendered directly in page flow — no dialog/fragment isolation — so filter state
    is reliably read by the catalog filter logic on the same rerun.
    """
    with st.container(key="mob-filter-panel"):
        # ── Top bar: title + X close button ──────────────────────────────────
        _th, _tc = st.columns([6, 1])
        with _th:
            st.markdown(
                "<h3 style='margin:0.1rem 0 0 0; font-size:1.05rem; font-weight:700;"
                "color:#e2e8f0;'>Filter & Sort</h3>",
                unsafe_allow_html=True,
            )
        with _tc:
            if st.button("✕", key="f_mob_close", use_container_width=True):
                st.session_state["_mob_filters_open"] = False
                st.rerun()

        # ── Sort (always visible) ─────────────────────────────────────────────
        st.selectbox(
            "Sort",
            ["Compatibility", "Chai Score", "Noel Score", "IMDb Score", "Newest First"],
            index=None,
            placeholder="Sort: Compatibility (default)",
            key="f_sort_mob",
        )

        # ── Collapsible filter sections ───────────────────────────────────────
        with st.expander("Services"):
            st.checkbox("Netflix",    key="f_mob_svc_netflix")
            st.checkbox("Max",        key="f_mob_svc_max")
            st.checkbox("Disney+",    key="f_mob_svc_disney")
            st.checkbox("Hulu",       key="f_mob_svc_hulu")
            st.checkbox("Apple TV+",  key="f_mob_svc_apple")
            st.checkbox("Peacock",    key="f_mob_svc_peacock")
            st.checkbox("Paramount+", key="f_mob_svc_paramount")

        with st.expander("Content Type"):
            st.checkbox("Movies",   key="f_mob_type_movies")
            st.checkbox("TV Shows", key="f_mob_type_tv")

        with st.expander("Chai Watch Status"):
            st.checkbox("Seen",     key="f_mob_w_chai_seen")
            st.checkbox("Not Seen", key="f_mob_w_chai_not_seen")

        with st.expander("Noel Watch Status"):
            st.checkbox("Seen",     key="f_mob_w_noel_seen")
            st.checkbox("Not Seen", key="f_mob_w_noel_not_seen")

        with st.expander("IMDb Score"):
            st.slider("Min IMDb", 0.0, 10.0, value=0.0, step=0.5, format="%.1f", key="f_mob_imdb")

        with st.expander("Release Year"):
            st.slider("Year", 1900, 2026, value=(1950, 2026), key="f_mob_yr")

        # ── Apply ─────────────────────────────────────────────────────────────
        if st.button("Apply", key="f_mob_apply", type="primary", use_container_width=True):
            st.session_state["_mob_filters_open"] = False
            st.rerun()


def _render_catalog_card(item) -> None:
    """Render one catalog card (HTML + a Details button)."""
    poster_url = str(item.get("poster_url") or "")
    title      = str(item.get("title") or "Unknown")
    year       = item.get("year")
    imdb       = item.get("imdb_score")
    chai_pct   = float(item.get("chai_pct") or 0)
    noel_pct   = float(item.get("noel_pct") or 0)
    service    = str(item.get("service") or "").lower()

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
      <span style='font-size:0.75rem; font-weight:700; color:{score_color(chai_pct)};'>Chai {f"{chai_pct:.1f}%" if chai_pct % 1 != 0 else f"{int(chai_pct)}%"}</span>
      <span style='font-size:0.75rem; font-weight:700; color:{score_color(noel_pct)};'>Noel {f"{noel_pct:.1f}%" if noel_pct % 1 != 0 else f"{int(noel_pct)}%"}</span>
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


@st.cache_data(show_spinner=False, ttl=300)
def _load_catalog() -> "pd.DataFrame":
    df = pd.read_parquet(CATALOG_PATH)

    # Load seen IDs: prefer local .numbers files (laptop), fall back to
    # data/seen_ids.json (pushed to GitHub so production has the data).
    _seen_json = os.path.join(_APP_DIR, "data", "seen_ids.json")
    if os.path.exists(CHAI_SEEN_FILE) or os.path.exists(NOEL_SEEN_FILE):
        chai_ids = _ids_from_seen_numbers(CHAI_SEEN_FILE)
        noel_ids = _ids_from_seen_numbers(NOEL_SEEN_FILE)
    elif os.path.exists(_seen_json):
        with open(_seen_json) as _f:
            _seen = json.load(_f)
        chai_ids = set(_seen.get("chai_seen", []))
        noel_ids = set(_seen.get("noel_seen", []))
    else:
        chai_ids, noel_ids = set(), set()

    df["chai_seen"] = df["imdb_id"].isin(chai_ids)
    df["noel_seen"] = df["imdb_id"].isin(noel_ids)
    return df
