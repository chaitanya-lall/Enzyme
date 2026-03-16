"""
Shared helpers for LLM categorical tag encoding and inference.
Used by: tag_movies.py (batch), predict.py (inference), training scripts.
"""
from __future__ import annotations

import json
import time

from config import TAG_TAXONOMY


def tag_col_name(category: str, value: str) -> str:
    """Convert (category, value) → a safe DataFrame column name."""
    safe = value.replace("&", "and").replace("/", "_").replace(" ", "_").replace("-", "_")
    return f"tag_{category}_{safe}"


# All one-hot column names, in deterministic order
ALL_TAG_COLS: list[str] = [
    tag_col_name(cat, val)
    for cat, vals in TAG_TAXONOMY.items()
    for val in vals
]

# Reverse map: col_name → (category, display_value)
_COL_TO_TAG: dict[str, tuple[str, str]] = {
    tag_col_name(cat, val): (cat, val)
    for cat, vals in TAG_TAXONOMY.items()
    for val in vals
}

# Human-readable category labels: "social_context" → "Social Context"
_CAT_LABELS = {cat: cat.replace("_", " ").title() for cat in TAG_TAXONOMY}


def tag_display_label(feat: str) -> str | None:
    """
    Convert a tag column name to a human-readable label.
    e.g. 'tag_pacing_Slow_Burn_Atmospheric' → 'Pacing: Slow Burn/Atmospheric'
    Returns None if the feature is not a tag column.
    """
    if feat in _COL_TO_TAG:
        cat, val = _COL_TO_TAG[feat]
        return f"{_CAT_LABELS[cat]}: {val}"
    return None


def compute_tag_interactions(
    tag_df: "pd.DataFrame",
    selected_tags: list | None = None,
    top_k: int = 10,
    min_support: int = 25,
) -> tuple["pd.DataFrame", list]:
    """
    Compute pairwise AND interaction features for the most discriminative tag columns.

    Training mode  (selected_tags=None):
        Picks the top_k tags whose frequency falls in the 10–80% range
        (discriminative, not too rare or too universal), then only keeps
        pairs that co-occur on at least min_support movies.
        Returns (interaction_df, selected_tags).

    Inference mode (selected_tags provided):
        Uses exactly the saved tag list so columns always match training.
        Returns (interaction_df, selected_tags).

    With top_k=10 this considers up to 10×9/2 = 45 pairs, filtered by
    min_support, producing a compact set of high-quality interaction columns
    named ix__{cat_A}__{cat_B} (leading "tag_" stripped for brevity).
    """
    import pandas as pd  # local to avoid circular at module level

    if selected_tags is None:
        freqs = tag_df.reindex(columns=ALL_TAG_COLS, fill_value=0).mean()
        discriminative = freqs[(freqs >= 0.10) & (freqs <= 0.80)]
        candidates = discriminative.nlargest(top_k).index.tolist()

        # Only keep pairs with enough co-occurrences
        kept: list = []
        for i, col_a in enumerate(candidates):
            a = tag_df[col_a].values if col_a in tag_df.columns else 0
            for col_b in candidates[i + 1:]:
                b = tag_df[col_b].values if col_b in tag_df.columns else 0
                if (a * b).sum() >= min_support:
                    if col_a not in kept:
                        kept.append(col_a)
                    if col_b not in kept:
                        kept.append(col_b)
        selected_tags = kept

    cols: dict = {}
    for i, col_a in enumerate(selected_tags):
        a = tag_df[col_a].values if col_a in tag_df.columns else 0
        for col_b in selected_tags[i + 1:]:
            b = tag_df[col_b].values if col_b in tag_df.columns else 0
            name = f"ix__{col_a[4:]}__{col_b[4:]}"   # strip leading "tag_"
            cols[name] = a * b

    ix_df = pd.DataFrame(cols, index=tag_df.index)
    return ix_df, selected_tags


def encode_tags(tags_dict: dict) -> dict[str, int]:
    """
    Convert {category: value_or_list} → {col_name: 0 or 1} for every tag column.
    Supports up to 2 values per category (list or single string).
    Unknown categories/values are silently ignored.
    """
    row: dict[str, int] = {col: 0 for col in ALL_TAG_COLS}
    for category, value in tags_dict.items():
        if category not in TAG_TAXONOMY:
            continue
        # Handle both single string and list of strings
        values = value if isinstance(value, list) else [value]
        for v in values[:2]:  # cap at 2 per category
            if v in TAG_TAXONOMY[category]:
                row[tag_col_name(category, v)] = 1
    return row


def _build_tagger_prompt(rec: dict) -> str:
    taxonomy_lines = "\n".join(
        f'- {cat}: {", ".join(vals)}'
        for cat, vals in TAG_TAXONOMY.items()
    )
    title    = rec.get("Title", "?")
    year     = rec.get("Year", "?")
    genre    = rec.get("Genre", "?")
    director = rec.get("Director", "?")
    plot     = (rec.get("Plot", "") or "")[:400]
    return (
        f"Classify this movie into each of the 7 categories below.\n"
        f"Pick 1-2 values per category from the allowed options. Return JSON only.\n\n"
        f"Movie: {title} ({year})\nGenre: {genre}\nDirector: {director}\nPlot: {plot}\n\n"
        f"Categories and allowed values:\n{taxonomy_lines}\n\n"
        f"Return JSON with exactly these 7 keys: social_context, pacing, cognitive_load, social_value, vibes, narrative_resolve, tension_profile.\n"
        f"Each value must be a JSON array of 1-2 strings chosen exactly from the allowed list."
    )


def _validate_tags(tags: dict) -> dict[str, list[str]]:
    """Keep only values that exactly match the taxonomy, normalised to list, capped at 2."""
    validated: dict[str, list[str]] = {}
    for cat, val in tags.items():
        if cat not in TAG_TAXONOMY:
            continue
        values = val if isinstance(val, list) else [val]
        good = [v for v in values if v in TAG_TAXONOMY[cat]][:2]
        if good:
            validated[cat] = good
    return validated


def _extract_retry_delay(err_str: str, default: float = 60.0) -> float:
    """Pull the retry delay in seconds from a Groq/Gemini 429 error message."""
    import re
    m = re.search(r'retry[^\d]*(\d+(?:\.\d+)?)\s*s', str(err_str), re.IGNORECASE)
    return float(m.group(1)) + 2 if m else default


# Ordered list of Gemini models to try, from cheapest to most capable
_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
    "gemini-3.1-flash-lite-preview",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
    "gemini-3.1-pro-preview",
]


def call_gemini_tagger(rec: dict, max_retries: int = 6) -> dict[str, list[str]]:
    """
    Use Gemini to classify a movie into the 7 tag categories.
    On rate-limit (429) waits the requested retry delay and tries again.
    Cycles through model variants if a model's daily quota is exhausted.
    Returns {category: [value1, value2]} with 1-2 values per category.
    """
    from google import genai
    from google.genai import types
    from config import GEMINI_API_KEY

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_tagger_prompt(rec)
    title  = rec.get("Title", "?")

    for model in _GEMINI_MODELS:
        attempts = 0
        while attempts < max_retries:
            try:
                resp = client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        max_output_tokens=800,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                raw = resp.text or ""
                if not raw.strip():
                    raise ValueError("Empty response")
                # Strip markdown code fences if present
                raw = raw.strip()
                if raw.startswith("```"):
                    raw = raw.split("```")[1]
                    if raw.startswith("json"):
                        raw = raw[4:]
                tags = json.loads(raw)
                return _validate_tags(tags)

            except Exception as e:
                err = str(e)
                if "429" in err:
                    # Check if it's a daily limit (no point retrying this model today)
                    if "PerDay" in err or "per_day" in err.lower():
                        print(f"  [gemini] Daily quota exhausted for {model}, trying next model")
                        break  # move to next model
                    delay = _extract_retry_delay(err)
                    print(f"  [gemini] Rate-limited on {model}, waiting {delay:.0f}s… ({title})")
                    time.sleep(delay)
                    attempts += 1
                else:
                    print(f"  [gemini-tagger] ERROR for '{title}': {err[:120]}")
                    return {}

    print(f"  [gemini-tagger] All models exhausted for '{title}'")
    return {}


def call_groq_tagger(rec: dict) -> dict[str, list[str]]:
    """
    Call Groq (llama-3.1-8b-instant) to classify a movie into the 5 tag categories.
    Returns {category: [value1, value2]} with 1-2 values per category.
    Returns {} on error.
    """
    from groq import Groq
    from config import GROQ_API_KEY

    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_tagger_prompt(rec)

    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        raw = resp.choices[0].message.content
        tags = json.loads(raw)
        return _validate_tags(tags)

    except Exception as e:
        print(f"  [tagger] ERROR for '{rec.get('Title','?')}': {e}")
        return {}
