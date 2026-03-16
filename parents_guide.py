"""
Parents Guide tagging pipeline.

For each movie across Chai's and Noel's datasets, tags the 5 parental-content
categories (Sex & Nudity, Violence & Gore, Profanity, Alcohol/Drugs, Intensity)
with one of: None / Mild / Moderate / Severe.

Strategy (per movie):
  1. Try scraping the IMDb Parents Guide page — authoritative, community-voted.
  2. If scraping fails, fall back to Gemini (with Groq as secondary fallback).

Output: data/parents_guide.csv
  Columns: Const, Title, sex_nudity, violence_gore, profanity, alcohol_drugs, intensity

Resumable: skips movies that already have all 5 fields populated.
"""
from __future__ import annotations

import json
import time
import re

import requests
import pandas as pd

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("WARNING: beautifulsoup4 not installed — IMDb scraping disabled, using LLM only.")
    print("Run: pip3 install beautifulsoup4")

from config import OMDB_CSV, NOEL_OMDB_CSV, PARENTS_GUIDE_CSV

VALID_RATINGS = {"None", "Mild", "Moderate", "Severe"}
PG_COLS = ["sex_nudity", "violence_gore", "profanity", "alcohol_drugs", "intensity"]

SCRAPE_SLEEP  = 1.5   # seconds between IMDb requests
LLM_SLEEP     = 6.5   # seconds between LLM calls

# IMDb section names → our column names
# IMDb category IDs in parentsGuide JSON → our column names
IMDB_CATEGORY_MAP = {
    "NUDITY":      "sex_nudity",
    "VIOLENCE":    "violence_gore",
    "PROFANITY":   "profanity",
    "ALCOHOL":     "alcohol_drugs",
    "FRIGHTENING": "intensity",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# ─── IMDb scraping ─────────────────────────────────────────────────────────────

def scrape_imdb_parents_guide(imdb_id: str) -> dict[str, str] | None:
    """
    Fetch the IMDb parentsGuide page and parse community-voted severity ratings
    from the embedded Next.js __NEXT_DATA__ JSON blob.

    Returns {col: rating} (e.g. {"sex_nudity": "Mild", ...}) if all 5 found,
    or None if the page is blocked / missing data.
    """
    if not BS4_AVAILABLE:
        return None

    url = f"https://www.imdb.com/title/{imdb_id}/parentalguide"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=12)
        if resp.status_code != 200:
            return None

        soup = BeautifulSoup(resp.text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if not script:
            return None

        data = json.loads(script.string)
        categories = (
            data.get("props", {})
                .get("pageProps", {})
                .get("contentData", {})
                .get("data", {})
                .get("title", {})
                .get("parentsGuide", {})
                .get("categories", [])
        )

        result: dict[str, str] = {}
        for cat in categories:
            cat_id  = (cat.get("category") or {}).get("id", "")
            col     = IMDB_CATEGORY_MAP.get(cat_id)
            rating  = (cat.get("severity") or {}).get("text", "").strip().title()
            if col and rating in VALID_RATINGS:
                result[col] = rating

        return result if len(result) == 5 else None

    except Exception:
        return None


# ─── LLM fallback ──────────────────────────────────────────────────────────────

_PG_DEFINITIONS = """
Sex & Nudity:
  None = No nudity or sexual references.
  Mild = Brief suggestive dialogue, kissing, or non-sexual nudity.
  Moderate = Partial nudity, passionate scenes, or heavy sexual innuendo.
  Severe = Full frontal nudity, graphic sexual acts, or pervasive sexual content.

Violence & Gore:
  None = No physical violence or blood.
  Mild = Slapstick violence, bloodless action hits, or brief scuffles.
  Moderate = Blood visible; stabbings, shootings, or realistic injury.
  Severe = Graphic gore (dismemberment, torture), excessive blood, brutal violence.

Profanity:
  None = No coarse language; G-rated dialogue.
  Mild = Occasional "hell," "damn," or mild epithets.
  Moderate = Frequent moderate swear words or a handful of F-bombs.
  Severe = Constant, pervasive strong profanity or offensive slurs.

Alcohol/Drugs:
  None = No substance use shown or mentioned.
  Mild = Social drinking or background smoking.
  Moderate = Characters shown intoxicated or using soft drugs (marijuana).
  Severe = Graphic drug use, glamorized addiction, or pervasive substance abuse.

Intensity:
  None = No scary elements or high-stress sequences.
  Mild = Brief suspenseful moments or mild spooky imagery.
  Moderate = Frequent jumpscares, intense peril, or emotionally heavy tear-jerker scenes.
  Severe = Extreme psychological terror, disturbing imagery, or deeply dark/depressing themes.
"""


def _build_pg_prompt(rec: dict) -> str:
    title   = rec.get("Title", "?")
    year    = rec.get("Year", "?")
    genre   = rec.get("Genre", "?")
    plot    = (rec.get("Plot", "") or "")[:500]
    rated   = rec.get("Rated", "?")
    director = rec.get("Director", "?")
    actors  = rec.get("Actors", "?")

    return (
        f"You are a parents-guide expert. Rate this movie on 5 content categories.\n\n"
        f"Movie: {title} ({year})\nGenre: {genre}\nMPAA Rating: {rated}\n"
        f"Director: {director}\nCast: {actors}\nPlot: {plot}\n\n"
        f"Definitions:\n{_PG_DEFINITIONS}\n"
        f"Return ONLY valid JSON with exactly these 5 keys:\n"
        f"  sex_nudity, violence_gore, profanity, alcohol_drugs, intensity\n"
        f"Each value must be exactly one of: None, Mild, Moderate, Severe\n"
        f"Example: {{\"sex_nudity\":\"Mild\",\"violence_gore\":\"Severe\","
        f"\"profanity\":\"Moderate\",\"alcohol_drugs\":\"None\",\"intensity\":\"Moderate\"}}"
    )


def _validate_pg(raw: dict) -> dict[str, str] | None:
    """Return dict only if all 5 keys are present and valid."""
    result = {}
    for col in PG_COLS:
        val = raw.get(col, "").strip().title()
        if val not in VALID_RATINGS:
            return None
        result[col] = val
    return result


_GEMINI_MODELS = [
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
    "gemini-2.0-flash",
]


def _llm_pg_gemini(rec: dict, max_retries: int = 5) -> dict[str, str] | None:
    try:
        from google import genai
        from google.genai import types
        from config import GEMINI_API_KEY
    except ImportError:
        return None

    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = _build_pg_prompt(rec)
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
                        max_output_tokens=200,
                        thinking_config=types.ThinkingConfig(thinking_budget=0),
                    ),
                )
                raw_text = (resp.text or "").strip()
                if raw_text.startswith("```"):
                    raw_text = raw_text.split("```")[1]
                    if raw_text.startswith("json"):
                        raw_text = raw_text[4:]
                parsed = json.loads(raw_text)
                result = _validate_pg(parsed)
                if result:
                    return result
                return None

            except Exception as e:
                err = str(e)
                if "429" in err:
                    if "PerDay" in err or "per_day" in err.lower():
                        print(f"  [gemini-pg] Daily quota for {model}, trying next model")
                        break
                    m = re.search(r'retry[^\d]*(\d+(?:\.\d+)?)\s*s', err, re.IGNORECASE)
                    delay = float(m.group(1)) + 2 if m else 60.0
                    print(f"  [gemini-pg] Rate-limited on {model}, waiting {delay:.0f}s… ({title})")
                    time.sleep(delay)
                    attempts += 1
                else:
                    print(f"  [gemini-pg] ERROR for '{title}': {err[:120]}")
                    return None
    return None


def _llm_pg_groq(rec: dict) -> dict[str, str] | None:
    try:
        from groq import Groq
        from config import GROQ_API_KEY
    except ImportError:
        return None

    client = Groq(api_key=GROQ_API_KEY)
    prompt = _build_pg_prompt(rec)
    try:
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            response_format={"type": "json_object"},
        )
        parsed = json.loads(resp.choices[0].message.content)
        return _validate_pg(parsed)
    except Exception as e:
        print(f"  [groq-pg] ERROR for '{rec.get('Title','?')}': {e}")
        return None


def tag_movie_pg(rec: dict) -> tuple[dict[str, str] | None, str]:
    """
    Returns (ratings_dict, source) where source is 'imdb', 'gemini', 'groq', or 'failed'.
    """
    imdb_id = rec.get("Const", "")

    # 1. Try IMDb scrape
    if imdb_id:
        result = scrape_imdb_parents_guide(imdb_id)
        if result:
            return result, "imdb"

    # 2. Try Gemini LLM
    result = _llm_pg_gemini(rec)
    if result:
        return result, "gemini"

    # 3. Try Groq LLM
    result = _llm_pg_groq(rec)
    if result:
        return result, "groq"

    return None, "failed"


# ─── Main pipeline ─────────────────────────────────────────────────────────────

def load_all_movies() -> pd.DataFrame:
    """Combine Chai + Noel enriched CSVs, deduplicate by Const."""
    chai = pd.read_csv(OMDB_CSV)
    noel = pd.read_csv(NOEL_OMDB_CSV)
    all_df = pd.concat([chai, noel], ignore_index=True).drop_duplicates(subset=["Const"])
    return all_df.reset_index(drop=True)


def load_existing() -> tuple[pd.DataFrame, set]:
    """Load existing parents_guide.csv; return (df, done_ids)."""
    import os
    if not os.path.exists(PARENTS_GUIDE_CSV):
        return pd.DataFrame(), set()
    df = pd.read_csv(PARENTS_GUIDE_CSV)
    # A row is "done" if all 5 PG columns are filled with valid values
    complete_mask = df[PG_COLS].apply(lambda col: col.isin(VALID_RATINGS)).all(axis=1)
    done = set(df.loc[complete_mask, "Const"].tolist())
    return df[complete_mask].reset_index(drop=True), done


def save(existing_df: pd.DataFrame, new_rows: list[dict]) -> pd.DataFrame:
    if not new_rows:
        return existing_df
    new_df = pd.DataFrame(new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["Const"])
    combined.to_csv(PARENTS_GUIDE_CSV, index=False)
    return combined


def main():
    movies   = load_all_movies()
    existing, done_ids = load_existing()

    pending = movies[~movies["Const"].isin(done_ids)].reset_index(drop=True)
    total   = len(pending)

    print(f"Total unique movies: {len(movies)}")
    print(f"Already tagged:      {len(done_ids)}")
    print(f"Pending:             {total}")

    if total == 0:
        print("All movies already tagged!")
        return

    source_counts = {"imdb": 0, "gemini": 0, "groq": 0, "failed": 0}
    new_rows: list[dict] = []
    last_was_llm = False

    for i, (_, row) in enumerate(pending.iterrows()):
        rec    = row.to_dict()
        const  = rec["Const"]
        title  = rec.get("Title", "?")

        ratings, source = tag_movie_pg(rec)
        source_counts[source] += 1

        if ratings:
            new_rows.append({"Const": const, "Title": title, **ratings})
            tag_str = " | ".join(f"{k}={v}" for k, v in ratings.items())
        else:
            # Store nulls so we can skip on resume but mark as attempted
            new_rows.append({"Const": const, "Title": title,
                             **{col: None for col in PG_COLS}})
            tag_str = "FAILED"

        print(f"[{i+1}/{total}] [{source}] {title} → {tag_str}")

        # Incremental save every 50
        if (i + 1) % 50 == 0 or i == total - 1:
            existing = save(existing, new_rows)
            new_rows = []
            print(f"  → Saved. Sources so far: {source_counts}")

        # Rate limiting — only sleep if we used IMDb scrape or LLM
        if source == "imdb":
            time.sleep(SCRAPE_SLEEP)
        else:
            if i < total - 1:
                time.sleep(LLM_SLEEP)

    print(f"\nDone! Results saved to {PARENTS_GUIDE_CSV}")
    print(f"Sources: IMDb={source_counts['imdb']}, Gemini={source_counts['gemini']}, "
          f"Groq={source_counts['groq']}, Failed={source_counts['failed']}")


if __name__ == "__main__":
    main()
