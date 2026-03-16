"""
Rotten Tomatoes page scraper: top-5 cast + production company.
Used by 2b_enrich_rt.py (batch) and predict.py (live inference).
No API key required — parses RT JSON-LD + HTML.
"""
from __future__ import annotations

import json
import re
import time
import requests
from bs4 import BeautifulSoup

RT_BASE  = "https://www.rottentomatoes.com/m/"
RT_SLEEP = 0.5   # polite delay between requests (seconds)

_SESSION = requests.Session()
_SESSION.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer":         "https://www.rottentomatoes.com/",
})


def make_rt_slug(title: str) -> str:
    """Convert movie title → RT URL slug."""
    s = title.lower()
    s = re.sub(r"['''\u2019]", "", s)      # strip apostrophes
    s = re.sub(r"[^a-z0-9 ]", "_", s)     # non-alphanum → _
    s = re.sub(r"\s+", "_", s.strip())
    s = re.sub(r"_+", "_", s)
    return s.rstrip("_")


def _parse_page(html: str) -> dict:
    """Parse RT movie page → {cast: [...], studio: str|None, year: int|None}."""
    soup   = BeautifulSoup(html, "html.parser")
    result: dict = {"cast": [], "studio": None, "year": None}

    # 1. JSON-LD: actors + year
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            d = json.loads(tag.string or "")
            actors = d.get("actor", [])
            if actors:
                result["cast"] = [a["name"] for a in actors if a.get("name")]
            for key in ("datePublished", "dateCreated"):
                val = str(d.get(key, ""))
                m = re.search(r"(\d{4})", val)
                if m:
                    result["year"] = int(m.group(1))
                    break
        except Exception:
            pass

    # 2. Production Co from <dt>Production Co</dt><dd>…</dd>
    for dt in soup.find_all("dt"):
        if "production co" in dt.get_text(strip=True).lower():
            dd = dt.find_next_sibling("dd")
            if dd:
                raw = re.split(r"[\n,]", dd.get_text(strip=True))[0].strip()
                if raw:
                    result["studio"] = raw
            break

    return result


def fetch_rt_data(title: str, year: int, sleep: bool = True) -> dict | None:
    """
    Fetch RT cast (top 5) + studio for a movie.

    Strategy:
      1. Try plain slug → validate JSON-LD year (±1 tolerance)
      2. If mismatch or miss, try slug_YEAR
      3. Return None if both fail or no cast found.

    Returns: {"cast_top5": [...], "studio": str|None}  or  None
    """
    slug_plain = make_rt_slug(title)
    slug_year  = f"{slug_plain}_{year}"

    for slug in (slug_plain, slug_year):
        try:
            resp = _SESSION.get(f"{RT_BASE}{slug}", timeout=12, allow_redirects=True)
            if sleep:
                time.sleep(RT_SLEEP)
            if resp.status_code != 200:
                continue
            parsed = _parse_page(resp.text)
            # Skip if we landed on the wrong movie (year off by > 1)
            rt_year = parsed.get("year")
            if rt_year is not None and abs(rt_year - year) > 1:
                continue
            if parsed["cast"]:
                return {
                    "cast_top5": parsed["cast"][:5],
                    "studio":    parsed["studio"],
                }
        except Exception:
            if sleep:
                time.sleep(RT_SLEEP)

    return None
