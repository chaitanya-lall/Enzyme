"""
catalog_sync.py — Background catalog refresh for the Enzyme app.

Usage (from app.py):
    from catalog_sync import start_background_sync, get_sync_status, catalog_age_days, CATALOG_PATH

Behaviour:
  - On app launch, checks the catalog parquet's modification time.
  - If the file is >7 days old (or missing), starts a background thread to reseed it.
  - The UI shows stale data immediately; the new catalog is hot-swapped once ready.
"""
from __future__ import annotations

import os
import threading
import time
from datetime import datetime
from pathlib import Path

BASE_DIR = Path(__file__).parent
CATALOG_PATH = str(BASE_DIR / "data" / "catalog_data.parquet")
SYNC_MAX_AGE_DAYS = 7

_lock = threading.Lock()
_state: dict = {
    "running":     False,
    "finished":    False,
    "error":       None,
    "started_at":  None,
    "finished_at": None,
    "new_count":   0,
}


def catalog_age_days() -> float | None:
    """Return age of catalog parquet in days, or None if it doesn't exist."""
    if not os.path.exists(CATALOG_PATH):
        return None
    return (time.time() - os.path.getmtime(CATALOG_PATH)) / 86400


def needs_refresh() -> bool:
    age = catalog_age_days()
    return age is None or age > SYNC_MAX_AGE_DAYS


def get_sync_status() -> dict:
    with _lock:
        return dict(_state)


def _has_watchmode_key() -> bool:
    for k in ("WATCHMODE_API_KEY", "WATCHMODE_API_KEY_2", "WATCHMODE_API_KEY_3"):
        if os.environ.get(k):
            return True
    secrets_path = BASE_DIR / ".streamlit" / "secrets.toml"
    if not secrets_path.exists():
        return False
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib  # type: ignore
        with open(secrets_path, "rb") as f:
            s = tomllib.load(f)
        return any(s.get(k) for k in ("WATCHMODE_API_KEY", "WATCHMODE_API_KEY_2", "WATCHMODE_API_KEY_3"))
    except Exception:
        return False


def _run_sync() -> None:
    global _state
    with _lock:
        _state.update({"running": True, "started_at": datetime.now().isoformat(),
                        "error": None, "finished": False})
    try:
        from catalog_seed import run_seed
        df = run_seed(skip_streaming_urls=True)
        with _lock:
            _state["new_count"] = len(df) if df is not None else 0
    except Exception as e:
        with _lock:
            _state["error"] = str(e)
    finally:
        with _lock:
            _state.update({"running": False, "finished": True,
                            "finished_at": datetime.now().isoformat()})


def start_background_sync() -> bool:
    """
    Start a background catalog refresh if the catalog is stale and a
    Watchmode API key is available. Returns True if a thread was started.
    """
    with _lock:
        if _state["running"]:
            return False
    if not needs_refresh():
        return False
    if not _has_watchmode_key():
        return False
    t = threading.Thread(target=_run_sync, daemon=True, name="catalog-sync")
    t.start()
    return True
