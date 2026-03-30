"""
ui_catalog.py — Recommend tab: filter bar, catalog grid, load more.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as _components

from catalog_sync import get_sync_status, catalog_age_days, CATALOG_PATH
from ui_components import _render_catalog_card, _mobile_filters_panel, _load_catalog


def render_recommend_tab() -> None:
    """Render the 🍿 Recommend tab content."""
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
    # Each filter ORs the desktop key with the mobile-dialog key so both UIs work
    _svc_netflix   = st.session_state.get("f_svc_netflix",  False) or st.session_state.get("f_mob_svc_netflix",  False)
    _svc_max       = st.session_state.get("f_svc_max",      False) or st.session_state.get("f_mob_svc_max",      False)
    _svc_disney    = st.session_state.get("f_svc_disney",   False) or st.session_state.get("f_mob_svc_disney",   False)
    _svc_hulu      = st.session_state.get("f_svc_hulu",     False) or st.session_state.get("f_mob_svc_hulu",     False)
    _svc_apple     = st.session_state.get("f_svc_apple",    False) or st.session_state.get("f_mob_svc_apple",    False)
    _svc_peacock   = st.session_state.get("f_svc_peacock",  False) or st.session_state.get("f_mob_svc_peacock",  False)
    _svc_paramount = st.session_state.get("f_svc_paramount",False) or st.session_state.get("f_mob_svc_paramount",False)
    _type_movies   = st.session_state.get("f_type_movies",  False) or st.session_state.get("f_mob_type_movies",  False)
    _type_tv       = st.session_state.get("f_type_tv",      False) or st.session_state.get("f_mob_type_tv",      False)
    _w_chai_seen     = st.session_state.get("f_w_chai_seen",     False) or st.session_state.get("f_mob_w_chai_seen",     False)
    _w_chai_not_seen = st.session_state.get("f_w_chai_not_seen", False) or st.session_state.get("f_mob_w_chai_not_seen", False)
    _w_noel_seen     = st.session_state.get("f_w_noel_seen",     False) or st.session_state.get("f_mob_w_noel_seen",     False)
    _w_noel_not_seen = st.session_state.get("f_w_noel_not_seen", False) or st.session_state.get("f_mob_w_noel_not_seen", False)
    # Take the more restrictive value from desktop/mobile for range filters
    _imdb_val = max(float(st.session_state.get("f_imdb", 0.0)), float(st.session_state.get("f_mob_imdb", 0.0)))
    _yr_desk  = st.session_state.get("f_yr",     (1950, 2026))
    _yr_mob   = st.session_state.get("f_mob_yr", (1950, 2026))
    _yr_val   = (max(_yr_desk[0], _yr_mob[0]), min(_yr_desk[1], _yr_mob[1]))

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
    _chai_watch_sel = [w for w, v in [("Chai Seen", _w_chai_seen), ("Chai Not Seen", _w_chai_not_seen)] if v]
    _noel_watch_sel = [w for w, v in [("Noel Seen", _w_noel_seen), ("Noel Not Seen", _w_noel_not_seen)] if v]

    _chai_watch_lbl = (
        "Chai: Both"   if (_w_chai_seen and _w_chai_not_seen)
        else ("Chai: Seen"   if _w_chai_seen
        else ("Chai: Unseen" if _w_chai_not_seen
        else  "Chai"))
    )
    _noel_watch_lbl = (
        "Noel: Both"   if (_w_noel_seen and _w_noel_not_seen)
        else ("Noel: Seen"   if _w_noel_seen
        else ("Noel: Unseen" if _w_noel_not_seen
        else  "Noel"))
    )
    _imdb_lbl = f"IMDb ≥ {_imdb_val:.1f}" if _imdb_val > 0 else "IMDb Score"
    _yr_lbl   = f"{_yr_val[0]}–{_yr_val[1]}" if _yr_val != (1950, 2026) else "Year"

    _any_active = bool(services or content_types or _watch_sel or _imdb_val > 0 or _yr_val != (1950, 2026))

    # Active pill highlights — columns in order: svc(1) type(2) chai-watch(3) noel-watch(4) imdb(5) yr(6)
    _pill_css = ""
    for _i, _active in enumerate([bool(services), bool(content_types),
                                   bool(_chai_watch_sel), bool(_noel_watch_sel),
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
        fc_svc, fc_type, fc_chai_w, fc_noel_w, fc_imdb, fc_yr, fc_sep, fc_clr, fc_srt = st.columns(
            [1.3, 1.3, 1.1, 1.1, 1.3, 1.35, 0.12, 0.75, 2.0], vertical_alignment="center"
        )
        with fc_svc:
            with st.popover(_svc_lbl, use_container_width=True):
                st.checkbox("Netflix",     key="f_svc_netflix")
                st.checkbox("Max",         key="f_svc_max")
                st.checkbox("Disney+",     key="f_svc_disney")
                st.checkbox("Hulu",        key="f_svc_hulu")
                st.checkbox("Apple TV+",   key="f_svc_apple")
                st.checkbox("Peacock",     key="f_svc_peacock")
                st.checkbox("Paramount+",  key="f_svc_paramount")
        with fc_type:
            with st.popover(_type_lbl, use_container_width=True):
                st.checkbox("Movies",   key="f_type_movies")
                st.checkbox("TV Shows", key="f_type_tv")
        with fc_chai_w:
            with st.popover(_chai_watch_lbl, use_container_width=True):
                st.checkbox("Seen",     key="f_w_chai_seen")
                st.checkbox("Not Seen", key="f_w_chai_not_seen")
        with fc_noel_w:
            with st.popover(_noel_watch_lbl, use_container_width=True):
                st.checkbox("Seen",     key="f_w_noel_seen")
                st.checkbox("Not Seen", key="f_w_noel_not_seen")
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
                for _k in [
                    "f_svc_netflix", "f_svc_max", "f_svc_disney",
                    "f_svc_hulu", "f_svc_apple", "f_svc_peacock", "f_svc_paramount",
                    "f_type_movies", "f_type_tv",
                    "f_w_chai_seen", "f_w_chai_not_seen", "f_w_noel_seen", "f_w_noel_not_seen",
                    "f_imdb", "f_yr",
                    "f_mob_svc_netflix", "f_mob_svc_max", "f_mob_svc_disney",
                    "f_mob_svc_hulu", "f_mob_svc_apple", "f_mob_svc_peacock", "f_mob_svc_paramount",
                    "f_mob_type_movies", "f_mob_type_tv",
                    "f_mob_w_chai_seen", "f_mob_w_chai_not_seen", "f_mob_w_noel_seen", "f_mob_w_noel_not_seen",
                    "f_mob_imdb", "f_mob_yr", "f_sort_mob",
                ]:
                    st.session_state.pop(_k, None)
                st.rerun()
        with fc_srt:
            sort_by = st.selectbox(
                "Sort",
                ["Compatibility", "Chai Score", "Noel Score", "IMDb Score", "Newest First"],
                format_func=lambda x: f"Sort: {x}",
                label_visibility="collapsed", key="f_sort",
            )

    # ── Mobile filter bar (CSS-hidden on desktop ≥769px) ──────────────────────
    _mob_active_count = sum([
        _svc_netflix, _svc_max, _svc_disney, _svc_hulu, _svc_apple, _svc_peacock, _svc_paramount,
        _type_movies, _type_tv,
        _w_chai_seen, _w_chai_not_seen, _w_noel_seen, _w_noel_not_seen,
        _imdb_val > 0, _yr_val != (1950, 2026),
    ])
    _mob_filter_lbl = f"Filter ({_mob_active_count}) ▾" if _mob_active_count else "Filter ▾"
    _mob_open = st.session_state.get("_mob_filters_open", False)
    # Show/hide the panel and the JS-injected backdrop element together.
    _panel_display = "block" if _mob_open else "none"
    st.markdown(
        f"<style>"
        f".st-key-mob-filter-panel {{ display: {_panel_display} !important; }}"
        f"#enzyme-mob-bd {{ display: {_panel_display} !important; }}"
        f"</style>",
        unsafe_allow_html=True,
    )
    with st.container(key="mob-filter-bar"):
        _mob_btn_lbl = (_mob_filter_lbl.rstrip("▾") + "▲") if _mob_open else _mob_filter_lbl
        if st.button(_mob_btn_lbl, key="f_mob_open", use_container_width=True):
            st.session_state["_mob_filters_open"] = not _mob_open
            st.rerun()

    # Active filter chips — horizontal scrollable row (CSS-hidden on desktop)
    _active_chips: list[tuple[str, str, str]] = []
    if _svc_netflix:     _active_chips.append(("Netflix",      "f_svc_netflix",     "f_mob_svc_netflix"))
    if _svc_max:         _active_chips.append(("Max",          "f_svc_max",         "f_mob_svc_max"))
    if _svc_disney:      _active_chips.append(("Disney+",      "f_svc_disney",      "f_mob_svc_disney"))
    if _svc_hulu:        _active_chips.append(("Hulu",         "f_svc_hulu",        "f_mob_svc_hulu"))
    if _svc_apple:       _active_chips.append(("Apple TV+",    "f_svc_apple",       "f_mob_svc_apple"))
    if _svc_peacock:     _active_chips.append(("Peacock",      "f_svc_peacock",     "f_mob_svc_peacock"))
    if _svc_paramount:   _active_chips.append(("Paramount+",   "f_svc_paramount",   "f_mob_svc_paramount"))
    if _type_movies:     _active_chips.append(("Movies",       "f_type_movies",     "f_mob_type_movies"))
    if _type_tv:         _active_chips.append(("TV Shows",     "f_type_tv",         "f_mob_type_tv"))
    if _w_chai_seen:     _active_chips.append(("Chai: Seen",   "f_w_chai_seen",     "f_mob_w_chai_seen"))
    if _w_chai_not_seen: _active_chips.append(("Chai: Unseen", "f_w_chai_not_seen", "f_mob_w_chai_not_seen"))
    if _w_noel_seen:     _active_chips.append(("Noel: Seen",   "f_w_noel_seen",     "f_mob_w_noel_seen"))
    if _w_noel_not_seen: _active_chips.append(("Noel: Unseen", "f_w_noel_not_seen", "f_mob_w_noel_not_seen"))
    if _imdb_val > 0:    _active_chips.append((f"IMDb ≥{_imdb_val:.0f}", "f_imdb", "f_mob_imdb"))
    if _yr_val != (1950, 2026):
        _active_chips.append((f"{_yr_val[0]}–{_yr_val[1]}", "f_yr", "f_mob_yr"))
    if _active_chips:
        with st.container(key="mob-chip-bar"):
            _chip_cols = st.columns(len(_active_chips))
            for _ccol, (_clbl, _cdk, _cmk) in zip(_chip_cols, _active_chips):
                with _ccol:
                    if st.button(f"{_clbl} ✕", key=f"chip_{_cdk}"):
                        st.session_state.pop(_cdk, None)
                        st.session_state.pop(_cmk, None)
                        st.rerun()

    # Always render the panel (CSS controls visibility so widget state persists)
    _mobile_filters_panel()
    # Prefer mobile sort if user explicitly picked one, otherwise desktop value
    sort_by = st.session_state.get("f_sort_mob") or sort_by

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
                _combined = _combined | _m
            dff = dff[_combined]
    if _imdb_val > 0:
        dff = dff[dff["imdb_score"].notna() & (dff["imdb_score"] >= _imdb_val)]
    year_from, year_to = _yr_val
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
    _filter_key = f"{services}|{content_types}|{_watch_sel}|{_imdb_val}|{_yr_val}|{sort_by}"
    if st.session_state.get("_catalog_filter_key") != _filter_key:
        st.session_state["_catalog_filter_key"] = _filter_key
        st.session_state["_catalog_visible"] = 8

    visible = st.session_state.get("_catalog_visible", 8)

    with st.container(key="catalog-grid"):
        if len(dff) == 0:
            st.markdown(
                "<div style='text-align:center; padding:2.5rem; color:#6b7280;'>"
                "No matches found — try adjusting your filters!</div>",
                unsafe_allow_html=True,
            )
        else:
            # Inject once: make catalog-card taps fire the hidden Details button on mobile.
            # Uses event delegation on parent.document so the listener survives Streamlit
            # reruns (the components iframe may be recreated, but the listener on
            # parent.document persists). The _enzymeCardTap guard prevents duplicates.
            _components.html("""<script>
(function() {
  if (parent.window._enzymeCardTap) return;
  parent.window._enzymeCardTap = true;
  parent.document.addEventListener('click', function(e) {
    var card = e.target && e.target.closest && e.target.closest('.catalog-card');
    if (!card) return;
    var col = card.closest('[data-testid="stColumn"]');
    if (col) {
      var btn = col.querySelector('button');
      if (btn) { e.stopPropagation(); btn.click(); }
    }
  }, true);
})();
</script>""", height=0)

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
