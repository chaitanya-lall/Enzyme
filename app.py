"""
Enzyme — Streamlit UI
Run with:  streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as _components
from streamlit_searchbox import st_searchbox

from predict import search_omdb
from catalog_sync import start_background_sync, get_sync_status
from ui_styles import APP_CSS
from ui_components import (
    preload_artifacts,
    preload_artifacts_noel,
    _render_movie_analysis,
    _load_catalog,
)
from ui_catalog import render_recommend_tab

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="The Personalized Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Styles ───────────────────────────────────────────────────────────────────
st.markdown(APP_CSS, unsafe_allow_html=True)


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
            _load_catalog.clear()   # invalidate the catalog cache

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

    # Dynamic CSS: button colours only
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

    # ── Content ───────────────────────────────────────────────────────────────
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
