APP_CSS = """<style>
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
  /* ── Header compaction (~50% reduction between logo/Recommend/Search) ── */
  /* Collapse the bottom of the logo area */
  .nav-btn { padding-bottom: 0 !important; padding-top: 0.15rem !important; }
  /* Tighten column gap inside nav block */
  div[data-testid="stHorizontalBlock"]:has(.nav-btn) {
    gap: 0.25rem !important;
    margin-bottom: -0.5rem !important;
  }
  /* Buttons: halve the top-margin that desktop adds */
  div[data-testid="stHorizontalBlock"]:has(.nav-btn) button {
    margin-top: 0.15rem !important;
    height: 2rem !important;
  }
  /* Gap between nav block and what follows (hr / filter bar) */
  div[data-testid="stHorizontalBlock"]:has(.nav-btn) + div,
  div[data-testid="stHorizontalBlock"]:has(.nav-btn) + hr {
    margin-top: -0.25rem !important;
  }
  /* ── Hide desktop filter bar; show mobile bar ── */
  .st-key-filter-bar { display: none !important; }
  /* ── Catalog grid: 2 cards per row in portrait ── */
  div[data-testid="stHorizontalBlock"]:has(.catalog-card) {
    flex-wrap: wrap !important;
    gap: 0.4rem !important;
  }
  div[data-testid="stHorizontalBlock"]:has(.catalog-card) > div[data-testid="stColumn"] {
    width: calc(50% - 0.2rem) !important;
    flex: 0 0 calc(50% - 0.2rem) !important;
    min-width: 0 !important;
  }
  /* Scale down card text in 2-up layout */
  .catalog-card-title { font-size: 0.73rem !important; }
  .catalog-card-year  { font-size: 0.65rem !important; }
  .catalog-card-scores span { font-size: 0.67rem !important; }
  /* Hide Details button on mobile — card tap opens the dialog instead */
  div[data-testid="stColumn"]:has(.catalog-card) .stButton { display: none !important; }
  .catalog-card { cursor: pointer; }
}

/* ── Mobile filter bar: hidden on desktop ── */
@media (min-width: 769px) {
  .st-key-mob-filter-bar { display: none !important; }
}
/* Mobile filter bar styles — single full-width "Filter and Sort" pill */
.st-key-mob-filter-bar {
  padding: 0 0 0.8rem 0;
  position: sticky;
  top: 0;
  z-index: 99;
  background: #0a0b0f;
}
.st-key-mob-filter-bar button {
  border-radius: 20px !important;
  font-size: 0.84rem !important;
  font-weight: 500 !important;
  height: 2.2rem !important;
  background: #ffffff !important;
  border: 1px solid #d1d5db !important;
  width: 100% !important;
}
.st-key-mob-filter-bar button p,
.st-key-mob-filter-bar button div,
.st-key-mob-filter-bar button span { color: #000000 !important; }

/* Mobile inline filter panel — full-screen overlay (dialog-style) on mobile.
   Rendered inline in the DOM for reliable state management, but CSS positions
   it as a fixed overlay so it looks and feels like the original @st.dialog. */
@media (max-width: 768px) {
  .st-key-mob-filter-panel {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    right: 0 !important;
    bottom: 0 !important;
    z-index: 1000 !important;
    background: #13161f !important;
    overflow-y: auto !important;
    padding: 1.5rem 1rem 5rem 1rem !important;
    margin: 0 !important;
    border-radius: 0 !important;
    border: none !important;
    /* Darken everything behind it */
    box-shadow: 0 0 0 200vw rgba(0,0,0,0.75) !important;
  }
}


/* ── Nav bar ─────────────────────────────────────────────────────────────── */
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
/* Column: clip to card border-radius, no padding */
div[data-testid="stColumn"]:has(.catalog-card) {
  padding: 0 !important;
  overflow: hidden !important;
  border-radius: 13px !important;
}
/* Zero out Streamlit's default ~1rem gap between the card HTML and the Details button */
div[data-testid="stColumn"]:has(.catalog-card) [data-testid="stVerticalBlock"] {
  gap: 0 !important;
}
/* Remove any residual margin/padding on the element containers */
div[data-testid="stColumn"]:has(.catalog-card) [data-testid="stElementContainer"] {
  margin: 0 !important;
  padding: 0 !important;
}
/* Streamlit injects margin-bottom:-16px on stMarkdownContainer, which makes the column
   think the card is 16px shorter than it is → overflow:hidden clips the bottom 16px.
   Zero it out so the column height correctly matches the card height. */
div[data-testid="stColumn"]:has(.catalog-card) [data-testid="stMarkdownContainer"] {
  margin-bottom: 0 !important;
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
/* Details button — flush against the card, full width */
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
/* ── Mobile: undo column clipping AFTER the desktop rule so it actually wins ── */
/* (both rules are !important same specificity — last in document wins)          */
@media (max-width: 768px) {
  div[data-testid="stColumn"]:has(.catalog-card) {
    overflow: visible !important;
    border-radius: 0 !important;
  }
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

</style>"""
