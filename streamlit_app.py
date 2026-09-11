"""Streamlit front end.

    streamlit run streamlit_app.py

Calls the agent in-process so the demo needs only the database. This module is
presentation only - it reads the state the agent returns and renders it. All
matching, scoring and verification logic lives in `app/`.

Every value interpolated into HTML goes through `_esc()`. The cards use
`unsafe_allow_html`, which does no sanitising, and chat content is user-supplied.
"""

from __future__ import annotations

import html
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import streamlit as st  # noqa: E402

from app.agent.graph import render_trace, run_turn  # noqa: E402
from app.agent.state import (  # noqa: E402
    ACTION_ANSWER_HELP,
    ACTION_CASE_STATUS,
    ACTION_OUT_OF_SCOPE,
    AWAITING_OWNERSHIP_PROOF,
)
from app.config import configure_logging, settings  # noqa: E402
from app.database.connection import database_is_reachable  # noqa: E402
from app.database.vector_support import current_backend  # noqa: E402
from app.observability import get_tracer  # noqa: E402

configure_logging()

st.set_page_config(
    page_title="Lost & Found AI",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)

CATEGORY_ICONS: dict[str, str] = {
    "headphones": "🎧",
    "earbuds": "🎧",
    "phone": "📱",
    "tablet": "📱",
    "wallet": "👛",
    "backpack": "🎒",
    "laptop": "💻",
    "watch": "⌚",
    "water_bottle": "🥤",
    "book": "📘",
    "charger": "🔌",
    "keys": "🔑",
    "glasses": "👓",
    "umbrella": "☂️",
    "jacket": "🧥",
}

THEME_CHOICES = ("System", "Light", "Dark")

LIGHT_TOKENS = """
  --lf-navy:      #0F1B33;
  --lf-navy-soft: #1B2C4E;
  --lf-navy-3:    #24406F;
  --lf-blue:      #2563EB;
  --lf-indigo:    #4F46E5;
  --lf-cyan:      #06B6D4;
  --lf-surface:   #FFFFFF;
  --lf-canvas:    #F6F8FC;
  --lf-text:      #0F172A;
  --lf-muted:     #64748B;
  --lf-border:    #E2E8F0;
  --lf-track:     #EEF2F7;
  --lf-code-bg:   #F1F5F9;
  --lf-icon-bg:   linear-gradient(140deg, #EEF4FF, #E0E9FF);
  --lf-user-bg:   #0F1B33;
  --lf-user-fg:   #EAF1FB;
  --lf-hint-bg:   #F1F5FF;
  --lf-hint-brd:  #DBE4FF;
  --lf-hint-fg:   #1B2C4E;
  --lf-green:     #059669;
  --lf-green-bg:  #ECFDF5;
  --lf-amber:     #B45309;
  --lf-amber-bg:  #FFFBEB;
  --lf-red:       #DC2626;
  --lf-red-bg:    #FEF2F2;
  --lf-shadow:    0 1px 2px rgba(15,27,51,.06), 0 8px 24px rgba(15,27,51,.06);
"""

DARK_TOKENS = """
  --lf-navy:      #0B1220;
  --lf-navy-soft: #16233D;
  --lf-navy-3:    #1E3357;
  --lf-blue:      #60A5FA;
  --lf-indigo:    #818CF8;
  --lf-cyan:      #22D3EE;
  --lf-surface:   #131C2E;
  --lf-canvas:    #0B1220;
  --lf-text:      #E6EDF7;
  --lf-muted:     #94A6BF;
  --lf-border:    #24334D;
  --lf-track:     #1B2740;
  --lf-code-bg:   #0E1729;
  --lf-icon-bg:   linear-gradient(140deg, #1C2B47, #243B63);
  --lf-user-bg:   #1D3A6B;
  --lf-user-fg:   #EAF1FB;
  --lf-hint-bg:   #16233D;
  --lf-hint-brd:  #24334D;
  --lf-hint-fg:   #C7D6EE;
  --lf-green:     #34D399;
  --lf-green-bg:  rgba(52,211,153,.12);
  --lf-amber:     #FBBF24;
  --lf-amber-bg:  rgba(251,191,36,.12);
  --lf-red:       #F87171;
  --lf-red-bg:    rgba(248,113,113,.12);
  --lf-shadow:    0 1px 2px rgba(0,0,0,.35), 0 10px 28px rgba(0,0,0,.35);
"""

# Streamlit's own chrome is themed by .streamlit/config.toml, which cannot change
# at runtime. These rules re-point it at the palette variables so the app frame
# follows the selected theme along with the cards.
CHROME_CSS = """
.stApp { background: var(--lf-canvas); color: var(--lf-text); }
section[data-testid="stSidebar"] { background: var(--lf-surface); border-right: 1px solid var(--lf-border); }
header[data-testid="stHeader"] { background: transparent; }
.stApp [data-testid="stExpander"] details {
  background: var(--lf-surface); border-color: var(--lf-border); border-radius: 14px;
}
.stApp [data-testid="stExpander"] summary p { color: var(--lf-text); }
.stApp [data-testid="stMarkdownContainer"] p { color: var(--lf-text); }
.stApp pre { background: var(--lf-code-bg) !important; border: 1px solid var(--lf-border); }
.stApp pre, .stApp pre code, .stApp pre span { color: var(--lf-text) !important; }
.stApp hr { border-color: var(--lf-border); }

/* The chat bar sits in its own container that paints an opaque background. */
[data-testid="stBottom"] > div { background: var(--lf-canvas); }
[data-testid="stChatInput"] {
  background: var(--lf-surface); border: 1px solid var(--lf-border); border-radius: 12px;
}
/* An unnamed wrapper inside the chat input paints its own white surface; let the
   themed container above show through instead. */
[data-testid="stChatInput"] div { background-color: transparent !important; }
[data-testid="stChatInputTextArea"] { color: var(--lf-text) !important; }
[data-testid="stChatInputSubmitButton"] svg { fill: var(--lf-blue) !important; }
[data-testid="stChatInputTextArea"]::placeholder { color: var(--lf-muted) !important; }

/* Buttons render their label through a markdown container, so the label colour
   above would otherwise sit on Streamlit's untouched light button surface. */
[data-testid="stBaseButton-secondary"] {
  background: var(--lf-surface) !important;
  color: var(--lf-text) !important;
  border-color: var(--lf-border) !important;
}
[data-testid="stBaseButton-secondary"]:hover { border-color: var(--lf-blue) !important; }
[data-testid="stButtonGroup"] button {
  background: var(--lf-surface) !important;
  color: var(--lf-muted) !important;
  border-color: var(--lf-border) !important;
}
[data-testid="stButtonGroup"] button[aria-checked="true"],
[data-testid="stButtonGroup"] button[data-selected="true"] {
  background: var(--lf-hint-bg) !important;
  color: var(--lf-blue) !important;
  border-color: var(--lf-blue) !important;
}
"""

BASE_CSS = """
:root { --lf-radius: 14px; }

/* Keep the reading column comfortable on wide monitors. */
.block-container { max-width: 1080px; padding-top: 1.2rem; padding-bottom: 6rem; }

@keyframes lf-rise   { from { opacity:0; transform: translateY(8px); } to { opacity:1; transform:none; } }
@keyframes lf-pop    { from { opacity:0; transform: scale(.96); }     to { opacity:1; transform:none; } }
@keyframes lf-grow   { from { width: 0; }                             to { width: var(--lf-target); } }
@keyframes lf-dot    { 0%,80%,100% { opacity:.25; transform: translateY(0); }
                       40%         { opacity:1;   transform: translateY(-3px); } }
@keyframes lf-halo   { 0% { transform: scale(.85); opacity:.55; }
                       70%{ transform: scale(1.5); opacity:0; }
                       100%{ transform: scale(1.5); opacity:0; } }
@keyframes lf-sweep  { from { background-position: 200% 0; } to { background-position: -200% 0; } }

/* ---------------------------------------------------------------- header */
.lf-header {
  position: relative;
  background: linear-gradient(120deg, var(--lf-navy) 0%, var(--lf-navy-soft) 55%, var(--lf-navy-3) 100%);
  border-radius: 18px;
  padding: 26px 30px;
  margin-bottom: 22px;
  overflow: hidden;
  animation: lf-rise .5s ease both;
}
.lf-header::after {
  content: ""; position: absolute; inset: 0;
  background: radial-gradient(680px 150px at 88% -30%, rgba(6,182,212,.30), transparent 70%);
  pointer-events: none;
}
.lf-header-row { display: flex; align-items: center; gap: 18px; position: relative; z-index: 1; }
.lf-title    { color: #fff; font-size: 1.65rem; font-weight: 700; letter-spacing: -.02em; margin: 0; }
.lf-subtitle { color: #A8BEDD; font-size: .95rem; margin: 5px 0 0; }

.lf-orb { position: relative; width: 46px; height: 46px; flex: 0 0 46px; }
.lf-orb-core {
  position: absolute; inset: 8px; border-radius: 50%;
  background: linear-gradient(140deg, var(--lf-cyan), var(--lf-blue));
  box-shadow: 0 0 18px rgba(6,182,212,.55);
}
.lf-orb-ring {
  position: absolute; inset: 0; border-radius: 50%;
  border: 2px solid rgba(6,182,212,.6);
  animation: lf-halo 2.6s ease-out infinite;
}
.lf-orb-ring:nth-child(2) { animation-delay: 1.3s; }

/* ------------------------------------------------------------------ chat */
.lf-turn { animation: lf-rise .35s ease both; margin-bottom: 14px; }
.lf-role {
  font-size: .72rem; font-weight: 700; letter-spacing: .09em;
  text-transform: uppercase; color: var(--lf-muted); margin-bottom: 6px;
  display: flex; align-items: center; gap: 7px;
}
.lf-role-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--lf-blue); }
.lf-bubble {
  background: var(--lf-surface); border: 1px solid var(--lf-border);
  border-radius: var(--lf-radius); padding: 14px 17px; color: var(--lf-text);
  line-height: 1.62; box-shadow: var(--lf-shadow); white-space: pre-wrap;
}
.lf-bubble-agent { border-left: 3px solid var(--lf-blue); }
.lf-turn-user { display: flex; flex-direction: column; align-items: flex-end; }
.lf-turn-user .lf-bubble {
  background: var(--lf-user-bg); color: var(--lf-user-fg); border-color: var(--lf-user-bg);
  max-width: 78%; border-left: none;
}
.lf-turn-user .lf-role { color: var(--lf-muted); }
.lf-turn-user .lf-role-dot { background: var(--lf-muted); }

/* ------------------------------------------------------------ processing */
.lf-working {
  display: flex; align-items: center; gap: 12px;
  background: var(--lf-surface); border: 1px solid var(--lf-border);
  border-left: 3px solid var(--lf-cyan);
  border-radius: var(--lf-radius); padding: 14px 17px;
  box-shadow: var(--lf-shadow); animation: lf-rise .3s ease both;
}
.lf-working-text { font-size: .92rem; color: var(--lf-text); font-weight: 600; }
.lf-working-sub  { font-size: .82rem; color: var(--lf-muted); margin-top: 2px; }
.lf-dots { display: inline-flex; gap: 5px; }
.lf-dots span {
  width: 7px; height: 7px; border-radius: 50%; background: var(--lf-blue);
  animation: lf-dot 1.25s infinite ease-in-out;
}
.lf-dots span:nth-child(2) { animation-delay: .16s; }
.lf-dots span:nth-child(3) { animation-delay: .32s; }

/* -------------------------------------------------------------- activity */
.lf-activity {
  background: var(--lf-surface); border: 1px solid var(--lf-border);
  border-radius: var(--lf-radius); padding: 15px 18px; margin: 12px 0;
  box-shadow: var(--lf-shadow);
}
.lf-activity-title {
  font-size: .68rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase;
  color: var(--lf-muted); margin-bottom: 11px;
}
.lf-step {
  display: flex; align-items: baseline; gap: 10px; padding: 4px 0;
  font-size: .89rem; color: var(--lf-text); animation: lf-rise .35s ease both;
}
.lf-step-mark { font-weight: 700; width: 15px; flex: 0 0 15px; }
.lf-step-detail { color: var(--lf-muted); font-size: .83rem; }
.lf-ok    .lf-step-mark { color: var(--lf-green); }
.lf-warn  .lf-step-mark { color: var(--lf-amber); }
.lf-fail  .lf-step-mark { color: var(--lf-red); }
.lf-live  .lf-step-mark { color: var(--lf-blue); }
.lf-live  { font-weight: 600; }

/* ------------------------------------------------------------ match card */
.lf-card {
  background: var(--lf-surface); border: 1px solid var(--lf-border);
  border-radius: var(--lf-radius); padding: 18px 20px; margin: 12px 0;
  box-shadow: var(--lf-shadow); animation: lf-pop .4s ease both;
}
.lf-card-head { display: flex; align-items: center; gap: 13px; margin-bottom: 13px; }
.lf-card-icon {
  width: 44px; height: 44px; flex: 0 0 44px; border-radius: 12px;
  background: var(--lf-icon-bg);
  display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
}
.lf-eyebrow {
  font-size: .68rem; font-weight: 700; letter-spacing: .11em;
  text-transform: uppercase; color: var(--lf-blue);
}
.lf-item-name { font-size: 1.08rem; font-weight: 700; color: var(--lf-text); margin-top: 1px; }
.lf-item-sub  { font-size: .87rem; color: var(--lf-muted); margin-top: 1px; }
.lf-meta { display: flex; flex-wrap: wrap; gap: 8px 20px; margin: 4px 0 14px; }
.lf-meta span { font-size: .87rem; color: var(--lf-muted); }

/* ------------------------------------------------------------ confidence */
.lf-conf-label {
  display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 6px;
}
.lf-conf-name {
  font-size: .68rem; font-weight: 700; letter-spacing: .11em;
  text-transform: uppercase; color: var(--lf-muted);
}
.lf-conf-value { font-size: 1.22rem; font-weight: 700; }
.lf-bar {
  height: 11px; border-radius: 99px; background: var(--lf-track); overflow: hidden;
  border: 1px solid var(--lf-border);
}
.lf-bar-fill {
  height: 100%; border-radius: 99px; width: var(--lf-target);
  animation: lf-grow .85s cubic-bezier(.2,.8,.2,1) both;
  background-image: repeating-linear-gradient(90deg,
      rgba(255,255,255,.28) 0 3px, transparent 3px 8px);
}
.lf-conf-note { font-size: .78rem; color: var(--lf-muted); margin-top: 6px; }
.lf-high  .lf-bar-fill  { background-color: var(--lf-green); }
.lf-high  .lf-conf-value{ color: var(--lf-green); }
.lf-medium .lf-bar-fill { background-color: var(--lf-amber); }
.lf-medium .lf-conf-value{ color: var(--lf-amber); }
.lf-low   .lf-bar-fill  { background-color: var(--lf-red); }
.lf-low   .lf-conf-value{ color: var(--lf-red); }

.lf-pill {
  display: inline-block; margin-top: 14px; padding: 5px 12px; border-radius: 99px;
  font-size: .76rem; font-weight: 700; letter-spacing: .03em;
  background: var(--lf-amber-bg); color: var(--lf-amber);
  border: 1px solid rgba(180,83,9,.22);
}

/* --------------------------------------------------------- result states */
.lf-success {
  background: linear-gradient(150deg, var(--lf-green-bg), var(--lf-surface) 65%);
  border: 1px solid rgba(5,150,105,.30); border-left: 4px solid var(--lf-green);
  border-radius: var(--lf-radius); padding: 20px 22px; margin: 12px 0;
  box-shadow: var(--lf-shadow); animation: lf-pop .45s cubic-bezier(.2,.8,.2,1) both;
}
.lf-success-head {
  display: flex; align-items: center; gap: 11px;
  font-size: .96rem; font-weight: 800; letter-spacing: .05em;
  text-transform: uppercase; color: var(--lf-green); margin-bottom: 10px;
}
.lf-check {
  width: 26px; height: 26px; border-radius: 50%; background: var(--lf-green); color: #fff;
  display: flex; align-items: center; justify-content: center; font-size: .85rem;
  animation: lf-pop .5s .12s cubic-bezier(.2,.8,.2,1) both;
}
.lf-detail-grid { display: flex; flex-wrap: wrap; gap: 18px 34px; margin-top: 14px; }
.lf-detail-key {
  font-size: .68rem; font-weight: 700; letter-spacing: .11em;
  text-transform: uppercase; color: var(--lf-muted);
}
.lf-detail-val { font-size: 1rem; font-weight: 700; color: var(--lf-text); margin-top: 2px; }
.lf-ref { font-family: ui-monospace, "Cascadia Mono", Menlo, monospace; letter-spacing: .04em; }

.lf-escalation {
  background: linear-gradient(150deg, var(--lf-amber-bg), var(--lf-surface) 65%);
  border: 1px solid rgba(180,83,9,.28); border-left: 4px solid var(--lf-amber);
  border-radius: var(--lf-radius); padding: 20px 22px; margin: 12px 0;
  box-shadow: var(--lf-shadow); animation: lf-pop .4s ease both;
}
.lf-escalation-head {
  display: flex; align-items: center; gap: 10px;
  font-size: .92rem; font-weight: 800; letter-spacing: .05em;
  text-transform: uppercase; color: var(--lf-amber); margin-bottom: 8px;
}
.lf-reason {
  margin-top: 12px; padding: 11px 14px; border-radius: 10px;
  background: var(--lf-surface); border: 1px solid rgba(180,83,9,.18);
  font-size: .87rem; color: var(--lf-text);
}
.lf-body { font-size: .93rem; color: var(--lf-text); line-height: 1.6; }

.lf-sources { display: flex; flex-direction: column; gap: 6px; margin-top: 10px; }
.lf-source { font-size: .9rem; color: var(--lf-text); }
.lf-source-score { font-size: .78rem; color: var(--lf-muted); margin-left: 6px; }
.lf-sample {
  display: inline-block; margin-left: 6px; padding: 1px 8px; border-radius: 99px;
  font-size: .68rem; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
  background: var(--lf-amber-bg); color: var(--lf-amber); border: 1px solid rgba(180,83,9,.22);
}

/* --------------------------------------------------------------- sidebar */
.lf-side-brand {
  background: linear-gradient(140deg, var(--lf-navy), var(--lf-navy-soft));
  border-radius: 13px; padding: 16px 17px; margin-bottom: 16px;
}
.lf-side-title { color: #fff; font-weight: 700; font-size: 1.02rem; }
.lf-side-sub   { color: #A8BEDD; font-size: .79rem; margin-top: 3px; }
.lf-side-heading {
  font-size: .68rem; font-weight: 700; letter-spacing: .11em; text-transform: uppercase;
  color: var(--lf-muted); margin: 18px 0 9px;
}
.lf-status-row {
  display: flex; align-items: center; gap: 9px; padding: 5px 0; font-size: .87rem;
  color: var(--lf-text);
}
.lf-status-dot {
  width: 8px; height: 8px; border-radius: 50%; flex: 0 0 8px; position: relative;
}
.lf-status-dot.up   { background: var(--lf-green); }
.lf-status-dot.down { background: var(--lf-red); }
.lf-status-dot.up::after {
  content: ""; position: absolute; inset: -3px; border-radius: 50%;
  border: 1px solid var(--lf-green); animation: lf-halo 2.4s ease-out infinite;
}
.lf-status-note { color: var(--lf-muted); font-size: .78rem; margin-left: auto; }
.lf-about { font-size: .82rem; color: var(--lf-muted); line-height: 1.55; }

/* ----------------------------------------------------------- signal table */
.lf-table { width: 100%; border-collapse: collapse; margin: 2px 0 14px; font-size: .85rem; }
.lf-table th, .lf-table td {
  text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--lf-border);
  color: var(--lf-text);
}
.lf-table th {
  color: var(--lf-muted); font-size: .68rem; letter-spacing: .1em;
  text-transform: uppercase; font-weight: 700;
}
.lf-table td.lf-num { text-align: right; font-variant-numeric: tabular-nums; }
.lf-table td.lf-unset { color: var(--lf-muted); font-style: italic; }

/* ------------------------------------------------------------ empty view */
.lf-empty {
  background: var(--lf-surface); border: 1px dashed var(--lf-border);
  border-radius: var(--lf-radius); padding: 30px 26px; text-align: center;
  animation: lf-rise .45s ease both;
}
.lf-empty-title { font-weight: 700; color: var(--lf-text); font-size: 1.02rem; }
.lf-empty-sub   { color: var(--lf-muted); font-size: .89rem; margin-top: 7px; line-height: 1.6; }
.lf-empty-hint {
  display: inline-block; margin-top: 16px; padding: 9px 15px; border-radius: 10px;
  background: var(--lf-hint-bg); border: 1px solid var(--lf-hint-brd); color: var(--lf-hint-fg);
  font-size: .87rem;
  background-image: linear-gradient(100deg, transparent 30%, rgba(37,99,235,.09) 50%, transparent 70%);
  background-size: 200% 100%; animation: lf-sweep 4.5s linear infinite;
}

/* ------------------------------------------------------------ responsive */
@media (max-width: 760px) {
  .block-container { padding-left: .9rem; padding-right: .9rem; }
  .lf-header { padding: 20px; }
  .lf-title { font-size: 1.3rem; }
  .lf-turn-user .lf-bubble { max-width: 92%; }
  .lf-detail-grid { gap: 14px 22px; }
}

@media (prefers-reduced-motion: reduce) {
  * { animation: none !important; transition: none !important; }
  .lf-bar-fill { width: var(--lf-target); }
}
"""


def build_styles(theme: str) -> str:
    """Assemble the stylesheet for the selected theme.

    "System" emits both palettes and lets `prefers-color-scheme` choose, so it
    tracks the OS with no round trip. "Light"/"Dark" pin one palette. Chrome
    rules come before the component rules so that, on equal specificity, the
    components win.
    """
    if theme == "Light":
        palette = f":root {{{LIGHT_TOKENS}}}"
    elif theme == "Dark":
        palette = f":root {{{DARK_TOKENS}}}"
    else:
        palette = (
            f":root {{{LIGHT_TOKENS}}}\n"
            f"@media (prefers-color-scheme: dark) {{ :root {{{DARK_TOKENS}}} }}"
        )
    return f"<style>\n{palette}\n{CHROME_CSS}\n{BASE_CSS}\n</style>"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _esc(value: Any) -> str:
    """Escape a value for interpolation into the card HTML."""
    return html.escape("" if value is None else str(value))


def _pretty(value: Any) -> str:
    return _esc(str(value).replace("_", " ")) if value else ""


def _level_class(level: str) -> str:
    return {"HIGH": "lf-high", "MEDIUM": "lf-medium"}.get((level or "").upper(), "lf-low")


def _icon_for(category: str | None) -> str:
    return CATEGORY_ICONS.get((category or "").lower(), "🔎")


def _format_found_time(raw: str | None) -> str:
    if not raw:
        return ""
    try:
        moment = datetime.fromisoformat(raw)
    except ValueError:
        return _esc(raw[:16].replace("T", " "))
    return _esc(moment.strftime("Found around %I:%M %p, %d %b").replace(" 0", " "))


def _item_title(candidate: dict[str, Any]) -> str:
    parts = [candidate.get("brand"), str(candidate.get("category") or "item").replace("_", " ")]
    return _esc(" ".join(str(part) for part in parts if part).title())


# ---------------------------------------------------------------------------
# Reusable renderers - each returns an HTML fragment
# ---------------------------------------------------------------------------


def render_confidence_indicator(
    score: float, level: str, note: str = "", name: str = "Match Confidence"
) -> str:
    """A labelled progress bar coloured by confidence level."""
    percent = max(0.0, min(1.0, float(score or 0.0))) * 100
    note_html = f'<div class="lf-conf-note">{_esc(note)}</div>' if note else ""
    return (
        f'<div class="{_level_class(level)}">'
        f'  <div class="lf-conf-label">'
        f'    <span class="lf-conf-name">{_esc(name)}</span>'
        f'    <span class="lf-conf-value">{percent:.0f}%</span>'
        f"  </div>"
        f'  <div class="lf-bar"><div class="lf-bar-fill" style="--lf-target:{percent:.1f}%"></div></div>'
        f"  {note_html}"
        f"</div>"
    )


def render_match_card(candidate: dict[str, Any], verification_required: bool = True) -> str:
    """A candidate item. Public attributes only - never ownership evidence."""
    confidence = candidate.get("confidence") or {}
    reasons = confidence.get("reasons") or []

    meta: list[str] = []
    if candidate.get("location"):
        meta.append(f'<span>📍 {_esc(candidate["location"])}</span>')
    found = _format_found_time(candidate.get("found_time"))
    if found:
        meta.append(f"<span>🕒 {found}</span>")

    colour = _pretty(candidate.get("color"))
    pill = (
        '<div class="lf-pill">🔒 Verification required</div>' if verification_required else ""
    )

    return (
        '<div class="lf-card">'
        '  <div class="lf-card-head">'
        f'    <div class="lf-card-icon">{_icon_for(candidate.get("category"))}</div>'
        "    <div>"
        '      <div class="lf-eyebrow">Potential match</div>'
        f'      <div class="lf-item-name">{_item_title(candidate)}</div>'
        f'      <div class="lf-item-sub">{colour}</div>'
        "    </div>"
        "  </div>"
        f'  <div class="lf-meta">{"".join(meta)}</div>'
        + render_confidence_indicator(
            confidence.get("score", 0.0),
            confidence.get("level", "LOW"),
            note=_esc(reasons[0]) if reasons else "",
        )
        + f"  {pill}"
        "</div>"
    )


def render_success_card(
    candidate: dict[str, Any],
    pickup: dict[str, Any],
    confidence: dict[str, Any] | None,
) -> str:
    """Shown once ownership has been verified and the item reserved."""
    label = str(candidate.get("category") or "item").replace("_", " ")
    verb = "have" if label.endswith("s") else "has"
    score = (confidence or {}).get("score")
    score_html = (
        f'<div><div class="lf-detail-key">Match confidence</div>'
        f'<div class="lf-detail-val">{float(score) * 100:.0f}%</div></div>'
        if score is not None
        else ""
    )

    return (
        '<div class="lf-success">'
        '  <div class="lf-success-head"><span class="lf-check">✓</span> Match verified</div>'
        f'  <div class="lf-body">Your {_esc(label)} {verb} been successfully identified '
        "and reserved for collection.</div>"
        '  <div class="lf-detail-grid">'
        '    <div><div class="lf-detail-key">📍 Pickup location</div>'
        f'         <div class="lf-detail-val">{_esc(pickup.get("location"))}</div></div>'
        '    <div><div class="lf-detail-key">🎫 Pickup request</div>'
        f'         <div class="lf-detail-val lf-ref">{_esc(pickup.get("pickup_request_id"))}</div></div>'
        f"    {score_html}"
        "  </div>"
        "</div>"
    )


def render_escalation_card(escalation: dict[str, Any]) -> str:
    """Shown when the agent refuses to make a claim and hands over to staff."""
    return (
        '<div class="lf-escalation">'
        '  <div class="lf-escalation-head">⚠ Human review required</div>'
        '  <div class="lf-body">This case has been passed to staff rather than risk a '
        "wrong match. They will follow up with you.</div>"
        '  <div class="lf-detail-grid">'
        '    <div><div class="lf-detail-key">🎫 Case reference</div>'
        f'         <div class="lf-detail-val lf-ref">{_esc(escalation.get("escalation_id"))}</div></div>'
        '    <div><div class="lf-detail-key">Status</div>'
        f'         <div class="lf-detail-val">{_pretty(escalation.get("status")).title()}</div></div>'
        "  </div>"
        f'  <div class="lf-reason"><strong>Reason:</strong> {_esc(escalation.get("reason"))}</div>'
        + render_contact_line(escalation.get("contact"), escalation.get("contact_is_sample", False))
        + "</div>"
    )


def render_contact_line(contact: str | None, is_sample: bool) -> str:
    if not contact:
        return ""
    sample = ' <span class="lf-sample">Sample data</span>' if is_sample else ""
    return f'<div class="lf-reason"><strong>Reach staff:</strong> {_esc(contact)}{sample}</div>'


def render_help_card(result: dict[str, Any]) -> str:
    """The evidence behind a help answer: which stored articles it came from."""
    sources = result.get("sources") or []
    if result.get("grounded"):
        head, note = "📚 Answered from the help desk knowledge base", ""
    else:
        head = "📚 Not in the help desk knowledge base"
        note = (
            '<div class="lf-body">No stored article was close enough to answer this, '
            "so the assistant did not guess.</div>"
        )
    rows = "".join(
        f'<div class="lf-source"><span class="lf-ref">[{_esc(source.get("n"))}]</span> '
        f"{_esc(source.get('title'))}"
        + (f' <span class="lf-source-score">similarity {source["similarity"]:.2f}</span>'
           if source.get("similarity") is not None else "")
        + (' <span class="lf-sample">Sample data</span>' if source.get("is_sample") else "")
        + "</div>"
        for source in sources
    )
    return (
        '<div class="lf-card">'
        f'  <div class="lf-eyebrow">{head}</div>'
        f"  {note}"
        f'  <div class="lf-sources">{rows}</div>'
        "</div>"
    )


def agent_activity(turn: dict[str, Any]) -> list[tuple[str, str, str]]:
    """Derive the activity list from the trace the agent actually produced.

    Returns (mark-class, label, detail) rows. Nothing here is invented: every
    row corresponds to a node that ran on this turn.
    """
    router_labels = {
        "VERIFY_OWNERSHIP": "Requesting ownership proof",
        "ASK_CLARIFICATION": "Asking for clarification",
    }
    rows: list[tuple[str, str, str]] = []

    for entry in turn.get("trace", []):
        step, detail = entry["step"], entry["detail"]

        if step == "CLASSIFY_INTENT":
            method = "LLM" if str(detail.get("method", "")).startswith("llm") else "rules"
            rows.append(("lf-ok", "Understanding intent", f"{detail.get('intent')} ({method})"))
        elif step == "SEARCH_HELP":
            grounded = detail.get("grounded")
            rows.append(
                (
                    "lf-ok" if grounded else "lf-warn",
                    "Searching help knowledge base",
                    f"top similarity {detail.get('top_similarity', 0):.2f}"
                    + ("" if grounded else " - below threshold, not answered"),
                )
            )
        elif step == "LOOKUP_CASE":
            found = detail.get("found")
            rows.append(
                ("lf-ok" if found else "lf-warn", "Looking up reference", str(detail.get("reference") or "none given"))
            )
        elif step == "OUT_OF_SCOPE":
            rows.append(("lf-ok", "Explaining what I can help with", ""))
        elif step == "UNDERSTAND_REQUEST":
            known = [
                f"{key}: {value}"
                for key, value in (detail.get("extracted") or {}).items()
                if value
            ]
            rows.append(("lf-ok", "Understanding request", ", ".join(known)))
        elif step == "NEED_MORE_INFORMATION":
            missing = ", ".join(detail.get("missing") or [])
            rows.append(("lf-warn", "Needs more information", f"missing {missing}" if missing else ""))
        elif step == "SEARCH_MATCHES":
            results = detail.get("results") or []
            top = f"top similarity {results[0]['similarity']:.2f}" if results else "no results"
            rows.append(("lf-ok", "Searching database", f"{len(results)} candidates"))
            rows.append(("lf-ok", "Semantic matching", top))
        elif step == "CALCULATE_CONFIDENCE":
            scores = detail.get("scores") or []
            best = f"best {scores[0]['score']:.0%} ({scores[0]['level']})" if scores else ""
            rows.append(("lf-ok", "Calculating confidence", best))
        elif step == "CONFIDENCE_ROUTER":
            decision = detail.get("decision", "")
            rows.append(("lf-ok", router_labels.get(decision, decision.replace("_", " ").title()), ""))
        elif step == "VERIFY_OWNERSHIP":
            passed = detail.get("result") == "PASS"
            rows.append(
                (
                    "lf-ok" if passed else "lf-fail",
                    "Ownership verified" if passed else "Ownership check failed",
                    f"match score {detail.get('match_score', 0):.2f}",
                )
            )
        elif step == "CREATE_PICKUP":
            rows.append(("lf-ok", "Pickup request created", str(detail.get("pickup_request_id", ""))))
        elif step == "NOTIFY_USER":
            rows.append(("lf-ok", "User notified", ""))
        elif step == "HUMAN_ESCALATION":
            rows.append(("lf-warn", "Escalated to staff", str(detail.get("escalation_id", ""))))

    # The turn ends waiting on the user; show that as the live step.
    if turn.get("awaiting") == AWAITING_OWNERSHIP_PROOF:
        rows.append(("lf-live", "Awaiting ownership proof", "verification pending"))

    return rows


def render_agent_status(turn: dict[str, Any]) -> str:
    rows = agent_activity(turn)
    if not rows:
        return ""

    marks = {"lf-ok": "✓", "lf-warn": "!", "lf-fail": "✗", "lf-live": "●"}
    steps = "".join(
        f'<div class="lf-step {css}" style="animation-delay:{index * 0.05:.2f}s">'
        f'<span class="lf-step-mark">{marks[css]}</span>'
        f"<span>{_esc(label)}</span>"
        + (f'<span class="lf-step-detail">{_esc(detail)}</span>' if detail else "")
        + "</div>"
        for index, (css, label, detail) in enumerate(rows)
    )
    return (
        '<div class="lf-activity">'
        '  <div class="lf-activity-title">Agent activity</div>'
        f"  {steps}"
        "</div>"
    )


def render_signal_table(confidence: dict[str, Any]) -> str:
    """The per-signal confidence breakdown, as a table the theme can style."""
    signals = confidence.get("signals") or {}
    if not signals:
        return ""
    weights = confidence.get("weights") or {}

    cells: list[str] = []
    for name, value in signals.items():
        score = (
            '<td class="lf-unset">not stated</td>'
            if value is None
            else f'<td class="lf-num">{value:.2f}</td>'
        )
        weight = f"{weights[name]:.0%}" if name in weights else "—"
        cells.append(
            f"<tr><td>{_pretty(name)}</td>{score}"
            f'<td class="lf-num">{weight}</td></tr>'
        )
    rows = "".join(cells)
    return (
        '<table class="lf-table">'
        "<tr><th>Signal</th><th>Score</th><th>Weight</th></tr>"
        f"{rows}</table>"
    )


def render_chat_bubble(role: str, content: str) -> str:
    is_user = role == "user"
    return (
        f'<div class="lf-turn {"lf-turn-user" if is_user else ""}">'
        f'  <div class="lf-role"><span class="lf-role-dot"></span>{"You" if is_user else "AI Agent"}</div>'
        f'  <div class="lf-bubble {"" if is_user else "lf-bubble-agent"}">{_esc(content)}</div>'
        "</div>"
    )


def render_working(agent_state: dict[str, Any] | None) -> str:
    """In-flight indicator. The sub-label reflects the stage that will actually run."""
    if agent_state and agent_state.get("awaiting") == AWAITING_OWNERSHIP_PROOF:
        stage = "Verifying ownership against stored evidence"
    else:
        stage = "Searching lost &amp; found database"
    return (
        '<div class="lf-working">'
        '  <div class="lf-dots"><span></span><span></span><span></span></div>'
        "  <div>"
        '    <div class="lf-working-text">AI Agent is analyzing your request</div>'
        f'    <div class="lf-working-sub">{stage}…</div>'
        "  </div>"
        "</div>"
    )


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------


def render_result_cards(turn: dict[str, Any]) -> None:
    """The visual outcome of a turn: help sources, match, success, or escalation."""
    pickup = turn.get("pickup_request")
    escalation = turn.get("escalation")
    candidate = turn.get("selected_candidate")
    action = turn.get("next_action")

    # Side turns leave any open case untouched, so don't re-render its cards.
    if action == ACTION_ANSWER_HELP:
        if turn.get("help_result"):
            st.markdown(render_help_card(turn["help_result"]), unsafe_allow_html=True)
        return
    if action in (ACTION_CASE_STATUS, ACTION_OUT_OF_SCOPE):
        return

    if pickup and candidate:
        st.markdown(
            render_success_card(candidate, pickup, turn.get("confidence")),
            unsafe_allow_html=True,
        )
    elif escalation:
        st.markdown(render_escalation_card(escalation), unsafe_allow_html=True)
    elif candidate and turn.get("awaiting") == AWAITING_OWNERSHIP_PROOF:
        st.markdown(render_match_card(candidate), unsafe_allow_html=True)
    else:
        for row in (turn.get("candidate_matches") or [])[:2]:
            st.markdown(render_match_card(row, verification_required=False), unsafe_allow_html=True)


def render_workflow_panel(turn: dict[str, Any]) -> None:
    """The detail view, collapsed by default so the chat stays clean."""
    with st.expander("How the agent worked", expanded=False):
        st.markdown("**Decision path**")
        st.code(render_trace(turn), language=None)

        intent = next((e["detail"] for e in turn.get("trace", []) if e["step"] == "CLASSIFY_INTENT"), None)
        if intent:
            st.caption(f"Intent: `{intent.get('intent')}` via `{intent.get('method')}` — {intent.get('reason')}")
        if turn.get("next_action") == ACTION_ANSWER_HELP and turn.get("help_result"):
            st.markdown("**Help retrieval**")
            st.code(json.dumps(turn["help_result"], indent=2, default=str), language="json")

        details = turn.get("item_details") or {}
        st.markdown("**Intent & entity extraction**")
        st.code(
            json.dumps(
                {
                    key: details.get(key)
                    for key in ("category", "brand", "color", "location", "time_phrase", "lost_time")
                },
                indent=2,
            ),
            language="json",
        )
        st.caption(f"Extracted by: `{details.get('extracted_by', 'n/a')}`")

        candidates = turn.get("candidate_matches") or []
        st.markdown(f"**Candidate matching & confidence** — {len(candidates)} retrieved")
        if not candidates:
            st.caption("No candidates retrieved.")
        for rank, row in enumerate(candidates, start=1):
            confidence = row.get("confidence") or {}
            st.markdown(
                f"`#{rank}` · item {row['id']} · {row['description'][:58]} · "
                f"similarity {row['similarity']:.2f} · "
                f"confidence {confidence.get('score', 0):.0%} ({confidence.get('level', '-')})"
            )
            table = render_signal_table(confidence)
            if table:
                st.markdown(table, unsafe_allow_html=True)

        st.markdown("**Ownership verification**")
        verification = turn.get("verification_result")
        if verification:
            st.code(json.dumps(verification, indent=2), language="json")
        else:
            st.caption("Not reached on this turn.")

        st.markdown("**Final action**")
        st.code(
            f"next_action = {turn.get('next_action') or '-'}\n"
            f"awaiting    = {turn.get('awaiting') or '-'}",
            language=None,
        )

        if turn.get("errors"):
            st.error("Errors: " + "; ".join(turn["errors"]))

        with st.expander("Raw tool calls", expanded=False):
            st.code(json.dumps(turn.get("trace", []), indent=2, default=str), language="json")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            '<div class="lf-side-brand">'
            '  <div class="lf-side-title">Lost &amp; Found AI</div>'
            '  <div class="lf-side-sub">AI-powered recovery assistant</div>'
            "</div>",
            unsafe_allow_html=True,
        )

        reachable = database_is_reachable()
        backend = current_backend()
        tracing = get_tracer().enabled

        rows = [
            (True, "Agent Online", "LangGraph"),
            (reachable, "Database Connected" if reachable else "Database Unreachable", "PostgreSQL"),
            (reachable, "Vector Search Ready" if reachable else "Vector Search Offline", backend),
            (
                settings.llm_enabled,
                "LLM Connected" if settings.llm_enabled else "Rule-based Extraction",
                settings.llm_model if settings.llm_enabled else "no LLM key",
            ),
            (tracing, "Tracing Active" if tracing else "Tracing Off", "Langfuse"),
        ]
        status_html = "".join(
            f'<div class="lf-status-row">'
            f'<span class="lf-status-dot {"up" if ok else "down"}"></span>'
            f"<span>{_esc(label)}</span>"
            f'<span class="lf-status-note">{_esc(note)}</span>'
            "</div>"
            for ok, label, note in rows
        )
        st.markdown(
            f'<div class="lf-side-heading">System status</div>{status_html}',
            unsafe_allow_html=True,
        )

        if not reachable:
            st.error(
                "Cannot reach the database. Check `DATABASE_URL` in `.env`, then run "
                "`python scripts/seed_database.py`."
            )

        st.markdown(
            '<div class="lf-side-heading">About</div>'
            '<div class="lf-about">Semantic search over items handed in at the desk, '
            "scored on multiple signals. Nothing is released until you prove ownership "
            "with a detail only the owner would know.</div>",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="lf-side-heading">Appearance</div>', unsafe_allow_html=True)
        chosen = st.segmented_control(
            "Theme",
            options=THEME_CHOICES,
            default=st.session_state.theme,
            label_visibility="collapsed",
            key="theme_control",
        )
        if chosen and chosen != st.session_state.theme:
            st.session_state.theme = chosen
            st.rerun()
        if st.session_state.theme == "System":
            st.caption("Follows your operating system setting.")

        st.divider()
        if st.button("New conversation", width="stretch"):
            reset_conversation()
            st.rerun()


# ---------------------------------------------------------------------------
# Session and flow
# ---------------------------------------------------------------------------


def init_state() -> None:
    st.session_state.setdefault("agent_state", None)
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending", None)
    st.session_state.setdefault("celebrate", False)
    st.session_state.setdefault("theme", "System")


def reset_conversation() -> None:
    st.session_state.agent_state = None
    st.session_state.messages = []
    st.session_state.pending = None
    st.session_state.celebrate = False


def render_history() -> None:
    for entry in st.session_state.messages:
        st.markdown(render_chat_bubble(entry["role"], entry["content"]), unsafe_allow_html=True)
        turn = entry.get("turn")
        if turn:
            status = render_agent_status(turn)
            if status:
                st.markdown(status, unsafe_allow_html=True)
            render_result_cards(turn)
            render_workflow_panel(turn)


def process_pending(message: str) -> None:
    """Render the in-flight state, run the agent, then store the result."""
    st.markdown(render_chat_bubble("user", message), unsafe_allow_html=True)
    working = st.empty()
    working.markdown(render_working(st.session_state.agent_state), unsafe_allow_html=True)

    try:
        result = run_turn(st.session_state.agent_state, message)
    except Exception as exc:  # noqa: BLE001 - keep the UI usable
        working.empty()
        st.session_state.messages.append({"role": "user", "content": message})
        st.session_state.messages.append(
            {"role": "assistant", "content": f"Something went wrong: {exc}", "turn": None}
        )
        return

    working.empty()
    st.session_state.agent_state = result
    # Fires once, on the rerun that first shows the success card.
    st.session_state.celebrate = bool(result.get("pickup_request"))
    st.session_state.messages.append({"role": "user", "content": message})
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": result.get("agent_response", ""),
            "turn": dict(result),
        }
    )


def render_empty_state() -> None:
    st.markdown(
        '<div class="lf-empty">'
        '  <div class="lf-empty-title">Describe what you lost</div>'
        '  <div class="lf-empty-sub">Include anything you remember - what it is, its colour '
        "or brand, roughly where and when. The agent searches everything handed in, scores "
        "how confident it is, and asks you to prove ownership before releasing anything.</div>"
        '  <div class="lf-empty-hint">e.g. “I lost my black Sony headphones near the library '
        "yesterday around 4 PM.”</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    st.markdown(build_styles(st.session_state.theme), unsafe_allow_html=True)

    st.markdown(
        '<div class="lf-header"><div class="lf-header-row">'
        '  <div class="lf-orb"><div class="lf-orb-ring"></div>'
        '       <div class="lf-orb-ring"></div><div class="lf-orb-core"></div></div>'
        "  <div>"
        '    <h1 class="lf-title">Lost &amp; Found AI</h1>'
        '    <p class="lf-subtitle">Your intelligent assistant for recovering lost belongings.</p>'
        "  </div>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    render_sidebar()
    render_history()

    if st.session_state.pending:
        message = st.session_state.pending
        st.session_state.pending = None
        process_pending(message)
        st.rerun()

    if not st.session_state.messages:
        render_empty_state()

    if st.session_state.celebrate:
        st.session_state.celebrate = False
        st.balloons()

    if prompt := st.chat_input("Describe the item you lost…"):
        st.session_state.pending = prompt
        st.rerun()


if __name__ == "__main__":
    main()
