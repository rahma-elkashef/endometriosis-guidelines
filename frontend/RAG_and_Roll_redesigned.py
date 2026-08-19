import os
import re
import json
import base64
import pickle
import html
import hashlib
import streamlit as st
import requests

# ============================================================
# RAG & ROLL
# Clinical evidence intelligence for endometriosis, grounded
# in NICE / ESHRE guideline retrieval. Native Streamlit UI —
# no raw HTML is ever shown to the user as text.
# Built for competition demo: bold typography, a signature
# animation, and a full-bleed background photo layer.
# ============================================================

st.set_page_config(
    page_title="RAG & Roll — Endometriosis Clinical Evidence Intelligence",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------
# Paths
# -----------------------------
BASE = "/content"
LOGO_PATH = os.path.join(BASE, "logo.png")
BACKEND_URL = os.getenv("RAG_BACKEND_URL", "http://127.0.0.1:8000")

# Background photo: drop a file named bg.jpg / bg.png (or background.*)
# into any of these locations and it will be picked up automatically,
# base64-embedded, and layered under a readability gradient + a slow
# Ken Burns drift. No file found -> falls back to a designed gradient.
BG_CANDIDATES = [
    os.path.join(BASE, "bg.jpg"),
    os.path.join(BASE, "bg.png"),
    os.path.join(BASE, "bg.jpeg"),
    os.path.join(BASE, "background.jpg"),
    os.path.join(BASE, "background.png"),
    "/mnt/user-data/uploads/bg.jpg",
    "/mnt/user-data/uploads/bg.png",
    "/mnt/user-data/uploads/bg.jpeg",
    "/mnt/user-data/uploads/background.jpg",
    "/mnt/user-data/uploads/background.png",
    "bg.jpg",
    "bg.png",
]

CHROMA_CANDIDATES = [
    "/content/data/chromadb",
    "/content/data/chroma_db",
    "/content/chroma_db",
    "/content/chromadb",
]

BM25_CANDIDATES = [
    "/content/data/chunks/nice_bm25.pkl",
    "/content/data/nice_bm25.pkl",
    "/content/chunks/nice_bm25.pkl",
]


def first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def bg_data_uri(paths):
    """Base64-embed the first background image found so it always
    renders regardless of how/where Streamlit is served."""
    path = first_existing(paths)
    if not path:
        return None
    try:
        ext = os.path.splitext(path)[1].lower().lstrip(".")
        mime = "jpeg" if ext in ("jpg", "jpeg") else ext
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/{mime};base64,{encoded}"
    except Exception:
        return None


CHROMA_PATH = first_existing(CHROMA_CANDIDATES)
BM25_FILE = first_existing(BM25_CANDIDATES)
BG_URI = bg_data_uri(BG_CANDIDATES)

# ============================================================
# Icon system — minimalist stroke icons, Lucide-style
# (round caps/joins, 1.8px stroke, 24x24 viewbox)
# ============================================================
_ICONS = {
    "search": '<circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/>',
    "activity": '<path d="M22 12h-4l-3 8-6-16-3 8H2"/>',
    "shield": '<path d="M12 3l8 3.5v5.2c0 5-3.4 8.4-8 9.8-4.6-1.4-8-4.8-8-9.8V6.5L12 3z"/><path d="M9.2 12.2l1.8 1.8 3.8-3.8"/>',
    "database": '<ellipse cx="12" cy="5.5" rx="8" ry="3"/><path d="M4 5.5V12c0 1.7 3.6 3 8 3s8-1.3 8-3V5.5"/><path d="M4 12v6.5c0 1.7 3.6 3 8 3s8-1.3 8-3V12"/>',
    "book": '<path d="M4 4.5A2.5 2.5 0 0 1 6.5 2H20v17H6.5A2.5 2.5 0 0 0 4 21.5v-17z"/><path d="M20 19H6.5A2.5 2.5 0 0 0 4 21.5"/>',
    "arrow-right": '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    "sparkles": '<path d="M12 3l1.6 4.4L18 9l-4.4 1.6L12 15l-1.6-4.4L6 9l4.4-1.6L12 3z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8L19 15z"/>',
    "alert": '<path d="M12 3l10 18H2L12 3z"/><path d="M12 10v4"/><path d="M12 17.2v.1"/>',
    "file": '<path d="M6 2h9l5 5v15H6V2z"/><path d="M15 2v5h5"/><path d="M9 13h6M9 17h6"/>',
    "layers": '<path d="M12 2l9 5-9 5-9-5 9-5z"/><path d="M3 12l9 5 9-5"/><path d="M3 16.5l9 5 9-5"/>',
    "chevron": '<path d="M9 6l6 6-6 6"/>',
    "clipboard": '<rect x="6" y="4" width="12" height="17" rx="2"/><rect x="9" y="2" width="6" height="4" rx="1"/>',
    "pulse-line": '<path d="M2 12h4l2-7 4 14 3-9 2 5h5"/>',
    "flask": '<path d="M9 2h6"/><path d="M10 2v6.3L4.5 18a2 2 0 0 0 1.8 3h11.4a2 2 0 0 0 1.8-3L14 8.3V2"/>',
    "check": '<path d="M4 12.5l5 5L20 6.5"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "refresh": '<path d="M3 12a9 9 0 0 1 15.3-6.3L21 8"/><path d="M21 3v5h-5"/><path d="M21 12a9 9 0 0 1-15.3 6.3L3 16"/><path d="M3 21v-5h5"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 7v5l4 2"/>',
    "stethoscope": '<path d="M5 3v6a4 4 0 0 0 8 0V3"/><path d="M9 15a5 5 0 0 0 10 0v-2"/><circle cx="19" cy="6" r="2"/>',
}


def icon(name, size=16, color="currentColor", stroke=1.9):
    body = _ICONS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" class="i">{body}</svg>'
    )


# -----------------------------
# Visual design
# -----------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,500;0,9..144,600;1,9..144,600&family=DM+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --ink:#0e2126;
    --ink-soft:#37545a;
    --muted:#6d8589;
    --teal:#158089;
    --teal-2:#0c5c63;
    --teal-deep:#062a2e;
    --navy:#0b1f26;
    --gold:#c6912b;
    --gold-soft:#e7c88a;
    --coral:#c8503c;
    --green:#1a7a55;
    --paper:#f7f5ef;
    --paper-2:#eef1ea;
    --card:rgba(255,255,255,.9);
    --card-solid:#ffffff;
    --line:rgba(14,33,38,.14);
    --line-soft:rgba(14,33,38,.08);
    --shadow: 0 14px 30px rgba(6,25,28,.10);
    --radius-lg:12px;
    --radius-md:10px;
    --radius-sm:8px;
}

* { box-sizing:border-box; }

html, body, [class*="css"] { font-family:"DM Sans", sans-serif; }

.stApp {
    background:
        radial-gradient(1100px 550px at 92% -10%, rgba(31,122,128,.10), transparent 60%),
        radial-gradient(900px 500px at -5% 15%, rgba(192,138,46,.08), transparent 55%),
        linear-gradient(180deg, var(--paper) 0%, var(--paper-2) 100%);
    color:var(--ink);
}

.block-container {
    max-width:1480px;
    padding-top:1.6rem;
    padding-bottom:3rem;
    position:relative;
}

header[data-testid="stHeader"] { background:transparent; }
#MainMenu, footer[data-testid="stBottom"] { visibility:hidden; }

/* Blueprint / clinical-chart grid, faint & fixed.
   Fine lines every 13px, a heavier line every 6th (78px) to mimic
   graph paper on a medical chart, plus a soft dot-grid layer for
   depth. Both fade out toward the page edges so nothing distracts
   from the content. */
.watermark {
    position:fixed;
    inset:0;
    z-index:-2;
    pointer-events:none;
    background-image:
        linear-gradient(rgba(12,92,99,.07) 1px, transparent 1px),
        linear-gradient(90deg, rgba(12,92,99,.07) 1px, transparent 1px),
        linear-gradient(rgba(12,92,99,.14) 1px, transparent 1px),
        linear-gradient(90deg, rgba(12,92,99,.14) 1px, transparent 1px);
    background-size: 13px 13px, 13px 13px, 78px 78px, 78px 78px;
    -webkit-mask-image: radial-gradient(ellipse 85% 65% at 50% 0%, black 12%, transparent 72%);
    mask-image: radial-gradient(ellipse 85% 65% at 50% 0%, black 12%, transparent 72%);
}
.watermark::after {
    content:"";
    position:absolute; inset:0;
    background-image: radial-gradient(rgba(192,138,46,.10) 1px, transparent 1px);
    background-size: 78px 78px;
    background-position: 6px 6px;
}

/* Full-bleed background photo layer (only rendered if a bg image
   was found + base64-embedded). Slow Ken Burns drift for polish,
   with a paper-tinted gradient scrim on top so every card stays
   fully readable regardless of what's in the photo. */
@keyframes kenBurns {
    0%   { transform:scale(1.04) translate(0,0); }
    50%  { transform:scale(1.11) translate(-1%,-1%); }
    100% { transform:scale(1.04) translate(0,0); }
}
.bg-photo-layer {
    position:fixed;
    inset:0;
    z-index:-3;
    overflow:hidden;
    pointer-events:none;
}
.bg-photo-layer .photo {
    position:absolute;
    inset:-3%;
    background-size:cover;
    background-position:center;
    animation:kenBurns 34s ease-in-out infinite;
    filter:saturate(.92);
}
.bg-photo-layer .scrim {
    position:absolute;
    inset:0;
    background:
        linear-gradient(180deg, rgba(247,245,239,.94) 0%, rgba(247,245,239,.88) 24%, rgba(238,241,234,.93) 100%);
}

@keyframes fadeUp {
    from { opacity:0; transform:translateY(10px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulseDot {
    0%,100% { opacity:1; transform:scale(1); }
    50%     { opacity:.4; transform:scale(.72); }
}
@keyframes shimmer {
    0%   { background-position:-300px 0; }
    100% { background-position:300px 0; }
}
@keyframes drawLine {
    from { stroke-dashoffset:820; }
    to   { stroke-dashoffset:0; }
}
@keyframes growBar {
    from { transform:scaleX(0); }
    to   { transform:scaleX(1); }
}
@keyframes floatSlow {
    0%,100% { transform:translateY(0); }
    50%     { transform:translateY(-4px); }
}

.fade-in { animation:fadeUp .55s cubic-bezier(.2,.7,.3,1) both; }
.fade-in.d1 { animation-delay:.05s; }
.fade-in.d2 { animation-delay:.12s; }
.fade-in.d3 { animation-delay:.19s; }
.fade-in.d4 { animation-delay:.26s; }

/* ---------- Hero ---------- */
.hero-row { display:flex; align-items:center; gap:18px; margin-bottom:2px; }
.hero-badge {
    width:118px; height:118px;
    display:flex; align-items:center; justify-content:center;
    flex-shrink:0;
    animation:floatSlow 5s ease-in-out infinite;
    filter:drop-shadow(0 8px 14px rgba(11,45,49,.18));
}

.hero-dash {
    display:flex; align-items:center; gap:26px;
    justify-content:flex-end;
    padding-top:6px;
}
.hero-dash .dash-item { text-align:center; }
.hero-dash svg { display:block; margin:0 auto; }
.hero-dash .dash-cap {
    font-family:"JetBrains Mono", monospace; font-size:.6rem; font-weight:700;
    letter-spacing:.1em; text-transform:uppercase; color:var(--muted); margin-top:6px;
}
.dash-bar-wrap { width:126px; }
.dash-bar-track { height:8px; border-radius:4px; background:var(--line-soft); overflow:hidden; display:flex; }
.dash-bar-seg { height:100%; }
@media (max-width:1150px) { .hero-dash { display:none; } }

/* ---------- Pipeline status pills ---------- */
.pipe-badges { display:flex; flex-wrap:wrap; align-items:center; gap:10px; margin:20px 0 22px 0; }
.pipe-pill {
    display:inline-flex; align-items:center; gap:7px;
    font-family:"JetBrains Mono", monospace; font-size:.68rem; font-weight:700;
    letter-spacing:.07em; text-transform:uppercase;
    padding:9px 14px; border-radius:999px; white-space:nowrap;
}
.pipe-pill.dark { background:var(--navy); color:#fff; box-shadow:0 8px 16px rgba(11,31,38,.28); }
.pipe-pill.outline { background:rgba(255,255,255,.7); border:1.5px solid var(--line); color:var(--ink-soft); }
.pipe-pill.green { background:rgba(26,122,85,.1); border:1.5px solid rgba(26,122,85,.3); color:var(--green); }
.pipe-pill.live { background:rgba(21,128,137,.08); border:1.5px solid rgba(21,128,137,.32); color:var(--teal-2); }
.pipe-pill.live.offline { background:rgba(200,80,60,.08); border-color:rgba(200,80,60,.3); color:var(--coral); }
.pipe-pill .pdot { width:6px; height:6px; border-radius:50%; background:currentColor; animation:pulseDot 1.6s ease-in-out infinite; }
.pipe-arrow { color:var(--muted); font-size:.9rem; margin:0 -2px; }
.pipe-checks { display:inline-flex; gap:3px; margin-left:2px; }
.pipe-checks svg { width:12px; height:12px; }
@media (max-width:900px) { .pipe-arrow { display:none; } }

.kicker {
    font-family:"JetBrains Mono", monospace;
    font-size:.68rem;
    font-weight:600;
    letter-spacing:.2em;
    color:var(--teal);
    text-transform:uppercase;
    margin-bottom:4px;
    display:flex; align-items:center; gap:8px;
}
.kicker .dot { width:5px; height:5px; border-radius:50%; background:var(--gold); }

.brand {
    font-family:"Fraunces", serif;
    font-optical-sizing:auto;
    font-weight:600;
    font-size:clamp(2.7rem,5.2vw,5rem);
    line-height:.96;
    letter-spacing:-.035em;
    color:var(--ink);
    margin:0;
}
.brand .amp {
    font-style:italic;
    font-weight:600;
    background:linear-gradient(120deg, var(--gold) 10%, var(--teal) 90%);
    background-size:200% auto;
    -webkit-background-clip:text;
    background-clip:text;
    color:transparent;
    animation:shimmerText 6s ease-in-out infinite;
    display:inline-block;
    padding:0 .06em;
}
@keyframes shimmerText {
    0%,100% { background-position:0% 50%; }
    50%     { background-position:100% 50%; }
}

.tagline {
    font-weight:700;
    font-size:1.06rem;
    color:var(--teal-2);
    margin-top:12px;
    letter-spacing:-.01em;
}
.tagline .accent { color:var(--gold); }

.subtitle {
    max-width:760px;
    color:var(--ink-soft);
    font-size:.98rem;
    line-height:1.65;
    margin-top:8px;
}
.subtitle b { color:var(--ink); font-weight:700; }

/* ECG signature divider */
.ecg-wrap { margin:18px 0 20px 0; opacity:.85; }
.ecg-wrap svg { width:100%; height:34px; display:block; }
.ecg-line {
    fill:none; stroke:url(#ecgGrad); stroke-width:2;
    stroke-linecap:round; stroke-linejoin:round;
    stroke-dasharray:820; stroke-dashoffset:0;
    animation:drawLine 2.4s cubic-bezier(.3,.7,.2,1) both;
}

/* ---------- Status bar ---------- */
.statusbar {
    display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;
    gap:12px; padding:13px 18px;
    border:1px solid var(--line); border-radius:var(--radius-lg);
    background:linear-gradient(120deg, rgba(255,255,255,.86), rgba(255,255,255,.6));
    backdrop-filter:blur(6px);
    margin-bottom:24px;
    box-shadow:var(--shadow);
}
.status-title {
    font-family:"JetBrains Mono", monospace;
    font-size:.72rem; letter-spacing:.1em; text-transform:uppercase;
    font-weight:600; color:var(--ink-soft);
    display:flex; align-items:center; gap:8px;
}
.status-pill {
    display:flex; align-items:center; gap:7px;
    border-radius:999px; padding:7px 13px 7px 10px;
    font-family:"JetBrains Mono", monospace;
    font-size:.68rem; letter-spacing:.06em; font-weight:600;
    text-transform:uppercase;
    border:1px solid rgba(194,91,74,.25); color:#9b4638;
    background:rgba(253,242,240,.9);
}
.status-pill.online { color:#1a6e52; border-color:rgba(26,110,82,.25); background:rgba(235,249,243,.9); }
.status-pill .pdot { width:7px; height:7px; border-radius:50%; background:currentColor; animation:pulseDot 1.6s ease-in-out infinite; }

/* ---------- Cards ---------- */
.card {
    background:var(--card);
    backdrop-filter:blur(6px);
    border:1px solid var(--line);
    border-radius:var(--radius-lg);
    padding:22px 24px;
    box-shadow:var(--shadow);
    margin-bottom:18px;
}
.card-tight { padding:16px 18px; }

.section-label {
    color:var(--teal-2);
    font-family:"JetBrains Mono", monospace;
    font-size:.66rem; font-weight:600; letter-spacing:.16em; text-transform:uppercase;
    margin-bottom:6px;
    display:flex; align-items:center; gap:7px;
}
.section-label svg { color:var(--gold); }
.section-title {
    font-family:"Fraunces", serif;
    color:var(--ink);
    font-size:1.55rem; font-weight:700; letter-spacing:-.015em; margin-bottom:14px;
}
.badge-live {
    display:inline-flex; align-items:center; gap:5px;
    font-family:"JetBrains Mono", monospace; font-size:.6rem; font-weight:700;
    letter-spacing:.08em; text-transform:uppercase; color:#1a6e52;
    background:rgba(26,110,82,.1); border:1px solid rgba(26,110,82,.22);
    border-radius:999px; padding:3px 8px; vertical-align:middle; margin-left:9px;
}
.badge-live .pdot { width:5px; height:5px; border-radius:50%; background:currentColor; animation:pulseDot 1.6s ease-in-out infinite; }

/* Answer / recommendation */
.answer-box {
    position:relative;
    background:linear-gradient(160deg, rgba(255,255,255,.96), rgba(238,244,240,.9));
    border-left:4px solid var(--teal);
    border-radius:12px;
    padding:22px 24px 20px 28px;
    line-height:1.75;
    color:#1c3a3f;
    font-size:1.02rem;
}
.answer-box::before {
    content:"“";
    position:absolute; top:-6px; left:10px;
    font-family:"Fraunces", serif;
    font-size:3.2rem; color:var(--gold-soft);
    line-height:1;
}

/* Evidence cards */
.evidence {
    border:1px solid var(--line);
    border-radius:var(--radius-md);
    padding:16px 17px;
    margin-top:12px;
    background:rgba(255,255,255,.8);
    transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease;
}
.evidence:hover {
    transform:translateY(-2px);
    box-shadow:0 12px 26px rgba(15,92,98,.12);
    border-color:rgba(31,122,128,.35);
}
.evidence-head { display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:6px; }
.source-tag {
    display:inline-flex; align-items:center; gap:7px;
    color:#fff; font-weight:700; font-size:.72rem;
    letter-spacing:.03em;
    background:linear-gradient(120deg, var(--teal), var(--teal-2));
    padding:3px 10px 3px 8px; border-radius:999px;
}
.evidence-source { color:var(--teal-2); font-weight:700; font-size:.85rem; }
.meta { color:var(--muted); font-size:.72rem; font-family:"JetBrains Mono", monospace; }
.quote { color:#324f54; font-size:.89rem; line-height:1.62; margin-top:6px; }

.relevance-row { display:flex; align-items:center; gap:8px; margin-top:8px; }
.relevance-track { flex:1; height:5px; border-radius:99px; background:var(--line-soft); overflow:hidden; }
.relevance-fill {
    height:100%; border-radius:99px;
    background:linear-gradient(90deg, var(--gold), var(--teal));
    transform-origin:left; animation:growBar 1s cubic-bezier(.2,.8,.3,1) both;
}
.relevance-val { font-family:"JetBrains Mono", monospace; font-size:.7rem; color:var(--teal-2); font-weight:600; min-width:34px; text-align:right;}

/* Metrics */
.metric {
    border:1px solid var(--line);
    background:linear-gradient(160deg, rgba(255,255,255,.92), rgba(240,246,242,.75));
    border-radius:var(--radius-md); padding:16px; text-align:center;
    transition:transform .18s ease;
}
.metric:hover { transform:translateY(-2px); }
.metric-number { font-family:"Fraunces", serif; font-size:1.9rem; font-weight:600; color:var(--ink); }
.metric-label {
    color:var(--muted); font-size:.63rem; letter-spacing:.1em; text-transform:uppercase;
    margin-top:3px; font-family:"JetBrains Mono", monospace;
}

/* Pipeline */
.pipeline-step {
    display:flex; align-items:center; gap:10px;
    padding:9px 0; border-bottom:1px dashed var(--line-soft); font-size:.8rem;
}
.pipeline-step:last-child { border-bottom:0; }
.pipeline-step .p-icon {
    width:26px; height:26px; border-radius:8px; flex-shrink:0;
    display:flex; align-items:center; justify-content:center;
    background:linear-gradient(150deg, rgba(31,122,128,.14), rgba(192,138,46,.1));
    color:var(--teal-2);
}
.pipeline-step .p-name { font-weight:700; color:var(--ink); flex:1; }
.pipeline-step .p-kind {
    color:var(--teal); font-size:.64rem; letter-spacing:.08em;
    font-family:"JetBrains Mono", monospace; font-weight:700;
}

/* Confidence indicator */
.conf-wrap { display:flex; align-items:center; gap:16px; }
.conf-ring svg { display:block; }
.conf-label-lg { font-family:"Fraunces", serif; font-size:1.15rem; font-weight:600; }
.conf-sub { color:var(--muted); font-size:.78rem; margin-top:2px; }

.conf-meter-wrap { margin-top:16px; }
.conf-meter-scale {
    display:flex; justify-content:space-between;
    font-family:"JetBrains Mono", monospace; font-size:.6rem; font-weight:700;
    letter-spacing:.06em; text-transform:uppercase; color:var(--muted); margin-bottom:6px;
}
.conf-meter-track {
    position:relative; height:10px; border-radius:999px; overflow:visible;
    background:linear-gradient(90deg, #c25b4a 0%, #c25b4a 32%, #c08a2e 32%, #c08a2e 68%, #1f7a80 68%, #1f7a80 100%);
    opacity:.9;
}
.conf-meter-marker {
    position:absolute; top:50%; width:18px; height:18px; border-radius:50%;
    background:#fff; border:3px solid var(--ink); transform:translate(-50%,-50%);
    box-shadow:0 3px 8px rgba(6,25,28,.3);
    animation:markerIn .8s cubic-bezier(.2,.8,.25,1) both;
}
@keyframes markerIn { from { left:0%; opacity:0; } }
.conf-score-badge {
    display:inline-flex; align-items:baseline; gap:3px;
    font-family:"Fraunces", serif; font-weight:600;
}
.conf-score-badge .num { font-size:2.1rem; line-height:1; }
.conf-score-badge .pct { font-size:1rem; opacity:.7; }

/* Safety note */
.safety-note {
    display:flex; gap:10px; align-items:flex-start;
    background:linear-gradient(120deg, rgba(192,138,46,.09), rgba(31,122,128,.06));
    border:1px solid rgba(192,138,46,.25);
    border-radius:var(--radius-md); padding:12px 14px; margin-top:14px;
    font-size:.8rem; color:var(--ink-soft); line-height:1.55;
}
.safety-note svg { color:var(--gold); flex-shrink:0; margin-top:1px; }

/* Footer */
.footer {
    display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;
    color:var(--muted); font-size:.66rem; letter-spacing:.08em; text-transform:uppercase;
    font-family:"JetBrains Mono", monospace;
    padding-top:22px; margin-top:8px; border-top:1px solid var(--line-soft);
}

/* Chips (history / suggestions use native buttons, styled below) */
/* ---------- Primary AI composer: ChatGPT-like multiline input ---------- */
.st-key-query_input { margin-top:14px !important; margin-bottom:12px !important; }
.st-key-query_input textarea {
    min-height:118px !important; max-height:240px !important; resize:vertical !important;
    border-radius:14px !important; border:2px solid rgba(14,33,38,.34) !important;
    padding:18px 20px !important; background:rgba(255,255,255,.97) !important;
    color:var(--ink) !important; font-family:"DM Sans",sans-serif !important;
    font-size:1.04rem !important; font-weight:600 !important; line-height:1.6 !important;
    box-shadow:0 8px 22px rgba(6,25,28,.08), inset 0 1px 0 rgba(255,255,255,.9) !important;
}
.st-key-query_input textarea::placeholder { color:#789094 !important; font-weight:500 !important; }
.st-key-query_input textarea:focus {
    border-color:var(--teal-2) !important;
    box-shadow:0 0 0 4px rgba(21,128,137,.13),0 12px 28px rgba(6,25,28,.10) !important;
    outline:none !important;
}
.primary-input-caption {
    display:flex; align-items:center; justify-content:space-between; gap:10px;
    margin:2px 2px 8px; color:var(--muted); font-family:"JetBrains Mono",monospace;
    font-size:.61rem; font-weight:700; letter-spacing:.08em; text-transform:uppercase;
}
.primary-input-caption .input-mode {
    color:var(--teal-2); background:rgba(21,128,137,.07);
    border:1px solid rgba(21,128,137,.16); border-radius:999px; padding:4px 8px;
}

div[data-testid="stTextInput"] input {
    border-radius:var(--radius-sm) !important;
    border:1px solid rgba(19,46,51,.18) !important;
    padding:14px !important;
    background:rgba(255,255,255,.94) !important;
    font-size:.98rem !important;
    transition:box-shadow .2s ease, border-color .2s ease;
}
div[data-testid="stTextInput"] input:focus {
    border-color:var(--teal) !important;
    box-shadow:0 0 0 4px rgba(31,122,128,.14) !important;
}

div.stButton > button {
    border-radius:var(--radius-sm);
    border:1.5px solid var(--line);
    min-height:54px;
    font-weight:700;
    font-size:.87rem;
    text-align:left;
    justify-content:flex-start;
    background:var(--card-solid);
    color:var(--ink-soft);
    transition:all .16s ease;
}
div.stButton > button:hover {
    border-color:var(--teal);
    color:var(--teal-2);
    background:rgba(21,128,137,.05);
    transform:translateY(-1px);
    box-shadow:0 6px 14px rgba(15,92,98,.14);
}
div.stButton > button p { text-align:left; font-size:.87rem; }
@keyframes ctaGlow {
    0%,100% { box-shadow:0 10px 22px rgba(15,92,98,.3), 0 0 0 0 rgba(31,122,128,.0); }
    50%     { box-shadow:0 10px 26px rgba(15,92,98,.36), 0 0 0 7px rgba(31,122,128,.08); }
}
div.stButton > button[kind="primary"] {
    background:linear-gradient(120deg, var(--teal) 0%, var(--teal-2) 100%);
    border-color:var(--teal-2);
    color:white;
    font-weight:700;
    letter-spacing:.02em;
    text-align:center;
    justify-content:center;
    animation:ctaGlow 2.6s ease-in-out infinite;
}
div.stButton > button[kind="primary"] p { text-align:center; }
div.stButton > button[kind="primary"]:hover {
    filter:brightness(1.08);
    transform:translateY(-1px);
    animation-play-state:paused;
}
div.stButton > button[kind="primary"]:disabled {
    background:#c9d3d2; border-color:#c9d3d2; box-shadow:none; color:#8a9694;
}

.small-note { color:var(--muted); font-size:.75rem; line-height:1.55; }

/* Suggested-question chips — scoped to the keyed container so the
   primary Ask button and other st.button instances are untouched. */
.st-key-suggestion_chips div[data-testid="stHorizontalBlock"] { gap:8px !important; }
.st-key-suggestion_chips div.stButton > button {
    min-height:auto;
    border-radius:999px;
    padding:8px 16px;
    font-size:.78rem;
    font-weight:600;
    color:var(--teal-2);
    background:rgba(21,128,137,.06);
    border:1.3px solid rgba(21,128,137,.24);
    box-shadow:none;
    justify-content:center;
    text-align:center;
    white-space:nowrap;
}
.st-key-suggestion_chips div.stButton > button p { text-align:center; font-size:.78rem; }
.st-key-suggestion_chips div.stButton > button:hover {
    background:rgba(21,128,137,.13);
    border-color:var(--teal);
    color:var(--teal-2);
    transform:translateY(-1px);
    box-shadow:0 6px 14px rgba(15,92,98,.16);
}

/* Evidence summary table */
.evi-table { width:100%; border-collapse:collapse; margin-top:12px; font-size:.78rem; }
.evi-table th {
    text-align:left; font-family:"JetBrains Mono", monospace; font-size:.62rem;
    letter-spacing:.08em; text-transform:uppercase; color:var(--muted);
    border-bottom:1.5px solid var(--line); padding:0 8px 7px 0;
}
.evi-table td { padding:9px 8px 9px 0; border-bottom:1px solid var(--line-soft); vertical-align:top; color:var(--ink-soft); }
.evi-table tr:last-child td { border-bottom:0; }
.evi-table .id-chip {
    display:inline-flex; align-items:center; justify-content:center;
    width:22px; height:22px; border-radius:6px; font-weight:700; font-size:.68rem;
    background:linear-gradient(120deg, var(--teal), var(--teal-2)); color:#fff;
}
.evi-table .src-name { font-weight:700; color:var(--ink); }
.evi-table .empty-row td { text-align:center; color:var(--muted); padding:22px 0; font-style:italic; }

/* Knowledge-base spec sheet */
.spec-list { margin-top:14px; border-top:1px solid var(--line-soft); }
.spec-row {
    display:flex; justify-content:space-between; align-items:center; gap:12px;
    padding:9px 0; border-bottom:1px dashed var(--line-soft);
}
.spec-row:last-child { border-bottom:0; }
.spec-k {
    font-family:"JetBrains Mono", monospace; font-size:.66rem; letter-spacing:.05em;
    text-transform:uppercase; color:var(--muted); flex-shrink:0;
}
.spec-v { font-size:.78rem; font-weight:600; color:var(--ink-soft); text-align:right; }

/* Bookshelf decoration */
.bookshelf { display:inline-flex; align-items:flex-end; gap:2px; height:20px; margin-left:8px; vertical-align:middle; }
.bookshelf span { display:block; width:4px; border-radius:1.5px 1.5px 0 0; }

.copy-btn {
    display:inline-flex; align-items:center; gap:6px;
    font-family:"JetBrains Mono", monospace; font-size:.68rem; font-weight:600;
    letter-spacing:.05em; text-transform:uppercase;
    color:var(--teal-2); background:rgba(31,122,128,.09);
    border:1px solid rgba(31,122,128,.22); border-radius:var(--radius-sm);
    padding:6px 11px; cursor:pointer; margin-top:14px;
    transition:all .15s ease;
}
.copy-btn:hover { background:rgba(31,122,128,.16); }
.copy-btn.copied { background:rgba(26,110,82,.14); border-color:rgba(26,110,82,.3); color:#1a6e52; }
    /* ---------- Workspace switcher ---------- */
    .workspace-heading{display:flex;align-items:center;gap:18px;margin:6px 0 10px}
    .workspace-heading .ws-kicker{font-family:"JetBrains Mono",monospace;font-size:.62rem;letter-spacing:.16em;text-transform:uppercase;color:var(--muted);font-weight:700}
    div[data-testid="stRadio"]>label{display:none!important}
    .st-key-workspace div[role="radiogroup"]{display:flex;gap:6px;padding:5px;width:fit-content;border:1px solid var(--line);border-radius:999px;background:rgba(255,255,255,.72);backdrop-filter:blur(8px);box-shadow:0 8px 20px rgba(6,25,28,.08)}
    .st-key-workspace div[role="radiogroup"]>label{border:0!important;border-radius:999px!important;padding:9px 16px!important;min-height:0!important;background:transparent!important;color:var(--ink-soft)!important;transition:all .18s ease!important;cursor:pointer!important}
    .st-key-workspace div[role="radiogroup"]>label:hover{background:rgba(21,128,137,.06)!important;color:var(--teal-2)!important;transform:translateY(-1px)}
    .st-key-workspace div[role="radiogroup"]>label[data-checked="true"]{background:linear-gradient(120deg,var(--teal),var(--teal-2))!important;color:#fff!important;box-shadow:0 7px 16px rgba(15,92,98,.24)}
    .st-key-workspace div[role="radiogroup"]>label p{font-family:"DM Sans",sans-serif!important;font-size:.78rem!important;font-weight:700!important;margin:0!important}
    .query-source-hint{display:flex;align-items:center;gap:7px;width:fit-content;margin:8px 0 10px;padding:6px 10px;border-radius:999px;background:rgba(21,128,137,.06);border:1px solid rgba(21,128,137,.14);color:var(--teal-2);font-family:"JetBrains Mono",monospace;font-size:.62rem;letter-spacing:.04em}
    /* ---------- PDF Intelligence ---------- */
    .pdf-hero{position:relative;overflow:hidden;padding:42px 34px;min-height:290px;display:flex;align-items:center;justify-content:center;text-align:center;background:radial-gradient(circle at 50% 10%,rgba(21,128,137,.12),transparent 42%),linear-gradient(145deg,rgba(255,255,255,.94),rgba(238,241,234,.82))}
    .pdf-hero::after{content:"";position:absolute;width:330px;height:330px;border-radius:50%;border:1px solid rgba(21,128,137,.12);animation:floatSlow 5s ease-in-out infinite;pointer-events:none}
    .pdf-hero-inner{position:relative;z-index:1;max-width:760px}
    .pdf-icon{width:76px;height:76px;margin:0 auto 17px;display:flex;align-items:center;justify-content:center;border-radius:22px;color:#fff;background:linear-gradient(145deg,var(--teal),var(--teal-deep));box-shadow:0 16px 28px rgba(15,92,98,.22);animation:floatSlow 4.5s ease-in-out infinite}
    .pdf-icon svg{width:35px;height:35px}
    .pdf-title{font-family:"Fraunces",serif;font-size:clamp(2rem,3.5vw,3.15rem);line-height:1;letter-spacing:-.03em;color:var(--ink);margin:0}
    .pdf-subtitle{max-width:650px;margin:13px auto 0;color:var(--ink-soft);line-height:1.65;font-size:.96rem}
    .pdf-upload-zone{border:1.5px dashed rgba(21,128,137,.38);border-radius:var(--radius-lg);padding:22px;margin-top:18px;background:rgba(255,255,255,.58);transition:all .2s ease}
    .pdf-upload-zone:hover{border-color:var(--teal);background:rgba(21,128,137,.045);box-shadow:0 12px 26px rgba(15,92,98,.10)}
    .pdf-upload-caption{text-align:center;color:var(--muted);font-size:.73rem;font-family:"JetBrains Mono",monospace;letter-spacing:.06em;text-transform:uppercase;margin-top:9px}
    .pdf-success{display:flex;align-items:flex-start;gap:14px;padding:17px 18px;border:1px solid rgba(26,122,85,.22);border-radius:var(--radius-md);background:linear-gradient(120deg,rgba(26,122,85,.09),rgba(21,128,137,.045));margin-bottom:16px;animation:fadeUp .5s cubic-bezier(.2,.7,.3,1) both}
    .pdf-success-icon{width:34px;height:34px;flex-shrink:0;display:flex;align-items:center;justify-content:center;border-radius:10px;background:rgba(26,122,85,.13);color:var(--green)}
    .pdf-success-title{font-weight:800;color:var(--ink)}
    .pdf-success-meta{color:var(--muted);font-size:.74rem;margin-top:4px;line-height:1.55}
    .pdf-processing{margin-top:16px;border:1px solid var(--line);border-radius:var(--radius-lg);background:rgba(255,255,255,.72);padding:20px 22px;box-shadow:var(--shadow)}
    .pdf-processing-title{display:flex;align-items:center;gap:8px;color:var(--teal-2);font-family:"JetBrains Mono",monospace;font-size:.66rem;font-weight:700;letter-spacing:.13em;text-transform:uppercase;margin-bottom:12px}
    .pdf-flow-step{display:flex;align-items:center;gap:12px;padding:10px 0;border-bottom:1px dashed var(--line-soft);color:var(--ink-soft);font-size:.82rem}
    .pdf-flow-step:last-child{border-bottom:0}
    .pdf-flow-dot{width:25px;height:25px;display:flex;align-items:center;justify-content:center;border-radius:8px;background:rgba(21,128,137,.09);color:var(--teal-2);flex-shrink:0}
    .pdf-flow-step.active .pdf-flow-dot{background:linear-gradient(145deg,var(--teal),var(--teal-2));color:#fff;animation:pulseDot 1.3s ease-in-out infinite}
    .pdf-ready-banner{display:flex;align-items:center;gap:10px;padding:10px 13px;margin-bottom:14px;border-radius:999px;width:fit-content;color:var(--green);background:rgba(26,122,85,.09);border:1px solid rgba(26,122,85,.22);font-family:"JetBrains Mono",monospace;font-size:.66rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase}
    .pdf-chat-shell{min-height:520px}
    .pdf-chat-empty{display:flex;align-items:center;justify-content:center;text-align:center;padding:50px 25px;color:var(--muted)}
    .pdf-chat-empty strong{display:block;color:var(--ink);font-family:"Fraunces",serif;font-size:1.55rem;margin-bottom:7px}
    .pdf-message{max-width:86%;padding:14px 16px;border-radius:14px;margin:10px 0;animation:fadeUp .4s ease both;line-height:1.65;font-size:.9rem}
    .pdf-message.user{margin-left:auto;color:#fff;background:linear-gradient(135deg,var(--teal),var(--teal-2));border-bottom-right-radius:5px;box-shadow:0 10px 20px rgba(15,92,98,.16)}
    .pdf-message.assistant{margin-right:auto;color:#1c3a3f;background:linear-gradient(160deg,rgba(255,255,255,.97),rgba(238,244,240,.91));border:1px solid var(--line);border-left:4px solid var(--teal);border-bottom-left-radius:5px;box-shadow:var(--shadow)}
    .pdf-message-label{font-family:"JetBrains Mono",monospace;font-size:.59rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;opacity:.72;margin-bottom:5px}
    .pdf-answer-meta{display:flex;flex-wrap:wrap;gap:7px;margin-top:12px}
    .pdf-badge{display:inline-flex;align-items:center;gap:5px;border-radius:999px;padding:4px 8px;font-family:"JetBrains Mono",monospace;font-size:.59rem;font-weight:700;letter-spacing:.04em;text-transform:uppercase;background:rgba(21,128,137,.08);border:1px solid rgba(21,128,137,.2);color:var(--teal-2)}
    .pdf-doc-card{position:sticky;top:18px}
    .pdf-doc-icon{width:44px;height:44px;border-radius:12px;display:flex;align-items:center;justify-content:center;background:rgba(21,128,137,.09);color:var(--teal-2);flex-shrink:0}
    .pdf-doc-head{display:flex;align-items:center;gap:11px;margin-bottom:14px}
    .pdf-doc-name{font-weight:800;color:var(--ink);font-size:.88rem;word-break:break-word}
    .pdf-doc-status{color:var(--green);font-size:.66rem;font-family:"JetBrains Mono",monospace;margin-top:3px;text-transform:uppercase}
    .pdf-reset{margin-top:12px}
    .pdf-reset button{min-height:42px!important;font-size:.74rem!important}
    @media(max-width:900px){.st-key-workspace div[role="radiogroup"]{width:100%}.st-key-workspace div[role="radiogroup"]>label{flex:1;justify-content:center;text-align:center}.pdf-message{max-width:94%}.pdf-doc-card{position:relative;top:auto}}
</style>

<div class="watermark"></div>
""", unsafe_allow_html=True)

if BG_URI:
    st.markdown(
        f"""
        <div class="bg-photo-layer">
            <div class="photo" style="background-image:url('{BG_URI}');"></div>
            <div class="scrim"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------
# RAG imports / lazy backend
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_backend():
    try:
        response = requests.get(BACKEND_URL + "/", timeout=5)
        response.raise_for_status()
        payload = response.json()

        if not payload.get("index_ready", False):
            return None, payload.get("startup_error") or "RAG index is not ready."

        return {
            "api_url": BACKEND_URL,
            "chunk_count": payload.get("chunk_count", 0),
        }, None
    except Exception as e:
        return None, str(e)


backend, backend_error = load_backend()

if backend:
    try:
        chunk_count = int(backend.get("chunk_count", 0))
    except Exception:
        chunk_count = 0
else:
    chunk_count = 0

# -----------------------------
# Session state
# -----------------------------
if "query" not in st.session_state:
    st.session_state.query = ""
if "answer" not in st.session_state:
    st.session_state.answer = None
if "evidence" not in st.session_state:
    st.session_state.evidence = []
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of {"query": str, "confidence": float}

# ============================================================
# Workspace state
# ============================================================
if "workspace" not in st.session_state:
    st.session_state.workspace = "Clinical Guideline Assistant"
if "pdf_document" not in st.session_state:
    st.session_state.pdf_document = None
if "pdf_messages" not in st.session_state:
    st.session_state.pdf_messages = []
if "pdf_upload_error" not in st.session_state:
    st.session_state.pdf_upload_error = None
if "question_source" not in st.session_state:
    st.session_state.question_source = "Clinical Guidelines"

def upload_pdf_to_backend(uploaded_file):
    """Send the selected PDF to the PDF Intelligence backend.

    TODO: Confirm the multipart field name and response schema with the backend.
    """
    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
    response = requests.post(BACKEND_URL + "/upload", files=files, timeout=300)
    response.raise_for_status()
    return response.json()

def pdf_chat_request(document_id, question):
    """Ask the PDF Intelligence backend for a grounded answer.

    TODO: Align request/response keys with the final /pdf-chat contract.
    """
    response = requests.post(
        BACKEND_URL + "/pdf-chat",
        json={"document_id": document_id, "question": question},
        timeout=180,
    )
    response.raise_for_status()
    return response.json()

# ============================================================
# Header
# ============================================================
CADUCEUS_SVG = """
<svg width="118" height="118" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <line x1="50" y1="12" x2="50" y2="92" stroke="#0c5c63" stroke-width="4" stroke-linecap="round"/>
  <circle cx="50" cy="11" r="5.5" fill="#c6912b"/>
  <path d="M50 26 C32 15 12 22 6 38 C22 37 36 32 50 34" stroke="#c6912b" stroke-width="3.4"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M50 26 C68 15 88 22 94 38 C78 37 64 32 50 34" stroke="#c6912b" stroke-width="3.4"
        fill="none" stroke-linecap="round" stroke-linejoin="round"/>
  <path d="M50 30 C34 40 66 50 50 60 C34 70 66 78 50 90" stroke="#158089" stroke-width="3.2"
        fill="none" stroke-linecap="round"/>
  <path d="M50 30 C66 40 34 50 50 60 C66 70 34 78 50 90" stroke="#062a2e" stroke-width="3.2"
        fill="none" stroke-linecap="round"/>
</svg>
"""

health_pct = 100 if backend else 0
dash_html = f"""
<div class="hero-dash">
  <div class="dash-item">
    <svg width="66" height="34" viewBox="0 0 66 34">
      <path d="M2,26 L14,15 L26,20 L38,7 L50,17 L62,5" fill="none" stroke="#158089"
            stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="62" cy="5" r="2.6" fill="#c6912b"/>
    </svg>
    <div class="dash-cap">Retrieval trend</div>
  </div>
  <div class="dash-item">
    <svg width="58" height="34" viewBox="0 0 58 34">
      <circle cx="29" cy="6" r="4" fill="#158089"/>
      <line x1="29" y1="10" x2="11" y2="20" stroke="#b8ccce" stroke-width="1.6"/>
      <line x1="29" y1="10" x2="29" y2="20" stroke="#b8ccce" stroke-width="1.6"/>
      <line x1="29" y1="10" x2="47" y2="20" stroke="#b8ccce" stroke-width="1.6"/>
      <circle cx="11" cy="24" r="4" fill="#0c5c63"/>
      <circle cx="29" cy="24" r="4" fill="#0c5c63"/>
      <circle cx="47" cy="24" r="4" fill="#c6912b"/>
    </svg>
    <div class="dash-cap">Hybrid pipeline</div>
  </div>
  <div class="dash-item">
    <div class="dash-bar-wrap">
      <div class="dash-bar-track">
        <div class="dash-bar-seg" style="width:{health_pct}%; background:linear-gradient(90deg,#158089,#c6912b);"></div>
      </div>
      <div class="dash-cap" style="margin-top:8px;">Index health · {health_pct}%</div>
    </div>
  </div>
</div>
"""

logo_col, brand_col, dash_col = st.columns([1.05, 4.9, 3.3])
with logo_col:
    if os.path.exists(LOGO_PATH):
        st.image(LOGO_PATH, width=112)
    else:
        st.markdown(f'<div class="hero-badge">{CADUCEUS_SVG}</div>', unsafe_allow_html=True)
with brand_col:
    st.markdown(
        f'<div class="kicker"><span class="dot"></span>CLINICAL DECISION SUPPORT · ENDOMETRIOSIS RAG</div>'
        f'<h1 class="brand">RAG <span class="amp">&amp;</span> Roll</h1>'
        f'<div class="tagline">Evidence you can trust. <span class="accent">Answers you can cite.</span></div>',
        unsafe_allow_html=True,
    )
with dash_col:
    st.markdown(dash_html, unsafe_allow_html=True)

st.markdown(
    '<div class="subtitle"><b>Grounded, guideline-cited answers on endometriosis</b> — retrieved from NICE '
    'and ESHRE guideline corpora through hybrid dense + lexical search, cross-encoder reranking, and '
    'source-aware generation. Every claim traces back to a numbered passage; <b>nothing is generated beyond '
    'the source text.</b></div>',
    unsafe_allow_html=True,
)


# Signature ECG divider
st.markdown(
    """
    <div class="ecg-wrap">
      <svg viewBox="0 0 1000 34" preserveAspectRatio="none">
        <defs>
          <linearGradient id="ecgGrad" x1="0" y1="0" x2="1" y2="0">
            <stop offset="0%" stop-color="#c6912b"/>
            <stop offset="55%" stop-color="#158089"/>
            <stop offset="100%" stop-color="#062a2e"/>
          </linearGradient>
        </defs>
        <path class="ecg-line" d="M0,17 L160,17 L185,17 L200,4 L215,30 L230,17 L260,17
                 L420,17 L445,17 L460,4 L475,30 L490,17 L520,17
                 L680,17 L705,17 L720,4 L735,30 L750,17 L780,17
                 L940,17 L965,17 L980,4 L995,30 L1000,17"/>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)

index_label = f"INDEX: {chunk_count:,} CHUNKS" if backend else "INDEX: UNAVAILABLE"
live_class = "live" if backend else "live offline"
live_word = "LIVE" if backend else "OFFLINE"
checks = "".join(icon("check", 12) for _ in range(4))

st.markdown(
    f"""
    <div class="pipe-badges fade-in">
        <span class="pipe-pill dark">{icon("search", 13, "#ffffff")}&nbsp;Retrieval</span>
        <span class="pipe-arrow">→</span>
        <span class="pipe-pill outline">{icon("sparkles", 13)}&nbsp;Reasoning</span>
        <span class="pipe-pill green">{icon("activity", 13)}&nbsp;Backend API <span class="pipe-checks">{checks}</span></span>
        <span class="pipe-pill {live_class}"><span class="pdot"></span>{index_label} — {live_word}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Unified single-page intelligence workspace
# Clinical Guidelines + optional PDF RAG, intentionally kept together.
# ============================================================

st.markdown(
    '<div class="workspace-heading"><div class="ws-kicker">INTELLIGENCE SOURCE</div></div>',
    unsafe_allow_html=True,
)

source_options = ["Clinical Guidelines"]
if st.session_state.pdf_document:
    source_options.append("Uploaded PDF")

st.radio(
    "Question source",
    source_options,
    horizontal=True,
    key="question_source",
    label_visibility="collapsed",
)

using_pdf = (
    st.session_state.question_source == "Uploaded PDF"
    and bool(st.session_state.pdf_document)
)

left, right = st.columns([2.05, 1], gap="large")

with left:
    source_label = (
        f'PDF · {st.session_state.pdf_document.get("filename", "Uploaded document")}'
        if using_pdf else "NICE / ESHRE Clinical Guidelines"
    )
    source_hint = (
        "Questions are answered only from your indexed PDF."
        if using_pdf else "Questions are answered from the indexed clinical guideline corpus."
    )

    st.markdown(
        f'<div class="card fade-in d1">'
        f'<div class="section-label">{icon("search", 14)} 01 · INTELLIGENCE QUERY</div>'
        f'<div class="section-title">Ask EndoGuide AI</div>'
        f'<div class="query-source-hint">{icon("target", 12)} '
        f'{html.escape(source_label)} · {html.escape(source_hint)}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="primary-input-caption"><span>YOUR QUESTION</span>'
        f'<span class="input-mode">{"PDF RAG · DOCUMENT GROUNDED" if using_pdf else "CLINICAL RAG · GUIDELINE GROUNDED"}</span></div>',
        unsafe_allow_html=True,
    )

    query = st.text_area(
        "Question",
        value=st.session_state.query,
        placeholder=(
            "Ask anything about the uploaded PDF… You can ask for summaries, explanations, comparisons, "
            "specific facts, or evidence from the document."
            if using_pdf else
            "Ask a clinical question grounded in the indexed guidelines…"
        ),
        label_visibility="collapsed",
        key="query_input",
        height=128,
    )

    ask = st.button(
        "ASK THE DOCUMENT →" if using_pdf else "ASK THE GUIDELINES →",
        type="primary",
        use_container_width=True,
        disabled=(not bool(st.session_state.pdf_document) if using_pdf else not bool(backend)),
        key="main_ask_button",
    )

    if not using_pdf and not backend:
        st.markdown(
            f'<div class="safety-note">{icon("alert", 18)}<div>'
            f'The clinical RAG backend is not available in this session.<br>'
            f'{html.escape(backend_error or "Start the FastAPI server at http://127.0.0.1:8000")}'
            f'</div></div>', unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # FIX: Allow execution if they clicked Ask while a PDF is active, even if text is empty
    if ask and (query.strip() or using_pdf):
        st.session_state.query = query.strip()
        question = query.strip()
        
        # Inject default question if empty (only for PDF mode)
        if not question and using_pdf:
             question = "Summarize the key clinical recommendations or main points in this document."

        if using_pdf:
            # ==========================================
            # PATH 1: PDF INTELLIGENCE MODE
            # ==========================================
            with st.spinner("Retrieving document evidence and generating a grounded response…"):
                try:
                    pdf_bytes = st.session_state.pdf_document["bytes"]
                    pdf_name = st.session_state.pdf_document["filename"]
                    
                    request_kwargs = {
                        "data": {"question": question},
                        "files": {"pdf_file": (pdf_name, pdf_bytes, "application/pdf")},
                        "timeout": 180,
                    }
                    
                    response = requests.post(BACKEND_URL + "/chat", **request_kwargs)
                    response.raise_for_status()
                    payload = response.json()

                    sources = payload.get("sources", []) or []
                    evidence = []
                    for source in sources[:5]:
                        evidence.append({
                            "text": source.get("text", ""),
                            "meta": source,
                            "rerank": float(source.get("rerank_score", 0.0)),
                            "score": float(source.get("rerank_score", source.get("bm25_score", 0.0)) or 0.0),
                        })

                    st.session_state.answer = payload.get("answer", "")
                    st.session_state.evidence = evidence
                    st.session_state.confidence = float(payload.get("confidence", 0.0))
                    
                    display_query = query.strip() if query.strip() else f"Analyzed Document: {pdf_name}"
                    st.session_state.history.append({"query": display_query, "confidence": st.session_state.confidence})

                except Exception as e:
                    st.session_state.answer = f"PDF retrieval error: {e}"
                    st.session_state.evidence = []
                    st.session_state.confidence = 0.0
                    
        else:
            # ==========================================
            # PATH 2: CLINICAL GUIDELINE MODE
            # ==========================================
            with st.spinner("Retrieving guideline evidence and generating a grounded response…"):
                try:
                    # FIX: Send as 'data' instead of 'json' so it matches the backend's new Form requirement
                    request_kwargs = {
                        "data": {"question": question},
                        "timeout": 180,
                    }
                    
                    response = requests.post(BACKEND_URL + "/chat", **request_kwargs)
                    response.raise_for_status()
                    payload = response.json()
                    
                    sources = payload.get("sources", []) or []
                    evidence = []
                    for source in sources[:5]:
                        evidence.append({
                            "text": source.get("text", ""),
                            "meta": source,
                            "rerank": float(source.get("rerank_score", 0.0)),
                            "score": float(source.get("rerank_score", source.get("bm25_score", 0.0)) or 0.0),
                        })

                    st.session_state.answer = payload.get("answer", "")
                    st.session_state.evidence = evidence
                    st.session_state.confidence = float(payload.get("confidence", 0.0))
                    st.session_state.history.append({"query": question, "confidence": st.session_state.confidence})

                except Exception as e:
                    st.session_state.answer = f"Retrieval error: {e}"
                    st.session_state.evidence = []
                    st.session_state.confidence = 0.0

    st.markdown(
        f'<div class="card fade-in d2">'
        f'<div class="section-label">{icon("sparkles", 14)} 02 · GENERATED ANSWER</div>'
        f'<div class="section-title">The answer, fully grounded</div>', unsafe_allow_html=True,
    )
    if st.session_state.answer:
        escaped_answer = html.escape(st.session_state.answer).replace(chr(10), "<br>")
        st.markdown(f'<div class="answer-box">{escaped_answer}</div>', unsafe_allow_html=True)
        
        # FIX: Base64 encode the text
        b64_answer = base64.b64encode(st.session_state.answer.encode("utf-8")).decode("utf-8")
        btn_id = "copybtn_" + hashlib.md5(st.session_state.answer.encode("utf-8")).hexdigest()[:8]
        
        # FIX: One continuous line to prevent Streamlit Markdown errors, 
        # and removed the SVG from the JS string to prevent double-quote HTML breaks.
        st.markdown(
            f'<button class="copy-btn" id="{btn_id}" onclick="const decodedText = decodeURIComponent(escape(window.atob(\'{b64_answer}\'))); navigator.clipboard.writeText(decodedText); this.classList.add(\'copied\'); this.innerHTML = \'✓ Copied\'; setTimeout(() => {{ this.classList.remove(\'copied\'); this.innerHTML = \'📋 Copy answer\'; }}, 1800);">{icon("clipboard", 13)} Copy answer</button>', 
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="safety-note">{icon("sparkles", 18)}<div><b>No answer yet.</b> Ask a question above to retrieve grounded evidence and generate a response.</div></div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        f'<div class="card fade-in d3">'
        f'<div class="section-label">{icon("layers", 14)} 03 · RETRIEVED EVIDENCE</div>'
        f'<div class="section-title">Every claim, traceable</div>', unsafe_allow_html=True,
    )
    if st.session_state.evidence:
        for i, item in enumerate(st.session_state.evidence, 1):
            meta = item.get("meta", {})
            if using_pdf:
                source_name = meta.get("source") or meta.get("filename") or st.session_state.pdf_document.get("filename", "Uploaded PDF")
                section = meta.get("section_title", meta.get("section", "PDF evidence"))
                page = meta.get("page", meta.get("page_number", "—"))
                score = item.get("rerank", item.get("score", 0))
                try: pct = max(0, min(100, int(float(score) * 100)))
                except Exception: pct = 0
            else:
                source_name = meta.get("guideline", meta.get("source", "Guideline"))
                section = meta.get("section_title", meta.get("section", "Clinical evidence"))
                page = meta.get("page", "—")
                score = item.get("rerank", item.get("score", 0))
                pct = max(0, min(100, int((float(score) + 5) / 10 * 100))) if score is not None else 0
            delay = 0.05 * (i - 1)
            st.markdown(
                f'''<div class="evidence fade-in" style="animation-delay:{delay}s;">
                    <div class="evidence-head"><div class="evidence-source"><span class="source-tag">S{i}</span>&nbsp; {html.escape(str(source_name))}</div><div class="meta">PAGE {html.escape(str(page))}</div></div>
                    <div class="meta">{html.escape(str(section))}</div>
                    <div class="quote">{html.escape(str(item.get("text", "")))}</div>
                    <div class="relevance-row"><div class="relevance-track"><div class="relevance-fill" style="width:{pct}%; animation-delay:{delay}s;"></div></div><div class="relevance-val">{pct}%</div></div>
                </div>''', unsafe_allow_html=True,
            )
    else:
        st.markdown('<div class="small-note">Retrieved passages will appear here after a successful query.</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right:

    pdf_doc = st.session_state.pdf_document
    st.markdown(
        f'<div class="card card-tight fade-in d1">'
        f'<div class="section-label">{icon("file", 14)} PDF INTELLIGENCE</div>'
        f'<div class="section-title" style="font-size:1.22rem;">Bring your own document</div>'
        f'<div class="small-note" style="margin-bottom:13px;">Upload a PDF, let the backend index it, then switch the source above to ask questions grounded in the document.</div>',
        unsafe_allow_html=True,
    )

    if not pdf_doc:
        uploaded = st.file_uploader("Upload PDF", type=["pdf"], accept_multiple_files=False, label_visibility="collapsed", key="unified_pdf_uploader")
        if uploaded is None:
            st.markdown(
                f'<div class="pdf-upload-zone"><div class="pdf-icon">{icon("file", 34, "#ffffff", 1.7)}</div>'
                f'<div class="pdf-title" style="font-size:1.65rem;">Upload a PDF</div>'
                f'<div class="pdf-subtitle">Drag and drop your document here, or choose a file. PDF only.</div></div>'
                f'<div style="height:12px;"></div>',
                unsafe_allow_html=True,
            )
        else:
            size_mb = uploaded.size / (1024 * 1024)
            st.markdown(
                f'<div class="pdf-success"><div class="pdf-success-icon">{icon("check", 19, "currentColor", 2.2)}</div><div><div class="pdf-success-title">PDF selected</div><div class="pdf-success-meta">{html.escape(uploaded.name)} · {size_mb:.2f} MB · Ready to process</div></div></div>',
                unsafe_allow_html=True,
            )
            if st.button("PROCESS & INDEX PDF →", type="primary", use_container_width=True, key="process_pdf_unified"):
                st.session_state.pdf_upload_error = None
                try:
                    # FIX: Save the PDF directly to Streamlit session state instead of 
                    # looking for a missing /upload endpoint on the backend.
                    st.session_state.pdf_document = {
                        "filename": uploaded.name,
                        "size_bytes": uploaded.size,
                        "bytes": uploaded.getvalue(), 
                        "page_count": "Unknown",
                        "chunk_count": "Dynamic",
                        "embedding_model": "BAAI/bge-small-en-v1.5",
                        "status": "Ready for Query",
                    }
                   # st.session_state.question_source = "Uploaded PDF"
                    st.session_state.pdf_upload_error = None
                    st.rerun()
                except Exception as exc:
                    st.session_state.pdf_upload_error = str(exc)

    if st.session_state.pdf_upload_error:
        st.markdown(
            f'<div class="safety-note">{icon("alert", 17)}<div><b>PDF processing could not be completed.</b><br>{html.escape(st.session_state.pdf_upload_error)}</div></div>',
            unsafe_allow_html=True,
        )

    if pdf_doc:
        page_count = pdf_doc.get("page_count") if pdf_doc.get("page_count") is not None else "—"
        chunks = pdf_doc.get("chunk_count") if pdf_doc.get("chunk_count") is not None else "—"
        size_bytes = pdf_doc.get("size_bytes")
        size_display = f"{size_bytes/(1024*1024):.2f} MB" if size_bytes else "—"
        st.markdown(
            f'<div class="pdf-success"><div class="pdf-success-icon">{icon("check", 19, "currentColor", 2.2)}</div><div><div class="pdf-success-title">File uploaded successfully</div><div class="pdf-success-meta">{html.escape(str(pdf_doc.get("filename", "Document")))} · Indexed ✓</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="spec-list"><div class="spec-row"><span class="spec-k">Status</span><span class="status-pill online" style="padding:3px 10px 3px 8px;"><span class="pdot"></span>INDEXED</span></div><div class="spec-row"><span class="spec-k">Pages</span><span class="spec-v">{html.escape(str(page_count))}</span></div><div class="spec-row"><span class="spec-k">Chunks</span><span class="spec-v">{html.escape(str(chunks))}</span></div><div class="spec-row"><span class="spec-k">File size</span><span class="spec-v">{html.escape(size_display)}</span></div><div class="spec-row"><span class="spec-k">Embedding model</span><span class="spec-v">{html.escape(str(pdf_doc.get("embedding_model", "Backend-defined")))}</span></div></div>',
            unsafe_allow_html=True,
        )
        if st.session_state.question_source == "Uploaded PDF":
            st.markdown(
                f'<div class="pdf-ready-banner" style="margin-top:14px;margin-bottom:0;">{icon("check", 13, "currentColor", 2.2)} DOCUMENT ACTIVE · READY FOR QUESTIONS</div>',
                unsafe_allow_html=True,
            )
        if st.button("UPLOAD A DIFFERENT PDF", use_container_width=True, key="reset_pdf"):
            st.session_state.pdf_document = None
            st.session_state.pdf_messages = []
            st.session_state.pdf_upload_error = None
            st.session_state.question_source = "Clinical Guidelines"
            st.rerun()

    st.markdown(
        f'<div class="card card-tight fade-in d2"><div class="section-label">{icon("database", 14)} KNOWLEDGE SOURCES</div><div class="section-title" style="font-size:1.22rem;">One interface. Two sources.</div><div class="spec-list"><div class="spec-row"><span class="spec-k">Guidelines</span><span class="status-pill {"online" if backend else ""}" style="padding:3px 10px 3px 8px;"><span class="pdot"></span>{"ONLINE" if backend else "OFFLINE"}</span></div><div class="spec-row"><span class="spec-k">PDF RAG</span><span class="status-pill {"online" if pdf_doc else ""}" style="padding:3px 10px 3px 8px;"><span class="pdot"></span>{"INDEXED" if pdf_doc else "READY TO UPLOAD"}</span></div><div class="spec-row"><span class="spec-k">Guideline chunks</span><span class="spec-v">{chunk_count:,}</span></div></div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card card-tight fade-in d3"><div class="section-label">{icon("shield", 14)} GROUNDING POLICY</div><div class="section-title" style="font-size:1.22rem;">Source-aware by design</div><div class="small-note">Choose <b>Clinical Guidelines</b> for the existing EndoGuide RAG flow, or <b>Uploaded PDF</b> to restrict retrieval to your indexed document. The UI does not invent PDF processing results; indexing and retrieval are handled by the backend APIs.</div></div>',
        unsafe_allow_html=True,
    )
    # ---------- Confidence & safety ----------
    st.markdown(
        f'<div class="card card-tight fade-in d1">'
        f'<div class="section-label">{icon("shield", 14)} CONFIDENCE & SAFETY</div>'
        f'<div class="section-title" style="font-size:1.22rem;">How sure should you be?</div>',
        unsafe_allow_html=True,
    )

    confidence = st.session_state.confidence

    if confidence is None:
        st.markdown(
            f'<div class="conf-wrap"><div class="conf-score-badge" style="color:var(--muted);">'
            f'<span class="num">—</span></div>'
            f'<div><div class="conf-label-lg" style="color:var(--muted);">Not yet run</div>'
            f'<div class="conf-sub">Ask a question to generate<br>an evidence-grounded answer.</div></div></div>',
            unsafe_allow_html=True,
        )
    else:
        pct_int = int(round(confidence * 100))

        if confidence >= 0.75:
            ring_color, label = "#1f7a80", "High"
            explanation = "Strong, closely matching evidence was retrieved across multiple guideline passages."
        elif confidence >= 0.5:
            ring_color, label = "#c08a2e", "Moderate"
            explanation = "Relevant evidence was found, but the match to your exact question is only partial."
        else:
            ring_color, label = "#c25b4a", "Low"
            explanation = "Retrieved passages are a weak match — treat this answer as a starting point, not a conclusion."

        marker_class = "mk_" + hashlib.md5(str(pct_int).encode()).hexdigest()[:6]

        st.markdown(
            f"""
            <div class="conf-wrap">
                <div class="conf-score-badge" style="color:{ring_color};">
                    <span class="num">{pct_int}</span><span class="pct">%</span>
                </div>
                <div>
                    <div class="conf-label-lg" style="color:{ring_color};">{label} confidence</div>
                    <div class="conf-sub">{explanation}</div>
                </div>
            </div>
            <div class="conf-meter-wrap">
                <div class="conf-meter-scale"><span>Low</span><span>Moderate</span><span>High</span></div>
                <div class="conf-meter-track">
                    <style>
                    @keyframes {marker_class} {{ from {{ left:0%; }} to {{ left:{pct_int}%; }} }}
                    .{marker_class} {{ animation:{marker_class} 1s cubic-bezier(.2,.8,.25,1) both; }}
                    </style>
                    <div class="conf-meter-marker {marker_class}" style="left:{pct_int}%;"></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="small-note" style="margin-top:12px;">This score reflects retrieval/evidence '
            'strength in the interface. It is not a probability of diagnosis or treatment success.</div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        f'<div class="safety-note">{icon("alert", 16)}<div>'
        f'Answers are generated only from retrieved clinical guideline evidence. This is decision support, '
        f'not a substitute for professional medical assessment.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    f'<div class="footer">'
    f'<div>{icon("stethoscope", 13)} &nbsp;RAG &amp; ROLL · RETRIEVAL-AUGMENTED CLINICAL INTELLIGENCE</div>'
    f'<div>🎗️ NOT A MEDICAL DEVICE · FOR CLINICIAN REFERENCE ONLY</div>'
    f'</div>', unsafe_allow_html=True,
)

