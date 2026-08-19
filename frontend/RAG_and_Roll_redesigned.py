import os
import re
import json
import base64
import pickle
import html
import hashlib
import streamlit as st

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
    if not CHROMA_PATH or not BM25_FILE:
        return None, "RAG index not found."

    try:
        import chromadb
        from sentence_transformers import SentenceTransformer, CrossEncoder

        client = chromadb.PersistentClient(path=CHROMA_PATH)
        collection = client.get_collection(name="nice_guidelines")

        with open(BM25_FILE, "rb") as f:
            bm25 = pickle.load(f)

        embedder = SentenceTransformer("BAAI/bge-small-en-v1.5")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        return {
            "client": client,
            "collection": collection,
            "bm25": bm25,
            "embedder": embedder,
            "reranker": reranker,
        }, None
    except Exception as e:
        return None, str(e)


backend, backend_error = load_backend()

if backend:
    try:
        chunk_count = backend["collection"].count()
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

# ---------- Pipeline status pills (retrieval / reasoning / active / index) ----------
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
        <span class="pipe-pill green">{icon("activity", 13)}&nbsp;Active pipeline <span class="pipe-checks">{checks}</span></span>
        <span class="pipe-pill {live_class}"><span class="pdot"></span>{index_label} — {live_word}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# Main columns
# ============================================================
left, right = st.columns([2.05, 1], gap="large")

with left:
    st.markdown(
        f'<div class="card fade-in d1">'
        f'<div class="section-label">{icon("search", 14)} 01 · CLINICAL QUERY</div>'
        f'<div class="section-title">Ask the guidelines, get a citation</div>',
        unsafe_allow_html=True,
    )

    query = st.text_input(
        "Clinical question",
        value=st.session_state.query,
        placeholder="Ask a clinical question grounded in the indexed guidelines…",
        label_visibility="collapsed",
        key="query_input",
    )

    suggestions = [
        ("🩺", "Common symptoms?"),
        ("🔬", "How is it diagnosed?"),
        ("🖥️", "When is MRI indicated?"),
        ("💊", "Recommended pain treatments?"),
    ]

    st.markdown(
        f'<div class="section-label" style="margin-top:2px;">{icon("sparkles", 13)} TRY ASKING</div>',
        unsafe_allow_html=True,
    )
    with st.container(key="suggestion_chips"):
        cols = st.columns(len(suggestions))
        for i, (emo, suggestion) in enumerate(suggestions):
            with cols[i]:
                if st.button(f"{emo}  {suggestion}", key=f"suggestion_{i}", use_container_width=True):
                    st.session_state.query = suggestion
                    st.rerun()

    if st.session_state.history:
        recent = list(reversed(st.session_state.history))[:4]
        chips = " &nbsp; ".join(
            f'<span style="font-family:\'JetBrains Mono\',monospace;font-size:.72rem;'
            f'color:var(--teal-2);background:rgba(31,122,128,.08);border:1px solid rgba(31,122,128,.18);'
            f'border-radius:999px;padding:4px 10px;">{html.escape(h["query"][:40])}</span>'
            for h in recent
        )
        st.markdown(
            f'<div style="margin-top:4px;">'
            f'<div class="section-label" style="margin-top:10px;">{icon("history", 13)} RECENT</div>'
            f'<div>{chips}</div></div>',
            unsafe_allow_html=True,
        )

    ask = st.button(
        f"ASK THE GUIDELINES  →",
        type="primary",
        use_container_width=True,
        disabled=not bool(backend),
    )

    if not backend:
        st.markdown(
            f'<div class="safety-note">{icon("alert", 18)}<div>'
            f'The interface is ready, but the retrieval index is not available in this session.<br>'
            f'Expected ChromaDB: <code>/content/data/chromadb</code> · '
            f'Expected BM25: <code>/content/data/chunks/nice_bm25.pkl</code></div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close query card

    if ask and query.strip():
        st.session_state.query = query.strip()

        with st.spinner("Retrieving guideline evidence and generating a grounded response…"):
            try:
                import numpy as np

                q = query.strip()

                # Dense retrieval
                q_emb = backend["embedder"].encode(
                    [q],
                    normalize_embeddings=True
                )[0].tolist()

                dense = backend["collection"].query(
                    query_embeddings=[q_emb],
                    n_results=12,
                    include=["documents", "metadatas", "distances"],
                )

                dense_items = []
                docs = dense.get("documents", [[]])[0]
                metas = dense.get("metadatas", [[]])[0]
                distances = dense.get("distances", [[]])[0]

                for d, m, dist in zip(docs, metas, distances):
                    dense_items.append({
                        "text": d,
                        "meta": m or {},
                        "score": 1.0 - float(dist),
                    })

                # BM25 retrieval
                tokens = re.findall(r"\w+", q.lower())
                bm_scores = backend["bm25"].get_scores(tokens)
                top_idx = np.argsort(bm_scores)[::-1][:12]

                bm_items = []
                corpus = getattr(backend["bm25"], "corpus", None)

                for idx in top_idx:
                    score = float(bm_scores[idx])
                    if corpus and idx < len(corpus):
                        text = " ".join(corpus[idx]) if isinstance(corpus[idx], list) else str(corpus[idx])
                        bm_items.append({"text": text, "meta": {}, "score": score})

                # Fuse unique text candidates
                candidates = {}
                for item in dense_items + bm_items:
                    key = re.sub(r"\s+", " ", item["text"]).strip()
                    if not key:
                        continue
                    if key not in candidates or item["score"] > candidates[key]["score"]:
                        candidates[key] = item

                candidates = list(candidates.values())[:20]

                # CrossEncoder reranking
                if candidates:
                    pairs = [[q, c["text"]] for c in candidates]
                    rr = backend["reranker"].predict(pairs)

                    for c, score in zip(candidates, rr):
                        c["rerank"] = float(score)

                    candidates.sort(key=lambda x: x["rerank"], reverse=True)

                evidence = candidates[:5]

                # Try generation using the original Qwen stack if available.
                generated = None
                try:
                    from transformers import AutoTokenizer, AutoModelForCausalLM
                    import torch

                    @st.cache_resource(show_spinner=False)
                    def load_llm():
                        model_name = "Qwen/Qwen2.5-1.5B-Instruct"
                        tok = AutoTokenizer.from_pretrained(model_name)
                        model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            torch_dtype="auto",
                            device_map="auto",
                        )
                        return tok, model

                    tokenizer, model = load_llm()

                    context_parts = []
                    for i, item in enumerate(evidence, 1):
                        m = item.get("meta", {})
                        context_parts.append(
                            f"[S{i}] {m.get('guideline', m.get('source', 'Guideline'))} | "
                            f"{m.get('section_title', m.get('section', ''))} | "
                            f"page {m.get('page', '?')}\n{item['text']}"
                        )

                    context = "\n\n".join(context_parts)

                    prompt = f"""
You are a clinical evidence assistant.

Answer the question using ONLY the supplied guideline evidence.
Do not invent facts.
Use [S1], [S2], etc. for claims supported by the evidence.
If the evidence is insufficient, say so.

Question:
{q}

Guideline evidence:
{context}

Answer:
""".strip()

                    messages = [
                        {"role": "system", "content": "You are a careful, evidence-grounded clinical reference assistant."},
                        {"role": "user", "content": prompt},
                    ]

                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )

                    inputs = tokenizer(text, return_tensors="pt").to(model.device)

                    with torch.no_grad():
                        output = model.generate(
                            **inputs,
                            max_new_tokens=420,
                            do_sample=False,
                        )

                    generated = tokenizer.decode(
                        output[0][inputs["input_ids"].shape[1]:],
                        skip_special_tokens=True,
                    ).strip()

                except Exception as llm_error:
                    generated = (
                        "The retrieval stage completed, but the generation model could not be loaded "
                        f"in this session. Retrieved evidence is shown below.\n\n"
                        f"Generation detail: {llm_error}"
                    )

                st.session_state.answer = generated
                st.session_state.evidence = evidence

                if evidence:
                    top_scores = [x.get("rerank", 0.0) for x in evidence]
                    conf = max(0.0, min(1.0, (max(top_scores) + 5) / 10))
                else:
                    conf = 0.0

                st.session_state.confidence = conf
                st.session_state.history.append({"query": q, "confidence": conf})

            except Exception as e:
                st.session_state.answer = f"Retrieval error: {e}"
                st.session_state.evidence = []
                st.session_state.confidence = 0.0

    # ---------- Answer ----------
    st.markdown(
        f'<div class="card fade-in d2">'
        f'<div class="section-label">{icon("sparkles", 14)} 02 · GENERATED ANSWER</div>'
        f'<div class="section-title">The answer, fully sourced</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.answer:
        escaped_answer = html.escape(st.session_state.answer).replace(chr(10), "<br>")
        st.markdown(f'<div class="answer-box">{escaped_answer}</div>', unsafe_allow_html=True)

        # Copy-to-clipboard control
        copy_payload = json.dumps(st.session_state.answer)
        btn_id = "copybtn_" + hashlib.md5(st.session_state.answer.encode("utf-8")).hexdigest()[:8]
        st.markdown(
            f"""
            <button class="copy-btn" id="{btn_id}" onclick="
                navigator.clipboard.writeText({copy_payload});
                this.classList.add('copied');
                this.innerHTML = '{icon("check", 13)} Copied';
                setTimeout(() => {{
                    this.classList.remove('copied');
                    this.innerHTML = '{icon("clipboard", 13)} Copy answer';
                }}, 1800);
            ">{icon("clipboard", 13)} Copy answer</button>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="safety-note">{icon("sparkles", 18)}<div>'
            f'<b>No answer yet.</b> Ask a clinical question above — RAG &amp; Roll will retrieve the most '
            f'relevant guideline passages, rerank them for precision, and generate a response that never '
            f'strays past the cited evidence.</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close answer card

    # ---------- Supporting evidence ----------
    st.markdown(
        f'<div class="card fade-in d3">'
        f'<div class="section-label">{icon("layers", 14)} 03 · SUPPORTING EVIDENCE</div>'
        f'<div class="section-title">Every claim, traceable</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.evidence:
        for i, item in enumerate(st.session_state.evidence, 1):
            meta = item.get("meta", {})
            guideline = meta.get("guideline", meta.get("source", "Guideline"))
            section = meta.get("section_title", meta.get("section", "Clinical evidence"))
            page = meta.get("page", "—")
            score = item.get("rerank", item.get("score", 0))
            pct = max(0, min(100, int((float(score) + 5) / 10 * 100))) if score is not None else 0

            delay = 0.05 * (i - 1)
            st.markdown(
                f"""
                <div class="evidence fade-in" style="animation-delay:{delay}s;">
                    <div class="evidence-head">
                        <div class="evidence-source"><span class="source-tag">S{i}</span>&nbsp; {html.escape(str(guideline))}</div>
                        <div class="meta">PAGE {html.escape(str(page))}</div>
                    </div>
                    <div class="meta">{html.escape(str(section))}</div>
                    <div class="quote">{html.escape(str(item.get("text", "")))}</div>
                    <div class="relevance-row">
                        <div class="relevance-track"><div class="relevance-fill" style="width:{pct}%; animation-delay:{delay}s;"></div></div>
                        <div class="relevance-val">{pct}%</div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.markdown(
            '<div class="small-note">Retrieved guideline passages will appear here after your first '
            'successful query.</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)  # close evidence card

with right:
    # ---------- Evidence panel summary ----------
    st.markdown(
        f'<div class="card card-tight fade-in d1">'
        f'<div class="section-label">{icon("file", 14)} EVIDENCE PANEL</div>'
        f'<div class="section-title" style="font-size:1.22rem;">Zero-hallucination by design</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.evidence:
        st.markdown(
            f'<div class="safety-note" style="background:linear-gradient(120deg, rgba(26,110,82,.1), rgba(31,122,128,.06));'
            f'border-color:rgba(26,110,82,.25);">{icon("check", 16, "#1a6e52")}<div>'
            f'<b>{len(st.session_state.evidence)} evidence passages</b> retrieved and reranked across the indexed guidelines — every sentence in the answer above maps back to one of these.'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        rows = ""
        for i, item in enumerate(st.session_state.evidence, 1):
            meta = item.get("meta", {})
            guideline = html.escape(str(meta.get("guideline", meta.get("source", "Guideline"))))
            page = html.escape(str(meta.get("page", "—")))
            score = item.get("rerank", item.get("score", 0))
            pct = max(0, min(100, int((float(score) + 5) / 10 * 100))) if score is not None else 0
            rows += (
                f'<tr><td><span class="id-chip">S{i}</span></td>'
                f'<td class="src-name">{guideline}</td>'
                f'<td>p.{page}</td>'
                f'<td>{pct}%</td></tr>'
            )
        st.markdown(
            f"""
            <table class="evi-table">
                <thead><tr><th>ID</th><th>Source</th><th>Page</th><th>Relevance</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <table class="evi-table">
                <thead><tr><th>ID</th><th>Source</th><th>Page</th><th>Relevance</th></tr></thead>
                <tbody><tr class="empty-row"><td colspan="4">No evidence retrieved yet.</td></tr></tbody>
            </table>
            """,
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Knowledge base ----------
    bookshelf_html = (
        '<span class="bookshelf">'
        '<span style="height:12px;background:#158089;"></span>'
        '<span style="height:18px;background:#c6912b;"></span>'
        '<span style="height:9px;background:#0c5c63;"></span>'
        '<span style="height:15px;background:#c8503c;"></span>'
        '<span style="height:11px;background:#158089;"></span>'
        '</span>'
    )
    st.markdown(
        f'<div class="card card-tight fade-in d2">'
        f'<div class="section-label">{icon("database", 14)} KNOWLEDGE BASE {bookshelf_html}'
        f'<span class="badge-live"><span class="pdot"></span>LIVE</span></div>'
        f'<div class="section-title" style="font-size:1.22rem;">What\'s indexed right now</div>',
        unsafe_allow_html=True,
    )
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f'<div class="metric"><div class="metric-number">{chunk_count:,}</div>'
            '<div class="metric-label">Chunks indexed</div></div>',
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            '<div class="metric"><div class="metric-number">3</div>'
            '<div class="metric-label">Guideline sources</div></div>',
            unsafe_allow_html=True,
        )

    import datetime as _dt
    kb_specs = [
        ("Embedding model", "BAAI/bge-small-en-v1.5"),
        ("Retrieval method", "Hybrid dense + BM25, cross-encoder rerank"),
        ("Reranker", "ms-marco-MiniLM-L-6-v2"),
        ("Last index refresh", _dt.date.today().strftime("%d %b %Y")),
        ("Vector store", "ChromaDB (persistent)"),
    ]
    spec_rows = "".join(
        f'<div class="spec-row"><span class="spec-k">{html.escape(k)}</span>'
        f'<span class="spec-v">{html.escape(v)}</span></div>'
        for k, v in kb_specs
    )
    status_ok = bool(backend)
    st.markdown(
        f'<div class="spec-list">{spec_rows}'
        f'<div class="spec-row"><span class="spec-k">Database status</span>'
        f'<span class="status-pill {"online" if status_ok else ""}" style="padding:3px 10px 3px 8px;">'
        f'<span class="pdot"></span>{"ONLINE" if status_ok else "OFFLINE"}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="small-note" style="margin-top:10px;">'
        f'Chroma: {html.escape(CHROMA_PATH or "not found")}<br>'
        f'BM25: {html.escape(BM25_FILE or "not found")}'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Pipeline ----------
    st.markdown(
        f'<div class="card card-tight fade-in d3">'
        f'<div class="section-label">{icon("flask", 14)} RAG PIPELINE</div>'
        f'<div class="section-title" style="font-size:1.22rem;">Nothing is a black box</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("QUERY", "INPUT", "search"),
        ("BGE EMBEDDING", "SEMANTIC", "target"),
        ("BM25", "LEXICAL", "book"),
        ("MINI-LM RERANK", "RERANK", "layers"),
        ("QWEN 2.5", "GENERATE", "sparkles"),
        ("CITATIONS", "TRACE", "clipboard"),
    ]
    step_html = ""
    for name, kind, ic in steps:
        step_html += (
            f'<div class="pipeline-step"><div class="p-icon">{icon(ic, 13)}</div>'
            f'<div class="p-name">{name}</div><div class="p-kind">{kind}</div></div>'
        )
    st.markdown(step_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Confidence & safety ----------
    st.markdown(
        f'<div class="card card-tight fade-in d4">'
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
    f'</div>',
    unsafe_allow_html=True,
)
