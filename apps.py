import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
try:
    import cv2
except ImportError:
    cv2 = None
import os
import json
import base64
import io
from datetime import datetime
import streamlit.components.v1 as components

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="B3 Detector",
    page_icon="♻️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# CSS — Mobile-first, clean & consistent
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

:root {
    --bg       : #f5f6fa;
    --surface  : #ffffff;
    --border   : #e4e7f0;
    --text-1   : #111827;
    --text-2   : #4b5563;
    --text-3   : #9ca3af;
    --red-bg   : #fff1f0;
    --red-bdr  : #fca5a5;
    --red-txt  : #dc2626;
    --green-bg : #f0fdf4;
    --green-bdr: #86efac;
    --green-txt: #16a34a;
    --blue     : #2563eb;
    --radius   : 14px;
    --shadow   : 0 2px 12px rgba(0,0,0,0.06);
}

*, html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: var(--bg) !important; }
.block-container {
    padding: 0 1rem 5rem !important;
    max-width: 520px !important;
    margin: 0 auto;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Topbar ── */
.topbar {
    background: var(--text-1);
    margin: 0 -1rem 1.5rem;
    padding: 1rem 1.2rem 0.9rem;
    display: flex; align-items: center; justify-content: space-between;
    position: sticky; top: 0; z-index: 99;
}
.tb-left { display: flex; align-items: center; gap: 0.65rem; }
.tb-dot {
    width: 34px; height: 34px; border-radius: 9px;
    background: #22c55e;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem; flex-shrink: 0;
}
.tb-title { font-size: 1rem; font-weight: 700; color: #fff; letter-spacing: -0.02em; }
.tb-sub   { font-size: 0.62rem; color: rgba(255,255,255,0.45); }
.tb-pill  {
    font-family: 'DM Mono', monospace;
    font-size: 0.55rem; font-weight: 500; letter-spacing: 0.08em;
    text-transform: uppercase; color: rgba(255,255,255,0.55);
    background: rgba(255,255,255,0.08); border: 1px solid rgba(255,255,255,0.12);
    padding: 0.25rem 0.65rem; border-radius: 20px;
}

/* ── Card base ── */
.card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--radius); box-shadow: var(--shadow);
    padding: 1.1rem; margin-bottom: 1rem;
}

/* ── Status bar ── */
.sbar {
    border-radius: 10px; padding: 0.6rem 0.9rem;
    font-size: 0.76rem; font-weight: 600; margin-bottom: 1rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.sbar-ok   { background: var(--green-bg); border: 1px solid var(--green-bdr); color: #15803d; }
.sbar-warn { background: #fffbeb; border: 1px solid #fcd34d; color: #92400e; }

/* ── Tabs (custom override) ── */
.stTabs [data-baseweb="tab-list"] {
    background: #f0f2f8 !important; border-radius: 10px !important;
    padding: 3px !important; gap: 2px !important;
    border: none !important; margin-bottom: 0.8rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-size: 0.8rem !important;
    font-weight: 600 !important; color: var(--text-3) !important;
    padding: 0.5rem 0.9rem !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: var(--surface) !important; color: var(--text-1) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.1) !important;
}

/* ── Upload zone ── */
.upload-zone {
    background: #f8f9ff; border: 2px dashed #c7d2fe;
    border-radius: 12px; padding: 1.5rem 1rem; text-align: center;
    margin-bottom: 0.75rem;
}
.uz-icon { font-size: 2.2rem; display: block; margin-bottom: 0.4rem; }
.uz-title { font-size: 0.88rem; font-weight: 700; color: var(--text-1); margin-bottom: 0.15rem; }
.uz-sub   { font-size: 0.7rem; color: var(--text-3); margin-bottom: 0.7rem; }
.uz-fmts  { display: flex; justify-content: center; gap: 0.35rem; }
.uz-fmt {
    background: #eff6ff; color: var(--blue);
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; padding: 0.18rem 0.55rem;
    border-radius: 5px; border: 1px solid #bfdbfe;
}

/* Streamlit file uploader */
[data-testid="stFileUploader"] {
    background: transparent !important;
    border: none !important; padding: 0 !important;
}
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    border: none !important; padding: 0 !important;
    display: none !important;
}

/* Camera button */
[data-testid="stCameraInput"] button {
    background: var(--blue) !important; color: white !important;
    border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; font-size: 0.85rem !important;
    padding: 0.7rem 1.2rem !important; width: 100% !important;
}
[data-testid="stCameraInput"] video {
    border-radius: 10px !important;
}
.cam-hint {
    background: #eff6ff; border: 1px solid #bfdbfe; border-radius: 9px;
    padding: 0.55rem 0.8rem; font-size: 0.72rem; color: #1d4ed8;
    margin-bottom: 0.75rem; font-weight: 500;
}

/* ── Verdict card ── */
.verdict {
    border-radius: 14px; padding: 1.4rem; text-align: center; margin-bottom: 0.8rem;
}
.verdict-b3   { background: var(--red-bg);   border: 1.5px solid var(--red-bdr); }
.verdict-safe { background: var(--green-bg); border: 1.5px solid var(--green-bdr); }
.v-icon   { font-size: 2.8rem; display: block; margin-bottom: 0.3rem; }
.v-name   { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.04em; line-height: 1; }
.v-name-b3   { color: var(--red-txt); }
.v-name-safe { color: var(--green-txt); }
.v-tag {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.14em;
    text-transform: uppercase; margin: 0.35rem 0 0.75rem;
}
.v-tag-b3   { color: #ef4444; }
.v-tag-safe { color: #22c55e; }
.v-desc { font-size: 0.8rem; color: var(--text-2); line-height: 1.6; }

/* ── Confidence pills ── */
.pills { display: flex; gap: 0.5rem; margin-bottom: 0.8rem; }
.pill {
    flex: 1; background: #f8f9fc; border: 1.5px solid var(--border);
    border-radius: 11px; padding: 0.75rem 0.5rem; text-align: center;
}
.pill-v { font-size: 1.1rem; font-weight: 800; display: block; letter-spacing: -0.03em; }
.pill-l { font-size: 0.55rem; font-weight: 700; letter-spacing: 0.09em;
          text-transform: uppercase; color: var(--text-3); display: block; margin-top: 0.1rem; }

/* ── Explanation box ── */
.explain {
    border-radius: 12px; padding: 1rem; margin-bottom: 0.8rem; font-size: 0.8rem; line-height: 1.7;
}
.explain-b3   { background: #fff7ed; border: 1.5px solid #fed7aa; }
.explain-safe { background: var(--green-bg); border: 1.5px solid var(--green-bdr); }
.explain-title { font-weight: 700; margin-bottom: 0.5rem; font-size: 0.82rem; }
.explain-b3   .explain-title { color: #c2410c; }
.explain-safe .explain-title { color: #15803d; }
.explain-item { display: flex; gap: 0.5rem; align-items: flex-start; margin-bottom: 0.3rem; }
.explain-dot  { flex-shrink: 0; margin-top: 0.2rem; }

/* ── YOLO objects ── */
.objs-title {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-3); margin: 0.75rem 0 0.4rem;
}
.obj-chip {
    display: inline-flex; align-items: center; gap: 0.35rem;
    background: #f1f5f9; border: 1px solid var(--border);
    border-radius: 20px; padding: 0.25rem 0.65rem;
    font-size: 0.72rem; font-weight: 600; color: var(--text-2);
    margin: 0 0.3rem 0.3rem 0;
}

/* ── Det label ── */
.det-label {
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: var(--text-3); margin: 0.75rem 0 0.4rem;
}
.hdiv { height: 1px; background: var(--border); margin: 1rem 0; }

/* ── History section ── */
.hist-wrap { margin-top: 1.5rem; }
.hist-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.75rem;
}
.hist-head-title { font-size: 0.95rem; font-weight: 700; color: var(--text-1); }
.hist-count {
    font-family: 'DM Mono', monospace; font-size: 0.62rem; color: var(--text-3);
    background: #f0f2f8; border-radius: 20px; padding: 0.18rem 0.6rem;
    border: 1px solid var(--border);
}
.stat-strip { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.stat-box {
    flex: 1; background: var(--surface); border: 1px solid var(--border);
    border-radius: 11px; padding: 0.7rem 0.5rem; text-align: center;
    box-shadow: var(--shadow);
}
.stat-v { font-size: 1.3rem; font-weight: 800; display: block; }
.stat-l { font-size: 0.55rem; font-weight: 700; letter-spacing: 0.09em;
          text-transform: uppercase; color: var(--text-3); display: block; margin-top: 0.1rem; }

/* History card */
.hcard {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 12px; padding: 0.8rem; margin-bottom: 0.6rem;
    display: flex; align-items: center; gap: 0.75rem;
    box-shadow: var(--shadow);
}
.hcard-b3   { border-left: 3px solid #ef4444; }
.hcard-safe { border-left: 3px solid #22c55e; }
.hcard-img {
    width: 54px; height: 54px; border-radius: 9px;
    object-fit: cover; flex-shrink: 0;
    background: #f0f2f8;
}
.hcard-info { flex: 1; min-width: 0; }
.hcard-verdict { font-size: 0.82rem; font-weight: 800; }
.hcard-verdict-b3   { color: var(--red-txt); }
.hcard-verdict-safe { color: var(--green-txt); }
.hcard-meta {
    display: flex; flex-wrap: wrap; gap: 0.35rem;
    align-items: center; margin-top: 0.25rem;
}
.hcard-conf {
    font-family: 'DM Mono', monospace; font-size: 0.65rem; font-weight: 500;
}
.hcard-conf-b3   { color: #ef4444; }
.hcard-conf-safe { color: #22c55e; }
.hcard-time { font-size: 0.6rem; color: var(--text-3); }
.hcard-src  {
    font-size: 0.57rem; background: #f0f2f8; color: var(--text-3);
    border-radius: 4px; padding: 0.1rem 0.35rem; border: 1px solid var(--border);
}
.hist-empty {
    text-align: center; padding: 2rem 1rem;
    background: #f8f9fc; border: 1.5px dashed var(--border);
    border-radius: 12px;
}
.hist-empty-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.hist-empty-text { font-size: 0.78rem; color: var(--text-3); line-height: 1.6; }

/* Filter row */
.filter-row { display: flex; gap: 0.5rem; margin-bottom: 0.75rem; }

/* Streamlit image */
[data-testid="stImage"] img { border-radius: 11px !important; }

/* Mobile safe bottom */
@media (max-width: 480px) {
    .block-container { padding: 0 0.75rem 5rem !important; }
    .topbar { padding: 0.9rem 0.9rem; }
    .tb-title { font-size: 0.95rem; }
}

/* Streamlit selectbox compact */
.stSelectbox > div > div {
    border-radius: 9px !important;
    font-size: 0.8rem !important;
}

/* Download button */
.stDownloadButton button {
    border-radius: 9px !important; font-size: 0.8rem !important;
    padding: 0.5rem 1rem !important; font-weight: 600 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-size: 0.82rem !important; font-weight: 600 !important;
    color: var(--text-2) !important; border-radius: 9px !important;
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# RIWAYAT — localStorage (per-device, persistent)
# ══════════════════════════════════════════════════════════════
STORAGE_KEY = "b3_detector_history_v2"

def render_localstorage_bridge():
    """
    Komponen JS yang:
    1. Membaca riwayat dari localStorage saat halaman load
    2. Mengirimnya ke Streamlit via query param ?hist_data=...
    Ini memungkinkan riwayat tetap ada setelah restart, per-device.
    """
    components.html(f"""
    <script>
    (function() {{
        const KEY = '{STORAGE_KEY}';
        const raw = localStorage.getItem(KEY);
        if (raw) {{
            // Kirim ke parent Streamlit via postMessage
            window.parent.postMessage({{
                type: 'b3_history',
                data: raw
            }}, '*');
        }}
    }})();
    </script>
    """, height=0)

def save_to_localstorage(entry_json: str):
    """Inject JS untuk simpan entry baru ke localStorage."""
    escaped = entry_json.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    components.html(f"""
    <script>
    (function() {{
        const KEY = '{STORAGE_KEY}';
        let hist = [];
        try {{ hist = JSON.parse(localStorage.getItem(KEY) || '[]'); }} catch(e) {{}}
        const entry = JSON.parse(`{escaped}`);
        hist.unshift(entry);
        if (hist.length > 100) hist = hist.slice(0, 100);
        localStorage.setItem(KEY, JSON.stringify(hist));
    }})();
    </script>
    """, height=0)

def clear_localstorage():
    components.html(f"""
    <script>
    localStorage.removeItem('{STORAGE_KEY}');
    </script>
    """, height=0)

def pil_to_thumb_b64(img: Image.Image, size: int = 100) -> str:
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()

def build_history_entry(img: Image.Image, is_b3: bool, pred_proba: float,
                         yolo_boxes: list, source: str) -> dict:
    conf = round((1 - pred_proba) if is_b3 else pred_proba, 4)
    objs = [b[5] for b in yolo_boxes if len(b) > 5] if yolo_boxes else []
    return {
        "id"        : datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
        "timestamp" : datetime.now().strftime("%d %b %Y, %H:%M"),
        "is_b3"     : is_b3,
        "verdict"   : "B3" if is_b3 else "non-B3",
        "confidence": conf,
        "source"    : source,
        "objects"   : objs,
        "thumb_b64" : pil_to_thumb_b64(img),
    }


# ══════════════════════════════════════════════════════════════
# SESSION STATE — riwayat disimpan di sini sementara
# ══════════════════════════════════════════════════════════════
if "history" not in st.session_state:
    st.session_state.history = []
if "last_saved_id" not in st.session_state:
    st.session_state.last_saved_id = None


# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_classifier():
    _STRIP = ["renorm","renorm_clipping","renorm_momentum","synchronized",
              "adjustment","quantization_config"]
    def _s(kw):
        for k in _STRIP: kw.pop(k, None)
        return kw
    class CBN(tf.keras.layers.BatchNormalization):
        def __init__(self, **kw): super().__init__(**_s(kw))
    class CIL(tf.keras.layers.InputLayer):
        def __init__(self, **kw):
            _s(kw)
            if "batch_shape" in kw:
                kw["input_shape"] = kw.pop("batch_shape")[1:]
            super().__init__(**kw)
    class CD(tf.keras.layers.Dense):
        def __init__(self, **kw): super().__init__(**_s(kw))
    class CC(tf.keras.layers.Conv2D):
        def __init__(self, **kw): super().__init__(**_s(kw))
    class CDC(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kw): super().__init__(**_s(kw))
    return tf.keras.models.load_model(
        os.path.join(BASE_DIR, "model_b3.h5"),
        custom_objects={"BatchNormalization":CBN,"InputLayer":CIL,"Dense":CD,
                        "Conv2D":CC,"DepthwiseConv2D":CDC},
        compile=False
    )

@st.cache_resource(show_spinner=False)
def load_yolo():
    try:
        from ultralytics import YOLO
        for fname in ["best.pt","yolov8n.pt"]:
            p = os.path.join(BASE_DIR, fname)
            if os.path.exists(p): return YOLO(p)
        return None
    except Exception:
        return None

def run_yolo(yolo_model, pil_img):
    if yolo_model is None: return []
    try:
        results = yolo_model(np.array(pil_img), verbose=False, conf=0.25)
        boxes = []
        for r in results:
            if r.boxes is None: continue
            for b in r.boxes:
                x1,y1,x2,y2 = b.xyxy[0].tolist()
                boxes.append((int(x1),int(y1),int(x2),int(y2),
                              float(b.conf[0]),
                              yolo_model.names.get(int(b.cls[0]),"obj")))
        return boxes
    except Exception:
        return []

def draw_overlay(pil_img, yolo_boxes, is_b3, pred_proba, yolo_ok):
    W, H   = pil_img.size
    result = pil_img.copy()
    draw   = ImageDraw.Draw(result)
    color  = (220,38,38) if is_b3 else (22,163,74)
    lw     = max(3, W//120)
    try:    font = ImageFont.truetype("arial.ttf", max(13, W//38))
    except: font = ImageFont.load_default()

    def corners(x1,y1,x2,y2):
        cl = max(18, W//28)
        for cx,cy,dx,dy in [(x1,y1,1,1),(x2,y1,-1,1),(x1,y2,1,-1),(x2,y2,-1,-1)]:
            draw.line([(cx,cy),(cx+dx*cl,cy)], fill=color, width=lw+2)
            draw.line([(cx,cy),(cx,cy+dy*cl)], fill=color, width=lw+2)

    def pill(x1,y1,label):
        bb = draw.textbbox((0,0),label,font=font)
        tw=bb[2]-bb[0]; th=bb[3]-bb[1]; ph=max(24,H//22)
        py=max(0,y1-ph-4)
        draw.rectangle([x1,py,x1+tw+16,py+ph], fill=color)
        draw.text((x1+8,py+(ph-th)//2), label, fill="white", font=font)

    verdict = "B3" if is_b3 else "non-B3"
    if yolo_ok and yolo_boxes:
        for (x1,y1,x2,y2,conf,_) in yolo_boxes:
            draw.rectangle([x1,y1,x2,y2], outline=color, width=lw)
            corners(x1,y1,x2,y2)
            pill(x1,y1,verdict)
        return result, "yolo"

    # Fallback saliency
    if cv2:
        gray = np.array(pil_img.convert("L"), dtype=np.float32)
        rows,cols = 7,7; ph2,pw2 = H//rows, W//cols
        heat = np.array([[np.var(gray[r*ph2:(r+1)*ph2,c*pw2:(c+1)*pw2])
                          for c in range(cols)] for r in range(rows)])
        if heat.max()>0: heat/=heat.max()
        mask = heat>0.4
        if not mask.any(): mask=heat==heat.max()
        yr,xc=np.where(mask)
        x1=max(0,xc.min()*pw2-10); y1=max(0,yr.min()*ph2-10)
        x2=min(W,(xc.max()+1)*pw2+10); y2=min(H,(yr.max()+1)*ph2+10)
    else:
        x1,y1,x2,y2=W//8,H//8,W*7//8,H*7//8

    draw.rectangle([x1,y1,x2,y2], outline=color, width=lw)
    corners(x1,y1,x2,y2)
    pill(x1,y1,verdict)
    return result, "fallback"


# ══════════════════════════════════════════════════════════════
# KONTEN PENJELASAN B3 & non-B3
# ══════════════════════════════════════════════════════════════
B3_ITEMS = [
    ("🔋", "Baterai", "Batu baterai, baterai HP, baterai laptop"),
    ("🧴", "Pembersih kimia", "Cairan pel, pembersih toilet, pemutih pakaian"),
    ("🧼", "Deterjen keras", "Sabun cuci baju berbahan kimia kuat"),
    ("💊", "Obat kadaluarsa", "Obat-obatan, suplemen, sirup yang sudah expired"),
    ("🛢️", "Oli & bensin", "Oli mesin, bensin, solar, cairan rem"),
    ("🖨️", "Elektronik rusak", "HP rusak, printer, kabel charger, lampu LED"),
    ("🎨", "Cat & thinner", "Cat tembok, pylox, lem, cairan thinner"),
    ("🌿", "Pestisida", "Obat nyamuk semprot, pembasmi hama"),
]

NON_B3_ITEMS = [
    ("📄", "Kertas & kardus", "Koran, majalah, dus bekas, kotak makanan"),
    ("🧴", "Plastik biasa", "Botol air, kantong belanja, bungkus makanan"),
    ("🥫", "Kaleng & logam", "Kaleng susu, kaleng cat (kosong), besi tua"),
    ("👕", "Pakaian & kain", "Baju bekas, handuk, kain perca"),
    ("🥬", "Sisa makanan", "Kulit buah, sayuran, nasi, tulang ayam"),
    ("👟", "Sepatu & tas", "Sepatu bekas, tas rusak, sandal"),
    ("📦", "Gabus & styrofoam", "Gabus bekas elektronik, wadah mie instan"),
    ("🌾", "Sampah dapur", "Ampas kopi, teh celup, cangkang telur"),
]

def render_b3_explanation():
    st.markdown("""
    <div class="explain explain-b3">
        <div class="explain-title">⚠️ Ini adalah Sampah B3 — Perlu Penanganan Khusus!</div>
        <p style="font-size:0.78rem;color:#78350f;margin-bottom:0.7rem">
        Sampah B3 adalah sampah yang <b>berbahaya dan beracun</b>. Jangan dibuang ke tempat sampah biasa
        karena bisa mencemari tanah, air, dan membahayakan kesehatan.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 Contoh sampah B3 lainnya yang perlu diwaspadai"):
        for emoji, nama, contoh in B3_ITEMS:
            st.markdown(f"**{emoji} {nama}** — {contoh}")
    st.markdown("""
    <div style="background:#fff7ed;border:1.5px solid #fed7aa;border-radius:12px;
                padding:0.9rem;margin-top:0.5rem;font-size:0.78rem;color:#78350f;line-height:1.7">
        <b>🗑️ Cara membuang yang benar:</b><br>
        1. Simpan di wadah tertutup rapat, jangan dicampur sampah lain<br>
        2. Bawa ke <b>bank sampah B3</b> atau <b>drop box limbah B3</b> terdekat<br>
        3. Bisa juga dititipkan ke toko elektronik atau bengkel resmi<br>
        4. Jangan dibakar — asapnya sangat berbahaya
    </div>
    """, unsafe_allow_html=True)

def render_safe_explanation():
    st.markdown("""
    <div class="explain explain-safe">
        <div class="explain-title">✅ Sampah Non-B3 — Aman Didaur Ulang!</div>
        <p style="font-size:0.78rem;color:#166534;margin-bottom:0.7rem">
        Sampah ini <b>tidak berbahaya</b> dan bisa dibuang ke tempat sampah biasa
        atau lebih baik lagi, didaur ulang agar bermanfaat kembali.</p>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("♻️ Contoh sampah non-B3 yang bisa didaur ulang"):
        for emoji, nama, contoh in NON_B3_ITEMS:
            st.markdown(f"**{emoji} {nama}** — {contoh}")
    st.markdown("""
    <div style="background:var(--green-bg);border:1.5px solid var(--green-bdr);border-radius:12px;
                padding:0.9rem;margin-top:0.5rem;font-size:0.78rem;color:#166534;line-height:1.7">
        <b>💡 Tips memilah sampah non-B3:</b><br>
        1. Pisahkan <b>organik</b> (sisa makanan) dan <b>anorganik</b> (plastik, kertas)<br>
        2. Cuci wadah plastik/kaca sebelum dibuang ke tempat daur ulang<br>
        3. Manfaatkan <b>bank sampah</b> di sekitar rumah<br>
        4. Sampah organik bisa dijadikan <b>kompos</b> untuk tanaman
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INIT MODELS
# ══════════════════════════════════════════════════════════════
with st.spinner("Memuat model AI..."):
    try: clf_model = load_classifier()
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {e}"); st.stop()

with st.spinner("Memuat YOLO..."):
    yolo_model = load_yolo()
    yolo_ok    = yolo_model is not None

THRESHOLD = 0.50


# ══════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════
yolo_label = "best.pt" if os.path.exists(os.path.join(BASE_DIR,"best.pt")) else "yolov8n"
st.markdown(f"""
<div class="topbar">
    <div class="tb-left">
        <div class="tb-dot">♻️</div>
        <div>
            <div class="tb-title">B3 Detector</div>
            <div class="tb-sub">Klasifikasi Sampah Berbahaya & Beracun</div>
        </div>
    </div>
    <div class="tb-pill">MobileNetV2</div>
</div>
""", unsafe_allow_html=True)

# Status YOLO
if yolo_ok:
    st.markdown(f'<div class="sbar sbar-ok">✅ YOLOv8 aktif (<b>{yolo_label}</b>) — Deteksi objek siap</div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="sbar sbar-warn">⚠️ YOLOv8 tidak tersedia — install: pip install ultralytics</div>',
                unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# MAIN TABS
# ══════════════════════════════════════════════════════════════
tab_detect, tab_history = st.tabs(["🔍  Deteksi", "🕘  Riwayat"])


# ════════════════════════════════
# TAB 1 — DETEKSI
# ════════════════════════════════
with tab_detect:
    img_source   = None
    input_source = "upload"

    # Cara input
    inp_tab1, inp_tab2 = st.tabs(["📁  Upload", "📷  Kamera"])

    with inp_tab1:
        st.markdown("""
        <div class="upload-zone">
            <span class="uz-icon">🗂️</span>
            <div class="uz-title">Upload Foto Sampah</div>
            <div class="uz-sub">Pilih dari galeri atau folder kamu</div>
            <div class="uz-fmts">
                <span class="uz-fmt">JPG</span>
                <span class="uz-fmt">PNG</span>
                <span class="uz-fmt">WEBP</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("up", type=["jpg","jpeg","png","webp"],
                                     label_visibility="collapsed")
        if uploaded:
            img_source   = Image.open(uploaded).convert("RGB")
            input_source = "upload"

    with inp_tab2:
        st.markdown('<div class="cam-hint">📸 Arahkan ke sampah lalu tekan <b>Take photo</b></div>',
                    unsafe_allow_html=True)
        snap = st.camera_input("cam", label_visibility="collapsed")
        if snap:
            img_source   = Image.open(snap).convert("RGB")
            input_source = "kamera"

    # ── Hasil deteksi ─────────────────────────────────────────
    if img_source is not None:
        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

        # Inference
        img_224    = img_source.resize((224,224))
        arr        = np.expand_dims(preprocess_input(
                         np.array(img_224, dtype=np.float32)), axis=0)
        pred_proba = float(clf_model.predict(arr, verbose=0)[0][0])
        is_b3      = pred_proba < THRESHOLD
        conf_b3    = 1.0 - pred_proba
        conf_safe  = pred_proba
        disp_conf  = conf_b3 if is_b3 else conf_safe

        with st.spinner("Menganalisis gambar..."):
            yolo_boxes = run_yolo(yolo_model, img_source)

        overlaid, mode = draw_overlay(img_source, yolo_boxes, is_b3, pred_proba, yolo_ok)

        # Detection image
        det_label = "🎯 YOLOv8 Detection" if mode=="yolo" else "🔍 Analisis Visual"
        st.markdown(f'<div class="det-label">{det_label}</div>', unsafe_allow_html=True)
        st.image(overlaid, use_container_width=True)

        # YOLO object chips
        if yolo_ok and yolo_boxes:
            st.markdown('<div class="objs-title">Objek Terdeteksi</div>', unsafe_allow_html=True)
            chips = "".join([
                f'<span class="obj-chip">#{i+1} {name} <b style="color:#6366f1">{conf:.0%}</b></span>'
                for i,(x1,y1,x2,y2,conf,name) in enumerate(yolo_boxes)
            ])
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

        # Verdict
        if is_b3:
            st.markdown(f"""
            <div class="verdict verdict-b3">
                <span class="v-icon">☣️</span>
                <div class="v-name v-name-b3">B3</div>
                <div class="v-tag v-tag-b3">⚠ BAHAN BERBAHAYA & BERACUN</div>
                <div class="v-desc">
                    Sampah ini mengandung bahan yang <b>berbahaya bagi kesehatan dan lingkungan</b>.
                    Tidak boleh dibuang sembarangan.
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict verdict-safe">
                <span class="v-icon">✅</span>
                <div class="v-name v-name-safe">non-B3</div>
                <div class="v-tag v-tag-safe">✓ AMAN — TIDAK BERBAHAYA</div>
                <div class="v-desc">
                    Sampah ini <b>tidak berbahaya</b> dan aman untuk didaur ulang
                    atau dibuang ke tempat sampah biasa.
                </div>
            </div>""", unsafe_allow_html=True)

        # Pills
        color_main = "#dc2626" if is_b3 else "#16a34a"
        st.markdown(f"""
        <div class="pills">
            <div class="pill">
                <span class="pill-v" style="color:{color_main}">{disp_conf:.1%}</span>
                <span class="pill-l">Keyakinan</span>
            </div>
            <div class="pill">
                <span class="pill-v" style="color:#6366f1">{pred_proba:.3f}</span>
                <span class="pill-l">Skor Model</span>
            </div>
            <div class="pill">
                <span class="pill-v" style="color:#64748b">{THRESHOLD:.2f}</span>
                <span class="pill-l">Ambang Batas</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Penjelasan B3 / non-B3
        if is_b3:
            render_b3_explanation()
        else:
            render_safe_explanation()

        # ── Simpan ke riwayat (session + localStorage) ────────
        entry = build_history_entry(img_source, is_b3, pred_proba,
                                     yolo_boxes, input_source)

        # Hindari duplikat dari re-render
        if st.session_state.last_saved_id != entry["id"]:
            st.session_state.history.insert(0, entry)
            if len(st.session_state.history) > 100:
                st.session_state.history = st.session_state.history[:100]
            st.session_state.last_saved_id = entry["id"]
            # Simpan ke localStorage browser (persistent per-device)
            save_to_localstorage(json.dumps(entry, ensure_ascii=False))


# ════════════════════════════════
# TAB 2 — RIWAYAT
# ════════════════════════════════
with tab_history:
    history  = st.session_state.history
    n_total  = len(history)
    n_b3     = sum(1 for h in history if h["is_b3"])
    n_safe   = n_total - n_b3

    # Header
    st.markdown(f"""
    <div class="hist-head">
        <span class="hist-head-title">📋 Riwayat Sesi Ini</span>
        <span class="hist-count">{n_total} deteksi</span>
    </div>
    """, unsafe_allow_html=True)

    # Info localStorage
    st.info("💾 Riwayat tersimpan di browser kamu. Akan tetap ada meski halaman di-refresh, "
            "tapi tidak terlihat di perangkat lain.", icon="ℹ️")

    # Komponen JS untuk membaca localStorage dan menampilkan riwayat lama
    components.html(f"""
    <div id="ls-history-root" style="font-family:'DM Sans',sans-serif;"></div>
    <script>
    (function() {{
        const KEY = '{STORAGE_KEY}';
        let hist = [];
        try {{ hist = JSON.parse(localStorage.getItem(KEY) || '[]'); }} catch(e) {{}}

        const root = document.getElementById('ls-history-root');
        if (!hist.length) {{
            root.innerHTML = '';
            return;
        }}

        const total = hist.length;
        const nb3   = hist.filter(h => h.is_b3).length;
        const nsafe = total - nb3;

        root.innerHTML = `
        <div style="background:#fff;border:1px solid #e4e7f0;border-radius:14px;
                    padding:1rem;margin-bottom:1rem;box-shadow:0 2px 12px rgba(0,0,0,0.06)">
            <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                      color:#9ca3af;margin:0 0 0.75rem">📱 Tersimpan di Perangkat Ini</p>
            <div style="display:flex;gap:0.5rem;margin-bottom:1rem">
                <div style="flex:1;background:#f8f9fc;border:1px solid #e4e7f0;border-radius:11px;
                            padding:0.65rem;text-align:center">
                    <span style="font-size:1.3rem;font-weight:800;color:#111827;display:block">${{total}}</span>
                    <span style="font-size:0.55rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;
                                 color:#9ca3af">Total</span>
                </div>
                <div style="flex:1;background:#fff5f5;border:1px solid #fca5a5;border-radius:11px;
                            padding:0.65rem;text-align:center">
                    <span style="font-size:1.3rem;font-weight:800;color:#dc2626;display:block">${{nb3}}</span>
                    <span style="font-size:0.55rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;
                                 color:#9ca3af">B3</span>
                </div>
                <div style="flex:1;background:#f0fdf4;border:1px solid #86efac;border-radius:11px;
                            padding:0.65rem;text-align:center">
                    <span style="font-size:1.3rem;font-weight:800;color:#16a34a;display:block">${{nsafe}}</span>
                    <span style="font-size:0.55rem;font-weight:700;letter-spacing:0.09em;text-transform:uppercase;
                                 color:#9ca3af">Non-B3</span>
                </div>
            </div>
            ${{hist.slice(0,20).map((h,i) => {{
                const bdr  = h.is_b3 ? '#ef4444' : '#22c55e';
                const bg   = h.is_b3 ? '#fff5f5' : '#f0fdf4';
                const icon = h.is_b3 ? '☣️' : '✅';
                const verd = h.is_b3 ? 'B3 — BERBAHAYA' : 'non-B3 — AMAN';
                const col  = h.is_b3 ? '#dc2626' : '#16a34a';
                const conf = ((h.confidence||0)*100).toFixed(1);
                const src  = h.source === 'kamera' ? '📷 Kamera' : '📁 Upload';
                const thumb= h.thumb_b64
                    ? `<img src="data:image/jpeg;base64,${{h.thumb_b64}}"
                           style="width:52px;height:52px;border-radius:9px;object-fit:cover;flex-shrink:0" />`
                    : `<div style="width:52px;height:52px;border-radius:9px;background:#f0f2f8;flex-shrink:0"></div>`;
                return `<div style="background:#fff;border:1px solid #e4e7f0;border-left:3px solid ${{bdr}};
                                    border-radius:12px;padding:0.75rem;margin-bottom:0.5rem;
                                    display:flex;align-items:center;gap:0.75rem">
                    ${{thumb}}
                    <div style="flex:1;min-width:0">
                        <div style="font-size:0.82rem;font-weight:800;color:${{col}}">${{icon}} ${{verd}}</div>
                        <div style="display:flex;flex-wrap:wrap;gap:0.35rem;align-items:center;margin-top:0.25rem">
                            <span style="font-family:monospace;font-size:0.65rem;font-weight:600;color:${{col}}">${{conf}}%</span>
                            <span style="font-size:0.6rem;color:#9ca3af">${{src}}</span>
                            <span style="font-size:0.6rem;color:#cbd5e1">${{h.timestamp||''}}</span>
                        </div>
                    </div>
                </div>`;
            }}).join('')}}
            ${{hist.length > 20 ? `<p style="text-align:center;font-size:0.72rem;color:#9ca3af">
                ... dan ${{hist.length-20}} entri lainnya</p>` : ''}}
            <button onclick="if(confirm('Hapus semua riwayat di perangkat ini?')){{
                localStorage.removeItem('{STORAGE_KEY}'); location.reload();}}"
                style="width:100%;margin-top:0.75rem;padding:0.6rem;border-radius:9px;
                       border:1.5px solid #e4e7f0;background:#f8f9fc;color:#64748b;
                       font-size:0.78rem;font-weight:600;cursor:pointer">
                🗑️ Hapus Riwayat Perangkat Ini
            </button>
        </div>`;
    }})();
    </script>
    """, height=600, scrolling=True)

    # Export riwayat sesi ini
    if st.session_state.history:
        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

        # Stats sesi
        st.markdown(f"""
        <div style="margin-bottom:0.75rem">
            <p style="font-size:0.7rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;
                      color:#9ca3af;margin-bottom:0.6rem">Sesi Sekarang ({n_total} deteksi)</p>
            <div class="stat-strip">
                <div class="stat-box">
                    <span class="stat-v" style="color:#111827">{n_total}</span>
                    <span class="stat-l">Total</span>
                </div>
                <div class="stat-box">
                    <span class="stat-v" style="color:#dc2626">{n_b3}</span>
                    <span class="stat-l">B3</span>
                </div>
                <div class="stat-box">
                    <span class="stat-v" style="color:#16a34a">{n_safe}</span>
                    <span class="stat-l">Non-B3</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Export JSON (tanpa thumb untuk ukuran kecil)
        export_data = [{k:v for k,v in h.items() if k != "thumb_b64"}
                       for h in st.session_state.history]
        json_str = json.dumps(export_data, ensure_ascii=False, indent=2)
        st.download_button(
            label="⬇️ Export Riwayat Sesi (.json)",
            data=json_str,
            file_name=f"riwayat_b3_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json",
            use_container_width=True
        )

        # Daftar kartu sesi
        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
        for entry in st.session_state.history:
            b3      = entry["is_b3"]
            card_c  = "hcard-b3" if b3 else "hcard-safe"
            v_c     = "hcard-verdict-b3" if b3 else "hcard-verdict-safe"
            c_c     = "hcard-conf-b3" if b3 else "hcard-conf-safe"
            icon    = "☣️" if b3 else "✅"
            verdict = "B3 — BERBAHAYA" if b3 else "non-B3 — AMAN"
            src_ico = "📷" if entry.get("source") == "kamera" else "📁"
            thumb   = entry.get("thumb_b64","")
            thumb_html = (f'<img class="hcard-img" src="data:image/jpeg;base64,{thumb}" />'
                          if thumb else '<div class="hcard-img"></div>')
            st.markdown(f"""
            <div class="hcard {card_c}">
                {thumb_html}
                <div class="hcard-info">
                    <div class="hcard-verdict {v_c}">{icon} {verdict}</div>
                    <div class="hcard-meta">
                        <span class="hcard-conf {c_c}">{entry['confidence']:.1%}</span>
                        <span class="hcard-src">{src_ico} {entry.get('source','—')}</span>
                        <span class="hcard-time">{entry.get('timestamp','')}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="hist-empty">
            <div class="hist-empty-icon">📂</div>
            <div class="hist-empty-text">Belum ada deteksi di sesi ini.<br>
            Mulai dari tab <b>🔍 Deteksi</b>.</div>
        </div>""", unsafe_allow_html=True)


# Footer
st.markdown("""
<div style="text-align:center;margin-top:2rem;font-size:0.62rem;color:#cbd5e1;
            font-family:'DM Mono',monospace;letter-spacing:0.06em">
    B3 Detector · MobileNetV2 + YOLOv8 · Streamlit
</div>""", unsafe_allow_html=True)