import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
try:
    import cv2
except ImportError as e:
    cv2 = None
import os
import json
import base64
import io
from datetime import datetime
import streamlit.components.v1 as components

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
STORAGE_KEY = "b3_detector_history_v2"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="B3 Waste Detector",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════
# CSS — Sembunyikan semua elemen Streamlit branding
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #f0f2f8 !important;
    color: #111827 !important;
}
.block-container { padding: 0 2rem 3rem !important; max-width: 1200px; }

/* ══════════════════════════════════════════════
   SEMBUNYIKAN SEMUA ELEMEN STREAMLIT BRANDING
   ══════════════════════════════════════════════ */

/* Navbar atas (hamburger menu) */
#MainMenu { visibility: hidden !important; display: none !important; }

/* Footer bawah Streamlit */
footer { visibility: hidden !important; display: none !important; }

/* Header Streamlit */
header { visibility: hidden !important; display: none !important; }

/* Toolbar kanan atas (deploy, settings, dll) */
[data-testid="stToolbar"] {
    visibility: hidden !important;
    display: none !important;
}

/* Badge Streamlit pojok kanan bawah (mahkota merah) */
[data-testid="stDecoration"] {
    visibility: hidden !important;
    display: none !important;
}

/* Deploy button */
.stDeployButton,
.stAppDeployButton {
    visibility: hidden !important;
    display: none !important;
}
            
/* Status widget (spinner running indicator) */
[data-testid="stStatusWidget"] {
    visibility: hidden !important;
    display: none !important;
}

/* Streamlit viewer badge / logo bottom right */
.viewerBadge_container__r5tak {
    visibility: hidden !important;
    display: none !important;
}
.viewerBadge_link__qRIco {
    visibility: hidden !important;
    display: none !important;
}

/* Catch-all untuk badge viewer dengan class yang berubah-ubah */
[class*="viewerBadge"] {
    visibility: hidden !important;
    display: none !important;
}

/* GitHub/profile avatar pojok kiri bawah */
[data-testid="stAppViewBlockContainer"] ~ div[style*="position: fixed"] {
    display: none !important;
}

/* Streamlit community cloud bottom bar */
[data-testid="collapsedControl"] {
    display: none !important;
}

/* Garis dekorasi merah di atas (Streamlit top bar) */
#stDecoration {
    display: none !important;
}

/* Semua elemen fixed position di pojok layar yang bukan milik kita */
.css-1dp5vir, .css-14xtw13, .e8zbici0 {
    display: none !important;
}

/* Iframe badge jika ada */
.streamlit-badge {
    display: none !important;
}

/* Powered by Streamlit text */
[class*="powered"] {
    display: none !important;
}

/* ══════════════════════════════════════════════
   STYLE KOMPONEN APP
   ══════════════════════════════════════════════ */

.topbar {
    background: linear-gradient(135deg, #1a1f3a 0%, #1e3a8a 60%, #1d4ed8 100%);
    margin: 0 -2rem 2rem; padding: 1.2rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-left  { display: flex; align-items: center; gap: 0.9rem; }
.topbar-icon  {
    width: 44px; height: 44px; border-radius: 12px;
    background: rgba(255,255,255,0.15); border: 1px solid rgba(255,255,255,0.2);
    display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
}
.topbar-title { font-size: 1.2rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
.topbar-sub   { font-size: 0.68rem; color: rgba(255,255,255,0.6); margin-top: 0.1rem; }
.topbar-badge {
    background: rgba(255,255,255,0.12); color: rgba(255,255,255,0.9);
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.35rem 0.9rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2);
}
.status-bar { border-radius: 10px; padding: 0.65rem 1rem; font-size: 0.75rem; font-weight: 600; margin-bottom: 1.5rem; }
.status-ok  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #15803d; }
.status-warn{ background: #fffbeb; border: 1px solid #fde68a; color: #b45309; }

.stTabs [data-baseweb="tab-list"] {
    background: #e8eaf0 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 2px !important; border: none !important; margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-size: 0.78rem !important;
    font-weight: 600 !important; color: #6b7280 !important;
    padding: 0.45rem 1.1rem !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #111827 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
.upload-zone {
    background: linear-gradient(135deg, #fafbff, #f0f4ff);
    border: 2.5px dashed #c7d2fe; border-radius: 16px;
    padding: 2rem 1.5rem 1.2rem; text-align: center; margin-bottom: 0.75rem;
}
.upload-icon  { font-size: 2.8rem; display: block; margin-bottom: 0.5rem; }
.upload-title { font-size: 0.92rem; font-weight: 700; color: #111827; margin-bottom: 0.2rem; }
.upload-sub   { font-size: 0.72rem; color: #6b7280; margin-bottom: 0.9rem; }
.fmt-row      { display: flex; justify-content: center; gap: 0.45rem; margin-bottom: 1rem; }
.fmt-badge {
    background: #eff6ff; color: #3b82f6; font-size: 0.6rem; font-weight: 700;
    padding: 0.22rem 0.6rem; border-radius: 6px; letter-spacing: 0.05em;
    text-transform: uppercase; border: 1px solid #bfdbfe;
}
.cam-hint {
    background: #f0f4ff; border: 1px solid #c7d2fe; border-radius: 10px;
    padding: 0.65rem 0.9rem; font-size: 0.73rem; color: #4338ca;
    margin-bottom: 0.8rem; font-weight: 500;
}
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, #1a56db, #5b8dee) !important;
    color: white !important; border: none !important;
    border-radius: 8px !important; font-weight: 600 !important; font-size: 0.8rem !important;
}
[data-testid="stFileUploader"] {
    background: #f9fafb !important; border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
}
.hdiv { height: 1px; background: linear-gradient(90deg,transparent,#e2e8f0,transparent); margin: 1rem 0; }
.det-label {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #6b7280; margin: 0.8rem 0 0.4rem;
}
.verdict-card { border-radius: 14px; padding: 1.5rem 1.4rem 1.2rem; text-align: center; margin-bottom: 1.2rem; }
.verdict-b3   { background: linear-gradient(135deg,#fff1f1,#ffe4e4); border: 2px solid #fca5a5; }
.verdict-safe { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 2px solid #86efac; }
.v-emoji  { font-size: 2.6rem; display: block; margin-bottom: 0.4rem; }
.v-label  { font-size: 2rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1; }
.v-label-b3   { color: #dc2626; }
.v-label-safe { color: #16a34a; }
.v-status { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em; text-transform: uppercase; margin: 0.4rem 0 0.8rem; }
.v-status-b3   { color: #ef4444; }
.v-status-safe { color: #22c55e; }
.v-desc { font-size: 0.75rem; color: #374151; line-height: 1.65; }
.pills { display: flex; gap: 0.7rem; margin-bottom: 1rem; }
.pill { flex: 1; background: #f8fafc; border: 1.5px solid #e2e8f0; border-radius: 12px; padding: 0.8rem 0.6rem; text-align: center; }
.pill-val { font-size: 1.15rem; font-weight: 800; display: block; letter-spacing: -0.03em; color: #111827; }
.pill-lbl { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #6b7280; display: block; margin-top: 0.2rem; }
.obj-row { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.5rem 0.8rem; margin-bottom: 0.35rem; font-size: 0.73rem; display: flex; align-items: center; justify-content: space-between; }
.obj-name { font-weight: 700; color: #111827; }
.obj-conf { color: #6366f1; font-weight: 600; }
.obj-size { color: #6b7280; font-size: 0.66rem; }
.explain-box { border-radius: 12px; padding: 1rem 1.1rem; font-size: 0.75rem; line-height: 1.75; margin-top: 0.75rem; margin-bottom: 0.5rem; }
.explain-box-b3   { background: #fffbeb; border: 1.5px solid #fde68a; }
.explain-box-safe { background: #f0fdf4; border: 1.5px solid #bbf7d0; }
.explain-title { font-size: 0.82rem; font-weight: 800; color: #111827; margin-bottom: 0.6rem; display: block; }
.tip { border-radius: 12px; padding: 0.85rem 1rem; font-size: 0.73rem; line-height: 1.75; margin-top: 0.5rem; }
.tip-b3   { background: #fffbeb; border: 1.5px solid #fde68a; color: #111827; }
.tip-safe { background: #f0fdf4; border: 1.5px solid #bbf7d0; color: #111827; }
.empty-state { background: #fff; border: 2px dashed #e2e8f0; border-radius: 18px; padding: 3.5rem 2rem; text-align: center; }
.empty-icon  { font-size: 3rem; margin-bottom: 0.8rem; display: block; }
.empty-title { font-size: 0.92rem; font-weight: 700; color: #6b7280; margin-bottom: 0.3rem; }
.empty-sub   { font-size: 0.72rem; color: #9ca3af; line-height: 1.6; }

/* History */
.hist-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.75rem; }
.hist-head-title { font-size: 0.95rem; font-weight: 700; color: #111827; }
.hist-count { font-family: 'JetBrains Mono', monospace; font-size: 0.62rem; color: #6b7280; background: #f1f5f9; border-radius: 20px; padding: 0.18rem 0.6rem; border: 1px solid #e2e8f0; }
.stat-strip { display: flex; gap: 0.5rem; margin-bottom: 1rem; }
.stat-box { flex: 1; background: #fff; border: 1.5px solid #e2e8f0; border-radius: 11px; padding: 0.75rem 0.5rem; text-align: center; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.stat-v { font-size: 1.3rem; font-weight: 800; display: block; color: #111827; }
.stat-l { font-size: 0.55rem; font-weight: 700; letter-spacing: 0.09em; text-transform: uppercase; color: #6b7280; display: block; margin-top: 0.1rem; }
.hcard { background: #fff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 0.8rem; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.75rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04); }
.hcard-b3   { border-left: 3px solid #ef4444; }
.hcard-safe { border-left: 3px solid #22c55e; }
.hcard-img  { width: 54px; height: 54px; border-radius: 9px; object-fit: cover; flex-shrink: 0; background: #f0f2f8; }
.hcard-verdict     { font-size: 0.82rem; font-weight: 800; }
.hcard-verdict-b3  { color: #dc2626; }
.hcard-verdict-safe{ color: #16a34a; }
.hcard-meta { display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center; margin-top: 0.25rem; }
.hcard-conf      { font-family: 'JetBrains Mono', monospace; font-size: 0.65rem; font-weight: 500; }
.hcard-conf-b3   { color: #ef4444; }
.hcard-conf-safe { color: #22c55e; }
.hcard-time { font-size: 0.6rem; color: #9ca3af; }
.hcard-src  { font-size: 0.57rem; background: #f1f5f9; color: #6b7280; border-radius: 4px; padding: 0.1rem 0.35rem; border: 1px solid #e2e8f0; }
.hist-empty { text-align: center; padding: 2.5rem 1rem; background: #fafbff; border: 1.5px dashed #e2e8f0; border-radius: 12px; }
.hist-empty-icon { font-size: 2rem; margin-bottom: 0.4rem; }
.hist-empty-text { font-size: 0.78rem; color: #6b7280; line-height: 1.6; }
.hcard-actions { display: flex; gap: 0.4rem; flex-shrink: 0; flex-direction: column; }
.streamlit-expanderHeader p { color: #111827 !important; font-weight: 600 !important; }
.js-plotly-plot .plotly { background: transparent !important; }
[data-testid="stImage"] img { border-radius: 12px !important; }
p, span, div, li, label, h1, h2, h3, h4, h5, h6 { color: #111827; }
.stMarkdown p { color: #111827 !important; }
.stCaption { color: #6b7280 !important; }
code { background: #f1f5f9 !important; color: #4338ca !important; border-radius: 4px; }
.stDownloadButton button {
    border-radius: 7px !important; font-size: 0.7rem !important;
    font-weight: 600 !important; padding: 0.3rem 0.6rem !important;
    min-height: unset !important; height: auto !important;
    background: #f0fdf4 !important; color: #15803d !important;
    border: 1px solid #bbf7d0 !important; width: 100% !important;
}
.stDownloadButton button:hover { background: #dcfce7 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# INJECT JS — Hapus elemen Streamlit branding via DOM
# (CSS saja kadang tidak cukup karena class name berubah tiap versi)
# ══════════════════════════════════════════════════════════════
components.html("""
<script>
(function hideBranding() {
    function remove() {
        // Toolbar kanan atas
        const toolbar = document.querySelector('[data-testid="stToolbar"]');
        if (toolbar) toolbar.style.display = 'none';

        // Deploy button
        const deploy = document.querySelector('.stDeployButton');
        if (deploy) deploy.style.display = 'none';

        // Status widget
        const status = document.querySelector('[data-testid="stStatusWidget"]');
        if (status) status.style.display = 'none';

        // Decoration bar merah di atas
        const deco = document.getElementById('stDecoration');
        if (deco) deco.style.display = 'none';

        // Semua elemen yang punya class 'viewerBadge'
        document.querySelectorAll('[class*="viewerBadge"]').forEach(el => {
            el.style.display = 'none';
        });

        // Cari semua fixed-position element di pojok layar
        // (logo Streamlit & avatar GitHub ada di sini)
        document.querySelectorAll('body > div').forEach(el => {
            const style = window.getComputedStyle(el);
            if (style.position === 'fixed' || style.position === 'absolute') {
                const rect = el.getBoundingClientRect();
                const isBottomCorner = rect.bottom >= window.innerHeight - 80;
                const isCorner = (rect.left < 80 || rect.right > window.innerWidth - 80);
                if (isBottomCorner && isCorner) {
                    // Pastikan bukan elemen milik app kita
                    if (!el.id || !el.id.startsWith('lp-')) {
                        el.style.display = 'none';
                    }
                }
            }
        });

        // Footer Streamlit
        const footer = document.querySelector('footer');
        if (footer) footer.style.display = 'none';

        // Header Streamlit  
        const header = document.querySelector('header');
        if (header) header.style.display = 'none';

        // MainMenu
        const mainMenu = document.getElementById('MainMenu');
        if (mainMenu) mainMenu.style.display = 'none';
    }

    // Jalankan sekarang
    remove();

    // Jalankan lagi setelah Streamlit selesai render
    setTimeout(remove, 500);
    setTimeout(remove, 1500);
    setTimeout(remove, 3000);

    // Observer untuk menangkap elemen yang muncul belakangan
    const observer = new MutationObserver(function() {
        remove();
    });
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
})();
</script>
""", height=0)


# ══════════════════════════════════════════════════════════════
# HELPER — Generate result card image untuk download
# ══════════════════════════════════════════════════════════════
def make_result_card(orig_img: Image.Image, overlaid_img: Image.Image,
                     is_b3: bool, confidence: float, timestamp: str,
                     yolo_boxes: list) -> bytes:
    CARD_W, CARD_H = 900, 420
    IMG_W           = 480
    PANEL_W         = CARD_W - IMG_W
    BG_COLOR        = (255, 255, 255)
    ACCENT          = (220, 38, 38) if is_b3 else (22, 163, 74)
    ACCENT_LIGHT    = (255, 241, 240) if is_b3 else (240, 253, 244)
    TEXT_DARK       = (17, 24, 39)
    TEXT_MID        = (107, 114, 128)

    card = Image.new("RGB", (CARD_W, CARD_H), BG_COLOR)
    draw = ImageDraw.Draw(card)

    img_resized = overlaid_img.copy()
    img_resized.thumbnail((IMG_W, CARD_H), Image.LANCZOS)
    iw, ih = img_resized.size
    bg_img = Image.new("RGB", (IMG_W, CARD_H), (240, 242, 248))
    bg_img.paste(img_resized, ((IMG_W - iw)//2, (CARD_H - ih)//2))
    card.paste(bg_img, (0, 0))

    px = IMG_W + 1
    draw.rectangle([px, 0, CARD_W, CARD_H], fill=BG_COLOR)
    draw.rectangle([px, 0, CARD_W, 5], fill=ACCENT)

    def font(size):
        for name in ["arialbd.ttf","arial.ttf","DejaVuSans-Bold.ttf","DejaVuSans.ttf"]:
            try: return ImageFont.truetype(name, size)
            except: pass
        return ImageFont.load_default()

    f_sm = font(13); f_xs = font(11)

    draw.text((px + 20, 20), "B3 Waste Detector", font=font(13), fill=TEXT_MID)
    verdict_txt = "BERBAHAYA" if is_b3 else "AMAN"
    icon_txt    = "☣" if is_b3 else "✓"
    draw.rectangle([px+20, 50, CARD_W-20, 130], fill=ACCENT_LIGHT, outline=ACCENT, width=2)
    draw.text((px + 30, 62), icon_txt, font=font(36), fill=ACCENT)
    label = "B3" if is_b3 else "non-B3"
    draw.text((px + 80, 62), label, font=font(34), fill=ACCENT)
    draw.text((px + 80, 102), verdict_txt, font=font(14), fill=ACCENT)

    conf_pct = f"{confidence:.1%}"
    draw.text((px + 20, 148), "Keyakinan Model", font=f_xs, fill=TEXT_MID)
    draw.text((px + 20, 164), conf_pct, font=font(28), fill=TEXT_DARK)

    bar_x1, bar_y = px+20, 200
    bar_w = PANEL_W - 40
    draw.rectangle([bar_x1, bar_y, bar_x1+bar_w, bar_y+8], fill=(230, 232, 240))
    fill_w = int(bar_w * confidence)
    if fill_w > 0:
        draw.rectangle([bar_x1, bar_y, bar_x1+fill_w, bar_y+8], fill=ACCENT)

    draw.line([(px+20, 222), (CARD_W-20, 222)], fill=(230, 232, 240), width=1)
    draw.text((px+20, 232), "Objek Terdeteksi", font=f_xs, fill=TEXT_MID)
    if yolo_boxes:
        obj_names = list(dict.fromkeys([b[5] for b in yolo_boxes if len(b)>5]))[:4]
        for i, name in enumerate(obj_names):
            draw.text((px+20, 250 + i*18), f"• {name}", font=f_sm, fill=TEXT_DARK)
    else:
        draw.text((px+20, 250), "—", font=f_sm, fill=TEXT_MID)

    draw.line([(px+20, 322), (CARD_W-20, 322)], fill=(230, 232, 240), width=1)
    draw.text((px+20, 332), timestamp, font=f_xs, fill=TEXT_MID)
    draw.text((CARD_W - 160, CARD_H - 22), "AI Classification Result", font=f_xs, fill=(200, 205, 215))
    draw.rectangle([0, 0, CARD_W-1, CARD_H-1], outline=(220, 224, 235), width=1)

    buf = io.BytesIO()
    card.save(buf, format="PNG", dpi=(150, 150))
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════
# localStorage helpers
# ══════════════════════════════════════════════════════════════
def save_to_localstorage(entry_json: str):
    escaped = entry_json.replace("\\","\\\\").replace("`","\\`").replace("$","\\$")
    components.html(f"""<script>
    (function(){{
        const K='{STORAGE_KEY}';
        let h=[];
        try{{h=JSON.parse(localStorage.getItem(K)||'[]');}}catch(e){{}}
        h.unshift(JSON.parse(`{escaped}`));
        if(h.length>100)h=h.slice(0,100);
        localStorage.setItem(K,JSON.stringify(h));
    }})();
    </script>""", height=0)

def delete_from_localstorage(entry_id: str):
    components.html(f"""<script>
    (function(){{
        const K='{STORAGE_KEY}';
        let h=[];
        try{{h=JSON.parse(localStorage.getItem(K)||'[]');}}catch(e){{}}
        h=h.filter(x=>x.id!=='{entry_id}');
        localStorage.setItem(K,JSON.stringify(h));
    }})();
    </script>""", height=0)

def clear_localstorage():
    components.html(f"""<script>
    localStorage.removeItem('{STORAGE_KEY}');
    </script>""", height=0)

def pil_to_thumb_b64(img: Image.Image, size: int = 110) -> str:
    thumb = img.copy()
    thumb.thumbnail((size, size), Image.LANCZOS)
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=75)
    return base64.b64encode(buf.getvalue()).decode()

def build_entry(img, is_b3, pred_proba, yolo_boxes, source):
    conf = round((1-pred_proba) if is_b3 else pred_proba, 4)
    objs = [b[5] for b in yolo_boxes if len(b)>5] if yolo_boxes else []
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

if "history"       not in st.session_state: st.session_state.history       = []
if "overlaid_imgs" not in st.session_state: st.session_state.overlaid_imgs = {}
if "orig_imgs"     not in st.session_state: st.session_state.orig_imgs     = {}
if "last_saved_id" not in st.session_state: st.session_state.last_saved_id = None
if "delete_id"     not in st.session_state: st.session_state.delete_id     = None


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
            if "batch_shape" in kw: kw["input_shape"] = kw.pop("batch_shape")[1:]
            super().__init__(**kw)
    class CD(tf.keras.layers.Dense):
        def __init__(self, **kw): super().__init__(**_s(kw))
    class CC(tf.keras.layers.Conv2D):
        def __init__(self, **kw): super().__init__(**_s(kw))
    class CDC(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, **kw): super().__init__(**_s(kw))
    return tf.keras.models.load_model(
        os.path.join(BASE_DIR,"model_b3_final.h5"),
        custom_objects={"BatchNormalization":CBN,"InputLayer":CIL,
                        "Dense":CD,"Conv2D":CC,"DepthwiseConv2D":CDC},
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
    except Exception: return None

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
    except Exception: return []

def draw_overlay(pil_img, yolo_boxes, is_b3, pred_proba, yolo_ok):
    W,H    = pil_img.size
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
        bb=draw.textbbox((0,0),label,font=font)
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

    gray=np.array(pil_img.convert("L"),dtype=np.float32)
    rows,cols=7,7; ph2,pw2=H//rows,W//cols
    heat=np.array([[np.var(gray[r*ph2:(r+1)*ph2,c*pw2:(c+1)*pw2])
                    for c in range(cols)] for r in range(rows)])
    if heat.max()>0: heat/=heat.max()
    mask=heat>0.4
    if not mask.any(): mask=heat==heat.max()
    yr,xc=np.where(mask)
    x1=max(0,xc.min()*pw2-10); y1=max(0,yr.min()*ph2-10)
    x2=min(W,(xc.max()+1)*pw2+10); y2=min(H,(yr.max()+1)*ph2+10)
    draw.rectangle([x1,y1,x2,y2], outline=color, width=lw)
    corners(x1,y1,x2,y2)
    pill(x1,y1,verdict)
    return result, "fallback"


# ══════════════════════════════════════════════════════════════
# PENJELASAN B3 / non-B3
# ══════════════════════════════════════════════════════════════
B3_ITEMS = [
    ("🔋","Baterai","Batu baterai, baterai HP, baterai laptop"),
    ("🧴","Pembersih kimia keras","Cairan pel, pembersih toilet, pemutih pakaian"),
    ("🧼","Deterjen berbahan kimia kuat","Sabun cuci tertentu dengan label peringatan"),
    ("💊","Obat-obatan kadaluarsa","Obat, suplemen, sirup yang sudah expired"),
    ("🛢️","Oli & cairan mesin","Oli mesin bekas, bensin, solar, cairan rem"),
    ("🖨️","Elektronik rusak","HP, printer, kabel, lampu neon, baterai lithium"),
    ("🎨","Cat, pylox & thinner","Cat tembok, pilox, lem tembak, cairan pelarut"),
    ("🌿","Pestisida & insektisida","Obat nyamuk semprot, pembasmi hama, pupuk kimia"),
]
NON_B3_ITEMS = [
    ("📄","Kertas & kardus","Koran, majalah, dus bekas, kotak makanan"),
    ("🧴","Plastik biasa","Botol air, kantong belanja, bungkus makanan"),
    ("🥫","Kaleng & logam biasa","Kaleng susu kosong, besi tua, aluminium"),
    ("👕","Pakaian & kain","Baju bekas, handuk lama, kain perca"),
    ("🥬","Sisa makanan & organik","Kulit buah, sayuran, nasi basi, tulang ayam"),
    ("👟","Sepatu & tas","Sepatu bekas, tas rusak, sandal"),
    ("📦","Gabus & styrofoam","Gabus bekas elektronik, wadah mie instan"),
    ("🌾","Sampah dapur lainnya","Ampas kopi, teh celup, cangkang telur"),
]

def render_b3_explanation():
    st.markdown("""
    <div class="explain-box explain-box-b3">
        <span class="explain-title">⚠️ Ini Sampah B3 — Wajib Penanganan Khusus!</span>
        Sampah B3 mengandung <b>bahan berbahaya dan beracun</b> yang bisa mencemari lingkungan
        dan membahayakan kesehatan. <b>Jangan dibuang ke tempat sampah biasa.</b>
    </div>
    """, unsafe_allow_html=True)
    with st.expander("📋 Contoh jenis sampah B3 lainnya"):
        for emoji,nama,contoh in B3_ITEMS:
            st.markdown(f"**{emoji} {nama}** — {contoh}")
    st.markdown("""
    <div class="tip tip-b3">
        <b>🗑️ Cara membuang sampah B3 yang benar:</b><br>
        • Simpan di wadah tertutup rapat, pisahkan dari sampah lain<br>
        • Bawa ke <b>bank sampah B3</b> atau <b>drop box limbah B3</b> terdekat<br>
        • Bisa dititipkan ke toko elektronik, bengkel, atau apotek resmi<br>
        • <b>Jangan dibakar</b> — asapnya sangat berbahaya bagi pernapasan
    </div>""", unsafe_allow_html=True)

def render_safe_explanation():
    st.markdown("""
    <div class="explain-box explain-box-safe">
        <span class="explain-title">✅ Sampah Non-B3 — Aman Didaur Ulang!</span>
        Sampah ini <b>tidak berbahaya</b> dan bisa dibuang ke tempat sampah biasa.
        Lebih baik lagi jika <b>dipilah dan didaur ulang</b> agar bermanfaat kembali.
    </div>""", unsafe_allow_html=True)
    with st.expander("♻️ Contoh sampah non-B3 yang bisa didaur ulang"):
        for emoji,nama,contoh in NON_B3_ITEMS:
            st.markdown(f"**{emoji} {nama}** — {contoh}")
    st.markdown("""
    <div class="tip tip-safe">
        <b>💡 Tips memilah sampah non-B3:</b><br>
        • Pisahkan <b>organik</b> (sisa makanan) dan <b>anorganik</b> (plastik, kertas, logam)<br>
        • Cuci wadah plastik atau kaca sebelum dibuang ke tempat daur ulang<br>
        • Manfaatkan <b>bank sampah</b> di sekitar rumah atau kelurahan<br>
        • Sampah organik bisa dijadikan <b>kompos</b> untuk menyuburkan tanaman
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# INIT MODELS
# ══════════════════════════════════════════════════════════════
with st.spinner("⚙️ Memuat model AI..."):
    try: model = load_classifier()
    except Exception as e:
        st.error(f"❌ Gagal memuat classifier: {e}"); st.stop()

with st.spinner("⚙️ Memuat YOLOv8..."):
    yolo_model = load_yolo()
    yolo_ok    = yolo_model is not None

THRESHOLD = 0.50

if st.session_state.delete_id:
    del_id = st.session_state.delete_id
    st.session_state.history = [h for h in st.session_state.history if h["id"] != del_id]
    st.session_state.overlaid_imgs.pop(del_id, None)
    st.session_state.orig_imgs.pop(del_id, None)
    delete_from_localstorage(del_id)
    st.session_state.delete_id = None

# ══════════════════════════════════════════════════════════════
# TOPBAR
# ══════════════════════════════════════════════════════════════
yolo_label = "best.pt (fine-tuned)" if os.path.exists(os.path.join(BASE_DIR,"best.pt")) else "yolov8n.pt (COCO)"
st.markdown(f"""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-icon">♻️</div>
        <div>
            <div class="topbar-title">B3 Waste Detector</div>
            <div class="topbar-sub">Klasifikasi sampah berbahaya berbasis AI</div>
        </div>
    </div>
    <div class="topbar-badge">MobileNetV2 · YOLOv8</div>
</div>
""", unsafe_allow_html=True)

if yolo_ok:
    st.markdown(f'<div class="status-bar status-ok">✅ &nbsp;YOLOv8 aktif — <b>{yolo_label}</b></div>',
                unsafe_allow_html=True)
else:
    st.markdown('<div class="status-bar status-warn">⚠️ &nbsp;YOLOv8 tidak tersedia — install: <code>pip install ultralytics</code></div>',
                unsafe_allow_html=True)

with st.expander("ℹ️ Apa itu Sampah B3? Pelajari lebih lanjut"):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**🔴 Sampah B3 (Berbahaya & Beracun)**")
        st.markdown("Mengandung bahan kimia berbahaya. Tidak boleh dibuang sembarangan.")
        for emoji,nama,_ in B3_ITEMS[:4]:
            st.markdown(f"{emoji} {nama}")
    with col_b:
        st.markdown("**🟢 Sampah Non-B3 (Aman)**")
        st.markdown("Sampah umum yang tidak berbahaya dan bisa didaur ulang.")
        for emoji,nama,_ in NON_B3_ITEMS[:4]:
            st.markdown(f"{emoji} {nama}")


# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab_detect, tab_history = st.tabs(["🔍  Deteksi Sampah", "🕘  Riwayat Klasifikasi"])


# ════════════════════════════════
# TAB 1 — DETEKSI
# ════════════════════════════════
with tab_detect:
    pred_proba=None; is_b3=None; img_source=None; input_source="upload"
    col_left, col_right = st.columns([1.1, 0.9], gap="large")

    with col_left:
        tab_upload, tab_camera = st.tabs(["📁  Upload Gambar","📷  Kamera Langsung"])

        with tab_upload:
            st.markdown("""
            <div class="upload-zone">
                <span class="upload-icon">🗂️</span>
                <div class="upload-title">Seret & lepas gambar sampah di sini</div>
                <div class="upload-sub">atau gunakan tombol Browse di bawah</div>
                <div class="fmt-row">
                    <span class="fmt-badge">JPG</span>
                    <span class="fmt-badge">PNG</span>
                    <span class="fmt-badge">WEBP</span>
                    <span class="fmt-badge">≤ 200MB</span>
                </div>
            </div>""", unsafe_allow_html=True)
            uploaded = st.file_uploader("upload", type=["jpg","jpeg","png","webp"],
                                         label_visibility="collapsed")
            if uploaded:
                img_source   = Image.open(uploaded).convert("RGB")
                input_source = "upload"

        with tab_camera:
            st.markdown('<div class="cam-hint">📸 &nbsp;Arahkan kamera ke sampah lalu tekan <b>Take photo</b></div>',
                        unsafe_allow_html=True)
            snap = st.camera_input("foto", label_visibility="collapsed")
            if snap:
                img_source   = Image.open(snap).convert("RGB")
                input_source = "kamera"

        if img_source is not None:
            st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

            img_224    = img_source.resize((224,224))
            arr        = np.expand_dims(preprocess_input(
                             np.array(img_224,dtype=np.float32)), axis=0)
            pred_proba = float(model.predict(arr,verbose=0)[0][0])
            is_b3      = pred_proba < THRESHOLD

            with st.spinner("🔍 Mendeteksi objek..."):
                yolo_boxes = run_yolo(yolo_model, img_source)

            overlaid, mode = draw_overlay(img_source, yolo_boxes, is_b3, pred_proba, yolo_ok)
            mode_label = "🎯 YOLOv8 Detection" if mode=="yolo" else "🔍 Saliency Map (fallback)"
            st.markdown(f'<div class="det-label">{mode_label}</div>', unsafe_allow_html=True)
            st.image(overlaid, use_container_width=True)

            warna = "red" if is_b3 else "green"
            label_txt = "B3 — BERBAHAYA" if is_b3 else "non-B3 — AMAN"
            if mode=="yolo":
                st.caption(f"YOLOv8 mendeteksi **{len(yolo_boxes)} objek**. "
                           f"Klasifikasi: **:{warna}[{label_txt}]**")
            else:
                st.caption(f"Klasifikasi MobileNetV2: **:{warna}[{label_txt}]**")

            conf_val = round((1-pred_proba) if is_b3 else pred_proba, 4)
            ts_now   = datetime.now().strftime("%d %b %Y, %H:%M")
            card_img = make_result_card(img_source, overlaid, is_b3,
                                         conf_val, ts_now, yolo_boxes)
            fname = f"hasil_{'B3' if is_b3 else 'nonB3'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            st.download_button(
                label="⬇️ Download Hasil Klasifikasi (PNG)",
                data=card_img,
                file_name=fname,
                mime="image/png",
                use_container_width=True,
            )

            entry = build_entry(img_source, is_b3, pred_proba, yolo_boxes, input_source)
            if st.session_state.last_saved_id != entry["id"]:
                st.session_state.history.insert(0, entry)
                st.session_state.overlaid_imgs[entry["id"]] = overlaid
                st.session_state.orig_imgs[entry["id"]]     = img_source
                if len(st.session_state.history) > 100:
                    oldest_id = st.session_state.history[-1]["id"]
                    st.session_state.history = st.session_state.history[:100]
                    st.session_state.overlaid_imgs.pop(oldest_id, None)
                    st.session_state.orig_imgs.pop(oldest_id, None)
                st.session_state.last_saved_id = entry["id"]
                save_to_localstorage(json.dumps(entry, ensure_ascii=False))

    with col_right:
        if img_source is None or pred_proba is None:
            st.markdown("""
            <div class="empty-state">
                <span class="empty-icon">🔬</span>
                <div class="empty-title">Belum ada gambar</div>
                <div class="empty-sub">Upload foto atau gunakan kamera<br>untuk memulai deteksi</div>
            </div>""", unsafe_allow_html=True)
        else:
            import plotly.graph_objects as go
            conf_b3   = 1.0-pred_proba; conf_safe = pred_proba
            disp_conf = conf_b3 if is_b3 else conf_safe
            color_main= "#dc2626" if is_b3 else "#16a34a"

            if is_b3:
                st.markdown("""
                <div class="verdict-card verdict-b3">
                    <span class="v-emoji">☣️</span>
                    <div class="v-label v-label-b3">B3</div>
                    <div class="v-status v-status-b3">⚠ BAHAN BERBAHAYA & BERACUN</div>
                    <div class="v-desc">Sampah ini terdeteksi mengandung bahan berbahaya.<br>
                    Diperlukan penanganan &amp; pembuangan khusus.</div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="verdict-card verdict-safe">
                    <span class="v-emoji">✅</span>
                    <div class="v-label v-label-safe">non-B3</div>
                    <div class="v-status v-status-safe">✓ AMAN — TIDAK BERBAHAYA</div>
                    <div class="v-desc">Sampah ini tidak terdeteksi berbahaya.<br>
                    Dapat diproses melalui daur ulang biasa.</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="pills">
                <div class="pill">
                    <span class="pill-val" style="color:{color_main}">{disp_conf:.1%}</span>
                    <span class="pill-lbl">{'Conf. B3' if is_b3 else 'Conf. Aman'}</span>
                </div>
                <div class="pill">
                    <span class="pill-val" style="color:#6366f1">{pred_proba:.3f}</span>
                    <span class="pill-lbl">Sigmoid</span>
                </div>
                <div class="pill">
                    <span class="pill-val" style="color:#374151">{THRESHOLD:.2f}</span>
                    <span class="pill-lbl">Threshold</span>
                </div>
            </div>""", unsafe_allow_html=True)

            gauge_val=conf_b3*100; gauge_color="#ef4444" if is_b3 else "#22c55e"
            fig_g=go.Figure(go.Indicator(
                mode="gauge+number", value=gauge_val,
                number={"suffix":"%","font":{"size":26,"color":gauge_color,"family":"Plus Jakarta Sans"}},
                gauge={"axis":{"range":[0,100],"showticklabels":False,"tickwidth":0},
                       "bar":{"color":gauge_color,"thickness":0.26},"bgcolor":"#f1f5f9","borderwidth":0,
                       "steps":[{"range":[0,40],"color":"#dcfce7"},{"range":[40,60],"color":"#fef9c3"},
                                 {"range":[60,100],"color":"#fee2e2"}],
                       "threshold":{"line":{"color":gauge_color,"width":3},"thickness":0.85,"value":gauge_val}},
                title={"text":"Risiko B3","font":{"size":11,"color":"#6b7280","family":"Plus Jakarta Sans"}},
                domain={"x":[0,1],"y":[0,1]}
            ))
            fig_g.update_layout(height=190,margin=dict(t=36,b=8,l=16,r=16),
                                paper_bgcolor="white",plot_bgcolor="white",font_color="#111827")
            st.plotly_chart(fig_g,use_container_width=True,config={"displayModeBar":False})

            fig_b=go.Figure()
            fig_b.add_trace(go.Bar(x=[conf_b3*100],y=["B3"],orientation="h",
                marker_color="#ef4444",marker_line_width=0,width=0.42,
                text=[f"{conf_b3:.1%}"],textposition="inside",
                insidetextfont=dict(color="white",size=11)))
            fig_b.add_trace(go.Bar(x=[conf_safe*100],y=["non-B3"],orientation="h",
                marker_color="#22c55e",marker_line_width=0,width=0.42,
                text=[f"{conf_safe:.1%}"],textposition="inside",
                insidetextfont=dict(color="white",size=11)))
            fig_b.update_layout(height=108,margin=dict(t=4,b=4,l=58,r=14),
                paper_bgcolor="white",plot_bgcolor="white",showlegend=False,
                xaxis=dict(range=[0,100],showticklabels=False,showgrid=False,zeroline=False),
                yaxis=dict(showgrid=False,tickfont=dict(size=11,color="#374151")),
                bargap=0.3,font_color="#111827")
            st.plotly_chart(fig_b,use_container_width=True,config={"displayModeBar":False})

            if yolo_ok and yolo_boxes:
                st.markdown('<div class="det-label">📦 Objek Terdeteksi YOLO</div>',unsafe_allow_html=True)
                for i,(x1,y1,x2,y2,conf,name) in enumerate(yolo_boxes):
                    st.markdown(f"""<div class="obj-row">
                        <span class="obj-name">#{i+1} &nbsp;{name}</span>
                        <span class="obj-conf">{conf:.0%}</span>
                        <span class="obj-size">{x2-x1}×{y2-y1}px</span>
                    </div>""", unsafe_allow_html=True)

            if is_b3: render_b3_explanation()
            else:     render_safe_explanation()


# ════════════════════════════════
# TAB 2 — RIWAYAT
# ════════════════════════════════
with tab_history:
    history = st.session_state.history
    n_total = len(history)
    n_b3    = sum(1 for h in history if h["is_b3"])
    n_safe  = n_total - n_b3

    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.markdown(f"""
        <div class="hist-head">
            <span class="hist-head-title">📋 Riwayat Klasifikasi</span>
            <span class="hist-count">{n_total} entri sesi ini</span>
        </div>""", unsafe_allow_html=True)
    with hcol2:
        if n_total > 0:
            if st.button("🗑️ Hapus Semua Sesi", use_container_width=True, type="secondary"):
                st.session_state.history       = []
                st.session_state.overlaid_imgs = {}
                st.session_state.orig_imgs     = {}
                st.session_state.last_saved_id = None
                clear_localstorage()
                st.rerun()

    st.info("💾 **Riwayat tersimpan di browser kamu** — tetap ada meski halaman di-refresh, "
            "tapi tidak terlihat di perangkat lain.", icon="ℹ️")

    components.html(f"""
    <div id="ls-root" style="font-family:'Plus Jakarta Sans',sans-serif;"></div>
    <script>
    (function(){{
        const KEY='{STORAGE_KEY}';
        let h=[];
        try{{h=JSON.parse(localStorage.getItem(KEY)||'[]');}}catch(e){{}}
        const root=document.getElementById('ls-root');
        if(!h.length){{ root.innerHTML=''; return; }}
        const nb3=h.filter(x=>x.is_b3).length, ns=h.length-nb3;
        root.innerHTML=`
        <div style="background:#fff;border:1.5px solid #e2e8f0;border-radius:14px;
                    padding:1rem;margin-bottom:1rem;box-shadow:0 1px 4px rgba(0,0,0,0.04)">
            <p style="font-size:0.64rem;font-weight:700;letter-spacing:0.12em;
                      text-transform:uppercase;color:#6b7280;margin:0 0 0.7rem">
                📱 Tersimpan di Perangkat Ini (localStorage)</p>
            <div style="display:flex;gap:0.5rem;margin-bottom:1rem">
                ${{[
                    [h.length,'Total','#111827','#f8fafc','#e2e8f0'],
                    [nb3,'B3','#dc2626','#fff1f1','#fca5a5'],
                    [ns,'Non-B3','#16a34a','#f0fdf4','#86efac']
                ].map(([v,l,c,bg,bd])=>`
                    <div style="flex:1;background:${{bg}};border:1.5px solid ${{bd}};
                                border-radius:11px;padding:0.65rem 0.5rem;text-align:center">
                        <span style="font-size:1.3rem;font-weight:800;color:${{c}};display:block">${{v}}</span>
                        <span style="font-size:0.55rem;font-weight:700;letter-spacing:0.09em;
                                     text-transform:uppercase;color:#6b7280">${{l}}</span>
                    </div>`).join('')}}
            </div>
            ${{h.slice(0,30).map(e=>{{
                const bdr=e.is_b3?'#ef4444':'#22c55e';
                const col=e.is_b3?'#dc2626':'#16a34a';
                const ico=e.is_b3?'☣️':'✅';
                const vrd=e.is_b3?'B3 — BERBAHAYA':'non-B3 — AMAN';
                const pct=((e.confidence||0)*100).toFixed(1);
                const src=e.source==='kamera'?'📷':'📁';
                const img=e.thumb_b64
                    ?`<img src="data:image/jpeg;base64,${{e.thumb_b64}}"
                           style="width:48px;height:48px;border-radius:8px;object-fit:cover;flex-shrink:0"/>`
                    :`<div style="width:48px;height:48px;border-radius:8px;background:#f0f2f8;flex-shrink:0"></div>`;
                return `
                <div style="background:#fff;border:1px solid #e2e8f0;border-left:3px solid ${{bdr}};
                             border-radius:11px;padding:0.65rem;margin-bottom:0.45rem;
                             display:flex;align-items:center;gap:0.65rem">
                    ${{img}}
                    <div style="flex:1;min-width:0">
                        <div style="font-size:0.8rem;font-weight:800;color:${{col}}">${{ico}} ${{vrd}}</div>
                        <div style="display:flex;gap:0.35rem;align-items:center;margin-top:0.2rem;flex-wrap:wrap">
                            <span style="font-size:0.65rem;font-weight:600;color:${{col}};font-family:monospace">${{pct}}%</span>
                            <span style="font-size:0.6rem;background:#f1f5f9;color:#6b7280;
                                         border-radius:4px;padding:0.1rem 0.3rem;border:1px solid #e2e8f0">${{src}} ${{e.source||''}}</span>
                            <span style="font-size:0.6rem;color:#9ca3af">${{e.timestamp||''}}</span>
                        </div>
                    </div>
                </div>`;}}).join('')}}
            ${{h.length>30?`<p style="text-align:center;font-size:0.72rem;color:#6b7280;margin:0.5rem 0">... dan ${{h.length-30}} entri lainnya</p>`:''}}
            <button onclick="if(confirm('Hapus semua riwayat di perangkat ini?')){{localStorage.removeItem('{STORAGE_KEY}');location.reload();}}"
                style="width:100%;margin-top:0.75rem;padding:0.6rem;border-radius:9px;
                       border:1.5px solid #e2e8f0;background:#f8fafc;color:#374151;
                       font-family:'Plus Jakarta Sans',sans-serif;font-size:0.78rem;
                       font-weight:600;cursor:pointer">
                🗑️ Hapus Semua Riwayat di Perangkat Ini
            </button>
        </div>`;
    }})();
    </script>
    """, height=620, scrolling=True)

    st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)

    if n_total > 0:
        st.markdown(f"""
        <p style="font-size:0.64rem;font-weight:700;letter-spacing:0.12em;
                  text-transform:uppercase;color:#6b7280;margin-bottom:0.6rem">
            Sesi Sekarang — {n_total} Deteksi</p>
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
        </div>""", unsafe_allow_html=True)

        for entry in history:
            b3     = entry["is_b3"]
            eid    = entry["id"]
            card_c = "hcard-b3" if b3 else "hcard-safe"
            v_c    = "hcard-verdict-b3" if b3 else "hcard-verdict-safe"
            c_c    = "hcard-conf-b3" if b3 else "hcard-conf-safe"
            icon   = "☣️" if b3 else "✅"
            verd   = "B3 — BERBAHAYA" if b3 else "non-B3 — AMAN"
            src_i  = "📷" if entry.get("source") == "kamera" else "📁"
            thumb  = entry.get("thumb_b64","")
            t_html = (f'<img class="hcard-img" src="data:image/jpeg;base64,{thumb}"/>'
                      if thumb else '<div class="hcard-img"></div>')

            st.markdown(f"""
            <div class="hcard {card_c}">
                {t_html}
                <div style="flex:1;min-width:0">
                    <div class="hcard-verdict {v_c}">{icon} {verd}</div>
                    <div class="hcard-meta">
                        <span class="hcard-conf {c_c}">{entry['confidence']:.1%}</span>
                        <span class="hcard-src">{src_i} {entry.get('source','—')}</span>
                        <span class="hcard-time">{entry.get('timestamp','')}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

            btn_a, btn_b = st.columns(2)
            with btn_a:
                overlaid_img = st.session_state.overlaid_imgs.get(eid)
                orig_img     = st.session_state.orig_imgs.get(eid)
                if overlaid_img is not None and orig_img is not None:
                    card_bytes = make_result_card(
                        orig_img, overlaid_img, b3,
                        entry["confidence"], entry.get("timestamp",""),
                        [(0,0,0,0,0,o) for o in entry.get("objects",[])]
                    )
                    dl_fname = (f"hasil_{'B3' if b3 else 'nonB3'}_"
                                f"{entry.get('timestamp','').replace(' ','_').replace(',','')}.png")
                    st.download_button(
                        label="⬇️ Download Gambar",
                        data=card_bytes, file_name=dl_fname,
                        mime="image/png", key=f"dl_{eid}",
                        use_container_width=True,
                    )
                elif thumb:
                    img_data  = base64.b64decode(thumb)
                    thumb_img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    card_bytes = make_result_card(
                        thumb_img, thumb_img, b3,
                        entry["confidence"], entry.get("timestamp",""),
                        [(0,0,0,0,0,o) for o in entry.get("objects",[])]
                    )
                    st.download_button(
                        label="⬇️ Download Gambar",
                        data=card_bytes, file_name=f"hasil_{eid}.png",
                        mime="image/png", key=f"dl_{eid}",
                        use_container_width=True,
                    )
            with btn_b:
                if st.button("🗑️ Hapus", key=f"del_{eid}", use_container_width=True):
                    st.session_state.delete_id = eid
                    st.rerun()
    else:
        st.markdown("""
        <div class="hist-empty">
            <div class="hist-empty-icon">📂</div>
            <div class="hist-empty-text">Belum ada deteksi di sesi ini.<br>
            Mulai dari tab <b>🔍 Deteksi Sampah</b>.</div>
        </div>""", unsafe_allow_html=True)

components.html("""
<script>
function hideStreamlitBranding() {

    // Semua kemungkinan selector Streamlit
    const selectors = [
        '#MainMenu',
        'header',
        'footer',
        '[data-testid="stToolbar"]',
        '[data-testid="stDecoration"]',
        '[data-testid="stStatusWidget"]',
        '.stDeployButton',
        '.stAppDeployButton',
        '[class*="viewerBadge"]',
        '[href*="streamlit.io"]',
        '[href*="github.com"]'
    ];

    selectors.forEach(selector => {
        document.querySelectorAll(selector).forEach(el => {
            el.style.display = 'none';
            el.style.visibility = 'hidden';
            el.style.opacity = '0';
            el.remove();
        });
    });

    // Hapus fixed floating element pojok
    document.querySelectorAll('div').forEach(el => {
        const style = window.getComputedStyle(el);

        if (
            style.position === 'fixed' &&
            (
                el.innerText.includes('Streamlit') ||
                el.innerHTML.includes('streamlit') ||
                el.innerHTML.includes('github')
            )
        ) {
            el.remove();
        }
    });
}

// Jalankan berkali-kali
hideStreamlitBranding();

setInterval(hideStreamlitBranding, 1000);

</script>
""", height=0)

st.markdown("""
<div style="text-align:center;margin-top:2rem;font-size:0.64rem;color:#9ca3af;
            font-family:'JetBrains Mono',monospace;letter-spacing:0.06em">
    B3 Waste Detector &nbsp;·&nbsp; MobileNetV2 + YOLOv8 &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Streamlit
</div>""", unsafe_allow_html=True)