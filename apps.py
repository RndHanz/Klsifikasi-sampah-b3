import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import os
import plotly.graph_objects as go

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

st.set_page_config(
    page_title="B3 Waste Detector",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #f0f2f8 !important;
}
.block-container { padding: 0 2rem 3rem !important; max-width: 1200px; }
#MainMenu, footer, header { visibility: hidden; }

.topbar {
    background: linear-gradient(135deg, #1a1f3a 0%, #1e3a8a 60%, #1d4ed8 100%);
    margin: 0 -2rem 2rem; padding: 1.2rem 2.5rem;
    display: flex; align-items: center; justify-content: space-between;
}
.topbar-left  { display: flex; align-items: center; gap: 0.9rem; }
.topbar-icon  {
    width: 44px; height: 44px; border-radius: 12px;
    background: rgba(255,255,255,0.15);
    display: flex; align-items: center; justify-content: center; font-size: 1.4rem;
    border: 1px solid rgba(255,255,255,0.2);
}
.topbar-title { font-size: 1.2rem; font-weight: 800; color: #fff; letter-spacing: -0.02em; }
.topbar-sub   { font-size: 0.68rem; color: rgba(255,255,255,0.6); margin-top: 0.1rem; }
.topbar-badge {
    background: rgba(255,255,255,0.12); color: rgba(255,255,255,0.9);
    font-size: 0.62rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase;
    padding: 0.35rem 0.9rem; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2);
}

.upload-zone {
    background: linear-gradient(135deg, #fafbff, #f0f4ff);
    border: 2.5px dashed #c7d2fe; border-radius: 16px;
    padding: 2rem 1.5rem 1.2rem; text-align: center; margin-bottom: 0.75rem;
}
.upload-icon  { font-size: 2.8rem; display: block; margin-bottom: 0.5rem; }
.upload-title { font-size: 0.92rem; font-weight: 700; color: #1e293b; margin-bottom: 0.2rem; }
.upload-sub   { font-size: 0.72rem; color: #94a3b8; margin-bottom: 0.9rem; }
.fmt-row      { display: flex; justify-content: center; gap: 0.45rem; margin-bottom: 1rem; }
.fmt-badge {
    background: #eff6ff; color: #3b82f6; font-size: 0.6rem; font-weight: 700;
    padding: 0.22rem 0.6rem; border-radius: 6px; letter-spacing: 0.05em;
    text-transform: uppercase; border: 1px solid #bfdbfe;
}

.cam-hint {
    background: #f0f4ff; border: 1px solid #c7d2fe; border-radius: 10px;
    padding: 0.65rem 0.9rem; font-size: 0.73rem; color: #4338ca;
    margin-bottom: 0.8rem;
}

.verdict-card { border-radius: 14px; padding: 1.5rem 1.4rem 1.2rem; text-align: center; margin-bottom: 1.2rem; }
.verdict-b3   { background: linear-gradient(135deg,#fff1f1,#ffe4e4); border: 2px solid #fca5a5; }
.verdict-safe { background: linear-gradient(135deg,#f0fdf4,#dcfce7); border: 2px solid #86efac; }
.v-emoji  { font-size: 2.6rem; display: block; margin-bottom: 0.4rem; }
.v-label  { font-size: 2rem; font-weight: 900; letter-spacing: -0.04em; line-height: 1; }
.v-label-b3   { color: #dc2626; }
.v-label-safe { color: #16a34a; }
.v-status { font-size: 0.65rem; font-weight: 700; letter-spacing: 0.14em;
            text-transform: uppercase; margin: 0.4rem 0 0.8rem; }
.v-status-b3   { color: #ef4444; }
.v-status-safe { color: #22c55e; }
.v-desc { font-size: 0.75rem; color: #64748b; line-height: 1.65; }

.pills { display: flex; gap: 0.7rem; margin-bottom: 1rem; }
.pill {
    flex: 1; background: #f8fafc; border: 1.5px solid #e2e8f0;
    border-radius: 12px; padding: 0.8rem 0.6rem; text-align: center;
}
.pill-val { font-size: 1.15rem; font-weight: 800; display: block; letter-spacing: -0.03em; }
.pill-lbl { font-size: 0.57rem; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
            color: #94a3b8; display: block; margin-top: 0.2rem; }

.det-label {
    font-size: 0.64rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #94a3b8; margin: 0.8rem 0 0.4rem;
}
.tip { border-radius: 12px; padding: 0.85rem 1rem; font-size: 0.73rem; line-height: 1.75; margin-top: 0.5rem; }
.tip-b3   { background: #fffbeb; border: 1.5px solid #fde68a; color: #78350f; }
.tip-safe { background: #f0fdf4; border: 1.5px solid #bbf7d0; color: #14532d; }

.obj-row {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
    padding: 0.5rem 0.8rem; margin-bottom: 0.35rem; font-size: 0.73rem;
    display: flex; align-items: center; justify-content: space-between;
}
.obj-name { font-weight: 700; color: #1e293b; }
.obj-conf { color: #6366f1; font-weight: 600; }
.obj-size { color: #94a3b8; font-size: 0.66rem; }

.hdiv { height: 1px; background: linear-gradient(90deg,transparent,#e2e8f0,transparent); margin: 1rem 0; }
.footer {
    text-align: center; margin-top: 3rem; font-size: 0.64rem; color: #cbd5e1;
    font-family: 'JetBrains Mono', monospace; letter-spacing: 0.06em;
}
.empty-state {
    background: #fff; border: 2px dashed #e2e8f0; border-radius: 18px;
    padding: 3.5rem 2rem; text-align: center;
}
.empty-icon { font-size: 3rem; margin-bottom: 0.8rem; display: block; }
.empty-title { font-size: 0.92rem; font-weight: 700; color: #94a3b8; margin-bottom: 0.3rem; }
.empty-sub   { font-size: 0.72rem; color: #cbd5e1; line-height: 1.6; }

.status-bar {
    border-radius: 10px; padding: 0.65rem 1rem; font-size: 0.75rem; font-weight: 600;
    margin-bottom: 1.5rem;
}
.status-ok  { background: #f0fdf4; border: 1px solid #bbf7d0; color: #15803d; }
.status-warn{ background: #fffbeb; border: 1px solid #fde68a; color: #b45309; }

[data-testid="stImage"] img { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_classifier():
    _STRIP = ["renorm","renorm_clipping","renorm_momentum","synchronized","adjustment","quantization_config"]
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
        "model_b3.h5",
        custom_objects={"BatchNormalization":CBN,"InputLayer":CIL,"Dense":CD,"Conv2D":CC,"DepthwiseConv2D":CDC},
        compile=False
    )


@st.cache_resource(show_spinner=False)
def load_yolo():
    try:
        from ultralytics import YOLO
        for p in ["best.pt","yolov8n.pt"]:
            if os.path.exists(p): return YOLO(p)
        return YOLO("yolov8n.pt")
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
                boxes.append((int(x1),int(y1),int(x2),int(y2),float(b.conf[0]),yolo_model.names.get(int(b.cls[0]),f"obj")))
        return boxes
    except Exception:
        return []


def draw_overlay(pil_img, yolo_boxes, is_b3, pred_proba, yolo_ok):
    W, H = pil_img.size
    result = pil_img.copy()
    draw = ImageDraw.Draw(result)

    color = (220,38,38) if is_b3 else (22,163,74)
    lw = max(3, W // 120)

    try:
        font = ImageFont.truetype("arial.ttf", max(13, W // 38))
    except:
        font = ImageFont.load_default()

    def corners(x1, y1, x2, y2):
        cl = max(18, W // 28)

        for cx, cy, dx, dy in [
            (x1, y1, 1, 1),
            (x2, y1, -1, 1),
            (x1, y2, 1, -1),
            (x2, y2, -1, -1)
        ]:
            draw.line([(cx, cy), (cx + dx * cl, cy)], fill=color, width=lw + 2)
            draw.line([(cx, cy), (cx, cy + dy * cl)], fill=color, width=lw + 2)

    def pill(x1, y1, label):
        bb = draw.textbbox((0,0), label, font=font)

        tw = bb[2] - bb[0]
        th = bb[3] - bb[1]

        ph = max(24, H // 22)
        py = max(0, y1 - ph - 4)

        draw.rectangle(
            [x1, py, x1 + tw + 16, py + ph],
            fill=color
        )

        draw.text(
            (x1 + 8, py + (ph - th) // 2),
            label,
            fill="white",
            font=font
        )

    # =========================
    # YOLO BOX ONLY (NO OBJECT NAME)
    # =========================
    if yolo_ok and yolo_boxes:

        verdict = "B3" if is_b3 else "non-B3"

        for (x1, y1, x2, y2, conf, _) in yolo_boxes:

            # kotak objek
            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=lw
            )

            corners(x1, y1, x2, y2)

            # hanya tampilkan B3 / non-B3
            pill(
                x1,
                y1,
                f"{verdict}"
            )

        return result, "yolo"

    # =========================
    # FALLBACK SALIENCY
    # =========================
    gray = np.array(
        pil_img.convert("L"),
        dtype=np.float32
    )

    rows, cols = 7, 7
    ph2, pw2 = H // rows, W // cols

    heat = np.array([
        [
            np.var(
                gray[
                    r*ph2:(r+1)*ph2,
                    c*pw2:(c+1)*pw2
                ]
            )
            for c in range(cols)
        ]
        for r in range(rows)
    ])

    if heat.max() > 0:
        heat /= heat.max()

    mask = heat > 0.4

    if not mask.any():
        mask = heat == heat.max()

    yr, xc = np.where(mask)

    pad = 10

    x1 = max(0, xc.min() * pw2 - pad)
    y1 = max(0, yr.min() * ph2 - pad)

    x2 = min(W, (xc.max()+1) * pw2 + pad)
    y2 = min(H, (yr.max()+1) * ph2 + pad)

    draw.rectangle(
        [x1, y1, x2, y2],
        outline=color,
        width=lw
    )

    corners(x1, y1, x2, y2)

    verdict = "B3" if is_b3 else "non-B3"

    pill(
        x1,
        y1,
        verdict
    )

    return result, "fallback"


# ── Load ──
with st.spinner("⚙️ Memuat model AI..."):
    try: model = load_classifier()
    except Exception as e:
        st.error(f"❌ Gagal memuat classifier: {e}"); st.stop()

with st.spinner("⚙️ Memuat YOLOv8..."):
    yolo_model = load_yolo()
    yolo_ok    = yolo_model is not None

THRESHOLD = 0.50

# ── Topbar ──
st.markdown("""
<div class="topbar">
    <div class="topbar-left">
        <div class="topbar-icon">♻️</div>
        <div>
            <div class="topbar-title">B3 Waste Detector</div>
            <div class="topbar-sub">AI-powered hazardous waste classification + YOLOv8 object detection</div>
        </div>
    </div>
    <div class="topbar-badge">MobileNetV2 · YOLOv8</div>
</div>
""", unsafe_allow_html=True)

if yolo_ok:
    st.markdown('<div class="status-bar status-ok">✅ &nbsp;YOLOv8 aktif — object detection siap digunakan</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-bar status-warn">⚠️ &nbsp;YOLOv8 tidak tersedia — install: <code>pip install ultralytics</code></div>', unsafe_allow_html=True)

# ── Layout ──
pred_proba = None; is_b3 = None; img_source = None
col_left, col_right = st.columns([1.1, 0.9], gap="large")

with col_left:
    tab_upload, tab_camera = st.tabs(["📁  Upload Gambar", "📷  Kamera Langsung"])

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
        </div>
        """, unsafe_allow_html=True)
        uploaded = st.file_uploader("upload", type=["jpg","jpeg","png","webp"], label_visibility="collapsed")
        if uploaded:
            img_source = Image.open(uploaded).convert("RGB")

    with tab_camera:
        st.markdown('<div class="cam-hint">📸 &nbsp;Arahkan kamera ke sampah lalu tekan <b>Take photo</b></div>', unsafe_allow_html=True)
        snap = st.camera_input("foto", label_visibility="collapsed")
        if snap:
            img_source = Image.open(snap).convert("RGB")

    if img_source is not None:
        st.markdown('<div class="hdiv"></div>', unsafe_allow_html=True)
        img_224    = img_source.resize((224,224))
        arr        = np.expand_dims(preprocess_input(np.array(img_224, dtype=np.float32)), axis=0)
        pred_proba = float(model.predict(arr, verbose=0)[0][0])
        is_b3      = pred_proba < THRESHOLD

        with st.spinner("🔍 Mendeteksi objek..."):
            yolo_boxes = run_yolo(yolo_model, img_source)

        overlaid, mode = draw_overlay(img_source, yolo_boxes, is_b3, pred_proba, yolo_ok)
        mode_label = "🎯 YOLOv8 Detection" if mode=="yolo" else "🔍 Saliency Map (fallback)"
        st.markdown(f'<div class="det-label">{mode_label}</div>', unsafe_allow_html=True)
        st.image(overlaid, use_container_width=True)

        label_txt = "B3 — BERBAHAYA" if is_b3 else "non-B3 — AMAN"
        warna = "red" if is_b3 else "green"
        if mode=="yolo":
            st.caption(f"YOLOv8 mendeteksi **{len(yolo_boxes)} objek**. Klasifikasi MobileNetV2: **:{warna}[{label_txt}]**")
        else:
            st.caption(f"Lokalisasi dari saliency map. Klasifikasi MobileNetV2: **:{warna}[{label_txt}]**")


with col_right:
    if img_source is None or pred_proba is None:
        st.markdown("""
        <div class="empty-state">
            <span class="empty-icon">🔬</span>
            <div class="empty-title">Belum ada gambar</div>
            <div class="empty-sub">Upload foto atau gunakan kamera<br>untuk memulai deteksi sampah B3</div>
        </div>""", unsafe_allow_html=True)
    else:
        conf_b3=1.0-pred_proba; conf_safe=pred_proba
        display_conf = conf_b3 if is_b3 else conf_safe
        color_main   = "#dc2626" if is_b3 else "#16a34a"

        if is_b3:
            st.markdown(f"""
            <div class="verdict-card verdict-b3">
                <span class="v-emoji">☣️</span>
                <div class="v-label v-label-b3">B3</div>
                <div class="v-status v-status-b3">⚠ BAHAN BERBAHAYA & BERACUN</div>
                <div class="v-desc">Sampah ini terdeteksi mengandung bahan berbahaya.<br>Diperlukan penanganan &amp; pembuangan khusus.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-card verdict-safe">
                <span class="v-emoji">✅</span>
                <div class="v-label v-label-safe">non-B3</div>
                <div class="v-status v-status-safe">✓ AMAN — TIDAK BERBAHAYA</div>
                <div class="v-desc">Sampah ini tidak terdeteksi berbahaya.<br>Dapat diproses melalui daur ulang biasa.</div>
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="pills">
            <div class="pill">
                <span class="pill-val" style="color:{color_main}">{display_conf:.1%}</span>
                <span class="pill-lbl">{'Conf. B3' if is_b3 else 'Conf. Aman'}</span>
            </div>
            <div class="pill">
                <span class="pill-val" style="color:#6366f1">{pred_proba:.3f}</span>
                <span class="pill-lbl">Sigmoid</span>
            </div>
            <div class="pill">
                <span class="pill-val" style="color:#64748b">{THRESHOLD:.2f}</span>
                <span class="pill-lbl">Threshold</span>
            </div>
        </div>""", unsafe_allow_html=True)

        gauge_val=conf_b3*100; gauge_color="#ef4444" if is_b3 else "#22c55e"
        fig_g=go.Figure(go.Indicator(
            mode="gauge+number", value=gauge_val,
            number={"suffix":"%","font":{"size":26,"color":gauge_color,"family":"Plus Jakarta Sans"}},
            gauge={"axis":{"range":[0,100],"showticklabels":False,"tickwidth":0},
                   "bar":{"color":gauge_color,"thickness":0.26},
                   "bgcolor":"#f1f5f9","borderwidth":0,
                   "steps":[{"range":[0,40],"color":"#dcfce7"},{"range":[40,60],"color":"#fef9c3"},{"range":[60,100],"color":"#fee2e2"}],
                   "threshold":{"line":{"color":gauge_color,"width":3},"thickness":0.85,"value":gauge_val}},
            title={"text":"Risiko B3","font":{"size":11,"color":"#94a3b8","family":"Plus Jakarta Sans"}},
            domain={"x":[0,1],"y":[0,1]}
        ))
        fig_g.update_layout(height=190, margin=dict(t=36,b=8,l=16,r=16), paper_bgcolor="white", plot_bgcolor="white")
        st.plotly_chart(fig_g, use_container_width=True, config={"displayModeBar":False})

        fig_b=go.Figure()
        fig_b.add_trace(go.Bar(x=[conf_b3*100],y=["B3"],orientation="h",marker_color="#ef4444",marker_line_width=0,width=0.42,text=[f"{conf_b3:.1%}"],textposition="inside",insidetextfont=dict(color="white",size=11)))
        fig_b.add_trace(go.Bar(x=[conf_safe*100],y=["non-B3"],orientation="h",marker_color="#22c55e",marker_line_width=0,width=0.42,text=[f"{conf_safe:.1%}"],textposition="inside",insidetextfont=dict(color="white",size=11)))
        fig_b.update_layout(height=108,margin=dict(t=4,b=4,l=58,r=14),paper_bgcolor="white",plot_bgcolor="white",showlegend=False,
            xaxis=dict(range=[0,100],showticklabels=False,showgrid=False,zeroline=False),
            yaxis=dict(showgrid=False,tickfont=dict(size=11,color="#374151")),bargap=0.3)
        st.plotly_chart(fig_b, use_container_width=True, config={"displayModeBar":False})

        if yolo_ok and img_source is not None:
            boxes2 = run_yolo(yolo_model, img_source)
            if boxes2:
                st.markdown('<div class="det-label">📦 Objek Terdeteksi YOLO</div>', unsafe_allow_html=True)
                for i,(x1,y1,x2,y2,conf,name) in enumerate(boxes2):
                    st.markdown(f"""<div class="obj-row">
                                <span class="obj-name">#{i+1} &nbsp;{name}</span>
                                <span class="obj-conf">{conf:.0%}</span>
                                <span class="obj-size">{x2-x1}×{y2-y1}px</span>
                                </div>""", unsafe_allow_html=True)

        if is_b3:
            st.markdown("""<div class="tip tip-b3">
                <b>⚠ Cara Penanganan Sampah B3</b><br>
                • Jangan buang ke tempat sampah rumah tangga biasa<br>
                • Simpan di wadah tertutup rapat dan berlabel jelas<br>
                • Serahkan ke fasilitas pengolahan limbah B3 resmi<br>
                • Contoh: baterai, cat, pestisida, oli bekas, lampu neon
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="tip tip-safe">
                <b>✓ Tips Daur Ulang non-B3</b><br>
                • Pisahkan organik, plastik, kertas, dan logam<br>
                • Cuci bersih wadah plastik/kaca sebelum dibuang<br>
                • Manfaatkan bank sampah atau TPS 3R terdekat
            </div>""", unsafe_allow_html=True)

st.markdown("""<div class="footer">
    B3 Waste Detector &nbsp;·&nbsp; MobileNetV2 + YOLOv8 &nbsp;·&nbsp; TensorFlow &nbsp;·&nbsp; Streamlit
</div>""", unsafe_allow_html=True)