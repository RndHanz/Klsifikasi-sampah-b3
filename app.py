import streamlit as st
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2
import os
import time
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
*, html, body, [class*="css"] { font-family: 'Plus Jakarta Sans', sans-serif !important; }
.stApp { background: #f7f8fc !important; }
.block-container { padding: 1.5rem 2rem 3rem !important; max-width: 1150px; }
#MainMenu, footer, header { visibility: hidden; }

.topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 1rem 0 1.5rem; border-bottom: 1.5px solid #e8eaf0; margin-bottom: 2rem;
}
.topbar-logo { display: flex; align-items: center; gap: 0.7rem; }
.topbar-logo-icon {
    width: 40px; height: 40px; border-radius: 10px;
    background: linear-gradient(135deg, #1a56db, #5b8dee);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; box-shadow: 0 4px 14px rgba(26,86,219,0.28);
}
.topbar-logo-text { font-size: 1.15rem; font-weight: 700; color: #111827; letter-spacing: -0.02em; }
.topbar-logo-sub { font-size: 0.67rem; color: #9ca3af; }
.topbar-badge {
    background: #eef2ff; color: #4f46e5; font-size: 0.64rem; font-weight: 600;
    letter-spacing: 0.06em; text-transform: uppercase;
    padding: 0.3rem 0.85rem; border-radius: 20px; border: 1px solid #c7d2fe;
}

.section-title {
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: #9ca3af; margin-bottom: 0.8rem;
}
.input-card {
    background: #fff; border: 1.5px solid #e8eaf0; border-radius: 16px;
    padding: 1.4rem; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.stTabs [data-baseweb="tab-list"] {
    background: #f3f4f6 !important; border-radius: 10px !important;
    padding: 4px !important; gap: 2px !important; border: none !important; margin-bottom: 1rem;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px !important; font-size: 0.78rem !important; font-weight: 600 !important;
    color: #6b7280 !important; padding: 0.45rem 1.1rem !important; border: none !important;
}
.stTabs [aria-selected="true"] {
    background: #fff !important; color: #111827 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}
[data-testid="stFileUploader"] {
    background: #f9fafb !important; border: 2px dashed #d1d5db !important;
    border-radius: 12px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #6366f1 !important; }
[data-testid="stCameraInput"] > div {
    border-radius: 12px !important; overflow: hidden !important;
    border: 1.5px solid #e8eaf0 !important;
}
[data-testid="stCameraInput"] button {
    background: linear-gradient(135deg, #1a56db, #5b8dee) !important;
    color: white !important; border: none !important; border-radius: 8px !important;
    font-weight: 600 !important; font-size: 0.8rem !important;
}

.verdict-card {
    border-radius: 16px; padding: 1.4rem 1.4rem 1.1rem;
    text-align: center; position: relative; overflow: hidden; margin-bottom: 1rem;
}
.verdict-card-b3 { background: linear-gradient(135deg,#fff5f5,#fff0f0); border: 1.5px solid #fca5a5; }
.verdict-card-safe { background: linear-gradient(135deg,#f0fdf4,#ecfdf5); border: 1.5px solid #86efac; }
.verdict-emoji { font-size: 2.4rem; display: block; margin-bottom: 0.35rem; }
.verdict-klass { font-size: 1.7rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.1rem; }
.verdict-klass-b3 { color: #dc2626; }
.verdict-klass-safe { color: #16a34a; }
.verdict-status { font-size: 0.67rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.7rem; }
.verdict-status-b3 { color: #ef4444; }
.verdict-status-safe { color: #22c55e; }
.verdict-desc { font-size: 0.74rem; color: #6b7280; line-height: 1.6; }

.metric-row { display: flex; gap: 0.6rem; margin-bottom: 0.9rem; }
.metric-pill { flex: 1; background: #fff; border: 1.5px solid #e8eaf0; border-radius: 12px; padding: 0.85rem 0.8rem; text-align: center; }
.metric-pill-val { font-size: 1.2rem; font-weight: 800; letter-spacing: -0.03em; display: block; }
.metric-pill-lbl { font-size: 0.58rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #9ca3af; display: block; margin-top: 0.1rem; }

.tip-card { background: #fffbeb; border: 1.5px solid #fcd34d; border-radius: 12px; padding: 0.9rem 1.1rem; font-size: 0.73rem; color: #78350f; line-height: 1.7; margin-top: 0.5rem; }
.tip-card-safe { background: #f0fdf4; border-color: #86efac; color: #14532d; }

.gradcam-label {
    font-size: 0.67rem; font-weight: 600; letter-spacing: 0.1em; text-transform: uppercase;
    color: #9ca3af; margin: 1rem 0 0.5rem;
}
.gradcam-note {
    font-size: 0.68rem; color: #9ca3af; margin-top: 0.4rem; line-height: 1.5;
    font-style: italic;
}
.hdivider { height: 1.5px; background: linear-gradient(90deg,transparent,#e8eaf0,transparent); margin: 1.2rem 0; }
.footer { text-align: center; margin-top: 3rem; font-size: 0.67rem; color: #d1d5db; font-family: 'JetBrains Mono',monospace; letter-spacing: 0.06em; }
[data-testid="stImage"] img { border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    class CompatBatchNorm(tf.keras.layers.BatchNormalization):
        def __init__(self, **kwargs):
            for k in ["renorm","renorm_clipping","renorm_momentum","synchronized","adjustment"]:
                kwargs.pop(k, None)
            super().__init__(**kwargs)
    return tf.keras.models.load_model(
        "model_b3.h5",
        custom_objects={"BatchNormalization": CompatBatchNorm},
        compile=False
    )

with st.spinner("Memuat model AI..."):
    model = load_model()

THRESHOLD = 0.50


# ── GradCAM++ ─────────────────────────────────────────────────────────────────
def compute_gradcam(_model, img_array):
    """
    GradCAM++ — lebih akurat dari GradCAM standar untuk lokalisasi objek.

    Perbedaan utama vs GradCAM biasa:
      GradCAM   : alpha_k = mean(grads)          → bobot per channel rata-rata
      GradCAM++ : alpha_k = f(grads, grads^2, grads^3, feat_map)
                            → bobot yang memperhitungkan kontribusi piksel secara
                              non-linear, hasilnya lebih tajam & presisi di objek kecil.

    Paper: Chattopadhyay et al., 2018 — "Grad-CAM++: Improved Visual Explanations
           for Deep Convolutional Networks"
    """

    # ── 1. Temukan base MobileNetV2 & layer Conv2D terakhirnya ────────────────
    base_model     = None
    last_conv_name = None

    for layer in _model.layers:
        if hasattr(layer, 'layers') and len(layer.layers) > 5:
            base_model = layer
            for sub in layer.layers:
                if isinstance(sub, tf.keras.layers.Conv2D):
                    last_conv_name = sub.name   # update terus → dapat yang terakhir
            break

    if base_model is None:          # fallback: tidak ada sub-model
        base_model = _model
        for layer in _model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_name = layer.name

    if last_conv_name is None:
        hm = np.zeros((224, 224)); hm[56:168, 56:168] = 0.8
        return hm, (48, 48, 176, 176)

    # ── 2. Feature extractor: input → conv terakhir ───────────────────────────
    feat_extractor = tf.keras.Model(
        inputs  = base_model.input,
        outputs = base_model.get_layer(last_conv_name).output
    )

    # ── 3. Tail layers (GAP → Dense → Dropout → output) ──────────────────────
    tail_layers = []
    found = False
    for layer in _model.layers:
        if found:
            tail_layers.append(layer)
        if layer.name == base_model.name:
            found = True

    def run_tail(feat):
        x = feat
        for layer in tail_layers:
            try:    x = layer(x, training=False)
            except: x = layer(x)
        return x

    # ── 4. GradCAM++ membutuhkan orde-1, 2, 3 gradien ─────────────────────────
    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape3:
        with tf.GradientTape() as tape2:
            with tf.GradientTape() as tape1:
                feat_map = feat_extractor(img_tensor, training=False)
                tape1.watch(feat_map)
                tape2.watch(feat_map)
                tape3.watch(feat_map)
                score = run_tail(feat_map)[:, 0]   # skor sigmoid kelas 0 (B3)

            grads1 = tape1.gradient(score, feat_map)   # dy/dA

        grads2 = tape2.gradient(grads1, feat_map)       # d²y/dA²

    grads3 = tape3.gradient(grads2, feat_map)           # d³y/dA³

    if grads1 is None:
        hm = np.zeros((224, 224)); hm[56:168, 56:168] = 0.8
        return hm, (48, 48, 176, 176)

    # ── 5. Hitung alpha (bobot GradCAM++) ─────────────────────────────────────
    # alpha_k^c = grad2 / (2*grad2 + feat_map * grad3 + eps)
    eps      = 1e-7
    g1  = grads1.numpy()
    g2  = grads2.numpy() if grads2 is not None else g1 ** 2
    g3  = grads3.numpy() if grads3 is not None else g1 ** 3
    fm  = feat_map.numpy()

    alpha_num   = g2
    alpha_den   = 2.0 * g2 + fm * g3 + eps
    alpha       = alpha_num / alpha_den                 # shape: (1, H, W, C)

    # Weights: w_k = sum_ij( alpha_k * ReLU(grad1) )
    relu_grad   = np.maximum(g1, 0)                    # ReLU pada gradien
    weights     = np.sum(alpha * relu_grad, axis=(1, 2), keepdims=True)  # (1,1,1,C)

    # CAM = ReLU( sum_k( w_k * A_k ) )
    cam = np.sum(weights * fm, axis=-1)[0]             # (H, W)
    cam = np.maximum(cam, 0)
    if cam.max() > 0:
        cam /= cam.max()

    # ── 6. Resize & haluskan dengan Gaussian blur ─────────────────────────────
    heatmap = cv2.resize(cam, (224, 224))
    heatmap = cv2.GaussianBlur(heatmap, (9, 9), sigmaX=2)   # ← haluskan noise
    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    # ── 7. Bounding box — ambil kontur dengan area terbesar ───────────────────
    # Pakai threshold adaptif: 50% dari nilai max heatmap
    thresh      = max(0.45, heatmap.mean() + heatmap.std() * 0.5)
    binary      = (heatmap >= thresh).astype(np.uint8)

    # Morphological closing → tutup lubang kecil dalam kontur
    kernel      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    binary      = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Gabung semua kontur besar (> 2% area gambar) menjadi 1 bounding box
        min_area = 224 * 224 * 0.02
        big      = [c for c in contours if cv2.contourArea(c) > min_area]
        if not big:
            big = [max(contours, key=cv2.contourArea)]

        xs = [cv2.boundingRect(c)[0]                       for c in big]
        ys = [cv2.boundingRect(c)[1]                       for c in big]
        x2s= [cv2.boundingRect(c)[0]+cv2.boundingRect(c)[2] for c in big]
        y2s= [cv2.boundingRect(c)[1]+cv2.boundingRect(c)[3] for c in big]

        pad = 12
        x1 = max(0,   min(xs)  - pad)
        y1 = max(0,   min(ys)  - pad)
        x2 = min(223, max(x2s) + pad)
        y2 = min(223, max(y2s) + pad)
    else:
        x1, y1, x2, y2 = 20, 20, 203, 203

    return heatmap, (x1, y1, x2, y2)


def overlay_gradcam(pil_img, heatmap, bbox, is_b3, pred_proba):
    """
    Overlay GradCAM heatmap + bounding box + label ke atas gambar asli.
    """
    orig_w, orig_h = pil_img.size

    # Scale heatmap ke ukuran gambar asli
    hm_resized = cv2.resize(heatmap, (orig_w, orig_h))
    hm_uint8   = np.uint8(255 * hm_resized)
    colormap   = cv2.COLORMAP_JET if is_b3 else cv2.COLORMAP_SUMMER
    hm_colored = cv2.applyColorMap(hm_uint8, colormap)
    hm_rgb     = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB)

    # Blend gambar asli + heatmap
    orig_np   = np.array(pil_img)
    blended   = cv2.addWeighted(orig_np, 0.55, hm_rgb, 0.45, 0)
    result    = Image.fromarray(blended)

    # Scale bbox dari 224 ke ukuran asli
    sx = orig_w / 224; sy = orig_h / 224
    x1, y1, x2, y2 = bbox
    bx1 = int(x1 * sx); by1 = int(y1 * sy)
    bx2 = int(x2 * sx); by2 = int(y2 * sy)

    # Gambar bounding box
    draw       = ImageDraw.Draw(result)
    box_color  = (220, 38, 38) if is_b3 else (22, 163, 74)
    lw         = max(3, orig_w // 120)

    draw.rectangle([bx1, by1, bx2, by2], outline=box_color, width=lw)

    # Corner accent (L-shape di 4 sudut)
    cl = max(16, orig_w // 30)
    for cx, cy, dx, dy in [
        (bx1, by1,  1,  1),
        (bx2, by1, -1,  1),
        (bx1, by2,  1, -1),
        (bx2, by2, -1, -1),
    ]:
        draw.line([(cx, cy), (cx + dx*cl, cy)], fill=box_color, width=lw+2)
        draw.line([(cx, cy), (cx, cy + dy*cl)], fill=box_color, width=lw+2)

    # Label pill di atas kotak
    label      = "☣ B3 — BERBAHAYA" if is_b3 else "✓ non-B3 — AMAN"
    conf_txt   = f"{(1-pred_proba if is_b3 else pred_proba):.1%}"
    pill_h     = max(24, orig_h // 22)
    pad_x      = 8
    font_size  = max(12, pill_h - 6)

    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    full_label = f"{label}  {conf_txt}"
    bbox_txt   = draw.textbbox((0, 0), full_label, font=font)
    tw         = bbox_txt[2] - bbox_txt[0]
    th         = bbox_txt[3] - bbox_txt[1]
    px         = bx1
    py         = max(0, by1 - pill_h - 4)
    pw         = tw + pad_x * 2

    draw.rectangle([px, py, px+pw, py+pill_h], fill=box_color)
    draw.text((px+pad_x, py + (pill_h-th)//2), full_label, fill="white", font=font)

    return result


# ══════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        <div class="topbar-logo-icon">♻️</div>
        <div>
            <div class="topbar-logo-text">B3 Waste Detector</div>
            <div class="topbar-logo-sub">AI-powered hazardous waste classification + GradCAM visualization</div>
        </div>
    </div>
    <div class="topbar-badge">MobileNetV2 · GradCAM</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1.05, 0.95], gap="large")

# ════════════════════════════════
# KOLOM KIRI — Input
# ════════════════════════════════
with col_left:
    st.markdown('<div class="section-title">📥 Input Gambar</div>', unsafe_allow_html=True)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)

    tab_upload, tab_camera = st.tabs(["📁  Upload File", "📷  Kamera"])
    img_source = None

    with tab_upload:
        uploaded = st.file_uploader(
            "Pilih gambar sampah", type=["jpg","jpeg","png","webp"],
            label_visibility="collapsed"
        )
        if uploaded:
            img_source = Image.open(uploaded).convert("RGB")

    with tab_camera:
        st.markdown(
            "<p style='font-size:0.75rem;color:#6b7280;margin-bottom:0.5rem'>"
            "Arahkan kamera ke sampah lalu tekan <b>Take photo</b></p>",
            unsafe_allow_html=True
        )
        camera_snap = st.camera_input(label="foto", label_visibility="collapsed")
        if camera_snap:
            img_source = Image.open(camera_snap).convert("RGB")

    st.markdown('</div>', unsafe_allow_html=True)

    # Preview + GradCAM overlay
    if img_source:
        st.markdown('<div class="hdivider"></div>', unsafe_allow_html=True)

        # Inference dulu (butuh untuk overlay)
        img_224   = img_source.resize((224, 224))
        arr       = np.expand_dims(preprocess_input(
                        np.array(img_224, dtype=np.float32)), axis=0)
        pred_proba = float(model.predict(arr, verbose=0)[0][0])
        is_b3      = pred_proba < THRESHOLD

        with st.spinner("Menghitung GradCAM..."):
            heatmap, bbox = compute_gradcam(model, arr)

        overlaid = overlay_gradcam(img_source, heatmap, bbox, is_b3, pred_proba)

        st.markdown('<div class="gradcam-label">🔥 GradCAM + Bounding Box</div>',
                    unsafe_allow_html=True)
        st.image(overlaid, use_container_width=True)

        box_color_css = "#dc2626" if is_b3 else "#16a34a"
        label_txt     = "B3 — BERBAHAYA" if is_b3 else "non-B3 — AMAN"
        st.markdown(f"""
        <p class="gradcam-note">
            Kotak <span style="color:{box_color_css};font-weight:700">{label_txt}</span>
            menunjukkan area yang paling diperhatikan model saat membuat prediksi.
            Warna panas = kontribusi tinggi terhadap keputusan model.
        </p>
        """, unsafe_allow_html=True)

# ════════════════════════════════
# KOLOM KANAN — Hasil
# ════════════════════════════════
with col_right:
    if img_source is None:
        st.markdown("""
        <div style="background:#fff;border:1.5px dashed #e8eaf0;border-radius:16px;
                    padding:3rem 2rem;text-align:center;margin-top:2.6rem;">
            <div style="font-size:3rem;margin-bottom:1rem">🔍</div>
            <div style="font-size:0.9rem;font-weight:600;color:#9ca3af;margin-bottom:0.4rem">
                Belum Ada Gambar</div>
            <div style="font-size:0.75rem;color:#d1d5db">
                Upload atau ambil foto sampah<br>untuk memulai deteksi</div>
        </div>""", unsafe_allow_html=True)
    else:
        conf_b3   = 1.0 - pred_proba
        conf_safe = pred_proba
        display_conf = conf_b3 if is_b3 else conf_safe

        st.markdown('<div class="section-title">🎯 Hasil Deteksi</div>', unsafe_allow_html=True)

        # Verdict card
        if is_b3:
            st.markdown(f"""
            <div class="verdict-card verdict-card-b3">
                <span class="verdict-emoji">☣️</span>
                <div class="verdict-klass verdict-klass-b3">B3</div>
                <div class="verdict-status verdict-status-b3">⚠ BAHAN BERBAHAYA & BERACUN</div>
                <div class="verdict-desc">Sampah ini terdeteksi mengandung bahan berbahaya.<br>
                Butuh penanganan dan pembuangan khusus.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-card verdict-card-safe">
                <span class="verdict-emoji">✅</span>
                <div class="verdict-klass verdict-klass-safe">non-B3</div>
                <div class="verdict-status verdict-status-safe">✓ AMAN — TIDAK BERBAHAYA</div>
                <div class="verdict-desc">Sampah ini tidak mengandung bahan berbahaya.<br>
                Dapat diproses melalui daur ulang biasa.</div>
            </div>""", unsafe_allow_html=True)

        # Metric pills
        color_main = "#dc2626" if is_b3 else "#16a34a"
        label_main = "Conf. B3" if is_b3 else "Conf. non-B3"
        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-pill">
                <span class="metric-pill-val" style="color:{color_main}">{display_conf:.1%}</span>
                <span class="metric-pill-lbl">{label_main}</span>
            </div>
            <div class="metric-pill">
                <span class="metric-pill-val" style="color:#6366f1">{pred_proba:.3f}</span>
                <span class="metric-pill-lbl">Raw sigmoid</span>
            </div>
            <div class="metric-pill">
                <span class="metric-pill-val" style="color:#374151">{THRESHOLD:.2f}</span>
                <span class="metric-pill-lbl">Threshold</span>
            </div>
        </div>""", unsafe_allow_html=True)

        # Gauge
        gauge_val   = conf_b3 * 100
        gauge_color = "#ef4444" if is_b3 else "#22c55e"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={"suffix":"%","font":{"size":28,"color":gauge_color,"family":"Plus Jakarta Sans"}},
            gauge={
                "axis":{"range":[0,100],"showticklabels":False,"tickwidth":0},
                "bar":{"color":gauge_color,"thickness":0.28},
                "bgcolor":"#f3f4f6","borderwidth":0,
                "steps":[
                    {"range":[0,40],"color":"#d1fae5"},
                    {"range":[40,60],"color":"#fef9c3"},
                    {"range":[60,100],"color":"#fee2e2"},
                ],
                "threshold":{"line":{"color":gauge_color,"width":3},"thickness":0.85,"value":gauge_val}
            },
            title={"text":"Probabilitas B3","font":{"size":12,"color":"#9ca3af","family":"Plus Jakarta Sans"}},
            domain={"x":[0,1],"y":[0,1]}
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(t=40,b=10,l=20,r=20),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar":False})

        # Bar chart
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[conf_b3*100], y=["B3"], orientation="h",
            marker_color="#ef4444", marker_line_width=0, width=0.45,
            text=[f"{conf_b3:.1%}"], textposition="inside",
            insidetextfont=dict(color="white",size=11),
        ))
        fig_bar.add_trace(go.Bar(
            x=[conf_safe*100], y=["non-B3"], orientation="h",
            marker_color="#22c55e", marker_line_width=0, width=0.45,
            text=[f"{conf_safe:.1%}"], textposition="inside",
            insidetextfont=dict(color="white",size=11),
        ))
        fig_bar.update_layout(
            height=110, margin=dict(t=6,b=6,l=60,r=16),
            paper_bgcolor="white", plot_bgcolor="white", showlegend=False,
            xaxis=dict(range=[0,100],showticklabels=False,showgrid=False,zeroline=False),
            yaxis=dict(showgrid=False,tickfont=dict(size=11,color="#374151")),
            bargap=0.3,
        )
        st.plotly_chart(fig_bar, use_container_width=True, config={"displayModeBar":False})

        # Tip
        if is_b3:
            st.markdown("""
            <div class="tip-card">
                <b>⚠ Cara Penanganan Sampah B3:</b><br>
                • Jangan buang ke tempat sampah biasa<br>
                • Simpan di wadah tertutup &amp; berlabel<br>
                • Serahkan ke fasilitas pengolahan limbah B3<br>
                • Contoh: baterai, cat, pestisida, oli bekas, lampu neon
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="tip-card tip-card-safe">
                <b>✓ Tips Daur Ulang non-B3:</b><br>
                • Pisahkan organik, plastik, kertas, dan logam<br>
                • Cuci wadah plastik/kaca sebelum dibuang<br>
                • Manfaatkan bank sampah atau TPS 3R terdekat
            </div>""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    B3 Waste Detector · MobileNetV2 + GradCAM · TensorFlow 2.10 · Streamlit
</div>""", unsafe_allow_html=True)