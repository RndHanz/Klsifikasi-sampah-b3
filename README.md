# 🗑️ Klasifikasi Sampah B3 vs Non-B3

Aplikasi deteksi sampah berbahaya (B3) menggunakan **Transfer Learning dengan MobileNetV2**.

---

## 📋 Daftar Isi
1. [Setup Environment](#setup-environment)
2. [Persiapan Dataset](#persiapan-dataset)
3. [Menjalankan Notebook](#menjalankan-notebook)
4. [Menjalankan Aplikasi Streamlit](#menjalankan-aplikasi-streamlit)
5. [File yang Dihasilkan](#file-yang-dihasilkan)

---

## 🔧 Setup Environment

### Install Dependencies

Semua library yang diperlukan sudah terinstall. Jika perlu install ulang:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn pillow streamlit
```

### Verifikasi Instalasi

```bash
python -c "import tensorflow, numpy, matplotlib, seaborn, sklearn, PIL, streamlit; print('✅ Semua library berhasil terinstall!')"
```

---

## 📂 Persiapan Dataset

Notebook ini memerlukan dataset dari **Roboflow** dengan struktur berikut:

```
dataset/
├── train/
│   ├── B3/          (gambar B3)
│   └── non-B3/      (gambar non-B3)
├── valid/
│   ├── B3/
│   └── non-B3/
└── test/
    ├── B3/
    └── non-B3/
```

### Cara Download Dataset

1. **Download dataset.zip** dari Roboflow/Google Drive
2. **Extract dan tempatkan** di folder `project/` dengan nama `dataset.zip`
3. Notebook akan otomatis mengekstrak saat dijalankan

### Alternatif: Path Kustom

Jika dataset ada di lokasi berbeda, edit cell 6 pada notebook:

```python
ZIP_PATH    = "path/ke/dataset.zip"  # ubah ke path Anda
EXTRACT_DIR = "path/ke/dataset"       # folder ekstraksi
```

---

## 📓 Menjalankan Notebook

### 1. Buka Notebook di VS Code
- File: `klasifikasi_sampah_b3_colab.ipynb`

### 2. Setup Notebook Kernel
- VS Code akan otomatis detect kernel
- Pilih Python 3.11 atau lebih tinggi

### 3. Jalankan Cell Secara Berurutan

**⚠️ PENTING:** Jalankan cell dengan urutan dari atas ke bawah:

| Cell | Deskripsi | Status |
|------|-----------|--------|
| 1-2 | Header & Info | Markdown |
| 3 | Cek GPU (Info saja) | ✅ Sudah diperbaiki |
| 4 | Setup Environment | ✅ Sudah diperbaiki |
| 5 | Extract Dataset | ✅ Sudah diperbaiki |
| 6 | Import Library | Jalankan |
| 7 | Konfigurasi Path | ✅ Sudah diperbaiki |
| 8-15 | Load & Augmentasi Data | Jalankan |
| 16-17 | Training Phase 1 | ⚠️ **LAMA** (~10-20 menit) |
| 18-19 | Training Phase 2 | ⚠️ **LAMA** (~20-30 menit) |
| 20-26 | Evaluasi & Visualisasi | Jalankan |
| 27-28 | Simpan Model | ✅ Sudah diperbaiki |
| 29 | Ringkasan Hasil | Info saja |

### ⏱️ Perkiraan Waktu
- **Total:** ~1-2 jam (tergantung GPU)
- **Tanpa GPU:** Bisa mencapai 4-5 jam

---

## 🌐 Deploy ke Streamlit Community
- Pastikan repo hanya berisi:
  - `app.py` atau `apps.py` (sesuaikan entrypoint Streamlit)
  - `requirements.txt`
  - `model_b3.h5`
  - `yolov8n.pt`
  - `README.md`
- Dataset besar tidak perlu dipush ke GitHub.
  - Folder `dataset/` sudah di-ignore oleh `.gitignore`.
  - Streamlit Community hanya membutuhkan model terlatih, bukan dataset pelatihan.
- Jika menggunakan repo baru, jalankan:

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<username>/<repo>.git
git push -u origin main
```
- Setelah repo berada di GitHub, buat Streamlit project dari repo tersebut.
  - Pilih branch `main`
  - Pastikan entrypoint adalah `apps.py`
  - Di `Advanced settings`, pilih Python versi `3.12`
  - Streamlit akan install `requirements.txt`

---

## 🚀 Menjalankan Aplikasi Streamlit

### Prasyarat
- Model `model_b3.h5` sudah dihasilkan dari notebook

### Jalankan Aplikasi

```bash
cd project/
python -m streamlit run app.py
```

Atau gunakan shortcut:
```bash
streamlit run app.py
```

### Akses Aplikasi
- **Local URL:** `http://localhost:8502`
- **Network URL:** `http://192.168.x.x:8502` (untuk akses dari device lain)

### Cara Menggunakan
1. **Upload Gambar** → Pilih file JPG/PNG
2. **Lihat Hasil** → Model akan mengklasifikasi sampah
3. **Interpretasi:**
   - 🔴 **B3 (Berbahaya)** → Butuh penanganan khusus
   - ✅ **non-B3 (Aman)** → Bisa didaur ulang biasa

---

## 📁 File yang Dihasilkan

Setelah menjalankan notebook, berikut file yang dihasilkan:

| File | Deskripsi |
|------|-----------|
| `model_b3.h5` | **Model terlatih** (dipakai Streamlit) |
| `checkpoint_best.h5` | Model checkpoint terbaik |
| `model_b3_savedmodel/` | SavedModel format (backup) |
| `training_curves.png` | Grafik loss, accuracy, AUC, recall |
| `confusion_matrix.png` | Confusion matrix test set |
| `threshold_f1.png` | Grafik optimal threshold |

### Lokasi File
Semua file tersimpan di folder `project/`

---

## 🐛 Troubleshooting

### Error: "ModuleNotFoundError: No module named 'tensorflow'"
```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
```

### Error: "No module named 'seaborn'"
```bash
pip install seaborn
```

### Error: "dataset folder not found"
- Pastikan `dataset.zip` sudah diletakkan di folder `project/`
- Atau ubah path di cell 6 notebook

### Streamlit App Tidak Muncul
```bash
# Kill process Streamlit jika masih berjalan
taskkill /im python.exe /f

# Jalankan ulang
python -m streamlit run app.py
```

### Out of Memory saat Training
- Kurangi `BATCH_SIZE` di cell 7: ubah dari 32 menjadi 16 atau 8
- Atau kurangi `EPOCHS_FT` dan `EPOCHS_UN`

---

## 📊 Model Architecture

- **Base Model:** MobileNetV2 (Pre-trained ImageNet)
- **Input Size:** 224×224 px
- **Output:** Binary Classification (B3 / non-B3)

### Two-Phase Training

**Fase 1: Head Training** (10 epochs)
- Base model frozen
- Training hanya head layer
- Learning rate: 1e-3

**Fase 2: Fine-Tuning** (20 epochs)
- Unfreeze 50 layer terakhir base
- Learning rate lebih kecil: 1e-4
- Callback: Early Stopping, ReduceLROnPlateau

---

## 📈 Expected Results

| Metrik | Target |
|--------|--------|
| Accuracy (Test) | ~92-95% |
| Precision (B3) | ~90-93% |
| Recall (B3) | ~88-92% |
| F1-Score (B3) | ~89-92% |
| Threshold Optimal | ~0.45-0.55 |

---

## 📝 Catatan Penting

1. **Dataset Balance:** Dataset imbalanced (B3: 3.960, non-B3: 10.811)
   - Solusi: Class weighting dalam training

2. **Threshold:** Default threshold = 0.50
   - Dapat disesuaikan melalui `THRESHOLD` variable
   - Lihat optimal threshold di grafik threshold_f1.png

3. **GPU:** Model training lebih cepat dengan GPU
   - Check: NVIDIA GPU + CUDA support

4. **Production:** Model H5 sudah siap untuk deployment
   - Gunakan di Streamlit, FastAPI, Flask, dll

---

## 🔗 Links

- **Dataset Roboflow:** https://roboflow.com/
- **MobileNetV2:** https://github.com/keras-team/keras/blob/master/applications/mobilenet_v2.py
- **Streamlit Docs:** https://docs.streamlit.io/

---

## 👨‍💻 Author
Disusun untuk Project Dosen — Klasifikasi Sampah B3

**Last Updated:** 2 Mei 2026
