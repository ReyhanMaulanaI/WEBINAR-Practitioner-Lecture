# Studi Kasus & Implementasi: Data, AI, dan Infrastruktur Komputasi untuk Dampak Nyata

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40%2B-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)

Repositori ini berisi implementasi lengkap kode starter dan studi kasus praktis dari **Webinar Data, AI, dan Infrastruktur Komputasi untuk Dampak Nyata**. Semua skrip siap dijalankan (*out-of-the-box*) menggunakan data sintetis yang realistis dan dapat langsung disubstitusi dengan data riil untuk keperluan tugas akhir, skripsi, maupun riset.

---

## 📁 Struktur Direktori

```text
WEBINAR/
├── .gitignore
├── requirements.txt
├── README.md
├── sesi1_analitik/
│   ├── sesi1.py
│   ├── ipk_hist.png              # Output grafik distribusi IPK
│   ├── outliers_review.csv       # Output deteksi pencilan (IQR)
│   └── hasil_rekomendasi.csv     # Output rekomendasi prescriptive
├── sesi2_hpc/
│   ├── sesi2.py
│   └── speedup.png               # Output grafik speedup dan efisiensi
├── sesi3_gis/
│   ├── sesi3.py
│   ├── kerawanan_banjir.geojson  # Output geospasial kerawanan banjir
│   └── sesi3_peta.html           # Output peta web interaktif Folium
├── sesi4_ml/
│   ├── sesi4.py
│   └── klaster.png               # Output scatter plot klaster PCA 2D
├── sesi5_dashboard/
│   ├── app.py                    # Aplikasi web interaktif Streamlit
│   ├── usability.py              # Skrip rekap evaluasi usability testing
│   └── usability_log.csv         # Log data pengujian usability
└── sesi6_deforestasi/
    └── sesi6.py                  # Benchmark Random Forest vs CNN (PyTorch)
```

---

## 🚀 Instalasi & Persiapan Lingkungan

### 1. Kloning Repositori (atau navigasi ke folder)
```bash
git clone https://github.com/<username-anda>/webinar-studi-kasus.git
cd webinar-studi-kasus
```

### 2. Buat & Aktifkan Virtual Environment (Disarankan)
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Pasang Dependensi
```bash
pip install -r requirements.txt
```

---

## 🔬 Rincian & Cara Menjalankan Masing-Masing Sesi

### 🔹 SESI 1: Patterns of Data with Analysis and AI
* **Pembicara:** Andrea Arisal (BRIN)
* **Topik:** Piramida analitik 4 tingkat (Descriptive, Diagnostic, Predictive, Prescriptive) untuk prediksi kelulusan tepat waktu mahasiswa.
* **Fitur Utama:**
  - Pembersihan data: deteksi nilai hilang, duplikat, dan pencilan metode IQR tanpa penghapusan otomatis (disimpan ke `outliers_review.csv` untuk ditinjau manusia).
  - Prediksi berbasis Logistic Regression dan Random Forest.
  - Aturan rekomendasi bimbingan akademik berdasarkan probabilitas risiko kelulusan.
* **Cara Menjalankan:**
  ```bash
  python sesi1_analitik/sesi1.py
  ```

---

### 🔹 SESI 2: Global Infrastructure to Address Local Challenge with HPC
* **Pembicara:** Dr. Worawan Diaz Carballo (Universitas Thammasat)
* **Topik:** Komputasi performa tinggi (HPC) untuk simulasi peringatan kualitas udara (PM2.5) menggunakan multi-processing core paralel.
* **Fitur Utama:**
  - Menjalankan puluhan skenario simulasi regresi.
  - Perbandingan eksekusi Serial vs Paralel (1, 2, 4, 8, 16 workers).
  - Pengukuran *Speedup* dan *Efisiensi* serta pembuktian Hukum Amdahl.
* **Cara Menjalankan:**
  ```bash
  python sesi2_hpc/sesi2.py
  ```

---

### 🔹 SESI 3: Career and Competition in Industrial Era with GIS
* **Pembicara:** Widi Primadepa (GIS Analyst, City Care Property, Selandia Baru)
* **Topik:** Pemodelan kerawanan banjir Kota Bengkulu menggunakan GIS sumber terbuka (GeoPandas, Folium) dan Machine Learning.
* **Fitur Utama:**
  - Pembuatan grid spasial 500 m dengan sistem koordinat WGS84 (EPSG:4326).
  - *Spatial block split* (mencegah kebocoran spasial / *spatial autocorrelation*).
  - Klasifikasi kerawanan (Rendah, Sedang, Tinggi) dan pembuatan peta interaktif (`sesi3_peta.html`).
* **Cara Menjalankan:**
  ```bash
  python sesi3_gis/sesi3.py
  ```
  *Buka `sesi3_gis/sesi3_peta.html` di peramban (browser) untuk melihat peta interaktif.*

---

### 🔹 SESI 4: Introduction to Machine Learning
* **Pembicara:** Taniza Tajudin
* **Topik:** Tiga pilar machine learning dalam satu studi kasus perkotaan.
* **Fitur Utama:**
  - **A. Supervised Learning (Regresi):** Prediksi harga tanah membandingkan Linear Regression, Random Forest, dan SVR dengan 5-fold cross validation.
  - **B. Unsupervised Learning (Clustering):** Klasterisasi profil kelurahan menggunakan K-Means, evaluasi *Silhouette Score* optimal, dan visualisasi PCA 2D.
  - **C. Anomaly Detection:** Deteksi transaksi anomali (1% outlier) menggunakan Isolation Forest.
* **Cara Menjalankan:**
  ```bash
  python sesi4_ml/sesi4.py
  ```

---

### 🔹 SESI 5: Interaksi Manusia-Komputer (IMK) dan Visualisasi Data
* **Topik:** Perancangan dasbor interaktif risiko bencana dengan prinsip Shneiderman (*Overview first, zoom & filter, details-on-demand*), aksesibilitas lansia, dan evaluasi usability testing.
* **Fitur Utama:**
  - Filter interaktif kecamatan, tanggal, dan tingkat risiko.
  - Mode aksesibilitas kontras tinggi dan font besar untuk lansia.
  - Grafik batang dengan sumbu mulai dari nol (jujur) dan skema warna ramah buta warna.
  - Unduh laporan dalam format CSV.
* **Cara Menjalankan Dasbor:**
  ```bash
  streamlit run sesi5_dashboard/app.py
  ```
* **Evaluasi Usability Testing:**
  ```bash
  python sesi5_dashboard/usability.py
  ```

---

### 🔹 SESI 6: Peringatan Deforestasi Hutan Tropis dengan Satelit dan AI
* **Pembicara:** Dr. Bart Slagter (Wageningen University and Research)
* **Topik:** Klasifikasi penyebab deforestasi (pertanian kecil, jalan, penebangan selektif, pertambangan) menggunakan tumpukan citra Sentinel-1 (radar) dan Sentinel-2 (optik).
* **Fitur Utama:**
  - Benchmark Random Forest (fitur statistik ringkasan) vs Convolutional Neural Network (PyTorch).
  - Mekanisme penolakan prediksi dengan ambang kepercayaan (*confidence thresholding* / kelas "Tidak Yakin").
  - *Learning curve* perbandingan performa model terhadap ketersediaan data latih berlabel.
* **Cara Menjalankan:**
  ```bash
  python sesi6_deforestasi/sesi6.py
  ```

---

## 💡 Keterkaitan Antar Sesi (Inspirasi Topik Tugas Akhir / Skripsi)

| Topik Riset / Skripsi | Sesi Terkait | Berkas Utama |
|---|---|---|
| Pemetaan Kerawanan Bencana Berbasis GIS & ML | Sesi 1, 3, 4 | `sesi3_gis/sesi3.py` |
| Klasifikasi Tutupan Lahan Sentinel Multi-Sensor (RF vs Deep CNN) | Sesi 3, 4, 6 | `sesi6_deforestasi/sesi6.py` |
| Sistem Informasi Eksekutif & Dasbor Bencana Berorientasi UX | Sesi 3, 5, 6 | `sesi5_dashboard/app.py` |
| Akselerasi Simulasi Prediksi Kualitas Lingkungan Berbasis HPC | Sesi 1, 2, 4 | `sesi2_hpc/sesi2.py` |
| Sistem Rekomendasi Intervensi Akademik Mahasiswa Berisiko | Sesi 1, 4, 5 | `sesi1_analitik/sesi1.py` + `app.py` |

---

## 🛡️ Catatan Etika & Reprodusibilitas
1. **Anonimisasi Data:** Semua data individu pada skrip ini menggunakan ID anonim sintetis. Pastikan data nyata telah memenuhi prinsip perlindungan data pribadi (PDP).
2. **Keterbatasan Model:** Model berbasis data sintetis ditujukan untuk demonstrasi alur kerja end-to-end. Pada implementasi skripsi, ganti variabel `X, y` dengan data primer/sekunder tervalidasi.
3. **Seed & Reprodusibilitas:** Semua generator bilangan acak dikunci menggunakan seed tetap (`random_state=42`) agar hasil eksperimen dapat diulang secara konsisten.

---

## 📤 Panduan Mengunggah ke GitHub

Untuk mengunggah seluruh proyek ini ke repositori GitHub Anda:

1. Buat repositori baru di [GitHub](https://github.com/new) (misalnya diberi nama `WEBINAR Practitioner Lecture`).
2. Jalankan perintah berikut di terminal:

```bash
# Inisialisasi git jika belum ada
git init

# Tambahkan semua berkas ke staging area
git add .

# Buat commit pertama
git commit -m "feat: implementasi lengkap 6 sesi webinar studi kasus data dan AI"

# Ubah nama branch utama menjadi main
git branch -M main

# Hubungkan dengan remote repository GitHub Anda (ganti URL dengan milik Anda)
git remote add origin https://github.com/<ReyhanMaulanaI>/<WEBINAR Practitioner Lecture>.git

# Unggah ke GitHub
git push -u origin main
```
