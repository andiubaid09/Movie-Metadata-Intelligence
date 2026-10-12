# 🎬 Movie Recommendation & Metadata Intelligence

Proyek Machine Learning yang mengeksplorasi sistem rekomendasi film, prediksi pendapatan, klasifikasi popularitas, dan segmentasi film menggunakan metadata film.

## 🎯 Gambaran Umum Proyek

Proyek ini bertujuan mengeksplorasi penerapan berbagai teknik Machine Learning pada metadata film untuk membangun sistem rekomendasi dan model prediktif.

Dengan menggunakan satu dataset metadata film yang berisi sekitar 45.000 data film, proyek ini mengembangkan beberapa pendekatan, mulai dari rekomendasi berbasis popularitas hingga Content-Based Filtering, supervised learning, dan unsupervised learning.

Setiap pendekatan dikembangkan dan dievaluasi secara terpisah untuk memahami metodologi, asumsi, kelebihan, keterbatasan, serta penerapannya.

Fokus proyek tidak hanya pada pembangunan model, tetapi juga pada proses preprocessing, evaluasi yang sesuai, analisis hasil eksperimen, dan dokumentasi keputusan teknis.

## 🎯 Tujuan Proyek

- Membangun sistem rekomendasi film berbasis popularitas sebagai baseline.
- Mengimplementasikan Content-Based Recommendation menggunakan TF-IDF dan cosine similarity.
- Memprediksi pendapatan film menggunakan metadata yang direncanakan tersedia sebelum film dirilis.
- Mengklasifikasikan film ke dalam kategori popularitas menggunakan supervised learning.
- Mengelompokkan film berdasarkan karakteristik metadata menggunakan unsupervised learning.
- Mengeksplorasi pendekatan hybrid content-based recommendation.
- Membandingkan performa model menggunakan metrik evaluasi yang sesuai.
- Mendokumentasikan proses preprocessing, eksperimen, hasil, dan keterbatasan setiap pendekatan.

## 📊 Dataset

**Nama dataset:** `movies_metadata.csv`

Dataset berisi sekitar 45.466 data film dengan 24 kolom yang mencakup informasi film, metadata produksi, pendapatan, dan metrik popularitas.

Beberapa fitur yang tersedia antara lain:

| Fitur | Deskripsi |
|---|---|
| `title` | Judul film |
| `genres` | Genre film |
| `overview` | Sinopsis film |
| `budget` | Anggaran produksi film |
| `revenue` | Pendapatan film yang tercatat |
| `runtime` | Durasi film |
| `release_date` | Tanggal rilis film |
| `original_language` | Bahasa asli film |
| `production_companies` | Metadata perusahaan produksi |
| `production_countries` | Metadata negara produksi |
| `popularity` | Skor popularitas yang tercatat |
| `vote_average` | Rata-rata penilaian pengguna |
| `vote_count` | Jumlah penilaian yang tercatat |

Dataset memiliki nilai kosong, tipe data yang tidak konsisten, dan beberapa kolom metadata dengan struktur yang memerlukan preprocessing.

### Kebijakan Penggunaan Data

- Menggunakan `movies_metadata.csv` sebagai satu-satunya dataset dalam proyek.
- Mempertahankan dataset asli dan melakukan preprocessing secara terpisah.
- Memeriksa nilai kosong, nol, dan tidak valid sebelum menentukan metode penanganannya.
- Tidak menggunakan informasi pascarilis sebagai fitur untuk eksperimen prediksi sebelum film dirilis.
- Mendokumentasikan asumsi, keputusan preprocessing, dan keterbatasan data.

## 🧠 Roadmap Proyek

| No. | Nama Notebook | Pendekatan | Status |
|---|---|---|---|
| 01 | `01_Popularity_Based_Movie_Recommender.ipynb` | Rekomendasi Berbasis Popularitas | ✅ Implementasi selesai |
| 02 | `02_Content_Based_Movie_Recommender_TFIDF.ipynb` | Content-Based Recommendation dengan TF-IDF | ✅ Implementasi selesai, pengujian kualitatif awal selesai |
| 03 | `03_Movie_Revenue_Prediction.ipynb` | Prediksi Pendapatan Film | 🚧 Direncanakan |
| 04 | `04_Movie_Popularity_Classification.ipynb` | Klasifikasi Popularitas Film | 🚧 Direncanakan |
| 05 | `05_Movie_Clustering_and_Segmentation.ipynb` | Clustering dan Segmentasi Film | 🚧 Direncanakan |
| 06 | `06_Hybrid_Content_Based_Recommendation.ipynb` | Hybrid Content-Based Recommendation | 🚧 Direncanakan |

Status akan diperbarui sesuai perkembangan implementasi dan hasil eksperimen yang benar-benar telah dilakukan.

## 🔎 Metodologi

### 01. Popularity-Based Movie Recommendation

Membangun baseline sistem rekomendasi dengan menggunakan informasi popularitas film yang tersedia.

Pendekatan ini menjadi titik awal untuk memahami sistem rekomendasi sebelum mengembangkan metode yang lebih kompleks.

### 02. Content-Based Movie Recommendation

Menggunakan TF-IDF dan cosine similarity untuk mengidentifikasi film yang memiliki kemiripan berdasarkan metadata dan deskripsi teks.

Fitur yang dieksplorasi mencakup genre, sinopsis, tagline, dan informasi koleksi film.

Pengujian kualitatif awal digunakan untuk memeriksa relevansi rekomendasi. Evaluasi kuantitatif akan dilakukan jika tersedia protokol evaluasi dan informasi relevansi yang sesuai.

### 03. Movie Revenue Prediction

Mengeksplorasi kemampuan model Machine Learning dalam memprediksi pendapatan film berdasarkan informasi yang direncanakan tersedia sebelum film dirilis.

Kandidat fitur meliputi:

- Anggaran produksi (`budget`).
- Genre film (`genres`).
- Bahasa asli (`original_language`).
- Bulan dan tahun rilis (`release_date`).
- Durasi film (`runtime`), jika sudah diketahui sebelum rilis.
- Metadata perusahaan dan negara produksi yang relevan.

Evaluasi menggunakan MAE, RMSE, dan R² dengan mempertimbangkan distribusi pendapatan film yang cenderung tidak merata.

Eksperimen juga akan mempertimbangkan transformasi target jika diperlukan dan membedakan metrik pada skala transformasi dengan metrik pada skala pendapatan asli.

### 04. Movie Popularity Classification

Mengembangkan model klasifikasi biner untuk mengategorikan film menjadi dua kelas:

- `0` — Tidak Populer (*Not Popular*).
- `1` — Populer (*Popular*).

Label dibuat berdasarkan skor `popularity` yang tercatat dalam dataset menggunakan ambang batas yang terdokumentasi, misalnya median dari data training.

Kandidat model meliputi:

- Logistic Regression.
- Decision Tree Classifier.
- Random Forest Classifier.
- HistGradientBoosting Classifier.

Fitur prediktor dibatasi pada informasi yang direncanakan tersedia sebelum film dirilis.

Skor `popularity` digunakan untuk membentuk label target, bukan sebagai fitur input. Informasi lain yang berpotensi baru tersedia setelah rilis, seperti statistik penilaian dan pendapatan aktual, juga tidak digunakan sebagai fitur prediktor dalam skenario ini.

Evaluasi menggunakan accuracy, precision, recall, F1-score, dan confusion matrix.

Kategori popularitas merupakan definisi operasional untuk eksperimen ini, bukan klasifikasi resmi dalam industri film.

### 05. Movie Clustering and Segmentation

Mengeksplorasi pengelompokan film berdasarkan karakteristik metadata dan representasi fitur yang sesuai.

Metode yang dapat dipertimbangkan meliputi K-Means dan algoritma clustering lainnya sesuai dengan karakteristik data serta kebutuhan komputasi.

Evaluasi dapat mencakup Silhouette Score, distribusi ukuran cluster, dan analisis kualitatif terhadap film-film dalam setiap kelompok.

Cluster yang dihasilkan akan diperlakukan sebagai kelompok berdasarkan pola data, bukan kategori film yang bersifat mutlak.

### 06. Hybrid Content-Based Recommendation

Mengeksplorasi penggabungan kemiripan konten dengan sinyal tambahan yang berasal dari metadata untuk menghasilkan peringkat rekomendasi.

Pendekatan hybrid akan dibandingkan dengan baseline content-based yang sudah ada apabila tersedia metode evaluasi yang sesuai.

Notebook ini tidak akan diklaim sebagai implementasi Collaborative Filtering atau Learning-to-Rank terlatih karena dataset tidak memiliki riwayat interaksi pengguna-film maupun label relevansi eksplisit yang diperlukan untuk pendekatan tersebut.

## 📈 Strategi Evaluasi

Metrik evaluasi dipilih berdasarkan tujuan masing-masing eksperimen.

| Tugas | Metrik Evaluasi |
|---|---|
| Prediksi pendapatan | MAE, RMSE, R² |
| Klasifikasi popularitas | Accuracy, Precision, Recall, F1-score, Confusion Matrix |
| Clustering film | Silhouette Score dan analisis kualitatif cluster |
| Sistem rekomendasi | Precision@K, Recall@K, NDCG@K, atau metrik ranking lain jika informasi relevansi tersedia |

Perbandingan model akan menggunakan pembagian data dan prosedur preprocessing yang konsisten jika sesuai dengan kebutuhan eksperimen.

Hasil kuantitatif tidak akan dicantumkan sebelum eksperimen benar-benar dijalankan.

## 🛠️ Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Matplotlib
- Seaborn
- Jupyter Notebook
- Google Colab

Library tambahan dapat digunakan jika diperlukan oleh eksperimen tertentu.

## 📁 Struktur Repository

```text
movie-recommendation-metadata-intelligence/
├── data/
│   └── raw/
│       └── movies_metadata.csv
├── notebooks/
│   ├── 01_Popularity_Based_Movie_Recommender.ipynb
│   ├── 02_Content_Based_Movie_Recommender_TFIDF.ipynb
│   ├── 03_Movie_Revenue_Prediction.ipynb
│   ├── 04_Movie_Popularity_Classification.ipynb
│   ├── 05_Movie_Clustering_and_Segmentation.ipynb
│   └── 06_Hybrid_Content_Based_Recommendation.ipynb
├── docs/
│   ├── 01_popularity_based.md
│   ├── 02_tfidf_content_based.md
│   ├── 02_content_based_deep_dive.md
│   ├── 03_revenue_prediction.md
│   ├── 04_movie_popularity_classification.md
│   ├── 05_movie_clustering.md
│   └── 06_hybrid_recommendation.md
├── src/
│   ├── preprocessing.py
│   ├── recommender.py
│   └── evaluation.py
├── models/
├── .gitignore
└── README.md
```

Struktur di atas merupakan struktur yang direncanakan. Folder dan file pendukung akan dibuat seiring dengan perkembangan implementasi.

Dataset asli akan dipertahankan tanpa perubahan. Data hasil preprocessing dan artefak model, jika diperlukan, akan disimpan secara terpisah.

## ⚠️ Keterbatasan Proyek

- Dataset tidak memiliki riwayat interaksi pengguna-film.
- Metadata saja mungkin tidak cukup untuk menjelaskan seluruh faktor yang memengaruhi pendapatan dan popularitas film.
- Data yang kosong, tidak konsisten, atau tidak valid dapat memengaruhi hasil model.
- Prediksi sebelum rilis bergantung pada ketersediaan setiap fitur pada waktu prediksi.
- Label popularitas yang berasal dari skor historis tidak membuktikan bahwa model mampu memprediksi permintaan penonton di dunia nyata secara akurat.
- Evaluasi kuantitatif sistem rekomendasi membutuhkan protokol evaluasi dan informasi relevansi yang sesuai.

## 📌 Status Proyek

**Status: Aktif Dikembangkan**

Notebook rekomendasi berbasis popularitas dan rekomendasi berbasis konten menggunakan TF-IDF telah diimplementasikan.

Pengujian kualitatif awal juga telah dilakukan pada pendekatan Content-Based Recommendation.

Tahap selanjutnya mencakup pengembangan prediksi pendapatan, klasifikasi popularitas, clustering, dan hybrid recommendation sesuai roadmap proyek.

## 💡 Catatan

Proyek ini dikembangkan sebagai portofolio Machine Learning yang berfokus pada sistem rekomendasi, pemodelan prediktif, dan analisis metadata.

Tujuan utamanya bukan sekadar membangun model, melainkan memahami asumsi, proses evaluasi, keterbatasan, dan alasan di balik pemilihan setiap pendekatan.

Setiap hasil eksperimen akan didokumentasikan berdasarkan implementasi dan evaluasi yang benar-benar dilakukan.