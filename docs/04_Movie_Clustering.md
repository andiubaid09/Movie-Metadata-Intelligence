# Clustering Profil Film Pra-Rilis

## 📌 Ringkasan Proyek

Proyek ini bertujuan membangun model segmentasi film menggunakan algoritma **K-Means Clustering** untuk memetakan karakteristik produksi dan potensi positioning komersial sebuah film sebelum dirilis (*pre-release*).

Untuk menghindari *data leakage*, model hanya menggunakan karakteristik yang tersedia pada tahap pra-rilis. Metrik pasca-rilis seperti `vote_average`, `vote_count`, dan `revenue` dikeluarkan dari fitur pemodelan.

Fitur yang digunakan meliputi:

- **Budget:** ditransformasi menggunakan `np.log1p()` dan nilai 0/NaN ditangani melalui imputasi median.
- **Belongs to Collection:** indikator keterkaitan film dengan waralaba (*franchise*).
- **Genres:** kategori genre film.
- **Production Companies:** perusahaan atau studio produksi.
- **Production Countries:** negara produksi film.

Model dikembangkan menggunakan `Pipeline` untuk mengintegrasikan preprocessing dan proses clustering ke dalam satu alur yang konsisten.

## 📊 1. Evaluasi dan Pemilihan Jumlah Cluster

Penentuan jumlah cluster (\(K\)) dilakukan melalui evaluasi **Silhouette Score** untuk membandingkan kualitas pemisahan antar-cluster pada beberapa kandidat nilai \(K\).

Berdasarkan hasil eksperimen dan grafik evaluasi, skor menunjukkan penurunan pada \(K=3\), kemudian kembali meningkat dan memperlihatkan titik lokal yang menonjol pada \(K=5\) dan \(K=6\).

Model dengan **\(K=6\)** dipilih karena memberikan keseimbangan antara kualitas pemisahan cluster secara matematis dan tingkat detail segmentasi yang dibutuhkan untuk analisis karakteristik film.

### Metrik Evaluasi Final — K = 6

| Metrik | Skor | Interpretasi |
|---|---:|---|
| Silhouette Score | **0.6220** | Menunjukkan pemisahan cluster yang cukup kuat berdasarkan struktur data. |
| Calinski-Harabasz Score | **27617.14** | Mengindikasikan variasi antar-cluster yang relatif tinggi dibandingkan variasi di dalam cluster. |
| Davies-Bouldin Score | **0.9895** | Memberikan ukuran kemiripan antar-cluster; nilai yang lebih rendah umumnya lebih baik. |

Ketiga metrik digunakan secara komplementer untuk mengevaluasi struktur cluster. Pemilihan \(K=6\) juga mempertimbangkan kebutuhan segmentasi agar karakteristik kelompok film dapat dianalisis secara lebih terperinci.

## 🌌 2. Visualisasi Reduksi Dimensi dengan PCA

Principal Component Analysis (PCA) digunakan untuk memproyeksikan data berdimensi tinggi ke ruang berdimensi lebih rendah agar struktur dan distribusi cluster lebih mudah diamati.

### PCA 2D Projection

![PCA 2 Dimensi](/docs/PCA%202D%20Clustering.png)

### PCA 3D Projection

![PCA 3 Dimensi](/docs/PCA%203D%20Clustering.png)

### Catatan Analisis PCA

Visualisasi 2D dan 3D memperlihatkan pola garis atau pita (*striations/bands*) pada distribusi titik data. Pola tersebut dapat muncul pada data berdimensi tinggi yang didominasi oleh fitur biner dan representasi hasil One-Hot Encoding dari genre, studio, serta negara produksi.

Struktur ini menunjukkan bahwa pola kombinasi fitur kategorikal berperan dalam membentuk distribusi data pada ruang hasil reduksi dimensi. Sementara itu, fitur numerik `budget_log` turut memberikan variasi berdasarkan skala anggaran produksi.

PCA pada bagian ini berfungsi sebagai alat bantu visualisasi. Hasil proyeksi dua atau tiga dimensi tidak sepenuhnya merepresentasikan seluruh informasi yang terdapat pada ruang fitur asli.

## 🎯 3. Profiling dan Analisis Cluster

Model K-Means dengan enam cluster menghasilkan segmentasi berdasarkan karakteristik anggaran, keterkaitan franchise, dan distribusi genre.

| Cluster | Profil Segmen | Jumlah Film | Karakteristik Utama | Top Genre Dominan |
|---:|---|---:|---|---|
| 0 | **General Mainstream (Mid-to-Low)** | 10.931 | Anggaran tergolong sangat rendah (mean: 0.016), dengan tingkat franchise moderat sebesar 9,8%. | Drama (50,4%), Comedy (28,5%), Thriller (14,6%) |
| 1 | **Studio Blockbuster & Commercial** | 8.602 | Memiliki rata-rata log-budget tertinggi (15,73) dan rasio franchise tertinggi, yaitu 18,3%. | Drama (47,0%), Comedy (31,5%), Thriller (27,2%), Action (24,2%) |
| 2 | **Indie Romance & Drama** | 3.445 | Anggaran minim atau tidak tercatat, dengan rasio franchise terendah untuk film fiksi (6,7%). | Drama (51,3%), Comedy (38,5%), Romance (28,8%) |
| 3 | **Foreign & Niche Intl. Drama** | 6.355 | Memiliki anggaran sangat kecil dan menonjol pada kategori pasar internasional. | Drama (55,5%), Comedy (27,4%), Foreign (8,8%) |
| 4 | **Documentary & Non-Fiction** | 6.043 | Memiliki rasio franchise terendah di seluruh kelompok (3,9%) dan menonjol pada konten non-fiksi. | Drama (26,2%), Documentary (20,8%), Comedy (19,4%) |
| 5 | **Genre Hybrid (Horror/Action)** | 10.087 | Anggaran relatif terbatas, dengan kombinasi genre hiburan yang beragam. | Drama (38,0%), Comedy (31,0%), Horror (13,9%) |

### Interpretasi Hasil Segmentasi

**Cluster 0 — General Mainstream (Mid-to-Low)**

Kelompok ini mencakup film dengan anggaran relatif rendah dan tingkat keterkaitan franchise yang moderat. Drama dan Comedy menjadi genre dominan, menunjukkan karakteristik konten yang cukup umum dalam katalog film.

**Cluster 1 — Studio Blockbuster & Commercial**

Cluster ini memiliki rata-rata `budget_log` tertinggi dan proporsi franchise terbesar. Karakteristik tersebut mengindikasikan kelompok film yang berorientasi pada skala produksi dan pengembangan waralaba.

**Cluster 2 — Indie Romance & Drama**

Kelompok ini ditandai oleh anggaran yang minim atau tidak tercatat serta dominasi Drama, Comedy, dan Romance. Profil ini memberikan gambaran segmen film dengan karakteristik produksi yang relatif sederhana.

**Cluster 3 — Foreign & Niche Intl. Drama**

Cluster ini memiliki anggaran relatif rendah dan proporsi genre Foreign yang menonjol. Hasil profiling mengindikasikan adanya kelompok film dengan karakteristik produksi yang lebih berorientasi pada pasar internasional.

**Cluster 4 — Documentary & Non-Fiction**

Cluster ini memiliki tingkat franchise paling rendah dan proporsi Documentary yang relatif tinggi dibandingkan beberapa kelompok lain. Hal tersebut mendukung identifikasi kelompok film yang berorientasi pada konten non-fiksi.

**Cluster 5 — Genre Hybrid (Horror/Action)**

Kelompok ini menunjukkan kombinasi genre yang beragam dengan proporsi Horror yang menonjol. Karakteristik tersebut memberikan gambaran segmen film hiburan dengan perpaduan elemen genre.

## 💡 4. Insight Bisnis dan Rekomendasi

### 4.1. Segmentasi Berdasarkan Skala Produksi

Perbandingan Cluster 1 dengan kelompok lainnya memperlihatkan perbedaan yang jelas dalam rata-rata log-budget dan keterkaitan franchise.

Cluster 1 dapat diposisikan sebagai segmen **Studio Blockbuster & Commercial**, sedangkan cluster lain menggambarkan kelompok film dengan karakteristik produksi dan konten yang berbeda. Segmentasi ini dapat membantu rumah produksi memahami posisi relatif sebuah proyek berdasarkan profil produksinya.

### 4.2. Identifikasi Spesialisasi Konten

Hasil clustering juga memperlihatkan diferensiasi karakteristik konten. Cluster 4 menonjol pada genre Documentary, sedangkan Cluster 3 memiliki proporsi genre Foreign yang lebih menonjol.

Pola tersebut menunjukkan bahwa K-Means tidak hanya membentuk kelompok berdasarkan karakteristik anggaran, tetapi juga berdasarkan kombinasi fitur produksi dan genre yang digunakan dalam model.

### 4.3. Potensi Penerapan Model

Pipeline clustering yang telah dikembangkan berpotensi digunakan sebagai alat bantu pemetaan proyek film pada tahap pra-rilis.

Dengan memasukkan estimasi budget, genre, perusahaan produksi, negara produksi, dan informasi franchise, sebuah proyek baru dapat dipetakan ke cluster yang paling sesuai berdasarkan model yang telah dilatih.

Hasil tersebut dapat menjadi bahan pertimbangan awal bagi tim produksi atau pemasaran dalam memahami karakteristik proyek dan menentukan positioning yang relevan. Penggunaan pada proyek baru tetap memerlukan konsistensi preprocessing serta validasi terhadap karakteristik data yang digunakan.

## 🛠️ Teknologi yang Digunakan

- **Python** — bahasa pemrograman.
- **Pandas & NumPy** — manipulasi dan preprocessing data.
- **Scikit-learn** — `Pipeline`, `ColumnTransformer`, preprocessing, K-Means, PCA, dan metrik evaluasi clustering.
- **Matplotlib** — visualisasi distribusi dan hasil reduksi dimensi.

## 🏁 Kesimpulan

Eksperimen ini menghasilkan model K-Means dengan enam cluster untuk melakukan segmentasi profil film berdasarkan karakteristik pra-rilis. Evaluasi menggunakan Silhouette Score, Calinski-Harabasz Score, dan Davies-Bouldin Score memberikan perspektif yang saling melengkapi terhadap struktur hasil clustering.

Melalui profiling cluster, model menghasilkan enam kelompok dengan karakteristik anggaran, keterkaitan franchise, dan distribusi genre yang berbeda. Hasil tersebut dapat menjadi dasar eksplorasi lebih lanjut untuk memahami segmentasi film dan potensi penerapan clustering sebagai alat bantu pengambilan keputusan pada tahap pra-rilis.