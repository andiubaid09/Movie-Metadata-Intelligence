# Hybrid Content-Based Movie Recommendation

## 1. Deskripsi Proyek

Notebook ini merupakan bagian dari proyek **Movie Recommender System** yang menggunakan metadata film untuk menghasilkan rekomendasi berdasarkan kemiripan konten.

Sistem menerima judul film sebagai masukan, kemudian mengurutkan film lain berdasarkan gabungan beberapa sinyal kemiripan, yaitu deskripsi cerita, tagline, genre, koleksi film, dan perusahaan produksi.

Notebook ini menggunakan pendekatan **Hybrid Content-Based Recommendation**, yaitu menggabungkan beberapa metode kemiripan konten dan metadata dalam satu skor rekomendasi.

## 2. Tujuan

Tujuan pengembangan notebook ini adalah:

- Menghasilkan rekomendasi film berdasarkan kemiripan konten dan metadata.
- Menggabungkan beberapa sinyal kemiripan agar proses rekomendasi tidak hanya bergantung pada satu fitur.
- Menggunakan bobot untuk mengatur kontribusi setiap fitur terhadap skor akhir.
- Mempelajari penerapan sistem rekomendasi berbasis konten menggunakan Python dan Scikit-learn.

## 3. Dataset

Dataset yang digunakan adalah `movies_metadata.csv`, yang berisi metadata film.

Fitur yang digunakan dalam rancangan sistem ini meliputi:

| Fitur | Fungsi |
|---|---|
| `title` | Mencari film masukan dan menampilkan judul rekomendasi |
| `overview` | Membandingkan kemiripan deskripsi cerita |
| `tagline` | Membandingkan kemiripan teks tagline |
| `genres` | Membandingkan genre film |
| `belongs_to_collection` | Memberikan sinyal kemiripan berdasarkan koleksi film |
| `production_companies` | Membandingkan perusahaan produksi |

Judul film digunakan untuk mencari film masukan, bukan sebagai sinyal utama dalam penghitungan skor kemiripan.

## 4. Metode yang Digunakan

### TF-IDF dan Cosine Similarity

TF-IDF digunakan untuk mengubah teks `overview` dan `tagline` menjadi representasi numerik. Cosine similarity digunakan untuk menghitung kemiripan antara representasi teks film masukan dan film kandidat.

### Jaccard Similarity

Jaccard similarity digunakan untuk membandingkan himpunan genre dan perusahaan produksi. Metode ini menghitung tingkat kesamaan anggota di antara dua himpunan.

### Kemiripan Koleksi Film

Fitur `belongs_to_collection` digunakan sebagai sinyal tambahan melalui fungsi kemiripan koleksi.

### Penggabungan Skor Hybrid

Skor akhir dirancang sebagai gabungan berbobot dari seluruh sinyal kemiripan.

Bobot yang digunakan dalam versi kode yang dibahas:

| Fitur | Bobot |
|---|---:|
| Overview | 0.35 |
| Tagline | 0.10 |
| Genres | 0.25 |
| Collection | 0.10 |
| Production Companies | 0.20 |
| **Total** | **1.00** |

Bobot tersebut merupakan konfigurasi yang digunakan dalam implementasi yang dibahas, bukan hasil optimasi yang telah terbukti.

## 5. Alur Sistem

Secara umum, alur rekomendasi adalah:

1. Pengguna memasukkan judul film.
2. Sistem mencari film berdasarkan judul.
3. Sistem menghitung kemiripan teks dan metadata terhadap film lain.
4. Setiap skor kemiripan dikalikan dengan bobotnya.
5. Seluruh skor digabungkan menjadi skor hybrid.
6. Film masukan dikeluarkan dari kandidat rekomendasi.
7. Kandidat diurutkan berdasarkan skor tertinggi.
8. Sistem menampilkan daftar rekomendasi beserta skor dan informasi film.

Hasil rekomendasi dirancang untuk menampilkan judul film, skor hybrid, genre, dan deskripsi cerita.

## 6. Pengujian

Pengujian yang telah dibahas menggunakan tiga judul film:

- The Dark Knight
- Toy Story
- Interstellar

Pada salah satu hasil pengujian `Toy Story`, film `Toy Story 2` dan `Toy Story 3` muncul dengan skor kemiripan tertinggi. Hasil tersebut terlihat masuk akal secara kualitatif karena keduanya merupakan sekuel dari film masukan.

Namun, selama proses debugging, hasil pengujian `The Dark Knight` dan `Interstellar` juga sempat menampilkan film masukan itu sendiri dalam daftar rekomendasi. Hal ini menunjukkan bahwa implementasi yang menghasilkan keluaran tersebut belum sepenuhnya sesuai dengan tujuan sistem.

Karena itu, hasil pengujian yang tersedia belum dapat dianggap sebagai evaluasi final dari sistem rekomendasi.

## 7. Evaluasi dan Keterbatasan

Evaluasi yang dibahas masih bersifat kualitatif dan terbatas pada beberapa contoh film.

Keterbatasan yang diketahui:

- Relevansi rekomendasi belum tervalidasi secara menyeluruh.
- Film masukan sempat muncul kembali dalam daftar rekomendasi.
- Keselarasan urutan baris antara data film dan struktur fitur perlu dijaga.
- Bobot kemiripan belum dibuktikan sebagai konfigurasi terbaik.
- Belum ada hasil evaluasi kuantitatif yang tercatat menggunakan metrik seperti Precision@K, Recall@K, atau NDCG@K.
- Belum ada dasar untuk menyatakan bahwa pendekatan hybrid ini lebih unggul daripada metode rekomendasi lain.

Skor hybrid merupakan skor kemiripan berdasarkan fitur dan bobot yang digunakan, bukan probabilitas bahwa pengguna akan menyukai film tersebut.

## 8. Penyimpanan Model

Implementasi yang dibahas merupakan sistem rekomendasi berbasis kemiripan dan tidak menunjukkan proses pelatihan model prediktif menggunakan `fit()`.

Komponen seperti vectorizer TF-IDF, matriks fitur, metadata film, dan kode rekomendasi dapat disimpan apabila diperlukan untuk menjalankan sistem tanpa membangun ulang seluruh representasi fitur. Namun, belum ada catatan bahwa komponen tersebut telah diekspor menjadi file.

Oleh karena itu, notebook ini tidak diklaim menghasilkan file model Machine Learning atau Deep Learning terlatih.

## 9. Kesimpulan

Notebook ini mengembangkan rancangan sistem rekomendasi film berbasis konten dengan menggabungkan kemiripan teks dan metadata melalui skor berbobot.

Beberapa hasil rekomendasi terlihat masuk akal secara kualitatif, tetapi masih terdapat masalah implementasi dan belum tersedia evaluasi menyeluruh untuk memastikan kualitas rekomendasi.

Notebook ini menjadi tahap eksplorasi penerapan metode hybrid content-based recommendation dalam proyek Movie Recommender System. Pengembangan lebih lanjut perlu berfokus pada konsistensi implementasi, relevansi rekomendasi, dan evaluasi yang dapat dipertanggungjawabkan.
