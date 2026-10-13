# 03 — Movie Revenue Prediction

## 1. Gambaran Umum

Notebook ini membahas eksperimen prediksi pendapatan film (*movie revenue prediction*) menggunakan dataset `movies_metadata.csv`.

Tujuan eksperimen adalah membangun model regresi yang memperkirakan nilai pendapatan film berdasarkan metadata yang tersedia, seperti anggaran produksi, genre, tahun dan waktu rilis, bahasa, perusahaan produksi, negara produksi, serta informasi koleksi film.

Eksperimen membandingkan tiga algoritma:

1. Ridge Regression
2. Random Forest Regressor
3. XGBoost Regressor

Pencarian hyperparameter dilakukan menggunakan `GridSearchCV` dengan **R² sebagai metrik scoring**. Evaluasi akhir dilakukan pada data pengujian dengan proporsi pembagian data sebesar 80:20.

> **Catatan:** Notebook ini menguji kemampuan prediksi berdasarkan fitur yang dipilih. Hasilnya tidak membuktikan hubungan sebab-akibat antara suatu fitur dan pendapatan film.

## 2. Tujuan Eksperimen

Tujuan utama eksperimen ini adalah:

- Membangun model regresi untuk memperkirakan `revenue`.
- Membandingkan performa Ridge Regression, Random Forest, dan XGBoost.
- Mencari kombinasi hyperparameter yang memberikan skor R² terbaik pada proses cross-validation.
- Mengukur kesalahan prediksi menggunakan MAE, MSE, dan RMSE.
- Mengamati fitur yang paling banyak digunakan atau dianggap penting oleh model berbasis pohon.
- Mengidentifikasi keterbatasan model dan peluang perbaikan pada eksperimen berikutnya.

## 3. Dataset dan Target Prediksi

Dataset yang digunakan adalah `movies_metadata.csv`, yang berisi metadata film.

Variabel target:

- `revenue`: pendapatan film yang hendak diprediksi.

Target tetap menggunakan **skala revenue asli**. Transformasi logaritmik terhadap target, seperti `np.log1p(revenue)`, tidak digunakan pada eksperimen yang didokumentasikan ini.

Baris yang memiliki nilai target `revenue` kosong dihapus sebelum pelatihan, berdasarkan informasi preprocessing yang telah disampaikan.

Nilai `budget` sebesar nol dipertahankan. Artinya, nilai tersebut tidak otomatis dianggap sebagai data hilang dan tidak otomatis diganti dengan hasil imputasi.

### Hal yang belum diketahui

- Jumlah baris akhir setelah preprocessing.
- Rentang nilai minimum dan maksimum `revenue`.
- Jumlah nilai `revenue` yang dihapus secara tepat.
- Distribusi target dan proporsi film dengan revenue nol.
- Apakah nilai `budget` nol merupakan nilai sebenarnya, nilai yang tidak tercatat, atau gabungan keduanya.

Hal-hal tersebut perlu diperiksa pada notebook sebelum dokumentasi menyertakan angka atau kesimpulan tambahan.

## 4. Fitur yang Digunakan

Fitur yang dilaporkan digunakan dalam eksperimen meliputi:

| Kelompok | Fitur |
|---|---|
| Anggaran dan koleksi | `budget`, `belongs_to_collection` |
| Genre | `Action`, `Adventure`, `Animation`, `Comedy`, `Crime`, `Documentary`, `Drama`, `Family`, `Fantasy`, `Foreign`, `History`, `Horror`, `Music`, `Mystery`, `Romance`, `Science Fiction`, `TV Movie`, `Thriller`, `War`, `Western` |
| Waktu rilis | `release_year`, `release_month`, `release_dayofweek` |
| Bahasa | `lang_clean` |
| Perusahaan produksi | `company_clean` |
| Negara produksi | `countries_clean` |

### Penjelasan fitur

**`budget`**

Anggaran produksi film. Fitur ini menjadi salah satu fitur dengan tingkat kepentingan tertinggi pada hasil Random Forest dan XGBoost.

**`belongs_to_collection`**

Indikator biner yang menunjukkan apakah film termasuk dalam sebuah koleksi film. Berdasarkan preprocessing yang dilaporkan, fitur ini direpresentasikan menggunakan nilai 0 dan 1.

**Genre**

Genre film direpresentasikan menggunakan kolom-kolom indikator. Sebagai contoh, sebuah film bergenre Action dan Adventure dapat memiliki nilai 1 pada kedua kolom tersebut.

**`release_year`, `release_month`, dan `release_dayofweek`**

Komponen tanggal rilis yang digunakan sebagai fitur numerik. Tujuannya adalah memberikan informasi mengenai waktu perilisan film kepada model.

**`lang_clean`, `company_clean`, dan `countries_clean`**

Fitur kategorikal yang telah melalui proses pembersihan. Kategori-kategori ini kemudian diproses menggunakan transformasi kategorikal pada pipeline.

### Catatan tentang pemilihan fitur

Daftar di atas adalah fitur yang dilaporkan digunakan dalam eksperimen. Daftar tersebut belum memastikan jumlah akhir kolom setelah seluruh transformasi, karena encoding kategorikal dapat menghasilkan beberapa kolom baru.

Belum ada informasi yang cukup untuk memastikan apakah fitur tambahan seperti `release_date_missing` ikut digunakan dalam model akhir. Keberadaan fitur tersebut perlu diverifikasi langsung pada notebook.

## 5. Preprocessing dan Pipeline

Preprocessing dilakukan menggunakan `ColumnTransformer` dan `Pipeline` dari scikit-learn.

Struktur kode yang dilaporkan:

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('log', log_transform),
        ('scaler', num_transform)
    ]), num_feat),
    ('cat', cat_transform, cat_feat)
], remainder='passthrough',
   verbose_feature_names_out=False)
```

### Penjelasan struktur pipeline

**1. Pipeline fitur numerik**

```python
Pipeline([
    ('log', log_transform),
    ('scaler', num_transform)
])
```

Pipeline ini menerapkan dua tahap terhadap kolom yang termasuk dalam `num_feat`:

- `log_transform`: tahap transformasi yang namanya menunjukkan transformasi logaritmik.
- `num_transform`: tahap transformasi numerik yang digunakan setelahnya; berdasarkan keterangan eksperimen, tahap ini berkaitan dengan scaling.

**Penting:** Implementasi dan konfigurasi aktual kedua objek tersebut belum tersedia dalam dokumentasi ini. Oleh karena itu, jenis transformasi log, penanganan nilai nol, dan metode scaling yang digunakan tidak dapat dipastikan hanya dari nama variabel.

Transformasi fitur numerik tersebut juga tidak berarti bahwa target `revenue` ikut ditransformasikan.

**2. Pipeline fitur kategorikal**

```python
('cat', cat_transform, cat_feat)
```

Kolom yang termasuk dalam `cat_feat` diproses menggunakan `cat_transform`. Berdasarkan informasi eksperimen, fitur kategorikal menggunakan One-Hot Encoding.

Jenis encoder, penanganan kategori yang belum pernah terlihat, dan pengaturan parameter encoder belum dikonfirmasi.

**3. `remainder='passthrough'`**

Kolom yang tidak tercantum secara eksplisit pada kelompok transformasi numerik atau kategorikal akan diteruskan tanpa transformasi tambahan oleh `ColumnTransformer`.

Penggunaan opsi ini membuat pemeriksaan daftar kolom menjadi penting, agar tidak ada fitur yang diteruskan secara tidak sengaja atau mengalami preprocessing yang tidak sesuai.

**4. `verbose_feature_names_out=False`**

Opsi ini mengatur penamaan fitur keluaran dari `ColumnTransformer` agar tidak selalu diberi awalan nama transformer.

Hal ini dapat mempermudah pembacaan nama fitur hasil transformasi, tetapi tidak mengubah algoritma regresi yang digunakan.

### Hal yang perlu diverifikasi

- Implementasi `log_transform` dan `num_transform`.
- Konfigurasi `cat_transform`.
- Apakah seluruh preprocessing dipasang di dalam pipeline yang digunakan oleh `GridSearchCV`.
- Apakah jumlah dan susunan fitur hasil transformasi sama pada seluruh model.

Preprocessing yang dipelajari dari data harus di-*fit* hanya pada data pelatihan di setiap fold cross-validation. Menempatkan preprocessing di dalam pipeline merupakan cara yang tepat untuk membantu menghindari kebocoran informasi, selama pipeline tersebut digunakan secara konsisten.

## 6. Pembagian Data dan Evaluasi

Data dibagi dengan rasio **80:20**, yaitu 80% untuk data pelatihan dan 20% untuk data pengujian.

`GridSearchCV` digunakan untuk mencari kombinasi hyperparameter dengan skor R² cross-validation terbaik.

### Metrik evaluasi

| Metrik | Arti |
|---|---|
| MAE | Rata-rata nilai absolut selisih antara prediksi dan nilai aktual. |
| MSE | Rata-rata kuadrat selisih antara prediksi dan nilai aktual. |
| RMSE | Akar kuadrat MSE, sehingga kembali ke satuan target. |
| R² | Mengukur kemampuan model menjelaskan variasi target dibandingkan baseline berupa prediksi rata-rata target. |

Interpretasi R²:

- R² = 1: prediksi cocok sempurna dengan nilai aktual pada data evaluasi.
- R² = 0: performa setara dengan baseline prediksi rata-rata target, berdasarkan definisi standar R².
- R² < 0: model memiliki performa lebih buruk daripada baseline tersebut pada data evaluasi.

Karena target menggunakan skala revenue asli, MAE dan RMSE juga menggunakan satuan revenue pada dataset. MSE menggunakan satuan revenue kuadrat.

**Informasi yang belum diketahui:** jumlah fold cross-validation, metode pengacakan data, `random_state`, dan apakah seluruh model menggunakan pembagian data serta preprocessing yang persis sama.

## 7. Eksperimen Ridge Regression

Ridge Regression merupakan model regresi linear yang menggunakan regularisasi L2 untuk membatasi besarnya koefisien model.

Hyperparameter yang dicari adalah `alpha`.

### Grid hyperparameter

```python
param_grid = {
    'model__regressor__alpha': [
        0.1, 1.0, 10.0, 100.0, 500.0, 1000.0,
        2000.0, 5000.0, 10000.0, 20000.0,
        50000.0, 70000.0, 100000.0
    ]
}
```

Hasil pencarian:

- Hyperparameter terbaik: `alpha = 0.1`
- Best CV R²: `-0.0157930709`

### Hasil evaluasi data pengujian

| Metrik | Hasil |
|---|---:|
| MAE | 12,372,935.80 |
| MSE | 4,816,187,994,342,705.00 |
| RMSE | 69,398,760.76 |
| R² | -0.0178 |

![Prediksi vs Aktual Ridge Regression](/docs/Prediksi%20vs%20Aktual%20Ridge%20Regression.png)<br>

### Analisis

Ridge Regression menghasilkan R² negatif pada cross-validation dan data pengujian. Pada data pengujian, skor tersebut menunjukkan bahwa model belum mengungguli baseline prediksi rata-rata target.

Nilai MAE dan RMSE juga menunjukkan kesalahan prediksi yang cukup besar dalam satuan revenue dataset.

Salah satu kemungkinan yang patut diuji adalah apakah hubungan antara fitur dan target cukup kompleks untuk ditangkap oleh model linear. Namun, hasil ini belum membuktikan bahwa kelemahan model semata-mata disebabkan oleh hubungan nonlinier. Distribusi target, kualitas fitur, penanganan nilai nol, dan pengaturan preprocessing juga perlu diperiksa.

## 8. Eksperimen Random Forest Regressor

Random Forest adalah metode ensemble yang menggabungkan prediksi dari sejumlah decision tree. Model ini dapat mempelajari hubungan nonlinier dan interaksi fitur tanpa mensyaratkan hubungan linear sederhana antara setiap fitur dan target.

### Grid hyperparameter

```python
params_grid = {
    'model__regressor__n_estimators': [50, 100, 200],
    'model__regressor__max_depth': [10, 20, 50, 100],
    'model__regressor__min_samples_split': [2, 5, 10]
}
```

Hasil pencarian:

- `n_estimators = 50`
- `max_depth = 20`
- `min_samples_split = 2`
- Best CV R²: `0.5418680294`

### Hasil evaluasi data pengujian

| Metrik | Hasil |
|---|---:|
| MAE | 8,900,251.07 |
| MSE | 2,083,478,721,583,889.25 |
| RMSE | 45,645,139.08 |
| R² | 0.5597 |

![Prediksi vs Aktual](/docs/Prediksi%20vs%20Aktual%20RandomForest.png)<br>

### Analisis

Random Forest memberikan hasil terbaik di antara ketiga model yang dilaporkan berdasarkan R² data pengujian.

R² sebesar 0.5597 menunjukkan bahwa model menjelaskan sekitar 55.97% variasi target pada data pengujian menurut definisi R². Angka ini bukan berarti model memprediksi 55.97% film dengan benar, karena tugas yang dilakukan adalah regresi, bukan klasifikasi.

MAE sekitar 8.90 juta dan RMSE sekitar 45.65 juta menunjukkan bahwa besarnya kesalahan sangat dipengaruhi oleh skala revenue. RMSE yang jauh lebih tinggi daripada MAE dapat menjadi alasan untuk memeriksa apakah terdapat sebagian prediksi dengan kesalahan yang sangat besar. Namun, distribusi residual harus diperiksa sebelum menyimpulkan penyebabnya.

## 9. Feature Importance Random Forest

![Fitur Importances Random Forest](/docs/Top%2010%20Fitur%20Random%20Forest.png)<br>

Sepuluh fitur teratas berdasarkan feature importance yang dilaporkan:

| Fitur | Importance |
|---|---:|
| `budget` | 0.574913 |
| `release_year` | 0.099828 |
| `release_month` | 0.059312 |
| `release_dayofweek` | 0.037236 |
| `company_clean_other` | 0.012505 |
| `belongs_to_collection` | 0.012367 |
| `Comedy` | 0.012269 |
| `Drama` | 0.012098 |
| `Romance` | 0.010481 |
| `countries_clean_United States of America` | 0.010240 |

### Interpretasi

**`budget` menjadi fitur dominan.**

Importance `budget` sebesar 0.574913 menunjukkan bahwa fitur tersebut memiliki kontribusi terbesar terhadap pembagian node dalam model Random Forest, menurut metode importance yang digunakan.

**Fitur waktu rilis juga menonjol.**

`release_year`, `release_month`, dan `release_dayofweek` masuk dalam sepuluh fitur teratas. Hasil ini menunjukkan bahwa model menggunakan informasi waktu rilis untuk membuat prediksi.

**Genre, koleksi, perusahaan, dan negara produksi juga berkontribusi.**

Fitur-fitur tersebut memiliki importance lebih rendah dibandingkan `budget` dan fitur waktu rilis pada daftar yang tersedia.

Feature importance bukan bukti hubungan sebab-akibat. Importance juga tidak menjelaskan secara langsung apakah peningkatan suatu fitur akan menaikkan atau menurunkan prediksi revenue.

## 10. Eksperimen XGBoost Regressor

XGBoost merupakan implementasi gradient boosting berbasis decision tree. Model ini membangun pohon secara bertahap untuk memperbaiki prediksi ensemble.

### Grid hyperparameter yang dilaporkan

```python
params_grid = {
    'model__regressor__n_estimators': [50, 100, 200, 300],
    'model__regressor__max_depth': [3, 10, 50, 100],
    'model__regressor__min_samples_split': [2, 5, 10, 20]
}
```

Hasil pencarian yang dilaporkan:

- `n_estimators = 50`
- `max_depth = 50`
- `min_samples_split = 2`
- Best CV R²: `0.5062665658`

**Catatan teknis tentang parameter:** `min_samples_split` merupakan parameter standar pada estimator seperti `DecisionTreeRegressor` dan `RandomForestRegressor`, tetapi bukan parameter standar `XGBRegressor` dari paket XGBoost. Parameter ini dicatat sesuai eksperimen yang dilaporkan, tetapi konfigurasi estimator dan hasil `GridSearchCV` perlu diperiksa untuk memastikan apakah parameter tersebut diterima dan benar-benar memengaruhi model. Jangan menganggapnya sebagai faktor yang telah terbukti memengaruhi hasil sebelum verifikasi.

Nilai `max_depth = 50` juga relatif tinggi untuk konfigurasi XGBoost yang lazim digunakan. Nilai tersebut tidak otomatis berarti model salah, tetapi layak dievaluasi dari sisi waktu pelatihan, kompleksitas, dan risiko overfitting.

### Hasil evaluasi data pengujian

| Metrik | Hasil |
|---|---:|
| MAE | 8,969,468.03 |
| MSE | 2,166,366,158,811,248.75 |
| RMSE | 46,544,238.73 |
| R² | 0.5422 |

![Prediksi vs Aktual XGBoost](/docs/Prediksi%20vs%20Aktual%20XGBoost.png)<br>

### Analisis

XGBoost menghasilkan R² positif sebesar 0.5422 pada data pengujian. Model ini lebih baik daripada Ridge Regression berdasarkan metrik yang dilaporkan, tetapi hasilnya sedikit di bawah Random Forest.

MAE dan RMSE XGBoost juga sedikit lebih tinggi daripada Random Forest pada hasil eksperimen ini. Dengan demikian, berdasarkan angka yang tersedia, Random Forest memberikan hasil pengujian terbaik dari tiga model tersebut.

Hasil ini tidak membuktikan bahwa Random Forest selalu lebih baik daripada XGBoost. Kesimpulan tersebut terbatas pada konfigurasi, data, dan evaluasi yang dilaporkan.

## 11. Feature Importance XGBoost

![Top Fitur XGBoost](/docs/Top%2010%20Fitur%20XGBoost.png)<br>

Sepuluh fitur teratas berdasarkan hasil yang dilaporkan:

| Fitur | Importance |
|---|---:|
| `budget` | 0.158071 |
| `company_clean_other` | 0.074482 |
| `company_clean_Unknown_company` | 0.069425 |
| `countries_clean_India` | 0.044895 |
| `TV Movie` | 0.032256 |
| `countries_clean_United States of America` | 0.024343 |
| `company_clean_RKO Radio Pictures` | 0.020181 |
| `countries_clean_Russia` | 0.017700 |
| `company_clean_United Artists` | 0.017446 |
| `countries_clean_Japan` | 0.017231 |

### Interpretasi

`budget` kembali menjadi fitur dengan importance tertinggi pada daftar yang diberikan. Beberapa fitur perusahaan dan negara produksi juga muncul di antara sepuluh fitur teratas.

Urutan importance berbeda dari Random Forest. Hal ini wajar karena kedua algoritma membangun ensemble dengan mekanisme yang berbeda. Nilai importance antarmodel juga tidak selalu dapat dibandingkan secara langsung tanpa memastikan definisi dan metode perhitungan importance yang digunakan.

Kategori seperti `other` atau `Unknown_company` tidak boleh langsung ditafsirkan sebagai karakteristik industri yang menyebabkan revenue tertentu. Kategori tersebut bisa mencerminkan strategi pengelompokan kategori langka atau nilai yang tidak diketahui. Untuk memahami maknanya, definisi preprocessing dan distribusi datanya perlu diperiksa.

## 12. Perbandingan Hasil Ketiga Model

| Model | Best CV R² | MAE | RMSE | Test R² |
|---|---:|---:|---:|---:|
| Ridge Regression | -0.0158 | 12,372,935.80 | 69,398,760.76 | -0.0178 |
| Random Forest | 0.5419 | 8,900,251.07 | 45,645,139.08 | 0.5597 |
| XGBoost | 0.5063 | 8,969,468.03 | 46,544,238.73 | 0.5422 |

*Semua metrik merupakan hasil eksperimen yang dilaporkan. Perbandingan mengasumsikan bahwa data pengujian dan prosedur evaluasi ketiga model memang sama; kesamaan tersebut tetap perlu dikonfirmasi dari notebook.*

### Temuan utama

1. **Random Forest memberikan hasil terbaik pada data pengujian dalam eksperimen ini.** Model tersebut memperoleh R² tertinggi, MAE terendah, dan RMSE terendah di antara ketiga model.
2. **XGBoost berada sedikit di bawah Random Forest.** Selisih hasilnya relatif kecil dibandingkan perbedaan terhadap Ridge Regression.
3. **Ridge Regression belum memberikan performa yang memadai.** R² negatif menunjukkan bahwa model kalah dari baseline prediksi rata-rata target pada data pengujian.
4. **`budget` menonjol pada kedua model berbasis pohon.** Fitur ini menjadi feature importance tertinggi dalam hasil Random Forest dan XGBoost yang dilaporkan.
5. **Hasil terbaik belum berarti model siap digunakan.** Validasi tambahan, pemeriksaan kesalahan prediksi, dan verifikasi preprocessing masih diperlukan sebelum membuat klaim tentang kemampuan generalisasi.

## 13. Keterbatasan Eksperimen

### 13.1 Target revenue menggunakan skala asli

Revenue pada skala asli dapat memiliki rentang nilai yang lebar. Jika distribusi target sangat menceng atau memiliki nilai ekstrem, sejumlah kesalahan prediksi besar dapat sangat memengaruhi MSE dan RMSE.

Distribusi target belum disertakan dalam informasi eksperimen ini. Oleh karena itu, pengaruh nilai ekstrem masih merupakan hal yang perlu diuji, bukan kesimpulan yang sudah terbukti.

Eksperimen lanjutan dapat membandingkan model dengan target `revenue` asli dan target `np.log1p(revenue)`. Jika menggunakan transformasi log pada target, prediksi perlu dikembalikan ke skala asli sebelum dibandingkan dengan metrik revenue asli. Hasil kedua pendekatan sebaiknya dievaluasi menggunakan data pengujian yang sama.

### 13.2 Makna `budget = 0` belum dipastikan

Nilai nol dipertahankan dalam eksperimen ini. Keputusan tersebut sesuai dengan preprocessing yang dilaporkan, tetapi makna nol belum dipastikan.

Jika nol merepresentasikan data yang tidak tercatat, model dapat mempelajari pola yang berasal dari proses pencatatan, bukan hanya karakteristik film. Hal ini perlu diteliti melalui pemeriksaan data, bukan dengan langsung mengasumsikan semua nol sebagai missing value.

### 13.3 Keterbatasan metadata

Prediksi hanya memanfaatkan fitur yang tersedia dan dipilih dari dataset. Informasi yang tidak tersedia pada dataset tidak dapat dimasukkan tanpa sumber data tambahan.

Akibatnya, kesalahan prediksi bisa terkait dengan informasi yang tidak terwakili dalam fitur, tetapi kontribusi masing-masing faktor belum diukur dalam eksperimen ini.

### 13.4 Evaluasi generalisasi belum lengkap

Hasil yang tersedia menggunakan satu pembagian data pengujian 80:20. Nilai performa dapat berubah jika pembagian data berubah.

Belum tersedia hasil dari repeated split, cross-validation independen pada konfigurasi final, atau pengujian pada dataset eksternal. Karena itu, stabilitas performa di luar pembagian data yang digunakan belum dapat dipastikan.

### 13.5 Interpretasi feature importance terbatas

Feature importance menjelaskan kepentingan relatif fitur menurut mekanisme model yang digunakan. Nilainya tidak membuktikan hubungan kausal, arah pengaruh, atau bahwa suatu fitur selalu meningkatkan akurasi prediksi.

Analisis tambahan seperti permutation importance, partial dependence, atau SHAP dapat dipertimbangkan, tetapi belum termasuk hasil yang dilaporkan dalam dokumentasi ini.

### 13.7 Baseline sederhana belum tersedia dalam hasil yang didokumentasikan

Belum ada hasil evaluasi baseline yang dilaporkan secara terpisah untuk dibandingkan dengan model-model regresi ini.

Baseline seperti `DummyRegressor` dapat digunakan untuk membangun titik pembanding yang jelas. Walaupun interpretasi R² sudah memiliki baseline prediksi rata-rata secara matematis, evaluasi baseline eksplisit menggunakan MAE dan RMSE juga akan membantu menilai apakah model memberikan perbaikan praktis.

## 14. Rencana Pengembangan Berikutnya

Prioritas perbaikan yang dapat dilakukan tanpa mengubah hasil eksperimen yang sudah dicatat:

1. **Verifikasi pipeline dan konfigurasi estimator.** Pastikan transformasi fitur, encoding, dan parameter masing-masing model sesuai dengan yang dimaksud.
2. **Evaluasi baseline.** Bandingkan model dengan prediktor sederhana, khususnya `DummyRegressor`.
3. **Periksa distribusi target dan residual.** Visualisasikan distribusi `revenue`, residual, serta hubungan antara nilai aktual dan prediksi untuk menemukan pola kesalahan.
4. **Uji target logaritmik sebagai eksperimen terpisah.** Bandingkan hasil dengan target asli menggunakan pembagian data dan metrik evaluasi yang konsisten.
5. **Tinjau nilai nol pada `budget`.** Periksa jumlah dan karakteristik baris dengan budget nol sebelum memutuskan apakah perlu strategi penanganan berbeda.
6. **Periksa kestabilan hasil.** Gunakan validasi tambahan atau beberapa pembagian data untuk melihat apakah peringkat model konsisten.
7. **Lakukan analisis interpretabilitas tambahan.** Gunakan metode yang sesuai untuk memahami kontribusi fitur dan pola kesalahan, tanpa menafsirkan importance sebagai hubungan sebab-akibat.

Rencana ini merupakan rekomendasi eksperimen berikutnya, bukan pekerjaan yang sudah selesai atau hasil yang sudah diperoleh.

## 15. Kesimpulan

Dalam eksperimen prediksi revenue film ini, tiga algoritma regresi dibandingkan menggunakan `GridSearchCV` dengan scoring R² dan pembagian data 80:20.

Berdasarkan hasil yang tersedia, Random Forest merupakan model dengan performa terbaik pada data pengujian, dengan R² sebesar 0.5597, MAE sekitar 8.90 juta, dan RMSE sekitar 45.65 juta dalam satuan revenue dataset. XGBoost memperoleh R² sebesar 0.5422, sedangkan Ridge Regression memperoleh R² negatif sebesar -0.0178.

`budget` menjadi fitur dengan importance tertinggi pada hasil Random Forest maupun XGBoost. Namun, hasil feature importance tidak menunjukkan hubungan sebab-akibat.

Performa ini merupakan hasil eksperimen awal, bukan jaminan akurasi untuk seluruh film atau data di luar dataset. Pemeriksaan konfigurasi XGBoost, validasi preprocessing, evaluasi baseline, serta analisis distribusi target dan residual merupakan langkah penting sebelum menarik kesimpulan yang lebih kuat.

---

## Catatan reproduksibilitas

Informasi berikut belum tersedia dan sebaiknya diisi setelah memeriksa notebook:

- Jumlah baris akhir setelah preprocessing.
- Jumlah fitur final setelah encoding.
- Implementasi dan parameter `log_transform`, `num_transform`, serta `cat_transform`.
- Metode pembagian data dan `random_state`.
- Jumlah fold `GridSearchCV`.
- Versi scikit-learn dan XGBoost.
- Hasil baseline `DummyRegressor`.
- Distribusi target dan residual.
- Konfirmasi apakah `min_samples_split` benar-benar digunakan oleh estimator XGBoost.

Bagian ini sengaja tidak diisi dengan asumsi agar dokumentasi tetap sesuai dengan eksperimen yang benar-benar dilakukan.