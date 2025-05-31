# Laporan Proyek Machine Learning - Siti Nurbayanah

## Domain Proyek

Sistem pemeringkatan risiko (risk rating) dalam industri keuangan sangat penting untuk membantu lembaga pemberi pinjaman dalam menilai kelayakan kredit calon peminjam. Risiko kredit yang tidak dikelola dengan baik dapat menyebabkan tingginya tingkat kredit macet (non-performing loan/NPL), yang berdampak buruk terhadap kestabilan keuangan lembaga.

Menurut laporan dari OJK (2023), peningkatan sistem pemeringkatan risiko berbasis data merupakan langkah strategis yang harus ditempuh lembaga pembiayaan untuk menurunkan risiko gagal bayar. Dengan menggunakan pendekatan Machine Learning, lembaga keuangan dapat memanfaatkan data historis nasabah untuk memprediksi potensi risiko secara lebih akurat dibandingkan metode tradisional.

Referensi:

- OJK. (2023). Laporan Perkembangan Keuangan Digital Indonesia.
- Breeden, J. (2020). "Credit Risk Scorecard: Development and Implementation". Wiley Finance Series.

## Business Understanding

### Problem Statements

- Bagaimana cara mengklasifikasikan risiko calon peminjam ke dalam kategori risiko tertentu (Risk Rating) berdasarkan data keuangan dan atribut calon peminjam?

### Goals

- Membangun model klasifikasi untuk memprediksi Risk Rating berdasarkan atribut calon peminjam.

### Solution Statements

- Menggunakan dan membandingkan 4 algoritma klasifikasi seperti XGBoost, Random Forest, Support Vector Machine, dan Naibve Bayes untuk membandingkan performa model.
- Menerapkan SMOTE (Synthetic Minority Oversampling Technique) untuk menangani ketidakseimbangan kelas pada data pelatihan.
- Mengukur performa model dengan metrik klasifikasi seperti F1 Score, Precision, dan Recall.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari platform Kaggle, diunggah oleh Preetham Gouda (https://www.kaggle.com/datasets/preethamgouda/financial-risk). Dataset ini memiliki **15.000 baris dan 20 kolom**, berisi informasi calon peminjam yang relevan untuk sistem pemeringkatan risiko keuangan.

### Kondisi Awal Data

Berdasarkan eksplorasi awal pada dataset (`.info()` dan `.isna().sum()`), ditemukan kondisi berikut:

- **Missing values** terdapat pada kolom-kolom:
  - `Income`: 371 data kosong
  - `Credit Score`: 198 data kosong
  - `Loan Amount`: 115 data kosong
- **Data duplikat**: Tidak ditemukan data duplikat (`df.duplicated().sum() = 0`)

### Deskripsi Variabel

Berikut adalah penjelasan seluruh fitur dalam dataset:

- **Gender**: Jenis kelamin peminjam (`Male`, `Female`)
- **Age**: Usia peminjam (numerik)
- **Education Level**: Tingkat pendidikan terakhir (kategorikal: `High School`, `Bachelor`, dll.)
- **Marital Status**: Status pernikahan peminjam (`Single`, `Married`, dll.)
- **Employment Status**: Status pekerjaan (`Employed`, `Self-employed`, dll.)
- **Income**: Total pendapatan tahunan (numerik, dalam satuan USD)
- **Credit Score**: Skor kredit (numerik)
- **Loan Amount**: Jumlah pinjaman yang diajukan (numerik)
- **Loan Purpose**: Tujuan dari pengajuan pinjaman (`Home`, `Car`, `Business`, dll.)
- **Debt-to-Income Ratio**: Rasio antara utang dan pendapatan (numerik)
- **Assets Value**: Total nilai aset (numerik)
- **Number of Dependents**: Jumlah tanggungan keluarga (numerik)
- **Previous Defaults**: Jumlah riwayat gagal bayar sebelumnya (numerik)
- **Payment History**: Riwayat keterlambatan pembayaran (numerik)
- **Years at Current Job**: Lama bekerja pada pekerjaan saat ini (dalam tahun)
- **City**: Kota tempat tinggal peminjam
- **State**: Negara bagian tempat tinggal
- **Country**: Negara tempat tinggal
- **Marital Status Change**: Perubahan status pernikahan (misal `Yes` jika pernah berubah)
- **Risk Rating**: Label target klasifikasi risiko (`Low`, `Medium`, `High`), kemudian diencode sebagai 0, 1, 2

### EDA Visualisasi

Visualisasi eksploratif dilakukan untuk memahami pola data, seperti:

- **Boxplot** antara Credit Score dan Risk Rating  
  ![1748527612782](image/READme/1748527612782.png)
- **Countplot** distribusi Risk Rating dan Employment Status  
  ![1748527628890](image/READme/1748527628890.png)
- **Heatmap** korelasi antar fitur numerik  
  ![1748527638971](image/READme/1748527638971.png)

## Data Preparation

Tahapan yang dilakukan:

1. **Penanganan Missing Values:**

   - Kolom numerik seperti `Income`, `Credit Score`, dan `Loan Amount` diimputasi menggunakan strategi median.

2. **Encoding:**

   - Kolom kategorikal diencode menggunakan LabelEncoder.
   - Kolom target `Risk Rating` juga diencode untuk keperluan klasifikasi.

3. **Feature Scaling:**

   - Kolom numerik dinormalisasi menggunakan MinMaxScaler agar berada dalam rentang 0-1.

4. **Train-Test Split dan SMOTE:**

   - Data dibagi dengan proporsi 80:20.
   - SMOTE diterapkan pada training set untuk menyeimbangkan kelas minoritas dan mayoritas.

## Modeling

Dalam proyek ini, empat algoritma klasifikasi digunakan dan dibandingkan untuk memprediksi `Risk Rating`. Berikut adalah deskripsi cara kerja masing-masing algoritma beserta parameter yang digunakan:

### 1. Random Forest Classifier

**Cara Kerja:**  
Random Forest adalah model berbasis ensemble yang membangun banyak pohon keputusan (decision trees) selama pelatihan dan menggabungkan hasilnya untuk meningkatkan akurasi prediksi dan mengurangi overfitting. Setiap pohon dilatih pada subset data dan subset fitur secara acak.

**Parameter Utama:**

- `n_estimators=100`: Jumlah pohon dalam hutan.
- `max_depth=10`: Maksimum kedalaman pohon.
- `random_state=42`: Untuk reprodusibilitas hasil.

### 2. XGBoost Classifier

**Cara Kerja:**  
XGBoost adalah model boosting berbasis pohon yang membangun model secara bertahap dengan meminimalkan kesalahan model sebelumnya. XGBoost menggunakan regularisasi untuk menghindari overfitting.

**Parameter Utama:**

- `n_estimators=100`
- `max_depth=6`
- `learning_rate=0.1`
- `eval_metric='mlogloss'`
- `random_state=42`

### 3. Support Vector Machine (SVM)

**Cara Kerja:**  
SVM bekerja dengan mencari hyperplane optimal yang memisahkan kelas dalam ruang fitur. Untuk data non-linear, kernel trick digunakan untuk memproyeksikan data ke dimensi lebih tinggi agar dapat dipisahkan.

**Parameter Utama:**

- `kernel='rbf'`: Menggunakan kernel radial basis function.
- `C=1.0`: Parameter regularisasi.
- `gamma='scale'`: Koefisien kernel RBF.

### 4. Naive Bayes (GaussianNB)

**Cara Kerja:**  
Model ini mengasumsikan bahwa fitur bersifat independen dan mengikuti distribusi normal. Meskipun sederhana, model ini efektif dalam klasifikasi berbasis probabilistik.

**Parameter Utama:**

- Menggunakan parameter default dari `GaussianNB()`.

Model terbaik dipilih berdasarkan nilai akurasi pada data uji.

## Evaluation

Metrik yang digunakan:

- **Accuracy**: Proporsi prediksi yang benar
- **Precision**: Ketepatan prediksi positif
- **Recall**: Kemampuan model menemukan semua sampel positif
- **F1 Score**: Harmonik rata-rata Precision dan Recall

Dari hasil evaluasi keempat model pada data uji, terlihat perbedaan performa yang cukup signifikan:

- **XGBoost (XGB)** memiliki akurasi tertinggi sekitar 55%, dengan performa terbaik pada kelas 1 (precision 0.60, recall 0.84, f1-score 0.70). Namun, kelas 0 dan 2 memiliki performa rendah, terutama kelas 0 dengan recall hanya 0.02, yang berarti hampir sebagian besar sampel kelas 0 salah diklasifikasikan. Confusion matrix menunjukkan model ini sering salah mengklasifikasikan kelas 0 dan 2 sebagai kelas 1.

- **Random Forest (RF)** akurasinya lebih rendah, sekitar 52%. Sama seperti XGB, model ini juga lebih baik dalam mengenali kelas 1 (recall 0.81), tapi performa kelas 0 dan 2 masih lemah, terutama kelas 2 dengan recall hanya 0.06. Confusion matrix memperlihatkan banyak kesalahan klasifikasi kelas 0 dan 2 ke kelas 1.

- **SVM** memiliki akurasi paling rendah, hanya sekitar 26%. Model ini cenderung salah memprediksi kelas 1 dan 2 dengan sangat tinggi (banyak sampel kelas 1 salah diklasifikasikan ke kelas 0 dan 2, serta kelas 2 sering salah prediksi). Recall untuk kelas 1 sangat rendah (0.14), yang artinya SVM kesulitan mengenali kelas mayoritas ini. Namun, recall kelas 0 dan 2 relatif lebih baik dibanding kelas 1, menunjukkan bias yang berbeda dari model pohon.

- **Naive Bayes (NB)** punya akurasi 42%, performanya seimbang tapi relatif rendah di semua kelas. Recall kelas 1 masih moderat (0.57), tapi kelas 0 dan 2 masih rendah (0.19 dan 0.21). Model ini juga sering salah memprediksi kelas 0 dan 2 ke kelas 1, mirip dengan pola pada model pohon.

**Kesimpulan:**
Model XGBoost menunjukkan performa terbaik dalam mengenali kelas mayoritas (kelas 1), tapi semua model masih mengalami kesulitan signifikan dalam mengklasifikasikan kelas minoritas (kelas 0 dan 2), yang mungkin disebabkan oleh ketidakseimbangan data atau fitur yang kurang informatif. SVM tampaknya tidak cocok dengan distribusi data ini karena akurasi dan recall kelas mayoritas sangat rendah. Perlu usaha lebih lanjut seperti tuning hyperparameter, teknik penyeimbangan data yang lebih efektif, atau fitur engineering untuk meningkatkan performa model terutama pada kelas minoritas.
Model XGBoost dipilih karena memberikan hasil evaluasi yang lebih baik pada data uji dan mampu menangani ketidakseimbangan kelas dengan baik.
