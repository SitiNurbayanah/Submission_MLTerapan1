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

- Menggunakan algoritma klasifikasi seperti Random Forest dan XGBoost untuk membandingkan performa model.
- Menerapkan SMOTE (Synthetic Minority Oversampling Technique) untuk menangani ketidakseimbangan kelas pada data pelatihan.
- Mengukur performa model dengan metrik klasifikasi seperti F1 Score, Precision, dan Recall.

## Data Understanding

Dataset yang digunakan dalam proyek ini berasal dari platform pembelajaran internal dan telah disesuaikan khusus untuk keperluan pembelajaran Machine Learning. Dataset ini memuat informasi penting terkait calon peminjam, seperti Age, Gender, Education Level, Marital Status, Income, Credit Score, Loan Amount, Loan Purpose, Employment Status, Years at Current Job, Payment History, Debt-to-Income Ratio, Assets Value, Number of Dependents, City, State, Country, Previous Defaults, Marital Status Change, serta label target berupa Risk Rating. Data ini mencerminkan berbagai aspek demografis dan finansial calon peminjam, yang sangat relevan untuk membangun model prediktif dalam konteks risiko peminjaman. Data diambil dari Kaggle (https://www.kaggle.com/datasets/preethamgouda/financial-risk)

### Variabel-variabel dalam dataset antara lain:

- Gender: Jenis kelamin peminjam
- Age: Usia peminjam
- Education Level: Tingkat pendidikan
- Marital Status: Status pernikahan
- Employment Status: Status pekerjaan
- Income: Pendapatan
- Credit Score: Skor kredit
- Loan Amount: Jumlah pinjaman
- Loan Purpose: Tujuan pinjaman
- Debt-to-Income Ratio: Rasio utang terhadap pendapatan
- Assets Value: Nilai aset
- Number of Dependents: Jumlah tanggungan
- Previous Defaults: Riwayat gagal bayar sebelumnya
- Payment History: Riwayat pembayaran
- Risk Rating: Target klasifikasi (rendah, sedang, tinggi, dll)

EDA dilakukan dengan visualisasi seperti:

- Boxplot antara Credit Score dan Risk Rating
  ![1748527612782](image/READme/1748527612782.png)
- Countplot distribusi Risk Rating dan Employment Status
  ![1748527628890](image/READme/1748527628890.png)
- Heatmap korelasi antar fitur numerik
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

Model yang digunakan:

- **Random Forest Classifier**
- **XGBoost Classifier**
- **Support Vector Machine**
- **Naive Bayes**

Model terbaik dipilih berdasarkan nilai F1 Score dan akurasi keseluruhan di data uji.

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
