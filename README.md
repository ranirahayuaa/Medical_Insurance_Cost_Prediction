# 🏥 Medical Insurance Cost Prediction

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Selamat datang di repositori **Medical Insurance Cost Prediction**! 🚀

Proyek ini bertujuan untuk memprediksi biaya asuransi kesehatan individu secara akurat menggunakan berbagai algoritma _Machine Learning_. Dengan memahami faktor-faktor demografis dan medis (seperti usia, BMI, status perokok, dan wilayah) yang memengaruhi biaya, model ini dapat menjadi alat bantu yang berharga untuk perencanaan finansial dan analisis risiko di industri asuransi.

---

## 📑 Daftar Isi

1. [Mengapa Proyek Ini Penting?](#-mengapa-proyek-ini-penting)
2. [Dataset](#-dataset)
3. [Fitur Utama](#-fitur-utama)
4. [Hasil & Performa Model](#-hasil--performa-model)
5. [Instalasi & Cara Menjalankan](#-instalasi--cara-menjalankan)
6. [Struktur Folder](#-struktur-folder)
7. [Teknologi yang Digunakan](#-teknologi-yang-digunakan)

---

## 💡 Mengapa Proyek Ini Penting?

Biaya kesehatan yang tidak terduga dapat menjadi beban finansial yang besar. Proyek ini tidak hanya sekadar memprediksi angka, tetapi juga:

- 🔍 **Mengungkap Pola Tersembunyi**: Menunjukkan fitur mana (misalnya: status perokok atau usia) yang paling dominan memengaruhi biaya.
- ⚖️ **Perbandingan Objektif**: Menguji 7 algoritma berbeda untuk menemukan keseimbangan terbaik antara akurasi dan generalisasi.
- 🛠️ **Siap Produksi**: Dilengkapi dengan _pipeline_ preprocessing yang tersimpan (`scaler` & `label encoder`), sehingga model dapat langsung digunakan untuk memprediksi data baru secara _real-time_.

---

## 📊 Dataset

Proyek ini menggunakan **Medical Insurance Dataset** yang populer. Dataset ini berisi informasi demografis dan medis dari sejumlah individu.

- **Sumber Dataset**: [Kaggle - Medical Insurance Dataset](https://www.kaggle.com/code/figolm10/medical-insurance-dataset/input)
- **Fitur (Variabel)**:
  - `age`: Usia individu (tahun)
  - `sex`: Jenis kelamin (male/female)
  - `bmi`: Body Mass Index (Indeks Massa Tubuh)
  - `children`: Jumlah anak/tanggungan
  - `smoker`: Status perokok (yes/no)
  - `region`: Wilayah tempat tinggal (southwest, southeast, northwest, northeast)
  - `charges`: **Target Variable** - Biaya asuransi medis (USD)

---

## 🚀 Fitur Utama

1. **Pipeline Otomatis**: Script `main.py` menjalankan seluruh alur kerja dari awal hingga akhir secara berurutan (Preprocessing → Training → Evaluasi → Prediksi).
2. **Hyperparameter Tuning**: Menggunakan `GridSearchCV` untuk mengoptimalkan model **Random Forest** agar mencapai performa terbaik.
3. **Evaluasi Komprehensif**: Menghitung metrik RMSE, MAE, MSE, R² Score, dan MAPE, lengkap dengan visualisasi _Feature Importance_ dan _Residual Analysis_.
4. **Kelas Predictor Kustom**: Kelas `InsurancePredictor` di `4_prediction.py` memudahkan prediksi untuk data tunggal maupun _batch_ (bulk prediction).

---

## 🏆 Hasil & Performa Model

Berikut adalah perbandingan performa dari 7 model yang diuji pada _Test Set_:

| Model                 | R² Score (Test) |  RMSE (Test)  | CV Score (R²) |           Status           |
| :-------------------- | :-------------: | :-----------: | :-----------: | :------------------------: |
| **Gradient Boosting** |   **0.8781**    | **$4,351.11** |    0.8404     |         ✅ Success         |
| **Random Forest**     |     0.8654      |   $4,571.50   |    0.8266     |         ✅ Success         |
| XGBoost               |     0.8502      |   $4,822.99   |    0.7938     |         ✅ Success         |
| Linear Regression     |     0.7833      |   $5,799.59   |    0.7339     |         ✅ Success         |
| Ridge Regression      |     0.7833      |   $5,800.16   |    0.7340     |         ✅ Success         |
| Lasso Regression      |     0.7833      |   $5,799.59   |    0.7339     |         ✅ Success         |
| Decision Tree         |     0.6950      |   $6,881.11   |    0.6942     | ✅ Success _(Overfitting)_ |

> **📌 Catatan Penting**:
> Meskipun _Gradient Boosting_ menunjukkan R² tertinggi pada model dasar, proyek ini secara khusus melakukan **Hyperparameter Tuning (Grid Search)** pada **Random Forest**. Model **Random Forest (Tuned)** inilah yang akhirnya dipilih dan disimpan sebagai `best_model.pkl` karena stabilitasnya yang sangat baik, risiko overfitting yang lebih rendah dibandingkan Decision Tree murni, dan kemudahan interpretasi _Feature Importance_.

### 🔑 Feature Importance (Top 3)

Berdasarkan model terbaik, faktor yang paling memengaruhi biaya asuransi adalah:

1. **Smoker** (Status Perokok) 🚬
2. **Age** (Usia) 🎂
3. **BMI** (Indeks Massa Tubuh) ⚖️

---

## 🛠️ Instalasi & Cara Menjalankan

Ikuti langkah-langkah berikut untuk menjalankan proyek ini di mesin lokal Anda:

### 1. Clone Repositori

```bash
git clone https://github.com/username-anda/medical-insurance-prediction.git
cd medical-insurance-prediction
```
