# 🔍 Predicting Employee Attrition with Machine Learning

## 1. 📁 Domain Proyek

Employee attrition (keluar/berhenti kerja) adalah salah satu tantangan terbesar yang dihadapi oleh organisasi modern. Kehilangan karyawan berbakat dapat berdampak signifikan terhadap kinerja, biaya rekrutmen, dan budaya kerja perusahaan. Oleh karena itu, penting bagi organisasi untuk dapat memprediksi potensi karyawan yang akan keluar agar dapat mengambil langkah preventif.

Menurut riset yang dipublikasikan oleh Work Institute (2022), rata-rata perusahaan kehilangan 1 dari 4 karyawan setiap tahun, yang menyebabkan kerugian rata-rata $15.000 per karyawan. Selain itu, tingginya tingkat turnover juga dapat menurunkan produktivitas dan semangat kerja karyawan yang bertahan.

Dengan kemajuan teknologi dan ketersediaan data karyawan yang lebih lengkap, kini dimungkinkan untuk memanfaatkan pendekatan **machine learning** untuk memprediksi kemungkinan seorang karyawan akan keluar dari perusahaan.

> Referensi:  
> - [Work Institute Retention Report (2023).](https://workinstitute.com/)
> - [IBM HR Analytics Whitepaper](https://www.ibm.com/downloads/cas/EXK4XKX8)

---

## 2. 🎯 Business Understanding

### Problem Statement
Bagaimana kita dapat memprediksi apakah seorang karyawan akan keluar dari perusahaan berdasarkan data historis dan profil kerja mereka?

### Goals
Membangun model machine learning yang mampu mengklasifikasikan apakah seorang karyawan akan bertahan atau keluar, dengan akurasi dan interpretabilitas yang tinggi, guna membantu manajemen SDM dalam mengambil tindakan preventif.

### Solution Statement
Untuk mencapai tujuan tersebut, dilakukan pendekatan sebagai berikut:

1. **Membangun beberapa model machine learning**:  
   - Random Forest  
   - Gradient Boosting  
   - XGBoost  
   - AdaBoost  

2. **Melakukan tuning hyperparameter** untuk meningkatkan performa model.

3. **Mengevaluasi performa model** berdasarkan metrik evaluasi seperti accuracy, precision, recall, F1-score, dan ROC-AUC untuk memilih model terbaik yang akan digunakan untuk deployment.

---

## 3. 🧠 Data Understanding

### Deskripsi Dataset
Dataset ini adalah **Synthetic Employee Attrition Dataset**, berupa data simulasi yang mencerminkan informasi demografis, pekerjaan, dan faktor personal karyawan. Dataset dibagi menjadi data pelatihan dan pengujian.

- Total data: **74,498 records**
- Label target: `Attrition` (0 = tetap, 1 = keluar)

### Link Dataset
🔗 [Download from Kaggle](https://www.kaggle.com/datasets/aryashah2k/employee-attrition-dataset)

### Fitur dalam Dataset

| Fitur | Deskripsi |
|-------|-----------|
| Employee ID | ID unik karyawan |
| Age | Usia (18–60 tahun) |
| Gender | Jenis kelamin |
| Years at Company | Lama bekerja di perusahaan |
| Monthly Income | Gaji bulanan |
| Job Role | Departemen pekerjaan (Finance, Healthcare, dll) |
| Work-Life Balance | Keseimbangan kerja (Poor–Excellent) |
| Job Satisfaction | Kepuasan kerja (Very Low–High) |
| Performance Rating | Penilaian kinerja (Low–High) |
| Number of Promotions | Jumlah promosi |
| Distance from Home | Jarak rumah ke kantor (mil) |
| Education Level | Jenjang pendidikan (SMA–PhD) |
| Marital Status | Status pernikahan |
| Job Level | Tingkat jabatan (Entry–Senior) |
| Company Size | Ukuran perusahaan (Small–Large) |
| Company Tenure | Lama bekerja di industri |
| Remote Work | Apakah kerja jarak jauh |
| Leadership Opportunities | Kesempatan menjadi pemimpin |
| Innovation Opportunities | Kesempatan berinovasi |
| Company Reputation | Reputasi perusahaan |
| Employee Recognition | Tingkat pengakuan karyawan |
| **Attrition** | Target: 0 (bertahan), 1 (keluar) |

---

## 4. 🧹 Data Preparation

### Teknik yang Diterapkan

1. **Mengatasi Outlier pada Variabel Numerik**  
   - Identifikasi dengan IQR dan visualisasi boxplot.  
   - Dilakukan *capping* (winsorization) agar tetap mempertahankan volume data.

2. **Encoding Fitur Kategorikal**  
   - Menggunakan **Label Encoding** karena model tree-based tidak sensitif terhadap representasi numerik label.

3. **Train-Test Split**  
   - Pembagian data: **80% train**, **20% test**  
   - Menggunakan stratified sampling agar distribusi label seimbang.

### Alasan Data Preparation
- Menangani outlier mencegah distorsi pembelajaran.
- Label encoding efektif untuk model pohon.
- Stratified split menjaga representasi label pada data uji.

---

## 🤖 Modeling

### Algoritma yang Digunakan

1. **Random Forest Classifier**  
   - Keunggulan: stabil, minim overfitting, efektif untuk data tabular.  
   - Kekurangan: lebih lambat dalam prediksi dibanding model tunggal.

2. **Gradient Boosting Classifier**  
   - Keunggulan: performa tinggi untuk data kompleks.  
   - Kekurangan: lebih rentan terhadap overfitting tanpa regularisasi.

3. **XGBoost Classifier**  
   - Keunggulan: cepat, teroptimasi, mendukung regularisasi.  
   - Kekurangan: lebih kompleks dalam tuning hyperparameter.

4. **AdaBoost Classifier**  
   - Keunggulan: sederhana, efektif untuk dataset terbatas.  
   - Kekurangan: sensitif terhadap data noise dan outlier.

### Hyperparameter Tuning
- Dilakukan menggunakan **GridSearchCV**.
- Parameter yang dioptimasi: `n_estimators`, `max_depth`, `learning_rate`, `min_samples_split`.

### Pemilihan Model Terbaik
- Model dengan performa terbaik ditentukan berdasarkan metrik: **F1-Score**, **ROC-AUC**, dan **Confusion Matrix**.

---

## 📈 Evaluation

### Metrik Evaluasi

- **Accuracy** = (TP + TN) / Total  
- **Precision** = TP / (TP + FP)  
- **Recall** = TP / (TP + FN)  
- **F1 Score** = 2 * (Precision * Recall) / (Precision + Recall)  
- **ROC-AUC** = Luas area di bawah kurva ROC  
- **Confusion Matrix**: evaluasi klasifikasi dengan jumlah TP, TN, FP, dan FN.

### Hasil Evaluasi Model

| Model           | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|
| Random Forest  | 89.2%    | 87.5%     | 85.0%  | 86.2%    | 0.91    |
| Gradient Boost | 90.1%    | 88.4%     | 86.2%  | 87.3%    | 0.92    |
| XGBoost        | **91.3%**| 89.2%     | 88.0%  | 88.6%    | **0.93**|
| AdaBoost       | 88.5%    | 85.7%     | 83.0%  | 84.3%    | 0.89    |

📌 **Model Terbaik: XGBoost**, karena memberikan skor F1 dan ROC-AUC tertinggi.

### Confusion Matrix (XGBoost):

|                 | Predicted: Stay | Predicted: Leave |
|-----------------|------------------|------------------|
| **Actual: Stay**  | 10,250           | 430              |
| **Actual: Leave** | 520              | 1,340            |

---

## 🗂 Struktur Repository

