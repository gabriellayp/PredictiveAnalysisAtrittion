# 🔍 Predicting Employee Attrition with Machine Learning
COHORT ID : MC008D5X2466

Nama : Gabriella Yoanda Pelawi

Email : mc008d5x2466@student.devacademy.id

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

- Total data: **14,900 records**
- Label target: `Attrition` (0 = tetap, 1 = keluar)

### Link Dataset
🔗 [Download from Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/employee-attrition-dataset/data)

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

## 5. 🤖 Modeling

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

## 6. 📈 Evaluation

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
| Random Forest  | 74.7%    | 75.0%     | 74.5%  | 74.5%    | 0.72    |
| Gradient Boost | 75.6%    | 75.5%     | 75.5%  | 75.5%    | 0.85    |
| XGBoost        | 75.7%    | 75.5%     | 75.5%  | 75.5%    | 0.85    |
| AdaBoost       | **76.6%**| **76.5%** | **76.0%**| **76.0%**| **0.86** |

📌 **Model Terbaik: AdaBoost**, karena memberikan skor F1 dan ROC-AUC tertinggi.

### Confusion Matrix (AdaBoost):

|                 | Predicted: Stay | Predicted: Leave |
|-----------------|------------------|------------------|
| **Actual: Stay**  | 1292           | 466              |
| **Actual: Leave** | 407              | 1560            |

- True Positives (TP) = 1560
→ Karyawan yang benar-benar resign (Leave) dan berhasil diprediksi dengan tepat sebagai resign.

- True Negatives (TN) = 1292
→ Karyawan yang benar-benar stay dan berhasil diprediksi dengan tepat sebagai stay.

- False Positives (FP) = 407
→ Karyawan yang sebenarnya resign, tetapi diprediksi akan stay.
⚠️ Ini cukup penting karena bisa berdampak pada kurangnya antisipasi perusahaan terhadap potensi turnover.

- False Negatives (FN) = 466
→ Karyawan yang sebenarnya stay, tetapi diprediksi akan resign.
⚠️ Ini bisa menyebabkan kesalahan alokasi sumber daya, seperti upaya retensi yang tidak perlu.
---

---

## 7. ✅ Kesimpulan

Proyek ini berhasil membangun model prediktif untuk mengklasifikasikan kemungkinan karyawan akan keluar dari perusahaan menggunakan berbagai algoritma machine learning.

Dari empat model yang dibangun — Random Forest, Gradient Boosting, XGBoost, dan AdaBoost — model **AdaBoost** menunjukkan performa terbaik berdasarkan metrik **Accuracy, F1 Score, dan ROC-AUC**. Hal ini menunjukkan bahwa boosting sederhana dengan algoritma dasar seperti decision tree bisa sangat efektif bila dipadukan dengan tuning parameter yang optimal.

Dengan memanfaatkan data historis karyawan dan pendekatan pembelajaran mesin, perusahaan kini memiliki alat yang lebih akurat untuk **mengidentifikasi risiko turnover lebih awal**, sehingga dapat mengambil langkah-langkah preventif dalam **retensi karyawan** dan **perencanaan SDM strategis**.

📌 **Rekomendasi selanjutnya**:  
- Integrasi model ke sistem HR secara real-time.  
- Update data secara berkala untuk menjaga akurasi prediksi.  
- Pertimbangkan explainable AI (XAI) untuk memahami lebih dalam faktor-faktor penyebab karyawan keluar.

---

