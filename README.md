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
| Overtime | Waktu Kerja di Luar Jam Normal |
| Number of Promotions | Jumlah promosi |
| Distance from Home | Jarak rumah ke kantor (mil) |
| Education Level | Jenjang pendidikan (SMA–PhD) |
| Marital Status | Status pernikahan |
| Number of Dependents | Jumlah Tanggungan |
| Job Level | Tingkat jabatan (Entry–Senior) |
| Company Size | Ukuran perusahaan (Small–Large) |
| Company Tenure | Lama bekerja di industri |
| Remote Work | Apakah kerja jarak jauh |
| Leadership Opportunities | Kesempatan menjadi pemimpin |
| Innovation Opportunities | Kesempatan berinovasi |
| Company Reputation | Reputasi perusahaan |
| Employee Recognition | Tingkat pengakuan karyawan |
| **Attrition** | Target: 0 (bertahan), 1 (keluar) |

### Exploratory Data Analysis
1. **Memeriksa Nilai yang Hilang (Missing Values)**
   - Menggunakan fungsi .isnull().sum(), namum pada data ini tidak ditemukan adanya missing values.
     
2. **Memeriksa Data Duplikat**
   - Menggunakan fungsi .duplicated().sum(), namun tidak ditemukan adanya indikasi duplikasi data.
   - 
3. **Distribusi Variabel Target**
     ![Distribusi Attrition](https://github.com/gabriellayp/PredictiveAnalysisAtrittion/blob/main/images/outputatrrition.png?raw=true)
     Visualisasi diatas menunjukkan bahwa distribusi label Attrition cukup seimbang, dengan jumlah karyawan yang bertahan (Stayed) sedikit lebih banyak dibanding yang resign (Left).
   
---

## 4. 🧹 Data Preparation

### Teknik yang Diterapkan

1. **Menghapus Kolom Identifikasi (`UserID`)**
   - Kolom `UserID` merupakan identifikasi unik yang tidak memiliki makna prediktif.
   - Karena kolom ini tidak memberikan informasi yang bermanfaat untuk proses pembelajaran model, kolom ini dihapus untuk mencegah noise dan potensi *data leakage*.

2. **Menangani Outlier pada Variabel Numerik**
   - Deteksi outlier dilakukan menggunakan metode **Interquartile Range (IQR)**.
     - Nilai batas bawah dihitung sebagai: `Q1 - 1.5 * IQR`
     - Nilai batas atas dihitung sebagai: `Q3 + 1.5 * IQR`
   - Setelah outlier teridentifikasi, dilakukan proses **capping (winsorization)**, yaitu mengganti nilai-nilai ekstrem dengan batas bawah atau batas atas IQR.
   - Tujuan: menjaga jumlah data tetap utuh (tidak menghapus data) sekaligus mengurangi pengaruh outlier terhadap performa model.

3. **Encoding Fitur Kategorikal**
   - Seluruh fitur kategorikal diubah menjadi nilai numerik menggunakan **Label Encoding**.
   - Alasan penggunaan Label Encoding:
     - Model berbasis pohon keputusan (seperti Random Forest dan XGBoost) tidak terpengaruh oleh skala atau urutan numerik dari kategori.
     - Label Encoding lebih efisien dalam memori dan waktu komputasi dibandingkan One-Hot Encoding, terutama jika kategori memiliki banyak nilai unik.

4. **Standarisasi Fitur Numerik**
   - Seluruh fitur numerik distandarisasi menggunakan **StandardScaler** dari Scikit-Learn.
   - Proses ini mengubah distribusi fitur sehingga memiliki **rata-rata = 0** dan **standar deviasi = 1**.
   - Meskipun model tree-based tidak membutuhkan standarisasi, langkah ini tetap dilakukan untuk memungkinkan penggunaan model lain yang sensitif terhadap skala data (misalnya SVM, KNN, Logistic Regression).

5. **Split Data: Train dan Test**
   - Dataset dibagi menjadi dua bagian: **75% untuk data latih (training set)** dan **25% untuk data uji (test set)**.
   - Proses pembagian dilakukan dengan teknik **Stratified Sampling** berdasarkan label target (*Attrition*).
   - Tujuan stratifikasi:
     - Menjaga distribusi kelas target tetap seimbang antara data latih dan data uji.
     - Mencegah bias model terhadap kelas mayoritas.

---

### Alasan dan Tujuan Data Preparation

| Langkah | Tujuan |
|--------|--------|
| Menghapus kolom `UserID` | Menghindari noise dan mencegah data leakage dari fitur non-prediktif |
| IQR Capping | Menangani outlier tanpa mengurangi jumlah data |
| Label Encoding | Efisiensi pemrosesan dan kompatibilitas dengan model tree-based |
| StandardScaler | Memastikan fitur numerik berada dalam skala yang sama, jika digunakan untuk model non-tree |
| Stratified Train-Test Split | Menjamin distribusi target seimbang, evaluasi model lebih representatif |

---

## 5. 🤖 Modeling

### Algoritma yang Digunakan dan Cara Kerjanya

1. **Random Forest Classifier**  
   Random Forest merupakan algoritma ensemble berbasis *bagging* yang membangun banyak pohon keputusan (decision trees) selama pelatihan, lalu menggabungkan prediksi dari semua pohon untuk meningkatkan akurasi dan mengurangi overfitting.  
   - **Keunggulan**: stabil, minim overfitting, efektif untuk data tabular.  
   - **Kekurangan**: lebih lambat dalam prediksi dibanding model tunggal.  
   - **Hyperparameter yang dituning dengan GridSearchCV**:
     - `n_estimators`: [100, 200, 300]  
     - `max_depth`: [None, 10, 20, 30]  
     - `min_samples_split`: [2, 5, 10]  
     - `min_samples_leaf`: [1, 2, 4]  
     - `max_features`: ['auto', 'sqrt', 'log2']

2. **Gradient Boosting Classifier**  
   Gradient Boosting membangun model secara bertahap, di mana setiap pohon baru mencoba memperbaiki kesalahan dari model sebelumnya dengan mengoptimalkan fungsi loss melalui pendekatan *gradient descent*.  
   - **Keunggulan**: performa tinggi untuk data kompleks.  
   - **Kekurangan**: rentan overfitting tanpa regularisasi.  
   - **Hyperparameter yang dituning dengan GridSearchCV**:
     - `n_estimators`: [100, 200]  
     - `learning_rate`: [0.01, 0.1]  
     - `max_depth`: [3, 5, 10]  
     - `subsample`: [0.8, 1.0]

3. **XGBoost Classifier**  
   XGBoost adalah implementasi yang efisien dari gradient boosting yang menggunakan pendekatan boosting berbasis *tree*. Ia mendukung regularisasi L1 dan L2 untuk menghindari overfitting, dan sangat dioptimalkan untuk kecepatan dan performa.  
   - **Keunggulan**: cepat, teroptimasi, mendukung regularisasi.  
   - **Kekurangan**: kompleks dalam tuning hyperparameter.  
   - **Hyperparameter yang dituning dengan GridSearchCV**:
     - `n_estimators`: [100, 200, 300]  
     - `max_depth`: [3, 6, 10]  
     - `learning_rate`: [0.001, 0.01, 0.1]  
     - `subsample`: [0.8, 1.0]

4. **AdaBoost Classifier**  
   AdaBoost bekerja dengan memberi bobot lebih besar pada instance yang salah diklasifikasikan oleh model sebelumnya, sehingga model selanjutnya fokus pada instance tersebut. Model akhir merupakan kombinasi dari semua model sebelumnya dengan bobot tertentu.  
   - **Keunggulan**: sederhana, efektif untuk dataset terbatas.  
   - **Kekurangan**: sensitif terhadap noise dan outlier.  
   - **Hyperparameter yang dituning dengan GridSearchCV**:
     - `n_estimators`: [50, 100, 200]  
     - `learning_rate`: [0.01, 0.1, 1]

### Hyperparameter Tuning
- Hyperparameter tuning dilakukan menggunakan **GridSearchCV** dengan 5-fold cross-validation dan pemanfaatan semua core (`n_jobs=-1`) untuk efisiensi komputasi.
- Penilaian performa tuning menggunakan metrik **accuracy**.
- Setiap model memiliki kombinasi parameter grid yang spesifik, seperti ditunjukkan di atas.

### Pemilihan Model Terbaik
- Model terbaik dipilih berdasarkan evaluasi pada data pengujian menggunakan metrik:
  - **F1-Score**
  - **ROC-AUC**
  - **Confusion Matrix**
- Model dengan performa terbaik digunakan untuk prediksi akhir terhadap data uji.

---

## 6. 📈 Evaluation

### Metrik Evaluasi

- **Accuracy (weighted average)** = (TP + TN) / Total  
- **Precision (weighted average)** = TP / (TP + FP)  
- **Recall (weighted average)** = TP / (TP + FN)  
- **F1 Score (weighted average)** = 2 * (Precision * Recall) / (Precision + Recall)  
- **ROC-AUC** = Luas area di bawah kurva ROC  
- **Confusion Matrix**: evaluasi klasifikasi dengan jumlah TP, TN, FP, dan FN.

### Hasil Evaluasi Model 

| Model           | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|----------------|----------|-----------|--------|----------|---------|
| Random Forest  | 75%    | 75%     | 75%  | 75%    | 0.84    |
| Gradient Boost | 76%    | 76%     | 76%  | 76%    | 0.85    |
| XGBoost        | 76%    | 76%     | 76%  | 76%    | 0.85    |
| **AdaBoost**       | **77%**| **77%** | **77%**| **77%**| **0.86** |

📌 **Model Terbaik: AdaBoost**, karena memberikan skor F1 dan ROC-AUC tertinggi.

### Confusion Matrix (AdaBoost):

|                 | Predicted: Left | Predicted: Stayed |
|-----------------|------------------|--------------------|
| **Actual: Left**   | 1292           | 466                |
| **Actual: Stayed** | 407            | 1560               |

- True Positives (TP) = 1292
→ Karyawan yang benar-benar resign (Leave) dan berhasil diprediksi dengan tepat sebagai resign.

- True Negatives (TN) = 1560
→ Karyawan yang benar-benar stay dan berhasil diprediksi dengan tepat sebagai stay.

- False Positives (FP) = 466
→ Karyawan yang sebenarnya resign, tetapi diprediksi akan stay.
⚠️ Hal ini cukup penting karena bisa berdampak pada kurangnya antisipasi perusahaan terhadap potensi turnover.

- False Negatives (FN) = 407
→ Karyawan yang sebenarnya stay, tetapi diprediksi akan resign.
⚠️ Hal ini bisa menyebabkan kesalahan alokasi sumber daya, seperti upaya retensi yang tidak perlu.

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

