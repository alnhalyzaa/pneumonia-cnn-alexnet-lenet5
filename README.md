# 🧠 Deteksi Pneumonia Menggunakan CNN (LeNet-5 & AlexNet)

Proyek ini bertujuan untuk membangun dan membandingkan dua arsitektur **Convolutional Neural Network (CNN)** — yaitu **LeNet-5** dan **AlexNet** — dalam mendeteksi pneumonia berdasarkan citra X-ray dada.

## 📂 Dataset
Dataset yang digunakan adalah **Chest X-Ray Images (Pneumonia)** yang berasal dari [Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia).

Dataset terdiri dari dua kelas:
- 🫁 **NORMAL**: Citra paru-paru sehat  
- 🩻 **PNEUMONIA**: Citra paru-paru dengan indikasi pneumonia


## 🧩 Tahapan Umum Analisis
1. **Exploratory Data Analysis (EDA)**  
    - Menghitung jumlah citra per kelas
    - Melihat distribusi dan ukuran citra
    - Menampilkan contoh citra dari tiap kelas  
      
2. **Preprocessing Data**  
   - Mengubah ukuran citra menjadi seragam  
   - Melakukan normalisasi nilai piksel  
   - Menerapkan augmentasi data 

3. **Pembangunan Model CNN**  
   - Menggunakan dua arsitektur: **LeNet-5** dan **AlexNet**  
   - Fungsi aktivasi **ReLU**, optimizer **Adam**, dan loss **Binary Crossentropy**  

4. **Training & Evaluasi**  
   - Model dilatih menggunakan data train dan dievaluasi menggunakan data validasi/test  
   - Metrik utama: **auc**, **loss**, dan **confusion matrix**

## 🧮 Arsitektur 1: LeNet-5

### 🔍 Tahapan Analisis
Notebook: `Arsitektur_LeNet5.ipynb`

1. **EDA & Preprocessing**
   - Ukuran citra diubah menjadi **32×32 piksel**
   - Nilai piksel dinormalisasi ke rentang [0, 1]
   - Data diubah ke grayscale

2. **Arsitektur Model**
   - 2 layer konvolusi (Conv2D)
   - 2 layer pooling (MaxPooling2D)
   - 2 fully connected layer (Dense)
   - 1 output neuron (sigmoid)

3. **Konfigurasi Model**
   - Optimizer: Adam  
   - Loss Function: Binary Crossentropy  
   - Activation Function: ReLU  
   - Output Activation: Sigmoid  

4. **Hasil**
   - Akurasi cukup baik, namun cenderung underfitting pada data kompleks

## 🧮 Arsitektur 2: AlexNet

### 🔍 Tahapan Analisis
Notebook: `Arsitektur_AlexNet.ipynb`

1. **EDA & Preprocessing**
   - Ukuran citra diubah menjadi **224×224 piksel**
   - Normalisasi nilai piksel ke rentang [0, 1]
   - Augmentasi data untuk meningkatkan generalisasi model

2. **Arsitektur Model**
   - 5 layer konvolusi dengan filter besar di awal (11×11, 5×5, 3×3)
   - 3 layer fully connected (4096, 4096, 1)
   - Batch Normalization & Dropout untuk mencegah overfitting

3. **Konfigurasi Model**
   - Optimizer: Adam  
   - Loss Function: Binary Crossentropy  
   - Activation Function: ReLU  
   - Output Activation: Sigmoid  

4. **Ringkasan Arsitektur**
   - Total Parameter: **46.753.577 (~178 MB)**  
   - Trainable Params: 46.752.361  
   - Non-trainable Params: 1.216  

5. **Hasil**
   - Model menunjukkan akurasi lebih tinggi daripada LeNet-5  
   - Lebih stabil karena Batch Normalization dan Dropout  
   - Waktu pelatihan lebih lama karena jumlah parameter yang besar

## 📊 Perbandingan Kinerja

| Model | Ukuran Input | Akurasi | Catatan |
|--------|---------------|------------------|----------|----------|
| **LeNet-5** | 32×32×3 | Sedang | Model sederhana, cepat, cocok baseline |
| **AlexNet** | 224×224×3 | Tinggi | Lebih akurat, stabil, tapi berat |

## ⚙️ Tools & Environment
- Python 3.10+
- TensorFlow / Keras
- NumPy, Matplotlib, Seaborn
- Jupyter Notebook / Google Colab

## 🏁 Kesimpulan
- **LeNet-5** cocok sebagai model dasar untuk pengenalan pola sederhana.  
- **AlexNet** memberikan hasil lebih baik karena kedalaman jaringan dan jumlah parameter yang besar.  
- Model AlexNet lebih cocok digunakan pada dataset citra medis berukuran besar dan kompleks.  

Dataset: [Chest X-Ray Images (Pneumonia) – Kaggle](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
