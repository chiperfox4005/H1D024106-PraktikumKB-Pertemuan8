# Praktikum KB Pertemuan 8 - Convolutional Neural Network (CNN)

## Deskripsi

Program ini merupakan implementasi **Convolutional Neural Network (CNN)** menggunakan TensorFlow dan Keras untuk melakukan klasifikasi gambar **Rock, Paper, Scissors**.

Model CNN digunakan untuk mengenali gambar berdasarkan pola visual dari dataset gambar yang telah disediakan.

---

## Dataset

Dataset yang digunakan:

* Rock
* Paper
* Scissors

Dataset diperoleh dari:
[Kaggle Rock Paper Scissors Dataset](https://www.kaggle.com/datasets/drgfreeman/rockpaperscissors?utm_source=chatgpt.com)

---

## Library yang Digunakan

```bash id="38p9jl"
pip install tensorflow numpy pandas matplotlib
```

---

## Struktur Folder

```txt id="ulr2n4"
PERTEMUAN 8
│
├── praktikum8_cnn.py
│
└── archive
    ├── rock
    ├── paper
    ├── scissors
```

---

## Import Library

```python id="4y3zef"
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
```

---

## Arsitektur CNN

Model CNN terdiri dari:

* 3 Layer Convolution
* 3 Layer MaxPooling
* Flatten Layer
* Dense Layer
* Output Layer Softmax

---

## Cara Menjalankan Program

Jalankan perintah berikut pada terminal:

```bash id="qfxl0r"
python praktikum8_cnn.py
```

---

## Proses Program

Program akan melakukan:

1. Membaca dataset gambar
2. Preprocessing menggunakan ImageDataGenerator
3. Membagi data training dan validation
4. Membangun model CNN
5. Melatih model
6. Menampilkan accuracy dan loss
7. Menampilkan hasil prediksi
8. Menyimpan model CNN

---

## Hasil Output

Program menghasilkan:

* Ringkasan model CNN
* Grafik Accuracy
* Grafik Loss
* Validation Accuracy
* Validation Loss
* File model:

```txt id="y3g6xq"
cnn_rockpaperscissors.h5
```

---

## Author

Nama : Fikry Mumtaz Pratama
NIM : H1D024106
Program Studi : Teknik Informatika
