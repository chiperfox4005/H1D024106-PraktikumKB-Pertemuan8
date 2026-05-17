import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

dataset_path = r"E:\INFORMATIKA\FILE VSC!!\PERTEMUAN 8\archive"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(
        32,
        (3,3),
        activation='relu',
        input_shape=(150,150,3)
    ),

    MaxPooling2D(2,2),

    Conv2D(
        64,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Conv2D(
        128,
        (3,3),
        activation='relu'
    ),

    MaxPooling2D(2,2),

    Flatten(),

    Dense(
        512,
        activation='relu'
    ),

    Dense(
        4,
        activation='softmax'
    )
])

print("\n===== ARSITEKTUR MODEL CNN =====")
model.summary()

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

print("\n===== MEMULAI PROSES TRAINING =====")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=10
)

print("\n===== EVALUASI MODEL =====")

val_loss, val_acc = model.evaluate(validation_generator)

print(f'Validation Loss     : {val_loss:.4f}')
print(f'Validation Accuracy : {val_acc:.4f}')

history_df = pd.DataFrame(history.history)

history_df[['accuracy', 'val_accuracy']].plot(figsize=(10,5))
plt.title('Grafik Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

history_df[['loss', 'val_loss']].plot(figsize=(10,5))
plt.title('Grafik Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

print("\n===== HASIL PREDIKSI =====")

predictions = model.predict(validation_generator)

print(predictions)

print("\n===== LABEL KELAS =====")
print(train_generator.class_indices)

model.save("cnn_rockpaperscissors.h5")

print("\nModel berhasil disimpan dengan nama:")
print("cnn_rockpaperscissors.h5")