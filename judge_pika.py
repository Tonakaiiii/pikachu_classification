import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np

# データセットのパス
dataset_path = 'dataset'
img_size = 100  # 画像のサイズを指定

# データセットの準備
def load_data():
    images = []
    labels = []

    for label in ['pikachu', 'not_pikachu']:
        folder_path = os.path.join(dataset_path, label)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (img_size, img_size))
            images.append(img)
            labels.append(1 if label == 'pikachu' else 0)  # ピカチュウなら1、そうでなければ0

    return np.array(images), np.array(labels)

# データの読み込み
X, y = load_data()
X = X / 255.0  # 正規化

# データの分割
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# モデルの作成
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # 0または1の二値分類
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# モデルの訓練
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# モデルの保存
model.save('pikachu_model.h5')
