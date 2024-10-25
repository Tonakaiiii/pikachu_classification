import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# データセットのパス
dataset_path = 'dataset'
img_size = 100  # 画像のサイズを指定

# データセットの準備
def load_data():
    images = []
    labels = []

    # ディレクトリを巡回
    for folder_name in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, folder_name)
        if os.path.isdir(folder_path):
            label = folder_name.split('_')[1]  # 数字の後の部分をラベルとする
            for img_name in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_name)
                img = cv2.imread(img_path)
                img = cv2.resize(img, (img_size, img_size))
                images.append(img)
                labels.append(label)  # ラベルを追加

    return np.array(images), np.array(labels)

# データの読み込み
X, y = load_data()
X = X / 255.0  # 正規化

# ラベルを数値に変換
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# データの分割
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# モデルの作成
num_classes = len(le.classes_)  # クラスの数を取得
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # 複数クラスの出力
])

# モデルのコンパイル
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# モデルの訓練
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# モデルの保存
model.save('model/pokemon_model.h5')

# ラベルエンコーダーを保存
import joblib
joblib.dump(le, 'model/label_encoder.pkl')
