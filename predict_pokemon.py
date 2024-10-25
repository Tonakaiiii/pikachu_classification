import tensorflow as tf
import cv2
import numpy as np
import joblib

# モデルとラベルエンコーダーの読み込み
model = tf.keras.models.load_model('pokemon_model.h5')
le = joblib.load('label_encoder.pkl')

# 画像のサイズ
img_size = 100

def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255.0  # 正規化
    img = np.expand_dims(img, axis=0)  # バッチサイズを追加

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)  # 最大の予測値のインデックスを取得
    return le.inverse_transform([predicted_class])[0]  # インデックスをラベルに変換

# 画像を判別
result = predict_image('test/4.jpg')  # 判別したい画像のパスを指定
print(result)

#testブランチで編集