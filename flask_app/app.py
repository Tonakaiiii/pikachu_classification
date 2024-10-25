from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
import cv2
import joblib
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # CORSを有効にする


# モデルとラベルエンコーダーの読み込み
model = tf.keras.models.load_model('model/pokemon_model.h5')
le = joblib.load('model/label_encoder.pkl')

# 画像のサイズ
img_size = 100

def predict_image(img):
    img = cv2.resize(img, (img_size, img_size))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    return le.inverse_transform([predicted_class])[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # 画像を読み込む
            img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            result = predict_image(img)
            return jsonify({'result': result})
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
