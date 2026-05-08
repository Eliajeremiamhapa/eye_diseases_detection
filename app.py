import os
import numpy as np
import tensorflow as tf
import threading
import time
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app) # Hii inaruhusu mobile apps na websites kuita API yako bila block

# Load Model
MODEL_PATH = 'eye_disease_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)
class_names = ['Cataract', 'Conjunctivitis', 'Normal', 'Trachoma']

# --- SEHEMU YA KUZUIA SERVER ISILALE ---
def keep_alive():
    """Inapiga picha server kila baada ya dakika 10 kuzuia isilale"""
    while True:
        try:
            # Badilisha 'your-app-name.onrender.com' na URL yako halisi ya Render
            url = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost:5000')}/health"
            requests.get(url)
        except Exception as e:
            print(f"Keep-alive error: {e}")
        time.sleep(600) # Inasubiri sekunde 600 (Dakika 10)

@app.route('/health')
def health():
    return "I am alive!", 200
# ----------------------------------------

@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "Eye Disease Detection API is running"
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Preprocessing
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction
        predictions = model.predict(img_array)
        score = np.max(predictions)
        result = class_names[np.argmax(predictions)]

        return jsonify({
            'prediction': result,
            'confidence': float(score), # Imerudishwa kama namba kwa urahisi wa mobile apps
            'confidence_percent': f"{score * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Anzisha thread ya kuzuia server isilale
    if os.environ.get('RENDER'):
        threading.Thread(target=keep_alive, daemon=True).start()
    
    # Render anatumia PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)