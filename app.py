import os
import numpy as np
import onnxruntime as ort
import threading
import time
import requests
from flask import Flask, request, jsonify  # Tumeondoa render_template hapa
from flask_cors import CORS
from PIL import Image

app = Flask(__name__)
CORS(app)  # Muhimu kwa ajili ya mobile app na web integration

# Load Model (.onnx)
MODEL_PATH = 'eye_disease_model.onnx'
session = ort.InferenceSession(MODEL_PATH)

# Kupata jina la input (lazima kwa ONNX)
input_name = session.get_inputs()[0].name

class_names = ['Cataract', 'Conjunctivitis', 'Normal', 'Trachoma']

# --- LOGIC YA KUAMSHA SERVER (KEEP-ALIVE) ---
def keep_alive():
    """Inapiga picha server kila baada ya dakika 10 kuzuia isilale kwenye Render"""
    while True:
        try:
            # Hii inajipiga yenyewe (Self-ping)
            # Render inatoa URL kwenye env variable RENDER_EXTERNAL_HOSTNAME
            host = os.environ.get('RENDER_EXTERNAL_HOSTNAME')
            if host:
                url = f"https://{host}/health"
                requests.get(url)
                print("Keep-alive: Ping sent!")
        except Exception as e:
            print(f"Keep-alive error: {e}")
        time.sleep(600)  # Dakika 10 (sekunde 600)

@app.route('/health')
def health():
    return "I am awake!", 200
# --------------------------------------------

# IMEBADILISHWA: Sasa inarudisha JSON safi badala ya kutafuta index.html
@app.route('/')
def index():
    return jsonify({
        'status': 'success',
        'message': 'Eye Diseases Detection API ipo hai na tayari kupokea picha kupitia endpoint ya /predict'
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Preprocessing (Logic ile ile)
        img = Image.open(file).convert('RGB')
        img = img.resize((224, 224))
        
        # ONNX inahitaji float32 na division ya 255.0
        img_array = np.array(img).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Prediction kwa kutumia ONNX
        outputs = session.run(None, {input_name: img_array})
        predictions = outputs[0]
        
        score = np.max(predictions)
        result = class_names[np.argmax(predictions)]

        return jsonify({
            'prediction': result,
            'confidence': f"{score * 100:.2f}%"
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Anza thread ya kuamsha server kama tuko kwenye Render
    if os.environ.get('RENDER'):
        threading.Thread(target=keep_alive, daemon=True).start()
    
    # Render inahitaji bind kwenye 0.0.0.0 na kutumia Port inayopewa
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
