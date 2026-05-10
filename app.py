"""
app.py — API REST Flask para SVM Fashion-MNIST con features HOG
===============================================================
Endpoints:
    GET  /                 -> sirve index.html
    GET  /health           -> estado del servicio
    POST /predict          -> predice 1 imagen  {"pixels": [784 floats]}
    POST /predict/batch    -> predice N imagenes {"images": [[784 floats], ...]}
"""

import os
import urllib.request
from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

# --------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------
MODEL_PATH = Path(os.getenv("MODEL_PATH", "model.pkl"))
MODEL_URL  = os.getenv("MODEL_URL", "")
PORT       = int(os.getenv("PORT", 5000))

CLASS_NAMES = [
    "T-shirt/top", "Trouser",  "Pullover", "Dress",  "Coat",
    "Sandal",       "Shirt",    "Sneaker",  "Bag",    "Ankle boot",
]

# --------------------------------------------------------------
# Cargar pipeline (descarga desde Hugging Face si no existe)
# --------------------------------------------------------------
pipeline = None

def load_pipeline() -> None:
    global pipeline

    if not MODEL_PATH.exists():
        if MODEL_URL:
            print(f"   Descargando modelo desde Hugging Face...")
            try:
                urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
                print(f"   Descarga completa: {MODEL_PATH} ({MODEL_PATH.stat().st_size // 1_000_000} MB)")
            except Exception as e:
                print(f"   Error al descargar el modelo: {e}")
                return
        else:
            print(f"   No se encontro '{MODEL_PATH}' y MODEL_URL no esta definida.")
            return

    try:
        pipeline = joblib.load(MODEL_PATH)
        print(f"   Pipeline cargado correctamente desde '{MODEL_PATH}'")
    except Exception as e:
        print(f"   Error al cargar el pipeline: {e}")

# --------------------------------------------------------------
# HOG + pixeles (misma funcion que en entrenamiento.py)
# --------------------------------------------------------------
def extract_hog_features(X: np.ndarray) -> np.ndarray:
    """Extrae HOG(324) + pixeles_norm(784) = 1108 features por imagen."""
    n = X.shape[0]
    hog_feats = np.zeros((n, 324), dtype=np.float32)
    for i in range(n):
        img = X[i].reshape(28, 28)
        hog_feats[i] = hog(
            img,
            orientations=9,
            pixels_per_cell=(7, 7),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
        )
    X_norm = X.astype(np.float32) / 255.0
    return np.hstack([hog_feats, X_norm])

# --------------------------------------------------------------
# App Flask
# --------------------------------------------------------------
app = Flask(__name__)
CORS(app)
load_pipeline()

# --------------------------------------------------------------
# Helpers
# --------------------------------------------------------------
def model_ready() -> bool:
    return pipeline is not None

def parse_pixels(data: list) -> np.ndarray:
    arr = np.array(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[1] != 784:
        raise ValueError(f"Se esperan 784 pixeles (28x28). Recibidos: {arr.shape[1]}")
    return arr

def run_inference(pixels: np.ndarray) -> list:
    features = extract_hog_features(pixels)
    preds    = pipeline.predict(features)
    return [
        {"class_index": int(p), "class_name": CLASS_NAMES[int(p)]}
        for p in preds
    ]

# --------------------------------------------------------------
# Rutas
# --------------------------------------------------------------

# GET / — sirve la interfaz web
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

# GET /health
@app.get("/health")
def health():
    return jsonify({
        "status":       "ok" if model_ready() else "model_not_loaded",
        "model_loaded": model_ready(),
        "model_path":   str(MODEL_PATH),
        "classes":      CLASS_NAMES,
    })

# POST /predict
@app.post("/predict")
def predict():
    if not model_ready():
        return jsonify({"error": "Modelo no cargado."}), 503

    body = request.get_json(silent=True)
    if not body or "pixels" not in body:
        return jsonify({"error": "Se requiere el campo 'pixels' en el body JSON."}), 400

    try:
        pixels  = parse_pixels(body["pixels"])
        results = run_inference(pixels)
        return jsonify({"prediction": results[0]})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 422

# POST /predict/batch
@app.post("/predict/batch")
def predict_batch():
    if not model_ready():
        return jsonify({"error": "Modelo no cargado."}), 503

    body = request.get_json(silent=True)
    if not body or "images" not in body:
        return jsonify({"error": "Se requiere el campo 'images' en el body JSON."}), 400

    images = body["images"]
    if not isinstance(images, list) or len(images) == 0:
        return jsonify({"error": "'images' debe ser una lista no vacia."}), 400

    try:
        all_pixels = np.vstack([parse_pixels(img) for img in images])
        results    = run_inference(all_pixels)
        return jsonify({"count": len(results), "predictions": results})
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 422

# --------------------------------------------------------------
# Entry-point local
# --------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("  API Flask -- SVM Fashion-MNIST")
    print("=" * 60)
    print(f"\n   Servidor en http://0.0.0.0:{PORT}")
    print(f"   GET  /")
    print(f"   GET  /health")
    print(f"   POST /predict")
    print(f"   POST /predict/batch\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)