"""
entrenamiento.py — SVM Fashion-MNIST con HOG + pixeles -> objetivo >= 92%
==========================================================================
Estrategia:
  * Features = HOG(324) concatenado con pixeles normalizados(784) = 1108 dims
    HOG captura bordes/texturas → diferencia mejor Shirt vs T-shirt vs Coat
  * StandardScaler -> PCA(300) -> SVC(rbf, C=100, gamma='scale')
  * class_weight='balanced' para compensar clases dificiles
  * 70,000 datos completos con split 80/20 estratificado

Split: 80% entrenamiento / 20% prueba (estratificado)

Uso:
    pip install scikit-image   (una sola vez, ademas de requirements.txt)
    python entrenamiento.py
"""

import time
from pathlib import Path

import joblib
import numpy as np
from skimage.feature import hog
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# --------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------
MODEL_PATH   = Path("model.pkl")
TEST_SIZE    = 0.20
RANDOM_STATE = 42
TARGET_ACC   = 0.92

CLASS_NAMES = [
    "T-shirt/top", "Trouser",  "Pullover", "Dress",  "Coat",
    "Sandal",       "Shirt",    "Sneaker",  "Bag",    "Ankle boot",
]


# --------------------------------------------------------------
# Extraccion de features HOG + pixeles
# --------------------------------------------------------------
def extract_hog_features(X: np.ndarray) -> np.ndarray:
    """
    Para cada imagen (vector de 784 pixeles):
      1. Reshape a (28, 28)
      2. Calcula HOG: orientaciones=9, celda=7x7px, bloque=2x2 celdas -> 324 features
      3. Concatena HOG(324) + pixeles_normalizados(784) = 1108 features

    HOG captura gradientes direccionales (bordes, texturas) que
    distinguen mejor Shirt de T-shirt/top y de Coat.
    """
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
        if i % 5000 == 0 and i > 0:
            print(f"      HOG: {i:,}/{n:,} imagenes procesadas …")

    # Normalizar pixeles a [0,1] y concatenar
    X_norm = X.astype(np.float32) / 255.0
    return np.hstack([hog_feats, X_norm])   # (n, 1108)


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=300, random_state=RANDOM_STATE)),
        ("svm",    SVC(
            kernel="rbf",
            C=100,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_STATE,
            cache_size=2000,
        )),
    ])


def main() -> None:
    print("=" * 64)
    print("  ENTRENAMIENTO -- SVM Fashion-MNIST  (objetivo >= 92%)")
    print("=" * 64)

    # 1. Cargar datos completos
    print("\n   Cargando Fashion-MNIST ... (primera vez ~30-60 s)")
    X, y = fetch_openml(
        "Fashion-MNIST", version=1,
        return_X_y=True, as_frame=False,
        parser="liac-arff",
    )
    y = y.astype(int)
    print(f"    Total: {X.shape[0]:,} imagenes x {X.shape[1]} pixeles")

    # 2. Split 80/20 estratificado
    print(f"\n    Split estratificado 80/20 ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    print(f"    Train: {X_train.shape[0]:,}  (80%)")
    print(f"    Test : {X_test.shape[0]:,}  (20%)")

    # 3. Extraer features HOG + pixeles
    print(f"\n    Extrayendo features HOG + pixeles ...")
    print(f"    Train ({X_train.shape[0]:,} imagenes) ...")
    t_hog = time.time()
    X_train_feat = extract_hog_features(X_train)
    X_test_feat  = extract_hog_features(X_test)
    print(f"    Completado en {time.time()-t_hog:.1f} s")
    print(f"    Dimensiones: {X_train_feat.shape[1]} features por imagen (HOG + pixeles)")

    # 4. Entrenar pipeline
    pipe = build_pipeline()
    print(f"\n    Entrenando: StandardScaler -> PCA(300) -> SVM(RBF, C=100) ...")
    print(f"    [Tiempo estimado: 8-20 min segun CPU]")
    t0 = time.time()
    pipe.fit(X_train_feat, y_train)
    elapsed = time.time() - t0
    print(f"    Completado en {elapsed:.1f} s  ({elapsed/60:.1f} min)")

    # 5. Evaluacion
    print("\n    Evaluando en conjunto de prueba ...")
    y_pred = pipe.predict(X_test_feat)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n{'='*64}")
    print(f"  Accuracy en test : {acc:.4f}  ({acc*100:.2f} %)")
    print(f"  Objetivo minimo  : {TARGET_ACC:.2f}  ({TARGET_ACC*100:.0f} %)")
    estado = "APROBADO" if acc >= TARGET_ACC else "NO ALCANZADO"
    print(f"  Estado           : {estado}")
    print(f"{'='*64}\n")

    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    # 6. Guardar pipeline (el HOG se aplica antes del pipeline, se guarda aparte)
    joblib.dump(pipe, MODEL_PATH)
    size_mb = MODEL_PATH.stat().st_size / 1_048_576
    print(f"    Pipeline guardado en '{MODEL_PATH}'  ({size_mb:.1f} MB)")
    print("    -> Ahora puedes ejecutar: python app.py")


if __name__ == "__main__":
    main()
