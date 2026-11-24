"""Entrena el clasificador visual con transferencia de aprendizaje (ResNet50).

Corrección del profesor (P3): división 70/15/15 en train / validación / PRUEBA.
- train: entrena la cabeza.
- validación: early stopping.
- prueba: SOLO para las métricas finales (el modelo nunca la ve al entrenar).

Uso (Colab con GPU):  python -m scripts.train_vision --data /content/drive/MyDrive/dataset_crustaceo
Salida: models/vision_head.npz, models/metrics_vision.json, models/split_vision.json,
        docs/metricas/confusion_vision.png
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from core import config
from core.vision import IMG_SIZE, cargar_imagen, preprocess_resnet
from scripts.train_nlp import _plot_cm

SEED = 42
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def listar(data_dir: Path):
    rutas, etiquetas = [], []
    clases = sorted(d.name for d in data_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    for c in clases:
        for f in sorted((data_dir / c).iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            try:
                cargar_imagen(f.read_bytes(), max_bytes=None)  # descarta archivos corruptos o de formato raro
            except Exception as e:
                print(f"  [omitida] {f.name}: {e}")
                continue
            rutas.append(f); etiquetas.append(c)
    return rutas, etiquetas, clases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

    import tensorflow as tf
    from tensorflow import keras
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    rutas, y, clases = listar(Path(args.data))
    print(f"{len(rutas)} imágenes, clases: { {c: y.count(c) for c in clases} }")
    r_tr, r_tmp, y_tr, y_tmp = train_test_split(rutas, y, test_size=0.30, stratify=y, random_state=SEED)
    r_va, r_te, y_va, y_te = train_test_split(r_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=SEED)
    print(f"train={len(r_tr)}  val={len(r_va)}  test={len(r_te)}")

    base = keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg",
                                       input_shape=IMG_SIZE + (3,))
    base.trainable = False

    def features(paths, flip=False):
        imgs = []
        for p in paths:
            x = cargar_imagen(Path(p).read_bytes(), max_bytes=None)
            imgs.append(x[:, ::-1] if flip else x)
        return base.predict(preprocess_resnet(np.stack(imgs)), batch_size=32, verbose=0)

    idx = {c: i for i, c in enumerate(clases)}
    # Aumento de datos simple: espejo horizontal del set de entrenamiento
    X_tr = np.concatenate([features(r_tr), features(r_tr, flip=True)])
    Y_tr = np.array([idx[c] for c in y_tr] * 2)
    X_va, Y_va = features(r_va), np.array([idx[c] for c in y_va])
    X_te, Y_te = features(r_te), np.array([idx[c] for c in y_te])

    head = keras.Sequential([
        keras.Input(shape=(2048,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(len(clases), activation="softmax"),
    ])
    head.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy",
                 metrics=["accuracy"])
    hist = head.fit(X_tr, Y_tr, validation_data=(X_va, Y_va), epochs=args.epochs, batch_size=16,
                    callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
                    verbose=2)

    def evaluar(X, Y):
        pred = head.predict(X, verbose=0).argmax(1)
        rep = classification_report(Y, pred, labels=range(len(clases)), target_names=clases,
                                    output_dict=True, zero_division=0)
        return pred, rep

    _, rep_va = evaluar(X_va, Y_va)
    pred_te, rep_te = evaluar(X_te, Y_te)
    cm = confusion_matrix(Y_te, pred_te, labels=range(len(clases)))
    print("\n=== MÉTRICAS EN PRUEBA (conjunto independiente) ===")
    print(classification_report(Y_te, pred_te, target_names=clases, zero_division=0))

    config.MODELS_DIR.mkdir(exist_ok=True)
    d1, d2 = head.layers[0], head.layers[2]
    np.savez(config.VISION_HEAD_PATH, W1=d1.get_weights()[0], b1=d1.get_weights()[1],
             W2=d2.get_weights()[0], b2=d2.get_weights()[1], classes=np.array(clases))
    metrics = {
        "modelo": "ResNet50 (ImageNet, congelada) + Dense128 + Dropout0.3 + softmax",
        "n_por_clase": {c: y.count(c) for c in clases},
        "split": {"train": len(r_tr), "val": len(r_va), "test": len(r_te)},
        "epocas_entrenadas": len(hist.history["loss"]),
        "val_accuracy": round(rep_va["accuracy"], 4),
        "test": {
            "accuracy": round(accuracy_score(Y_te, pred_te), 4),
            "macro_precision": round(rep_te["macro avg"]["precision"], 4),
            "macro_recall": round(rep_te["macro avg"]["recall"], 4),
            "macro_f1": round(rep_te["macro avg"]["f1-score"], 4),
            "por_clase": {c: {m: round(rep_te[c][m], 4) for m in ("precision", "recall", "f1-score", "support")}
                          for c in clases},
            "confusion_matrix": cm.tolist(),
        },
        "historial": {k: [round(float(v), 4) for v in vals] for k, vals in hist.history.items()},
        "clases": clases,
    }
    (config.MODELS_DIR / "metrics_vision.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    (config.MODELS_DIR / "split_vision.json").write_text(json.dumps(
        {"train": [str(p) for p in r_tr], "val": [str(p) for p in r_va], "test": [str(p) for p in r_te]},
        ensure_ascii=False, indent=1))
    _plot_cm(cm, clases, "Matriz de confusión – visión (prueba independiente)", "confusion_vision.png")
    print(json.dumps(metrics["test"] | {"confusion_matrix": None}, indent=2))


if __name__ == "__main__":
    main()
