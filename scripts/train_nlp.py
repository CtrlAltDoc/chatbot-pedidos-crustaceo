"""Entrena el clasificador de intenciones (TF-IDF + SVM) y guarda modelo + métricas.

Uso:  python -m scripts.train_nlp
Salida: models/intent_pipeline.joblib, models/metrics_nlp.json, docs/metricas/confusion_nlp.png
"""
import json
import random

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from core import config
from core.text import motor_nlp, preprocesar

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def main():
    df = pd.read_csv(config.DATA_DIR / "intenciones.csv")
    df["limpio"] = df["mensaje"].map(preprocesar)
    X_tr, X_te, y_tr, y_te = train_test_split(df["limpio"], df["intencion"], test_size=0.2,
                                              stratify=df["intencion"], random_state=SEED)
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
        # SVM lineal calibrado para obtener probabilidades (confianza) por intención
        ("svm", CalibratedClassifierCV(LinearSVC(C=1.0, random_state=SEED), cv=3)),
    ])
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    labels = sorted(df["intencion"].unique())
    report = classification_report(y_te, y_pred, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    print(classification_report(y_te, y_pred, labels=labels, zero_division=0))

    config.MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, config.INTENT_MODEL_PATH)
    errores = [{"mensaje": df.loc[i, "mensaje"], "esperada": t, "predicha": p}
               for i, t, p in zip(X_te.index, y_te, y_pred) if t != p]
    metrics = {
        "motor_preprocesamiento": motor_nlp(),
        "n_total": int(len(df)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "accuracy": round(accuracy_score(y_te, y_pred), 4),
        "macro_precision": round(report["macro avg"]["precision"], 4),
        "macro_recall": round(report["macro avg"]["recall"], 4),
        "macro_f1": round(report["macro avg"]["f1-score"], 4),
        "por_clase": {k: {m: round(v[m], 4) for m in ("precision", "recall", "f1-score", "support")}
                      for k, v in report.items() if k in labels},
        "labels": labels, "confusion_matrix": cm.tolist(), "errores_test": errores,
    }
    (config.MODELS_DIR / "metrics_nlp.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2))
    _plot_cm(cm, labels, "Matriz de confusión – intenciones (test)", "confusion_nlp.png")
    print(json.dumps({k: metrics[k] for k in ("accuracy", "macro_f1", "n_test")}, indent=2))


def _plot_cm(cm, labels, title, filename):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = config.ROOT / "docs" / "metricas"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicha"); ax.set_ylabel("Real"); ax.set_title(title)
    fig.tight_layout(); fig.savefig(out / filename, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
