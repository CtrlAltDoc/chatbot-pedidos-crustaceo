"""Clasificador de intenciones (TF-IDF + SVM) y extracción de entidades del menú."""
import re

import joblib
import numpy as np

from core import config
from core.text import normalizar, preprocesar


class IntentClassifier:
    def __init__(self, model_path=config.INTENT_MODEL_PATH):
        self.pipeline = joblib.load(model_path)  # TfidfVectorizer + SVM calibrado
        self.labels = list(self.pipeline.classes_)

    def predict(self, texto: str) -> dict:
        limpio = preprocesar(texto)
        proba = self.pipeline.predict_proba([limpio])[0]
        order = np.argsort(proba)[::-1]
        return {
            "intent": self.labels[order[0]],
            "confidence": round(float(proba[order[0]]), 4),
            "top_k": [{"intent": self.labels[i], "confidence": round(float(proba[i]), 4)}
                      for i in order[:3]],
            "preprocesado": limpio,
        }


_NUMEROS = {"un": 1, "una": 1, "uno": 1, "dos": 2, "tres": 3, "cuatro": 4, "cinco": 5}


def extraer_productos(texto: str, menu: dict) -> list[dict]:
    """Detecta productos del menú y cantidades ("dos papas" -> papas x2)."""
    t = f" {normalizar(texto)} "
    encontrados = {}
    # Primero los sinónimos largos ("hamburguesa doble") y se borran del texto,
    # así "cangreburger doble" no cuenta además como "cangreburger".
    pares = sorted(((s, pid) for pid, p in menu["productos"].items() for s in p["sinonimos"]),
                   key=lambda x: -len(x[0]))
    numeros = "|".join(_NUMEROS)
    for sin, pid in pares:
        patron = rf"\b(?:(\d+|{numeros})\s+(?:\w+\s+)?)?{re.escape(sin)}\b"
        while (m := re.search(patron, t)):
            cant_txt = m.group(1)
            cantidad = int(cant_txt) if cant_txt and cant_txt.isdigit() else _NUMEROS.get(cant_txt, 1)
            encontrados[pid] = encontrados.get(pid, 0) + cantidad
            t = t[:m.start()] + " " + t[m.end():]
    return [{"producto": pid, "cantidad": c} for pid, c in encontrados.items()]
