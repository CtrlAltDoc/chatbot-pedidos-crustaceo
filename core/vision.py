"""Clasificador visual: ResNet50 (ImageNet, congelada) + cabeza entrenada.

Para que el repo sea ligero, solo se guarda la cabeza entrenada en
`models/vision_head.npz` (pesos Dense + nombres de clase). La base
ResNet50 se descarga de Keras la primera vez que arranca la app.
"""
import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from core import config

IMG_SIZE = (224, 224)


class InvalidImageError(ValueError):
    """La imagen no se pudo leer o no es un formato permitido."""


def cargar_imagen(data: bytes, max_bytes: int | None = config.MAX_IMAGE_BYTES) -> np.ndarray:
    if max_bytes and len(data) > max_bytes:
        raise InvalidImageError(f"La imagen pesa más de {max_bytes // 2**20} MB.")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        img.verify()  # detecta archivos corruptos
        img = Image.open(io.BytesIO(data))
    except (UnidentifiedImageError, OSError) as e:
        raise InvalidImageError("El archivo no es una imagen válida.") from e
    if fmt not in config.ALLOWED_IMAGE_FORMATS:
        raise InvalidImageError(f"Formato {fmt} no permitido. Usa JPG, PNG o WEBP.")
    img = img.convert("RGB").resize(IMG_SIZE)
    return np.asarray(img, dtype=np.float32)


def preprocess_resnet(x: np.ndarray) -> np.ndarray:
    """Igual a keras.applications.resnet50.preprocess_input (modo 'caffe')."""
    x = x[..., ::-1].copy()  # RGB -> BGR
    x -= np.array([103.939, 116.779, 123.68], dtype=np.float32)
    return x


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class VisionClassifier:
    def __init__(self, head_path=config.VISION_HEAD_PATH, extractor=None):
        head = np.load(head_path, allow_pickle=False)
        self.W1, self.b1 = head["W1"], head["b1"]
        self.W2, self.b2 = head["W2"], head["b2"]
        self.classes = [str(c) for c in head["classes"]]
        # extractor: función imagen(224x224x3) -> vector 2048. Se inyecta en pruebas.
        self.extractor = extractor or self._resnet_extractor()

    @staticmethod
    def _resnet_extractor():
        from tensorflow.keras.applications import ResNet50  # import perezoso: TF es pesado
        base = ResNet50(weights="imagenet", include_top=False, pooling="avg",
                        input_shape=IMG_SIZE + (3,))
        return lambda x: base.predict(preprocess_resnet(x)[None, ...], verbose=0)[0]

    def predict_array(self, x: np.ndarray) -> dict:
        feats = self.extractor(x)
        h = np.maximum(feats @ self.W1 + self.b1, 0)  # Dense(128, relu)
        proba = _softmax(h @ self.W2 + self.b2)       # Dense(n, softmax)
        order = np.argsort(proba)[::-1]
        return {
            "label": self.classes[order[0]],
            "confidence": round(float(proba[order[0]]), 4),
            "top_k": [{"label": self.classes[i], "confidence": round(float(proba[i]), 4)}
                      for i in order[:3]],
        }

    def predict(self, data: bytes) -> dict:
        return self.predict_array(cargar_imagen(data))
