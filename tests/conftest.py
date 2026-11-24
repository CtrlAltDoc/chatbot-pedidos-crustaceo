"""Fixtures: modelo NLP real (entrenado con data/) y un clasificador visual de prueba.

El visual usa un extractor falso (color promedio) para no depender de TensorFlow
en las pruebas; la lógica de la cabeza, umbrales y errores es la misma.
"""
import io

import numpy as np
import pytest
from PIL import Image

from core import config

COLORES = {"cangreburger": (150, 90, 40), "malteada": (120, 200, 120), "papas": (240, 200, 60)}


def png(color, fmt="PNG"):
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture(scope="session")
def nlp():
    from core.nlp import IntentClassifier
    if not config.INTENT_MODEL_PATH.exists():
        from scripts import train_nlp
        train_nlp.main()
    return IntentClassifier()


@pytest.fixture(scope="session")
def vision(tmp_path_factory):
    from core.vision import VisionClassifier
    clases = list(COLORES)
    # extractor: vector de 2048 con el color promedio repetido (normalizado)
    extractor = lambda x: np.tile(x.reshape(-1, 3).mean(0) / 255.0, 683)[:2048]
    # cabeza que separa las clases por color: W1 = identidad sobre 3 canales
    W1 = np.zeros((2048, 128), np.float32); W1[0, 0] = W1[1, 1] = W1[2, 2] = 1
    centros = np.array([np.array(COLORES[c]) / 255 for c in clases])
    W2 = np.zeros((128, 3), np.float32); W2[:3, :] = centros.T * 40
    b2 = -0.5 * (centros ** 2).sum(1) * 40
    path = tmp_path_factory.mktemp("m") / "head.npz"
    np.savez(path, W1=W1, b1=np.zeros(128, np.float32), W2=W2, b2=b2.astype(np.float32),
             classes=np.array(clases))
    return VisionClassifier(head_path=path, extractor=extractor)


@pytest.fixture
def client(nlp, vision):
    from api.server import create_app
    app = create_app(nlp=nlp, vision=vision, load_models=False)
    return app.test_client()
