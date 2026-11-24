"""Arranca la API Flask en un hilo de fondo dentro del mismo contenedor que Streamlit.

Así todo corre en Streamlit Community Cloud con una sola URL pública, y la interfaz
se comunica con los servicios de inferencia por HTTP (REST), como pide la rúbrica.
"""
import logging
import threading
import time

import requests
from werkzeug.serving import make_server

from api.server import create_app
from core import config

log = logging.getLogger("crustaceo.runner")
logging.getLogger("werkzeug").setLevel(logging.WARNING)  # nuestro logger ya registra cada petición


def start_api_in_background(port: int = config.API_PORT, timeout: float = 180) -> str:
    """Levanta Flask una sola vez y espera a que /health responda. Devuelve la URL base."""
    base = f"http://{config.API_HOST}:{port}"
    try:  # ¿ya hay una API corriendo en ese puerto? (reinicio del script de Streamlit)
        requests.get(f"{base}/health", timeout=1)
        return base
    except requests.RequestException:
        pass

    def _serve():
        app = create_app()  # carga modelos (la primera vez descarga ResNet50)
        make_server(config.API_HOST, port, app, threaded=True).serve_forever()

    threading.Thread(target=_serve, daemon=True, name="flask-api").start()
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if requests.get(f"{base}/health", timeout=1).ok:
                log.info("API lista en %s (%.1fs)", base, time.time() - t0)
                return base
        except requests.RequestException:
            time.sleep(0.5)
    raise RuntimeError("La API no arrancó a tiempo.")
