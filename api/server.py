"""API REST (Flask) con los servicios de inferencia.

Endpoints:
  GET  /health            estado de los modelos
  GET  /metrics           monitoreo: peticiones, errores y latencias por endpoint
  POST /nlp/intent        {"text": "..."} -> intención
  POST /vision/classify   multipart "image" -> clase visual
  POST /agent/step        multipart/JSON: session_id, text, image -> respuesta del agente
Todos los errores salen como JSON: {"error", "code", "detail"}.
"""
import logging
import time
from collections import defaultdict

from flask import Flask, g, jsonify, request
from werkzeug.exceptions import HTTPException

from core import config
from core.agent import HybridAgent
from core.vision import InvalidImageError

log = logging.getLogger("crustaceo.api")


class ApiError(Exception):
    def __init__(self, code: int, error: str, detail: str):
        super().__init__(detail)
        self.code, self.error, self.detail = code, error, detail


class Monitor:
    """Contadores simples en memoria para el endpoint /metrics."""

    def __init__(self):
        self.start = time.time()
        self.requests = defaultdict(int)
        self.errors = defaultdict(int)
        self.latencies = defaultdict(list)

    def record(self, endpoint, ms, status):
        self.requests[endpoint] += 1
        if status >= 400:
            self.errors[endpoint] += 1
        lat = self.latencies[endpoint]
        lat.append(ms)
        del lat[:-500]  # conserva las últimas 500

    def snapshot(self):
        out = {}
        for ep, n in self.requests.items():
            lat = sorted(self.latencies[ep])
            pct = lambda p: round(lat[min(len(lat) - 1, int(p * len(lat)))], 1) if lat else None
            out[ep] = {"requests": n, "errors": self.errors[ep],
                       "p50_ms": pct(0.50), "p95_ms": pct(0.95)}
        return {"uptime_s": round(time.time() - self.start), "endpoints": out}


def _load_models():
    nlp = vision = None
    status = {}
    try:
        from core.nlp import IntentClassifier
        nlp = IntentClassifier()
        status["nlp"] = "ok"
    except Exception as e:  # modelo ausente o corrupto: la API sigue viva
        status["nlp"] = f"no disponible: {e.__class__.__name__}"
        log.exception("No se pudo cargar el modelo NLP")
    try:
        from core.vision import VisionClassifier
        vision = VisionClassifier()
        status["vision"] = "ok"
    except Exception as e:
        status["vision"] = f"no disponible: {e.__class__.__name__}"
        log.exception("No se pudo cargar el modelo visual")
    return nlp, vision, status


def create_app(nlp=None, vision=None, load_models=True) -> Flask:
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.MAX_IMAGE_BYTES + 1024 * 1024
    status = {"nlp": "ok" if nlp else "no cargado", "vision": "ok" if vision else "no cargado"}
    if load_models and nlp is None and vision is None:
        nlp, vision, status = _load_models()
    agent = HybridAgent(nlp, vision)
    monitor = Monitor()
    app.extensions.update(agent=agent, monitor=monitor)

    # ---------- monitoreo ----------
    @app.before_request
    def _start():
        g.t0 = time.perf_counter()

    @app.after_request
    def _log(resp):
        ms = (time.perf_counter() - g.t0) * 1000
        monitor.record(request.path, ms, resp.status_code)
        log.info('{"path": "%s", "status": %d, "ms": %.1f}', request.path, resp.status_code, ms)
        return resp

    # ---------- manejo de errores ----------
    @app.errorhandler(ApiError)
    def _api_error(e):
        return jsonify(error=e.error, code=e.code, detail=e.detail), e.code

    @app.errorhandler(InvalidImageError)
    def _bad_image(e):
        return jsonify(error="imagen_invalida", code=415, detail=str(e)), 415

    @app.errorhandler(HTTPException)
    def _http(e):
        return jsonify(error=e.name.lower().replace(" ", "_"), code=e.code, detail=e.description), e.code

    @app.errorhandler(Exception)
    def _unexpected(e):
        log.exception("Error inesperado")
        return jsonify(error="error_interno", code=500, detail="Ocurrió un error inesperado."), 500

    # ---------- helpers de entrada ----------
    def _text(required: bool):
        data = request.get_json(silent=True) or {}
        text = data.get("text", request.form.get("text", ""))
        if not isinstance(text, str):
            raise ApiError(400, "texto_invalido", "El campo 'text' debe ser texto.")
        text = text.strip()
        if required and not text:
            raise ApiError(400, "texto_vacio", "Envía un mensaje en el campo 'text'.")
        if len(text) > config.MAX_TEXT_CHARS:
            raise ApiError(413, "texto_largo", f"Máximo {config.MAX_TEXT_CHARS} caracteres.")
        return text

    def _image(required: bool):
        f = request.files.get("image")
        if f is None:
            if required:
                raise ApiError(400, "imagen_faltante", "Envía una imagen en el campo 'image'.")
            return None
        return f.read()

    def _need(component, name):
        if component is None:
            raise ApiError(503, "modelo_no_disponible", f"El modelo de {name} no está cargado.")

    # ---------- endpoints ----------
    @app.get("/health")
    def health():
        ok = all(v == "ok" for v in status.values())
        return jsonify(status="ok" if ok else "degradado", modelos=status), 200

    @app.get("/metrics")
    def metrics():
        return jsonify(monitor.snapshot())

    @app.post("/nlp/intent")
    def nlp_intent():
        _need(nlp, "lenguaje")
        return jsonify(nlp.predict(_text(required=True)))

    @app.post("/vision/classify")
    def vision_classify():
        _need(vision, "visión")
        return jsonify(vision.predict(_image(required=True)))

    @app.post("/agent/step")
    def agent_step():
        text = _text(required=False)
        image = _image(required=False)
        sid = (request.get_json(silent=True) or {}).get("session_id") or request.form.get("session_id")
        if image is not None:
            _need(vision, "visión")
        if text and nlp is None:
            _need(nlp, "lenguaje")
        return jsonify(agent.step(session_id=sid, text=text, image=image))

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    create_app().run(host=config.API_HOST, port=config.API_PORT)
