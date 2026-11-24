import io

from tests.conftest import png


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json["status"] == "ok"


def test_nlp_intent(client):
    r = client.post("/nlp/intent", json={"text": "a que hora abren"})
    assert r.status_code == 200 and r.json["intent"] == "consultar_horario"


def test_nlp_texto_vacio(client):
    r = client.post("/nlp/intent", json={"text": "   "})
    assert r.status_code == 400 and r.json["error"] == "texto_vacio"


def test_nlp_texto_largo(client):
    r = client.post("/nlp/intent", json={"text": "a" * 600})
    assert r.status_code == 413


def test_vision_classify(client):
    r = client.post("/vision/classify", data={"image": (io.BytesIO(png((120, 200, 120))), "m.png")})
    assert r.status_code == 200 and r.json["label"] == "malteada"


def test_vision_sin_imagen(client):
    assert client.post("/vision/classify").status_code == 400


def test_vision_archivo_invalido(client):
    r = client.post("/vision/classify", data={"image": (io.BytesIO(b"%PDF-1.4"), "x.pdf")})
    assert r.status_code == 415 and r.json["error"] == "imagen_invalida"


def test_agent_multimodal(client):
    r = client.post("/agent/step", data={"session_id": "t", "text": "quiero esto",
                                         "image": (io.BytesIO(png((240, 200, 60))), "p.png")})
    assert r.status_code == 200 and r.json["pedido"] == {"papas": 1}


def test_ruta_inexistente_json(client):
    r = client.get("/no-existe")
    assert r.status_code == 404 and "error" in r.json


def test_metrics_registra(client):
    client.get("/health")
    assert "/health" in client.get("/metrics").json["endpoints"]


def test_modelo_no_disponible():
    from api.server import create_app
    c = create_app(load_models=False).test_client()
    r = c.post("/nlp/intent", json={"text": "hola"})
    assert r.status_code == 503 and r.json["error"] == "modelo_no_disponible"
