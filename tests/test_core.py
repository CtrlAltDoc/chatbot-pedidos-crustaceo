import pytest

from core import config
from core.agent import HybridAgent
from core.nlp import extraer_productos
from core.text import preprocesar
from core.vision import InvalidImageError, cargar_imagen
from tests.conftest import png

MENU = config.load_menu()


def test_stopwords_conservan_negacion():
    assert preprocesar("No entendí la explicación") != preprocesar("Entendí la explicación")


def test_normaliza_acentos_y_signos():
    assert preprocesar("¡¡HOLA!!") == preprocesar("hola")


@pytest.mark.parametrize("texto,esperado", [
    ("quiero dos papas y una malteada", {"papas": 2, "malteada": 1}),
    ("una cangreburger doble", {"cangreburger_doble": 1}),
    ("3 cangreburgers", {"cangreburger": 3}),
    ("hola", {}),
])
def test_extraer_productos(texto, esperado):
    assert {p["producto"]: p["cantidad"] for p in extraer_productos(texto, MENU)} == esperado


@pytest.mark.parametrize("texto,intent", [
    ("quiero pedir una cangreburger", "ordenar"),
    ("cual es el menu", "consultar_menu"),
    ("a que hora cierran", "consultar_horario"),
    ("cuanto cuesta la malteada", "consultar_precio"),
])
def test_intenciones_basicas(nlp, texto, intent):
    assert nlp.predict(texto)["intent"] == intent


def test_vision_clasifica(vision):
    assert vision.predict(png((240, 200, 60)))["label"] == "papas"


def test_imagen_invalida():
    with pytest.raises(InvalidImageError):
        cargar_imagen(b"esto no es una imagen")


def test_formato_no_permitido():
    with pytest.raises(InvalidImageError):
        cargar_imagen(png((1, 2, 3), fmt="BMP"))


def test_agente_imagen_alimenta_al_chatbot(nlp, vision):
    """Intercambio de información: la etiqueta visual se usa para responder el texto."""
    a = HybridAgent(nlp, vision, MENU)
    r = a.step("s1", text="cuanto cuesta esto", image=png((120, 200, 120)))
    assert r["action"] == "dar_precio" and "Malteada" in r["response"]
    assert any(p["fase"] == "memoria" for p in r["trace"])


def test_agente_foto_y_luego_si(nlp, vision):
    a = HybridAgent(nlp, vision, MENU)
    r1 = a.step("s2", image=png((150, 90, 40)))
    assert r1["action"] == "describir_imagen"
    r2 = a.step("s2", text="si")
    assert r2["action"] == "agregar_por_imagen" and r2["pedido"] == {"cangreburger": 1}


def test_agente_pedido_y_confirmacion(nlp, vision):
    a = HybridAgent(nlp, vision, MENU)
    a.step("s3", text="quiero dos papas y una malteada")
    r = a.step("s3", text="eso es todo")
    assert r["action"] == "confirmar_pedido" and "$80 MXN" in r["response"]


def test_agente_baja_confianza_pide_aclaracion(nlp, vision):
    a = HybridAgent(nlp, vision, MENU)
    r = a.step("s4", text="xkcd zzz qwerty")
    assert r["action"] in {"aclarar", "pedir_producto"}
