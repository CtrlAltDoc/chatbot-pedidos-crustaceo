"""Preprocesamiento de texto (mismo enfoque que el Avance 2).

1. Normalización: minúsculas, sin acentos, sin signos.
2. Tokenización.
3. Eliminación de stopwords PERSONALIZADA: se conservan negaciones e
   intensificadores ("no", "muy", "poco", "pero") para no perder polaridad
   (recomendación de la Sesión 3).
4. Lematización con spaCy (es_core_news_sm). Si spaCy no está instalado
   (p. ej. en pruebas locales), se usa un stemming ligero de respaldo.
"""
import re
import unicodedata
from functools import lru_cache

KEEP_WORDS = {"no", "muy", "poco", "pero", "nada", "nunca"}

# Lista base de stopwords en español (subconjunto de la de spaCy) para el respaldo
_FALLBACK_STOPWORDS = {
    "a", "al", "algo", "con", "de", "del", "el", "ella", "en", "es", "esta", "este",
    "esto", "la", "las", "le", "lo", "los", "me", "mi", "para", "por", "que", "se",
    "su", "sus", "te", "tu", "un", "una", "unas", "unos", "y", "o", "ya", "yo",
    "favor", "porfa", "hay", "son", "sera", "ser", "estan", "estoy", "eres",
} - KEEP_WORDS

_SUFFIXES = ("amientos", "imientos", "amiento", "imiento", "aciones", "adoras", "adores",
             "ancias", "logias", "acion", "adora", "ador", "ancia", "mente", "ando", "iendo",
             "aron", "ieron", "aban", "ados", "idos", "ado", "ido", "ar", "er", "ir",
             "as", "es", "os", "a", "e", "o", "s")


def normalizar(texto: str) -> str:
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(c for c in texto if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9ñ\s]", " ", texto)
    return re.sub(r"\s+", " ", texto).strip()


def tokenizar(texto: str) -> list[str]:
    return re.findall(r"[a-z0-9ñ]+", texto)


@lru_cache(maxsize=1)
def _spacy():
    try:
        import spacy
        return spacy.load("es_core_news_sm", disable=["parser", "ner"])
    except Exception:
        return None


def _stem(token: str) -> str:
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            return token[: -len(suf)]
    return token


def preprocesar(texto: str) -> str:
    """Devuelve el texto limpio y lematizado, listo para TF-IDF."""
    limpio = normalizar(texto)
    nlp = _spacy()
    if nlp is not None:
        doc = nlp(limpio)
        tokens = [t.lemma_ for t in doc
                  if not t.is_punct and (not t.is_stop or t.text in KEEP_WORDS)]
        return " ".join(normalizar(t) for t in tokens)
    tokens = [t for t in tokenizar(limpio) if t not in _FALLBACK_STOPWORDS]
    return " ".join(_stem(t) for t in tokens)


def motor_nlp() -> str:
    return "spacy" if _spacy() is not None else "fallback"
