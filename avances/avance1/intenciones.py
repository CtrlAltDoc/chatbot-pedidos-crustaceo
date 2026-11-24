
import re
import random
import unicodedata
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.stem import SnowballStemmer

tokenizer = RegexpTokenizer(r'\w+')
stemmer = SnowballStemmer("spanish")

def normalizar_texto(texto):
    """
    Convierte el texto a minusculas, elimina acentos y caracteres especiales.
    """
    texto = texto.lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = texto.encode("ascii", "ignore").decode("utf-8")
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

def tokenizar_y_stem(texto):
    """
    Tokeniza el texto y aplica stemming para reducir variaciones morfologicas.
    """
    texto = normalizar_texto(texto)
    tokens = tokenizer.tokenize(texto)
    stems = [stemmer.stem(token) for token in tokens]
    return tokens, stems

intenciones = {
    "saludo": {
        "ejemplos": [
            "hola",
            "buenos dias",
            "buenas tardes",
            "que tal",
            "hay alguien ahi",
            "hola calamar"
        ],
        "respuestas": [
            "Bienvenido al Crustaceo Cascarudo. Donde el dinero siempre tiene la razon. Ordene rapido.",
            "Habla Calamardo. Bienvenido al Crustaceo Cascarudo. Que va a ordenar para que pueda volver a mi siesta?"
        ]
    },
    "es_el_crustaceo": {
        "ejemplos": [
            "es el crustaceo cascarudo",
            "es ahi el crustaceo cascarudo",
            "hablo al crustaceo cascarudo",
            "es este el crustaceo cascarudo",
            "es este el crustaceo",
            "es el crustaceo"
        ],
        "respuestas": [
            "No, es patricio!"
        ]
    },
    "consultar_menu": {
        "ejemplos": [
            "que tienen de comer",
            "cual es el menu",
            "que platillos venden",
            "muestrame la carta",
            "que hay para comer",
            "que venden aqui"
        ],
        "respuestas": [
            "Nuestro menu: Cangreburger clasica ($1.25), Cangreburger doble ($2.00), Papas de algas ($0.75), Malteada de algas ($1.00) y Trozos de coral ($0.50).",
            "Tenemos Cangreburgers clasicas y dobles, papas de algas, trozos de coral y malteadas. Decidete rapido."
        ]
    },
    "consultar_horario": {
        "ejemplos": [
            "a que hora abren",
            "a que hora cierran",
            "cual es el horario",
            "horario de atencion",
            "estan abiertos hoy"
        ],
        "respuestas": [
            "Abrimos de lunes a sabado de 8:00 a 22:00 horas. Don Cangrejo no permite cerrar antes mientras haya monedas en circulacion.",
            "Estamos abiertos de 8:00 a.m. a 10:00 p.m. Si llega un segundo tarde, la puerta estara cerrada."
        ]
    },
    "consultar_precio": {
        "ejemplos": [
            "cuanto cuesta la cangreburger",
            "precio de las papas",
            "cuanto vale la malteada",
            "cual es el costo",
            "precios del menu",
            "cuanto dinero cuesta"
        ],
        "respuestas": [
            "Para el articulo '{item}': Cangreburger $1.25, Doble $2.00, Papas $0.75, Malteada $1.00. Prepare su dinero en efectivo.",
            "El costo detectado para '{item}' esta dentro de la lista oficial aprobada por Don Cangrejo. No aceptamos cheques de Calamardo."
        ]
    },
    "ordenar": {
        "ejemplos": [
            "quiero una cangreburger",
            "dame unas papas",
            "voy a pedir una malteada",
            "deseo ordenar comida",
            "quiero hacer un pedido",
            "me da una cangreburger doble",
            "quiero servicio a domicilio"
        ],
        "respuestas": [
            "Registrando pedido de: {item} para la modalidad: {servicio}. Bob Esponja ya esta calentando la parrilla.",
            "Orden de '{item}' tomada bajo la modalidad '{servicio}'. Pase a la caja de inmediato."
        ]
    },
    "formula_secreta": {
        "ejemplos": [
            "cual es la formula secreta",
            "dame la receta secreta",
            "como se hace la cangreburger",
            "que ingredientes lleva la cangreburger",
            "dame la formula de don cangrejo",
            "revelame los ingredientes"
        ],
        "respuestas": [
            "ALERTA ROJA: Buen intento, Plankton! Jamas obtendras la formula secreta. Sal de aqui antes de que active el extractor!",
            "Intento de espionaje corporativo detectado. Contactando a la Policia de Fondo de Bikini y a Don Cangrejo!"
        ]
    },
    "despedida": {
        "ejemplos": [
            "gracias",
            "adios",
            "hasta luego",
            "eso es todo",
            "nos vemos"
        ],
        "respuestas": [
            "Gracias por su dinero. Vuelva pronto al Crustaceo Cascarudo.",
            "Terminamos? Perfecto. No olvide dejar su propina en el frasco de Calamardo."
        ]
    }
}

catalogo_items = {
    "cangreburger doble": ["cangreburger doble", "hamburguesa doble", "doble"],
    "cangreburger clasica": ["cangreburger", "hamburguesa", "cangreburguer"],
    "papas de algas": ["papas", "papas de algas", "patatas"],
    "malteada de algas": ["malteada", "malteada de algas", "batido", "bebida"],
    "trozos de coral": ["trozos de coral", "coral", "bocadillos"]
}

catalogo_servicios = {
    "a domicilio": ["domicilio", "envio", "mandar", "entrega", "lancha"],
    "para llevar": ["llevar", "para llevar", "recoger", "mostrador"],
    "en el local": ["aqui", "mesa", "comedor", "restaurante"]
}

