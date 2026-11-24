"""Agente inteligente HÍBRIDO que coordina el chatbot NLP y el clasificador visual.

Ciclo por turno:  PERCIBIR -> CAPA REACTIVA -> CAPA DELIBERATIVA -> ACTUAR
- Capa reactiva: reglas inmediatas (entrada vacía, confirmar/cancelar, despedida,
  baja confianza) que responden sin planear.
- Capa deliberativa: con la intención (NLP), la etiqueta visual (CNN) y la memoria
  de la sesión decide qué acción ejecutar y combina la información.
- Memoria de sesión: guarda el pedido y la última imagen reconocida. Así la visión
  alimenta al chatbot ("¿cuánto cuesta esto?" + foto) y viceversa.
Cada decisión queda en `trace` para mostrar el razonamiento en la interfaz.
"""
import time
import uuid
from dataclasses import dataclass, field

from core import config
from core.nlp import extraer_productos
from core.text import normalizar

AFIRMACIONES = {"si", "sip", "claro", "va", "dale", "ok", "okay", "sale", "simon", "agregala", "agregalo"}
NEGACIONES = {"no", "nel", "nop", "mejor no"}
CONFIRMAR = {"confirmar", "confirmo", "es todo", "eso es todo", "seria todo", "listo", "pagar", "cerrar pedido"}
CANCELAR = {"cancelar", "cancela", "borra todo", "olvidalo", "vaciar"}
DESPEDIDA = {"adios", "gracias", "bye", "hasta luego", "nos vemos"}


@dataclass
class Session:
    id: str
    pedido: dict = field(default_factory=dict)   # producto -> cantidad
    ultima_vision: dict | None = None             # {"producto", "label", "confidence"}
    pendiente: str | None = None                  # acción que espera un "sí"
    turnos: int = 0
    actualizado: float = field(default_factory=time.time)


class HybridAgent:
    def __init__(self, nlp, vision, menu=None):
        self.nlp = nlp          # IntentClassifier (o None si no cargó)
        self.vision = vision    # VisionClassifier (o None si no cargó)
        self.menu = menu or config.load_menu()
        self.sessions: dict[str, Session] = {}

    # ---------- utilidades ----------
    def session(self, sid: str | None) -> Session:
        sid = sid or str(uuid.uuid4())
        if sid not in self.sessions:
            self.sessions[sid] = Session(id=sid)
        return self.sessions[sid]

    def _p(self, pid):
        return self.menu["productos"][pid]

    def _precio(self, pid):
        return f"${self._p(pid)['precio']} {self.menu['moneda']}"

    def _resumen(self, s: Session) -> str:
        if not s.pedido:
            return "Tu pedido está vacío."
        lineas = [f"{c} x {self._p(pid)['nombre']}" for pid, c in s.pedido.items()]
        total = sum(self._p(pid)["precio"] * c for pid, c in s.pedido.items())
        return "Tu pedido: " + ", ".join(lineas) + f". Total: ${total} {self.menu['moneda']}."

    def _agregar(self, s: Session, items):
        for it in items:
            s.pedido[it["producto"]] = s.pedido.get(it["producto"], 0) + it["cantidad"]

    # ---------- ciclo principal ----------
    def step(self, session_id=None, text: str | None = None, image: bytes | None = None) -> dict:
        s = self.session(session_id)
        s.turnos += 1
        s.actualizado = time.time()
        trace = []
        text = (text or "").strip()

        # 1) PERCIBIR
        vision_out = None
        if image is not None:
            if self.vision is None:
                trace.append({"fase": "percepción", "detalle": "Clasificador visual no disponible"})
            else:
                vision_out = self.vision.predict(image)  # puede lanzar InvalidImageError
                trace.append({"fase": "percepción", "componente": "CNN ResNet50",
                              "detalle": f"{vision_out['label']} ({vision_out['confidence']:.0%})"})
                pid = self.menu["clase_visual_a_producto"].get(vision_out["label"])
                if pid and vision_out["confidence"] >= config.VISION_THRESHOLD:
                    s.ultima_vision = {"producto": pid, **vision_out}
                    trace.append({"fase": "memoria", "detalle": f"Se guarda en la sesión: {pid}"})
                else:
                    s.ultima_vision = None
                    trace.append({"fase": "memoria", "detalle": "Imagen no reconocida con confianza suficiente"})

        nlp_out = None
        if text:
            if self.nlp is None:
                trace.append({"fase": "percepción", "detalle": "Clasificador de intenciones no disponible"})
            else:
                nlp_out = self.nlp.predict(text)
                trace.append({"fase": "percepción", "componente": "NLP TF-IDF + SVM",
                              "detalle": f"{nlp_out['intent']} ({nlp_out['confidence']:.0%})"})

        # 2) CAPA REACTIVA
        reactiva = self._reactiva(s, text, vision_out, nlp_out, trace)
        if reactiva:
            action, response = reactiva
            capa = "reactiva"
        else:
            # 3) CAPA DELIBERATIVA
            action, response = self._deliberativa(s, text, nlp_out, vision_out, trace)
            capa = "deliberativa"
        trace.append({"fase": "acción", "capa": capa, "detalle": action})

        return {
            "session_id": s.id,
            "action": action,
            "response": response,
            "intent": nlp_out,
            "vision": vision_out,
            "pedido": dict(s.pedido),
            "trace": trace,
        }

    def _reactiva(self, s, text, vision_out, nlp_out, trace):
        t = normalizar(text)
        if not t and vision_out is None:
            trace.append({"fase": "regla", "detalle": "Entrada vacía"})
            return "pedir_entrada", "Escribe tu mensaje o sube una foto de un producto."
        if t in CANCELAR:
            s.pedido.clear(); s.pendiente = None
            trace.append({"fase": "regla", "detalle": "Palabra clave de cancelación"})
            return "cancelar_pedido", "Pedido cancelado. Calamardo vuelve a su siesta."
        if t in CONFIRMAR:
            trace.append({"fase": "regla", "detalle": "Palabra clave de confirmación"})
            if not s.pedido:
                return "confirmar_vacio", "No hay nada que confirmar todavía. ¿Qué vas a ordenar?"
            msg = self._resumen(s) + " ¡Pedido confirmado! Pase a recogerlo antes de que cierre."
            s.pedido.clear(); s.pendiente = None
            return "confirmar_pedido", msg
        if t in DESPEDIDA:
            trace.append({"fase": "regla", "detalle": "Despedida"})
            return "despedida", "Gracias por elegir El Crustáceo Cascarudo. Donde el dinero siempre tiene la razón."
        if s.pendiente == "agregar_vision" and (t in AFIRMACIONES or t in NEGACIONES):
            trace.append({"fase": "regla", "detalle": "Respuesta a la pregunta pendiente"})
            s.pendiente = None
            if t in AFIRMACIONES and s.ultima_vision:
                self._agregar(s, [{"producto": s.ultima_vision["producto"], "cantidad": 1}])
                return "agregar_por_imagen", "Agregado. " + self._resumen(s)
            return "rechazar_sugerencia", "Va, no lo agrego. ¿Algo más?"
        if nlp_out and nlp_out["confidence"] < config.INTENT_THRESHOLD and vision_out is None:
            trace.append({"fase": "regla", "detalle": f"Confianza baja (< {config.INTENT_THRESHOLD:.0%})"})
            return "aclarar", ("No entendí bien. Puedo tomar tu pedido, darte el menú, precios u horario. "
                               "También puedes mandarme una foto del producto.")
        return None

    def _deliberativa(self, s, text, nlp_out, vision_out, trace):
        intent = nlp_out["intent"] if nlp_out else None
        productos = extraer_productos(text, self.menu) if text else []
        if productos:
            trace.append({"fase": "entidades", "detalle": ", ".join(f"{p['cantidad']}x {p['producto']}" for p in productos)})

        # Solo imagen: describir y proponer (la visión alimenta el diálogo)
        if intent is None and vision_out is not None:
            if s.ultima_vision:
                pid = s.ultima_vision["producto"]
                s.pendiente = "agregar_vision"
                trace.append({"fase": "plan", "detalle": "Imagen sin texto -> describir y ofrecer agregar"})
                return "describir_imagen", (f"Eso parece una {self._p(pid)['nombre']} "
                                            f"({vision_out['confidence']:.0%} de confianza). Cuesta {self._precio(pid)}. "
                                            "¿La agrego a tu pedido?")
            return "imagen_no_reconocida", ("No reconozco ese producto en nuestro menú. "
                                            "Prueba con una foto clara de una cangreburger, papas o malteada.")

        # Si el texto no nombra producto pero hay imagen reconocida, el agente la usa
        if not productos and s.ultima_vision and intent in {"ordenar", "consultar_precio"}:
            productos = [{"producto": s.ultima_vision["producto"], "cantidad": 1}]
            trace.append({"fase": "plan", "detalle": "Texto sin producto -> se usa el producto de la imagen (memoria)"})

        if intent == "ordenar":
            if not productos:
                return "pedir_producto", "¿Qué va a ordenar? " + self._menu_txt()
            self._agregar(s, productos)
            return "agregar_pedido", "Anotado. " + self._resumen(s) + " Escriba 'eso es todo' para confirmar."
        if intent == "consultar_precio":
            if not productos:
                return "listar_precios", self._menu_txt()
            return "dar_precio", " ".join(f"{self._p(p['producto'])['nombre']}: {self._precio(p['producto'])}." for p in productos)
        if intent == "consultar_menu":
            return "mostrar_menu", self._menu_txt()
        if intent == "consultar_horario":
            return "dar_horario", self.menu["horario"]
        if intent == "saludo":
            return "saludar", "Bienvenido al Crustáceo Cascarudo. Ordene rápido, que Calamardo quiere dormir."
        if intent == "es_el_crustaceo":
            return "broma_patricio", "¡No, este es Patricio!"
        if intent == "formula_secreta":
            return "proteger_formula", "La fórmula secreta no se comparte. ¿Plankton, eres tú?"
        if intent == "queja":
            return "registrar_queja", ("Lamentamos la experiencia. Su queja quedó registrada y el señor Cangrejo "
                                       "la revisará. ¿Le gustaría ordenar algo más?")
        return "aclarar", "No entendí. ¿Me lo repites de otra forma?"

    def _menu_txt(self):
        items = [f"{p['nombre']} ({self._precio(pid)})" for pid, p in self.menu["productos"].items()]
        return "Nuestro menú: " + ", ".join(items) + "."
