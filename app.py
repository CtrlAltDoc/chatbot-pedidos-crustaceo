"""Interfaz web (Streamlit) del asistente de pedidos El Crustáceo Cascarudo.

La interfaz NO llama a los modelos directamente: todo pasa por la API REST Flask
(api/server.py), que se levanta en segundo plano en el mismo contenedor.
"""
import logging
import uuid

import requests
import streamlit as st

from api.runner import start_api_in_background

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
st.set_page_config(page_title="El Crustáceo Cascarudo", page_icon="🍔", layout="centered")


@st.cache_resource(show_spinner="Encendiendo la parrilla (cargando modelos)…")
def api_base() -> str:
    return start_api_in_background()


def call_agent(base, text, image_file):
    data = {"session_id": st.session_state.sid, "text": text or ""}
    files = {"image": (image_file.name, image_file.getvalue(), image_file.type)} if image_file else None
    try:
        r = requests.post(f"{base}/agent/step", data=data, files=files, timeout=60)
    except requests.Timeout:
        return None, "La API tardó demasiado en responder. Intenta de nuevo."
    except requests.ConnectionError:
        return None, "No hay conexión con la API de inferencia. Recarga la página en unos segundos."
    if not r.ok:
        try:
            err = r.json()
            return None, f"{err.get('detail', 'Error')} (código {err.get('code', r.status_code)})"
        except ValueError:
            return None, f"Error {r.status_code} de la API."
    return r.json(), None


# ---------------- estado ----------------
if "sid" not in st.session_state:
    st.session_state.sid = str(uuid.uuid4())
    st.session_state.chat = [{"role": "assistant",
                              "content": "Habla Calamardo, Crustáceo Cascarudo. ¿Qué va a ordenar? "
                                         "Puede escribir o mandar una foto del producto."}]
    st.session_state.uploader_key = 0

# ---------------- arranque de la API ----------------
try:
    BASE = api_base()
except Exception as e:  # la UI informa en vez de romperse
    st.error(f"No se pudo iniciar la API de inferencia: {e}")
    st.stop()

# ---------------- barra lateral: monitoreo ----------------
with st.sidebar:
    st.header("Estado del sistema")
    try:
        h = requests.get(f"{BASE}/health", timeout=5).json()
        icono = "🟢" if h["status"] == "ok" else "🟠"
        st.write(f"{icono} API: **{h['status']}**")
        for comp, est in h["modelos"].items():
            st.caption(f"{comp}: {est}")
    except requests.RequestException:
        st.write("🔴 API sin respuesta")
    with st.expander("Métricas de la API"):
        try:
            st.json(requests.get(f"{BASE}/metrics", timeout=5).json())
        except requests.RequestException:
            st.caption("Sin datos")
    mostrar_traza = st.toggle("Mostrar razonamiento del agente", value=True)
    if st.button("Nueva conversación"):
        st.session_state.clear()
        st.rerun()
    st.caption(f"Sesión: {st.session_state.sid[:8]}")

# ---------------- chat ----------------
st.title("🍔 El Crustáceo Cascarudo")
st.caption("Asistente de pedidos con NLP (TF-IDF + SVM), visión (ResNet50) y un agente híbrido.")

for msg in st.session_state.chat:
    with st.chat_message(msg["role"]):
        if msg.get("image"):
            st.image(msg["image"], width=180)
        st.markdown(msg["content"])
        if msg.get("trace") and mostrar_traza:
            with st.expander("Razonamiento del agente"):
                for paso in msg["trace"]:
                    st.write(f"**{paso['fase']}** · " + " · ".join(
                        str(v) for k, v in paso.items() if k != "fase"))

imagen = st.file_uploader("Foto del producto (opcional)", type=["jpg", "jpeg", "png", "webp"],
                          key=f"img_{st.session_state.uploader_key}")
texto = st.chat_input("Escribe tu mensaje… (ej. '¿cuánto cuesta esto?' con una foto)")

if texto is not None or (imagen is not None and st.button("Enviar solo la foto")):
    st.session_state.chat.append({"role": "user", "content": texto or "📷 (foto)",
                                  "image": imagen.getvalue() if imagen else None})
    with st.spinner("Calamardo está pensando…"):
        res, err = call_agent(BASE, texto, imagen)
    if err:
        st.session_state.chat.append({"role": "assistant", "content": f"⚠️ {err}"})
    else:
        extra = ""
        if res.get("pedido"):
            extra = "\n\n🧾 " + ", ".join(f"{c}× {p}" for p, c in res["pedido"].items())
        st.session_state.chat.append({"role": "assistant", "content": res["response"] + extra,
                                      "trace": res["trace"]})
    st.session_state.uploader_key += 1  # limpia el uploader para el siguiente turno
    st.rerun()
