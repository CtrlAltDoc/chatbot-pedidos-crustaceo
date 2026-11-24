"""Pruebas funcionales y métricas del chatbot (conversaciones completas por la API).

Métricas: exactitud de intención por turno, exactitud de acción del agente,
éxito de tarea por conversación, tasa de aclaración (fallback) y latencia p50/p95.

Uso:
  python -m scripts.eval_chatbot                       # API en proceso, modelos de models/
  python -m scripts.eval_chatbot --url https://...     # contra la app desplegada (solo texto)
  python -m scripts.eval_chatbot --images /ruta/dataset_crustaceo   # incluye turnos con imagen
Salida: docs/metricas/metricas_chatbot.json y docs/pruebas/tabla_pruebas_chatbot.csv
"""
import argparse
import csv
import io
import json
import statistics
import time
from pathlib import Path

from core import config

FALLBACKS = {"aclarar", "imagen_no_reconocida"}


def imagen_de_prueba(images_dir: Path | None, clase: str):
    if images_dir is None:
        return None
    split = config.MODELS_DIR / "split_vision.json"
    if split.exists():  # preferir imágenes del conjunto de PRUEBA
        for p in json.loads(split.read_text())["test"]:
            if Path(p).parent.name == clase and Path(p).exists():
                return Path(p).read_bytes()
    for p in sorted((images_dir / clase).glob("*")):
        return p.read_bytes()
    return None


def make_caller(url):
    if url:
        import requests

        def call(data, img):
            files = {"image": ("img.jpg", img, "image/jpeg")} if img else None
            r = requests.post(f"{url}/agent/step", data=data, files=files, timeout=60)
            return r.status_code, r.json()
        return call
    from api.server import create_app
    client = create_app().test_client()

    def call(data, img):
        if img:
            data = {**data, "image": (io.BytesIO(img), "img.jpg")}
        r = client.post("/agent/step", data=data)
        return r.status_code, r.get_json()
    return call


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--images", type=Path)
    args = ap.parse_args()
    call = make_caller(args.url)
    convs = json.loads((config.DATA_DIR / "conversaciones_prueba.json").read_text(encoding="utf-8"))

    filas, lat, ok_conv, n_conv = [], [], 0, 0
    for conv in convs:
        sid = f"eval-{conv['id']}-{time.time()}"
        exito, omitida = True, False
        for i, turno in enumerate(conv["turnos"], 1):
            img = None
            if "image" in turno:
                img = imagen_de_prueba(args.images, turno["image"])
                if img is None:
                    omitida = True
                    break
            t0 = time.perf_counter()
            status, res = call({"session_id": sid, "text": turno.get("text", "")}, img)
            ms = (time.perf_counter() - t0) * 1000
            lat.append(ms)
            intent = (res.get("intent") or {}).get("intent") if status == 200 else None
            action = res.get("action") if status == 200 else f"HTTP {status}"
            ok_int = turno.get("intent") is None or intent == turno["intent"]
            ok_act = action == turno["action"]
            exito &= ok_act
            filas.append({"conversacion": conv["id"], "objetivo": conv["objetivo"], "turno": i,
                          "entrada": turno.get("text", "") + (f" [foto: {turno['image']}]" if "image" in turno else ""),
                          "intencion_esperada": turno.get("intent", "-"), "intencion_obtenida": intent or "-",
                          "accion_esperada": turno["action"], "accion_obtenida": action,
                          "respuesta": res.get("response", res.get("detail", "")),
                          "latencia_ms": round(ms, 1), "resultado": "PASA" if ok_act and ok_int else "FALLA"})
        if omitida:
            filas = [f for f in filas if f["conversacion"] != conv["id"]]
            print(f"{conv['id']}: omitida (no hay imágenes; usa --images)")
            continue
        n_conv += 1
        ok_conv += exito

    con_int = [f for f in filas if f["intencion_esperada"] != "-"]
    lat_sorted = sorted(lat)
    metricas = {
        "conversaciones_evaluadas": n_conv,
        "turnos_evaluados": len(filas),
        "exito_de_tarea": round(ok_conv / n_conv, 4) if n_conv else None,
        "exactitud_intencion": round(sum(f["intencion_esperada"] == f["intencion_obtenida"] for f in con_int) / len(con_int), 4) if con_int else None,
        "exactitud_accion_agente": round(sum(f["accion_esperada"] == f["accion_obtenida"] for f in filas) / len(filas), 4),
        "tasa_aclaracion_fallback": round(sum(f["accion_obtenida"] in FALLBACKS for f in filas) / len(filas), 4),
        "latencia_p50_ms": round(statistics.median(lat), 1),
        "latencia_p95_ms": round(lat_sorted[min(len(lat_sorted) - 1, int(0.95 * len(lat_sorted)))], 1),
        "fallas": [f for f in filas if f["resultado"] == "FALLA"],
    }
    out_m = config.ROOT / "docs" / "metricas"; out_m.mkdir(parents=True, exist_ok=True)
    out_p = config.ROOT / "docs" / "pruebas"; out_p.mkdir(parents=True, exist_ok=True)
    (out_m / "metricas_chatbot.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=2))
    with open(out_p / "tabla_pruebas_chatbot.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader(); w.writerows(filas)
    print(json.dumps({k: v for k, v in metricas.items() if k != "fallas"}, ensure_ascii=False, indent=2))
    for f in metricas["fallas"]:
        print(f"FALLA {f['conversacion']}.{f['turno']}: '{f['entrada']}' -> {f['accion_obtenida']} (esperada {f['accion_esperada']})")


if __name__ == "__main__":
    main()
