"""Análisis del umbral de confianza del clasificador visual.

Hallazgo de las pruebas: con umbral 0.60 una foto de un perro se aceptó como
"cangreburger" (79 %). Este script mide el efecto de distintos umbrales sobre:
- cobertura: % de fotos REALES del menú (conjunto de prueba) que el agente acepta
- exactitud sobre las aceptadas
- falsa aceptación: % de fotos AJENAS al menú (perros, autos, personas...) aceptadas

Uso (Colab):  python -m scripts.eval_threshold --ood /content/ood
Salida: docs/metricas/umbral_vision.json y docs/metricas/umbral_vision.png
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import requests
from PIL import Image

from core import config
from core.vision import VisionClassifier
from scripts.descargar_dataset import HEADERS, iter_resultados

UMBRALES = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
OOD_QUERIES = ['deepcategory:"Dogs"', 'deepcategory:"Cats"', 'deepcategory:"Automobiles"',
               "portrait person face", "office desk laptop", "landscape mountains"]


def descargar_ood(dest: Path, por_query=6):
    dest.mkdir(parents=True, exist_ok=True)
    if len(list(dest.glob("*.jpg"))) >= por_query * len(OOD_QUERIES):
        return
    for qi, q in enumerate(OOD_QUERIES):
        n = 0
        for item in iter_resultados(q):
            if n >= por_query:
                break
            try:
                r = requests.get(item["url"], headers=HEADERS, timeout=30)
                img = Image.open(io.BytesIO(r.content)).convert("RGB")
            except Exception:
                continue
            img.save(dest / f"ood_{qi}_{n}.jpg", quality=90)
            n += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ood", type=Path, default=Path("/content/ood"))
    args = ap.parse_args()
    vc = VisionClassifier()
    test = json.loads((config.MODELS_DIR / "split_vision.json").read_text())["test"]

    menu = []  # (clase_real, clase_pred, confianza)
    for p in test:
        r = vc.predict(Path(p).read_bytes())
        menu.append((Path(p).parent.name, r["label"], r["confidence"]))
    descargar_ood(args.ood)
    ood = [vc.predict(p.read_bytes())["confidence"] for p in sorted(args.ood.glob("*.jpg"))]

    filas = []
    for u in UMBRALES:
        acept = [(t, p) for t, p, c in menu if c >= u]
        filas.append({
            "umbral": u,
            "cobertura_menu": round(len(acept) / len(menu), 4),
            "exactitud_aceptadas": round(sum(t == p for t, p in acept) / len(acept), 4) if acept else None,
            "falsa_aceptacion_ajenas": round(sum(c >= u for c in ood) / len(ood), 4) if ood else None,
        })
    out = {"n_prueba_menu": len(menu), "n_ajenas": len(ood), "umbral_en_produccion": config.VISION_THRESHOLD,
           "confianza_media_menu": round(float(np.mean([c for *_, c in menu])), 4),
           "confianza_media_ajenas": round(float(np.mean(ood)), 4) if ood else None,
           "tabla": filas}
    dst = config.ROOT / "docs" / "metricas"; dst.mkdir(parents=True, exist_ok=True)
    (dst / "umbral_vision.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [f["umbral"] for f in filas]
    ax.plot(xs, [f["cobertura_menu"] for f in filas], "o-", label="Cobertura (fotos del menú aceptadas)")
    ax.plot(xs, [f["exactitud_aceptadas"] for f in filas], "s-", label="Exactitud sobre aceptadas")
    if ood:
        ax.plot(xs, [f["falsa_aceptacion_ajenas"] for f in filas], "^-", label="Falsa aceptación (fotos ajenas)")
    ax.axvline(config.VISION_THRESHOLD, color="gray", ls="--", lw=1)
    ax.set_xlabel("Umbral de confianza"); ax.set_ylim(0, 1.05); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("Efecto del umbral del clasificador visual")
    fig.tight_layout(); fig.savefig(dst / "umbral_vision.png", dpi=150)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
