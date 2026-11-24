"""Amplía el dataset de imágenes con fotos libres de Wikimedia Commons (corrección P2).

- Agrega fotos a las carpetas existentes (cangreburger, papas, malteada) sin borrar las tuyas.
- Filtro CLIP zero-shot: solo acepta la foto si CLIP cree que es el producto
  (rechaza personas, lugares, logos, dibujos y otras bebidas).
- Reanudable: si se corta, al volver a correr continúa donde iba.
- Guarda creditos_wikimedia.csv (autor, licencia, URL) para citar en el documento.

Uso (Colab):  python -m scripts.descargar_dataset --data /content/drive/MyDrive/dataset_crustaceo --meta 75
"""
import argparse
import csv
import hashlib
import io
import re
import time
from pathlib import Path

import requests
from PIL import Image

API_URL = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "CrustaceoDatasetEscolar/1.0 (proyecto academico; github.com/CtrlAltDoc/chatbot-pedidos-crustaceo)"}
THUMB_WIDTH = 512
MIN_SIDE = 200
PAGE_SIZE = 50
MAX_PAGES_PER_QUERY = 10
SLEEP = 0.2
EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# Carpeta del dataset -> consultas (primero categorías curadas, luego texto)
CONSULTAS = {
    "cangreburger": ['deepcategory:"Hamburgers"', 'deepcategory:"Cheeseburgers"', "hamburger burger bun", "cheeseburger"],
    "papas": ['deepcategory:"French fries"', 'deepcategory:"Chips (food)"', "french fries plate", "pommes frites"],
    "malteada": ['deepcategory:"Milkshakes"', "milkshake glass", "chocolate milkshake", "strawberry milkshake"],
}

# Etiquetas para CLIP: las clases objetivo + "trampas" que deben rechazarse
CLIP_LABELS = {
    "cangreburger": "a photo of a hamburger or cheeseburger",
    "papas": "a photo of french fries",
    "malteada": "a photo of a milkshake in a glass or cup",
    "_people": "a photo of people, a crowd, a party or a performer",
    "_place": "a photo of a restaurant, a street or a building",
    "_other": "a photo of a logo, a menu, text, a drawing or a cartoon",
    "_drink": "a photo of a smoothie, juice, coffee, beer or a cocktail",
}


class ClipFilter:
    def __init__(self, min_prob=0.5):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device).eval()
        self.proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.keys, self.texts = list(CLIP_LABELS), list(CLIP_LABELS.values())
        self.min_prob = min_prob

    def es_clase(self, img: Image.Image, target: str) -> bool:
        with self.torch.no_grad():
            inputs = self.proc(text=self.texts, images=img, return_tensors="pt", padding=True).to(self.device)
            probs = self.model(**inputs).logits_per_image.softmax(dim=1)[0]
        best = int(probs.argmax())
        return self.keys[best] == target and float(probs[best]) >= self.min_prob


def _limpiar_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s or "").strip()


def iter_resultados(query: str):
    params = {"action": "query", "format": "json", "generator": "search",
              "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6, "gsrlimit": PAGE_SIZE,
              "prop": "imageinfo", "iiprop": "url|mime|extmetadata", "iiurlwidth": THUMB_WIDTH}
    for _ in range(MAX_PAGES_PER_QUERY):
        r = requests.get(API_URL, params=params, headers=HEADERS, timeout=30)
        r.raise_for_status()
        data = r.json()
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [{}])[0]
            if info.get("mime") in ("image/jpeg", "image/png"):
                meta = info.get("extmetadata", {})
                yield {"url": info.get("thumburl") or info.get("url"),
                       "pagina": info.get("descriptionurl", ""),
                       "titulo": page.get("title", ""),
                       "autor": _limpiar_html(meta.get("Artist", {}).get("value")),
                       "licencia": meta.get("LicenseShortName", {}).get("value", "")}
        if "continue" not in data:
            return
        params.update(data["continue"])


def ampliar_clase(dest: Path, clase: str, meta: int, clip, creditos_path: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    existentes = [p for p in dest.iterdir() if p.suffix.lower() in EXTS]
    hashes = {hashlib.md5(p.read_bytes()).hexdigest() for p in existentes}
    n = len(existentes)
    siguiente = len(list(dest.glob("wm_*.jpg"))) + 1
    rechazadas = 0
    print(f"\n{clase}: {n} imágenes actuales, meta {meta}")
    vistos = set()
    for query in CONSULTAS[clase]:
        if n >= meta:
            break
        print(f"  buscando: {query}")
        for item in iter_resultados(query):
            if n >= meta:
                break
            if not item["url"] or item["url"] in vistos:
                continue
            vistos.add(item["url"])
            time.sleep(SLEEP)
            try:
                resp = requests.get(item["url"], headers=HEADERS, timeout=30)
                resp.raise_for_status()
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            except Exception:
                continue
            digest = hashlib.md5(resp.content).hexdigest()
            if digest in hashes or min(img.size) < MIN_SIDE:
                continue
            if clip and not clip.es_clase(img, clase):
                rechazadas += 1
                continue
            nombre = f"wm_{clase}_{siguiente:03d}.jpg"
            img.save(dest / nombre, quality=90)
            hashes.add(digest)
            siguiente += 1
            n += 1
            with open(creditos_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([clase, nombre, item["titulo"], item["autor"], item["licencia"], item["pagina"]])
            print(f"   ✓ {n}/{meta}")
    print(f"  CLIP rechazó {rechazadas} imágenes")
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--meta", type=int, default=75, help="imágenes totales por clase")
    ap.add_argument("--sin-clip", action="store_true")
    args = ap.parse_args()

    creditos = args.data / "creditos_wikimedia.csv"
    if not creditos.exists():
        creditos.write_text("clase,archivo,titulo,autor,licencia,url\n", encoding="utf-8")
    clip = None if args.sin_clip else ClipFilter()
    resumen = {c: ampliar_clase(args.data / c, c, args.meta, clip, creditos) for c in CONSULTAS}
    print("\n=== RESUMEN ===")
    for c, n in resumen.items():
        print(f"{c:<14}{n:>4} imágenes  [{'OK' if n >= args.meta else 'faltan, vuelve a correr'}]")


if __name__ == "__main__":
    main()
