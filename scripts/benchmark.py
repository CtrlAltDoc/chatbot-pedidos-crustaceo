"""Prueba de desempeño y escalabilidad de la API (latencia y throughput con concurrencia).

Uso:
  python -m scripts.benchmark                          # levanta la API local en segundo plano
  python -m scripts.benchmark --url http://127.0.0.1:5000 --image foto.jpg
Salida: docs/metricas/benchmark.json
"""
import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from core import config

FRASES = ["hola", "quiero dos papas y una malteada", "cuanto cuesta la cangreburger",
          "a que hora cierran", "que tienen de comer"]


def medir(fn, n, workers):
    def uno(i):
        t0 = time.perf_counter()
        ok = fn(i)
        return (time.perf_counter() - t0) * 1000, ok
    t0 = time.perf_counter()
    with ThreadPoolExecutor(workers) as ex:
        res = list(ex.map(uno, range(n)))
    total = time.perf_counter() - t0
    lat = sorted(r[0] for r in res)
    return {"concurrencia": workers, "peticiones": n,
            "exito": round(sum(r[1] for r in res) / n, 4),
            "throughput_rps": round(n / total, 2),
            "p50_ms": round(statistics.median(lat), 1),
            "p95_ms": round(lat[min(n - 1, int(0.95 * n))], 1),
            "max_ms": round(lat[-1], 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    ap.add_argument("--image", type=Path)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()
    if args.url:
        base = args.url
    else:
        from api.runner import start_api_in_background
        base = start_api_in_background()

    texto = lambda i: requests.post(f"{base}/agent/step", timeout=60,
                                    data={"session_id": f"bench-{i}", "text": FRASES[i % len(FRASES)]}).ok
    resultados = {"agent_texto": [medir(texto, args.n, w) for w in (1, 5, 10)]}
    if args.image:
        img = args.image.read_bytes()
        vision = lambda i: requests.post(f"{base}/vision/classify", timeout=120,
                                         files={"image": ("i.jpg", img, "image/jpeg")}).ok
        resultados["vision"] = [medir(vision, max(10, args.n // 3), w) for w in (1, 5)]
    resultados["metrics_api"] = requests.get(f"{base}/metrics", timeout=10).json()

    out = config.ROOT / "docs" / "metricas"; out.mkdir(parents=True, exist_ok=True)
    (out / "benchmark.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2))
    for k, filas in resultados.items():
        if k == "metrics_api":
            continue
        print(f"\n{k}")
        for f in filas:
            print(f"  c={f['concurrencia']:>2}  rps={f['throughput_rps']:>7}  p50={f['p50_ms']}ms  "
                  f"p95={f['p95_ms']}ms  éxito={f['exito']:.0%}")


if __name__ == "__main__":
    main()
