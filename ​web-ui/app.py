"""
MUDU-VLLM — Minimal Web UI sunucusu
Çalıştır:  python app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT.parent / "yolo"
if not YOLO_DIR.exists():
    YOLO_DIR = ROOT / "yolo"  # Docker image layout
sys.path.insert(0, str(YOLO_DIR))

app = FastAPI(title="MUDU-VLLM")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DecisionCore:
    def __init__(self, num_features: int = 4):
        self.num_features = num_features
        self.qubits = np.ones((num_features, 2)) / np.sqrt(2)

    def encode(self, movement: float, audio: float, proximity: float):
        tm, ta, tp = (movement * np.pi / 2, audio * np.pi / 2, proximity * np.pi / 2)
        self.qubits[0] = [np.cos(tm), np.sin(tm)]
        self.qubits[1] = [np.cos(ta), np.sin(ta)]
        self.qubits[2] = [np.cos(tp), np.sin(tp)]
        tt = (tm + ta + tp) / 3
        self.qubits[3] = [np.cos(tt), np.sin(tt)]

    def evaluate(self, iterations: int = 200) -> float:
        results = []
        for _ in range(iterations):
            row = []
            for i in range(self.num_features):
                p = self.qubits[i, 1] ** 2
                row.append(1 if np.random.rand() < p else 0)
            results.append(row)
        mean = np.mean(results, axis=0)
        return float(np.dot(mean, [0.35, 0.35, 0.15, 0.15]))


def build_result(yolo: dict) -> dict:
    anomalies = yolo.get("anomalies") or []
    classes = yolo.get("detected_classes") or []
    types = {a.get("anomaly_type", "") for a in anomalies}
    has_prox = any("yakinsama" in t.lower() for t in types)
    has_still = any("hareketsiz" in t.lower() for t in types)
    has_speed = any("hiz" in t.lower() for t in types)

    move_sc = 0.85 if anomalies else 0.10
    prox_sc = 0.75 if has_prox else 0.10
    core = DecisionCore()
    core.encode(move_sc, 0.10, prox_sc)
    q = round(core.evaluate(), 4)

    events = [
        {
            "time": a.get("time", "00:00"),
            "event": a.get("description") or a.get("anomaly_type", "Olay"),
        }
        for a in anomalies[:12]
    ]

    if has_prox or has_still:
        risk = "Yuksek" if (has_prox and has_still) or q > 0.55 else "Orta"
    elif has_speed or anomalies:
        risk = "Orta"
    else:
        risk = "Dusuk"

    cls_tr = ", ".join(classes[:8]) if classes else "belirgin nesne"
    if anomalies:
        summary = (
            f"Videoda {len(anomalies)} hareket anomalisi izlendi "
            f"({', '.join(sorted(types)) or 'genel'}). Tespit: {cls_tr}."
        )
        actions = [
            "Kritik kareleri operatör paneline ilet",
            "Olay zaman damgalarını kayıt altına al",
        ]
        if has_prox:
            actions.append("Yakınsama bölgesini güvenlik şeridine al")
        if has_still:
            actions.append("Hareketsiz varlık için sağlık/güvenlik kontrolü başlat")
    else:
        summary = (
            f"Anormal hareket sinyali yok. Görüntüde izlenen sınıflar: {cls_tr}. "
            "Risk düşük; rutin izleme yeterli."
        )
        actions = ["Rutin izlemeye devam et", "Periyodik kare örneklemeyi sürdür"]
        events = []

    return {
        "summary": summary,
        "events": events,
        "risk": risk,
        "actions": actions,
        "quantum_inspired_risk_score": q,
    }


def _line(obj: dict) -> bytes:
    return (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.post("/api/analyze")
async def analyze(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(400, "Dosya yok")
    suffix = Path(video.filename).suffix or ".mp4"
    raw = await video.read()
    if not raw:
        raise HTTPException(400, "Boş dosya")

    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(raw)

    def generate():
        try:
            yield _line({"type": "log", "pct": 1, "msg": f"yükleme tamam · {video.filename}"})
            yield _line({"type": "log", "pct": 3, "msg": "YOLO pipeline açılıyor"})

            from yolo_pipeline import YoloPipeline

            import queue
            import threading

            pipeline = YoloPipeline()
            yield _line({"type": "log", "pct": 4, "msg": "ağırlıklar yüklendi · tarama başlıyor"})

            q: queue.Queue = queue.Queue()

            def on_progress_q(pct, msg):
                q.put(("log", pct, msg))

            def worker():
                try:
                    yolo = pipeline.run(path, conf=0.35, on_progress=on_progress_q)
                    q.put(("yolo", None, yolo))
                except Exception as e:
                    q.put(("error", None, str(e)))

            t = threading.Thread(target=worker, daemon=True)
            t.start()

            yolo = None
            while True:
                kind, pct, payload = q.get()
                if kind == "log":
                    yield _line({"type": "log", "pct": pct, "msg": payload})
                elif kind == "yolo":
                    yolo = payload
                    break
                elif kind == "error":
                    yield _line({"type": "error", "pct": 0, "msg": payload})
                    return

            yield _line({"type": "log", "pct": 96, "msg": "quantum risk füzyonu"})
            result = build_result(yolo)
            yield _line({"type": "log", "pct": 100, "msg": "tamam · JSON hazır"})
            yield _line({"type": "done", "pct": 100, "result": result})
        except Exception as e:
            yield _line({"type": "error", "pct": 0, "msg": str(e)})
        finally:
            Path(path).unlink(missing_ok=True)

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
    )
