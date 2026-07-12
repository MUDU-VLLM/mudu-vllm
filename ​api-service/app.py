"""
MUDU-VLLM API Service
=====================
Mimari (girdi → katmanlar → çıktı):

  GİRDİ   POST /v1/analyze   multipart/form-data  field=video
      │
      ├─[1] Algı      YOLO + ByteTrack   → anomaliler, sınıflar
      ├─[2] Füzyon    DecisionCore       → quantum_inspired_risk_score
      └─[3] Çıktı     DecisionResponse   → summary / events / risk / actions

  SAĞLIK  GET  /health
  ŞEMA    GET  /v1/schema

Çalıştır:  python app.py   # veya: uvicorn app:app --host 0.0.0.0 --port 8000
Docker:    docker compose up -d
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from typing import List, Literal

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
YOLO_DIR = ROOT.parent / "yolo"
if not YOLO_DIR.exists():
    YOLO_DIR = ROOT / "yolo"  # Docker image layout
sys.path.insert(0, str(YOLO_DIR))

RiskLevel = Literal["Dusuk", "Orta", "Yuksek"]


# ---------- Çıktı sözleşmesi ----------
class EventItem(BaseModel):
    time: str = Field(..., examples=["00:15"], description="MM:SS")
    event: str = Field(..., description="Türkçe olay açıklaması")


class DecisionResponse(BaseModel):
    summary: str
    events: List[EventItem]
    risk: RiskLevel
    actions: List[str]
    quantum_inspired_risk_score: float = Field(..., ge=0.0, le=1.0)


class HealthResponse(BaseModel):
    status: str
    service: str
    pipeline: List[str]


class SchemaResponse(BaseModel):
    input: dict
    output: dict
    flow: List[str]


# ---------- Füzyon ----------
class DecisionCore:
    def __init__(self, n: int = 4):
        self.n = n
        self.qubits = np.ones((n, 2)) / np.sqrt(2)

    def encode(self, movement: float, audio: float, proximity: float):
        tm, ta, tp = movement * np.pi / 2, audio * np.pi / 2, proximity * np.pi / 2
        self.qubits[0] = [np.cos(tm), np.sin(tm)]
        self.qubits[1] = [np.cos(ta), np.sin(ta)]
        self.qubits[2] = [np.cos(tp), np.sin(tp)]
        tt = (tm + ta + tp) / 3
        self.qubits[3] = [np.cos(tt), np.sin(tt)]

    def evaluate(self, iterations: int = 200) -> float:
        rows = []
        for _ in range(iterations):
            rows.append([1 if np.random.rand() < self.qubits[i, 1] ** 2 else 0 for i in range(self.n)])
        return float(np.dot(np.mean(rows, axis=0), [0.35, 0.35, 0.15, 0.15]))


def fuse(yolo: dict) -> DecisionResponse:
    anomalies = yolo.get("anomalies") or []
    classes = yolo.get("detected_classes") or []
    types = {a.get("anomaly_type", "") for a in anomalies}
    has_prox = any("yakinsama" in t.lower() for t in types)
    has_still = any("hareketsiz" in t.lower() for t in types)
    has_speed = any("hiz" in t.lower() for t in types)

    core = DecisionCore()
    core.encode(0.85 if anomalies else 0.10, 0.10, 0.75 if has_prox else 0.10)
    q = round(core.evaluate(), 4)

    events = [
        EventItem(
            time=a.get("time", "00:00"),
            event=a.get("description") or a.get("anomaly_type", "Olay"),
        )
        for a in anomalies[:12]
    ]

    if has_prox or has_still:
        risk: RiskLevel = "Yuksek" if (has_prox and has_still) or q > 0.55 else "Orta"
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

    return DecisionResponse(
        summary=summary,
        events=events,
        risk=risk,
        actions=actions,
        quantum_inspired_risk_score=q,
    )


# ---------- App ----------
app = FastAPI(
    title="MUDU-VLLM API",
    version="1.4.0",
    description="Video girdi → Algı → Füzyon → Türkçe karar JSON",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        service="mudu-vllm-api",
        pipeline=["yolo+bytetrack", "decision-core", "json"],
    )


@app.get("/v1/schema", response_model=SchemaResponse)
def schema():
    return SchemaResponse(
        input={
            "method": "POST",
            "path": "/v1/analyze",
            "content_type": "multipart/form-data",
            "field": "video",
            "accept": [".mp4", ".avi", ".mov", ".mkv"],
        },
        output=DecisionResponse.model_json_schema(),
        flow=[
            "video upload",
            "algı: YOLO+ByteTrack",
            "füzyon: DecisionCore",
            "çıktı: DecisionResponse",
        ],
    )


@app.post("/v1/analyze", response_model=DecisionResponse)
async def analyze(video: UploadFile = File(..., description="Güvenlik kamerası videosu")):
    if not video.filename:
        raise HTTPException(400, detail="video alanı zorunlu")
    suffix = Path(video.filename).suffix.lower() or ".mp4"
    if suffix not in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        raise HTTPException(400, detail=f"desteklenmeyen format: {suffix}")

    raw = await video.read()
    if not raw:
        raise HTTPException(400, detail="boş dosya")

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(raw)

        from yolo_pipeline import YoloPipeline

        yolo = YoloPipeline().run(path, conf=0.35)
        return fuse(yolo)
    except ImportError as e:
        raise HTTPException(500, detail=f"algı katmanı yok: {e}") from e
    except Exception as e:
        raise HTTPException(500, detail=f"analiz hatası: {e}") from e
    finally:
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
