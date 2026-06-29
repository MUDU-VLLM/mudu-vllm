"""
MUDU-VLLM — YOLO + ByteTrack Pipeline
Anomali tespiti: hareketsiz nesne, ani hiz degisimi, arac-varlik yakinsamasi.
"""

import cv2
import json
import numpy as np
from collections import defaultdict
from pathlib import Path

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("UYARI: ultralytics yuklu degil — pip install ultralytics")


class AnomalyDetector:
    """
    ByteTrack track gecmisini frame-frame analiz eder.
    Tespit ettigi anomaliler:
      - Hareketsiz kalma  (dusme / kaza / bilinc kaybi)
      - Ani hiz degisimi  (carpma ani)
      - Arac-varlik yakinsamasi (carpma riski)
    """

    def __init__(self, fps: float = 30.0):
        self.fps = fps
        self.history: dict = defaultdict(list)
        self.anomalies: list = []

    def update(self, track_id: int, cls_name: str,
               cx: float, cy: float, frame_idx: int):
        ts = self._ts(frame_idx)
        self.history[track_id].append(
            {"cx": cx, "cy": cy, "frame": frame_idx,
             "time": ts, "cls": cls_name}
        )
        if len(self.history[track_id]) >= 2:
            self._check(track_id, frame_idx, ts)

    def check_proximity(self, detections: list, frame_idx: int):
        """Arac ile kisi/hayvan arasindaki mesafeyi kontrol et."""
        ts = self._ts(frame_idx)
        vehicles = [d for d in detections
                    if d["cls"] in {"car", "truck", "motorcycle", "bus"}]
        vulns    = [d for d in detections
                    if d["cls"] in {"cat", "dog", "person"}]
        for v in vehicles:
            for u in vulns:
                dist = np.hypot(v["cx"] - u["cx"], v["cy"] - u["cy"])
                if dist < 80:
                    self._add(
                        ts, u.get("track_id", -1), u["cls"],
                        "Arac-varlik yakinsamasi",
                        f"{u['cls']} ve {v['cls']} cok yakin "
                        f"({dist:.0f}px) — carpma riski yuksek",
                        score=0.90, frame_idx=frame_idx
                    )

    def _ts(self, frame_idx: int) -> str:
        secs = frame_idx / self.fps
        return f"{int(secs // 60):02d}:{int(secs % 60):02d}"

    def _check(self, tid: int, frame_idx: int, ts: str):
        hist = self.history[tid]
        cls  = hist[-1]["cls"]

        # Kural 1 — Hareketsiz kalma (>1 saniye)
        window = int(self.fps)
        if len(hist) >= window:
            last   = hist[-window:]
            spread = (np.std([h["cx"] for h in last]) +
                      np.std([h["cy"] for h in last]))
            if spread < 6 and cls in {"cat", "dog", "person"}:
                self._add(ts, tid, cls,
                          "Hareketsiz nesne",
                          f"{cls} 1+ saniyedir hareketsiz "
                          "(dusme/kaza/bilinc kaybi olasiligi)",
                          score=0.85, frame_idx=frame_idx)

        # Kural 2 — Ani hiz degisimi
        if len(hist) >= 6:
            def spd(i):
                return np.hypot(hist[-i]["cx"] - hist[-i-1]["cx"],
                                hist[-i]["cy"] - hist[-i-1]["cy"])
            recent   = spd(1)
            baseline = np.mean([spd(i) for i in range(2, 6)])
            if abs(recent - baseline) > 18:
                self._add(ts, tid, cls,
                          "Ani hiz degisimi",
                          f"{cls} nesnede ani hiz degisimi "
                          "(carpma/dusme ani)",
                          score=0.75, frame_idx=frame_idx)

    def _add(self, ts, tid, cls, atype, desc, score, frame_idx):
        for a in self.anomalies:
            if (a["track_id"] == tid and
                    a["anomaly_type"] == atype and
                    abs(frame_idx - a["frame_idx"]) < self.fps * 2):
                return
        entry = {
            "time": ts, "track_id": tid, "class": cls,
            "anomaly_type": atype, "description": desc,
            "score": round(score, 2), "frame_idx": frame_idx
        }
        self.anomalies.append(entry)
        print(f"  [ANOMALİ] {ts} | {atype} | {cls} (track {tid})")


class YoloPipeline:
    """
    YOLOv8 + ByteTrack ile video analizi.
    Her frame'deki nesneleri tespit eder, takip eder,
    anomali dedektorune bildirir.
    """

    def __init__(self, model_size: str = "yolov8n.pt"):
        if not YOLO_OK:
            raise ImportError("ultralytics yuklu degil — pip install ultralytics")
        print(f"YOLO modeli yukleniyor: {model_size}")
        self.model = YOLO(model_size)
        self.detector: AnomalyDetector | None = None
        self.all_frames: list = []
        self.critical_frames: list = []

    def run(self, video_path: str, conf: float = 0.35) -> dict:
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        self.detector = AnomalyDetector(fps=fps)

        print("YOLO + ByteTrack analizi basladi...")
        results = self.model.track(
            source=video_path,
            stream=True,
            conf=conf,
            tracker="bytetrack.yaml",
            persist=True,
            verbose=False
        )

        frame_idx = 0
        for r in results:
            ts   = self.detector._ts(frame_idx)
            dets = []

            if r.boxes is not None and len(r.boxes):
                for box in r.boxes:
                    cid   = int(box.cls[0])
                    cname = r.names[cid]
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    tid = int(box.id[0]) if box.id is not None else -1
                    det = {
                        "cls": cname,
                        "conf": round(float(box.conf[0]), 2),
                        "bbox": [round(x1, 1), round(y1, 1),
                                 round(x2, 1), round(y2, 1)],
                        "cx": round(cx, 1), "cy": round(cy, 1),
                        "track_id": tid
                    }
                    dets.append(det)
                    if tid >= 0:
                        self.detector.update(tid, cname, cx, cy, frame_idx)

                self.detector.check_proximity(dets, frame_idx)

            self.all_frames.append(
                {"frame": frame_idx, "time": ts, "detections": dets}
            )
            frame_idx += 1

        self._select_critical()
        return self._build_json()

    def _select_critical(self):
        """Her anomali icin ±2 kare sec."""
        aframes    = {a["frame_idx"] for a in self.detector.anomalies}
        candidates = set()
        for f in aframes:
            for off in range(-2, 3):
                t = f + off
                if 0 <= t < len(self.all_frames):
                    candidates.add(t)
        self.critical_frames = sorted(candidates)

    def _build_json(self) -> dict:
        all_cls = {d["cls"]
                   for fd in self.all_frames
                   for d in fd["detections"]}
        return {
            "detected_classes": sorted(all_cls),
            "anomalies": self.detector.anomalies,
            "critical_frames": self.critical_frames,
            "critical_frame_data": [
                self.all_frames[i] for i in self.critical_frames
            ],
            "summary": {
                "total_frames": len(self.all_frames),
                "total_anomalies": len(self.detector.anomalies),
                "anomaly_types": sorted({
                    a["anomaly_type"]
                    for a in self.detector.anomalies
                })
            }
        }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Kullanim: python yolo_pipeline.py <video_yolu> [conf=0.35]")
        sys.exit(0)

    video = sys.argv[1]
    conf  = float(sys.argv[2]) if len(sys.argv) > 2 else 0.35

    pipeline = YoloPipeline()
    result   = pipeline.run(video, conf=conf)

    out = Path(video).stem + "_yolo.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nTespit edilen siniflar : {result['detected_classes']}")
    print(f"Anomali sayisi         : {result['summary']['total_anomalies']}")
    print(f"Kritik kare sayisi     : {len(result['critical_frames'])}")
    print(f"Sonuc kaydedildi       : {out}")
