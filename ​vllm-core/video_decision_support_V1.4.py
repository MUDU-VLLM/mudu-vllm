"""
MUDU-VLLM — Senaryo 3: Uctan uca video karar destek pipeline'i (7B) V1.4

Katmanlar:
    [HAREKET]  YOLO + ByteTrack -> nesne takibi + sinif etiketleri + hareket anomalileri
    [KUANTUM]  Quantum inspired -> olasiliksal hafif hibrit risk fuzyonu (ultra hizli)
    [GORSEL ]  Qwen2.5-VL-7B    -> semantik tehdit yorumu + Turkce JSON
    [SES    ]  Whisper          -> sesli ipuclari (imdat/help/ates)

Servis: Ollama (yerel/offline). Final/GPU: vLLM. OpenAI-uyumlu, tek degisiklik BASE_URL+MODEL.
Lisans: Apache License 2.0

ÖN KOSULLAR (bir kerelik):
    1) ollama pull qwen2.5vl:7b
    2) Modelfile:  FROM qwen2.5vl:7b
                   PARAMETER num_ctx 16384
       ollama create qwen2.5vl-16k -f Modelfile
    3) sudo apt-get install -y ffmpeg
    4) python3 -m venv venv && source venv/bin/activate
       pip install ultralytics opencv-python requests numpy faster-whisper

CALISTIRMA:  source venv/bin/activate && python3 video_decision_support_V1.4.py
"""

import os
import re
import json
import base64
import subprocess
from collections import defaultdict

import cv2
import numpy as np
import requests

try:
    from ultralytics import YOLO
    YOLO_OK = True
except ImportError:
    YOLO_OK = False
    print("UYARI: ultralytics yok -> YOLO atlanir. (pip install ultralytics)")

# YAPILANDIRMA
BASE_URL = "http://localhost:11434/v1"
MODEL    = "qwen2.5vl-16k"

YOLO_WEIGHTS   = "yolov8m.pt"
MAX_VL_FRAMES  = 8
SEND_WIDTH     = 768
KEEP_EVERY_SEC = 0.4
NUM_CTX        = 16384
VID_STRIDE     = 6
ENABLE_AUDIO   = True
WHISPER_SIZE   = "small"


# [HAREKET] YOLO + ByteTrack
class AnomalyDetector:
    def __init__(self, fps=30.0):
        self.fps = fps
        self.history = defaultdict(list)
        self.anomalies = []

    def update(self, track_id, cls_name, cx, cy, frame_idx):
        ts = self._ts(frame_idx)
        self.history[track_id].append({"cx": cx, "cy": cy, "frame": frame_idx,
                                       "time": ts, "cls": cls_name})
        if len(self.history[track_id]) >= 2:
            self._check(track_id, frame_idx, ts)

    def check_proximity(self, detections, frame_idx):
        ts = self._ts(frame_idx)
        vehicles = [d for d in detections if d["cls"] in {"car", "truck", "motorcycle", "bus"}]
        vulns    = [d for d in detections if d["cls"] in {"cat", "dog", "person"}]
        for v in vehicles:
            for u in vulns:
                dist = np.hypot(v["cx"] - u["cx"], v["cy"] - u["cy"])
                if dist < 80:
                    self._add(ts, u.get("track_id", -1), u["cls"], "Arac-varlik yakinsamasi",
                              f"{u['cls']} ve {v['cls']} cok yakin ({dist:.0f}px) -- carpma riski",
                              0.90, frame_idx)

    def _ts(self, frame_idx):
        s = frame_idx / self.fps
        return f"{int(s // 60):02d}:{int(s % 60):02d}"

    def _check(self, tid, frame_idx, ts):
        hist = self.history[tid]
        cls = hist[-1]["cls"]
        window = int(self.fps)
        if len(hist) >= window:
            last = hist[-window:]
            spread = np.std([h["cx"] for h in last]) + np.std([h["cy"] for h in last])
            if spread < 6 and cls in {"cat", "dog", "person"}:
                self._add(ts, tid, cls, "Hareketsiz nesne",
                          f"{cls} 1+ saniyedir hareketsiz (dusme/bilinc kaybi olasiligi)",
                          0.85, frame_idx)
        if len(hist) >= 6:
            def spd(i):
                return np.hypot(hist[-i]["cx"] - hist[-i-1]["cx"],
                                hist[-i]["cy"] - hist[-i-1]["cy"])
            if abs(spd(1) - np.mean([spd(i) for i in range(2, 6)])) > 18:
                self._add(ts, tid, cls, "Ani hiz degisimi",
                          f"{cls} nesnede ani hiz degisimi (carpma/dusme ani)", 0.75, frame_idx)

    def _add(self, ts, tid, cls, atype, desc, score, frame_idx):
        for a in self.anomalies:
            if (a["track_id"] == tid and a["anomaly_type"] == atype
                    and abs(frame_idx - a["frame_idx"]) < self.fps * 2):
                return
        self.anomalies.append({"time": ts, "track_id": tid, "class": cls,
                               "anomaly_type": atype, "description": desc,
                               "score": round(score, 2), "frame_idx": frame_idx})
        print(f"  [HAREKET] {ts} | {atype} | {cls} (track {tid})")


def run_yolo(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    detector = AnomalyDetector(fps=fps)
    frames_store = {}
    seen_classes = defaultdict(set)
    keep_stride = max(int(KEEP_EVERY_SEC * fps / VID_STRIDE), 1)

    results = YOLO(YOLO_WEIGHTS).track(source=video_path, persist=True, stream=True,
                                       tracker="bytetrack.yaml", verbose=False,
                                       vid_stride=VID_STRIDE)
    for step_idx, r in enumerate(results):
        frame_idx = step_idx * VID_STRIDE
        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            ids  = r.boxes.id.cpu().numpy().astype(int)
            clss = r.boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), tid, c in zip(xyxy, ids, clss):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                name = r.names[int(c)]
                detector.update(int(tid), name, float(cx), float(cy), frame_idx)
                dets.append({"cls": name, "cx": float(cx), "cy": float(cy), "track_id": int(tid)})
                seen_classes[detector._ts(frame_idx)].add(name)
        detector.check_proximity(dets, frame_idx)
        if step_idx % keep_stride == 0 and r.orig_img is not None:
            frames_store[frame_idx] = _downscale(r.orig_img)
    return detector.anomalies, frames_store, fps, seen_classes


# [SES] Whisper
AUDIO_KEYWORDS = ["imdat", "imdad", "yardim", "help", "yetisin", "kurtar",
                  "ates", "silah", "kacin", "dikkat", "yangin", "vuruldu", "saldiri"]


def transcribe_audio_cues(video_path):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("UYARI: faster-whisper yok -> ses atlandi.")
        return []
    wav = "/tmp/mudu_audio.wav"
    try:
        subprocess.run(["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000",
                        wav, "-loglevel", "error"], check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("UYARI: ffmpeg yok -> ses atlandi. (sudo apt install ffmpeg)")
        return []
    if not os.path.exists(wav):
        return []
    model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(wav, language="tr")
    cues = []
    for seg in segments:
        text = seg.text.strip()
        hit = next((k for k in AUDIO_KEYWORDS if k in text.lower()), None)
        if hit:
            ts = f"{int(seg.start // 60):02d}:{int(seg.start % 60):02d}"
            cues.append({"time": ts, "event": f'Sesli tehdit/yardim ifadesi: "{text}"'})
            print(f"  [SES] {ts} | '{hit}' -> {text}")
    return cues


# [KUANTUM] Quantum inspired speeded output
class DecisionCore:
    """
    Hafif ve ultra hizli Kuantum esintili Karar Motoru.
    Agir LLM katmanina gitmeden once verileri hizlica suzmek icin
    olasiliksal kuantum durumlari (Qubit) kullanarak milisaniyede karar uretir.
    """
    def __init__(self, num_features: int = 4):
        self.num_features = num_features
        self.qubits = np.ones((num_features, 2)) / np.sqrt(2)

    def encode_classical_to_quantum(self, movement_score: float, audio_score: float, proximity_score: float):
        theta_move = movement_score * (np.pi / 2)
        theta_audio = audio_score * (np.pi / 2)
        theta_prox = proximity_score * (np.pi / 2)

        self.qubits[0] = [np.cos(theta_move), np.sin(theta_move)]
        self.qubits[1] = [np.cos(theta_audio), np.sin(theta_audio)]
        self.qubits[2] = [np.cos(theta_prox), np.sin(theta_prox)]

        total_theta = (theta_move + theta_audio + theta_prox) / 3
        self.qubits[3] = [np.cos(total_theta), np.sin(total_theta)]

    def measure_collapse(self) -> list:
        collapsed_state = []
        for i in range(self.num_features):
            prob_of_one = self.qubits[i, 1] ** 2
            state = 1 if np.random.rand() < prob_of_one else 0
            collapsed_state.append(state)
        return collapsed_state

    def evaluate_risk(self, iterations):
        results = []
        for _ in range(iterations):
            results.append(self.measure_collapse())
        mean_states = np.mean(results, axis=0)
        final_risk_score = np.dot(mean_states, [0.35, 0.35, 0.15, 0.15])
        return float(final_risk_score)


# Kare yardimcilari
def _downscale(bgr):
    h, w = bgr.shape[:2]
    if w > SEND_WIDTH:
        bgr = cv2.resize(bgr, (SEND_WIDTH, int(h * SEND_WIDTH / w)))
    return bgr


def _ts(frame_idx, fps):
    s = frame_idx / fps
    return f"{int(s // 60):02d}:{int(s % 60):02d}"


def uniform_sample(video_path, n=MAX_VL_FRAMES):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    store = {}
    if total > 0:
        for idx in np.linspace(0, total - 1, n, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok:
                store[int(idx)] = _downscale(frame)
    cap.release()
    return store, fps


def pick_vl_frames(anomalies, frames_store, fps, n=MAX_VL_FRAMES):
    available = sorted(frames_store.keys())
    if not available:
        return []
    chosen = set()
    for a in anomalies:
        chosen.add(min(available, key=lambda f: abs(f - a["frame_idx"])))
    for idx in np.linspace(available[0], available[-1], max(n - len(chosen), 0) + 2, dtype=int):
        chosen.add(min(available, key=lambda f: abs(f - int(idx))))
        if len(chosen) >= n:
            break
    ordered = sorted(chosen)[:n]
    return [(idx, _ts(idx, fps), frames_store[idx]) for idx in ordered]


def _b64_jpeg(bgr):
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf).decode() if ok else None


# [GORSEL] VL prompt
CATEGORIES = """Su kategorilere dikkat et ve YALNIZCA gercekten gordugunu isaretle:
- INSAN TEHDIDI: kavga/arbede, fiziksel saldiri veya darp, hirsizlik, silahla yaralanma,
  gizlenen/saklanan supheli kisi, yerde hareketsiz kisi, dusme
- SILAH: tufek, tabanca, bicak, roketatar/fuze, patlayici
- ARAC / IS KAZASI: arac veya forklift devrilmesi, arac-yaya carpma riski
- HAYVAN TEHDIDI: yilan, akrep, yaban domuzu, kurt/ayi gibi vahsi hayvan, saldirgan kopek
- CEVRE: yangin, yogun duman"""

EXAMPLE_JSON = """{
  "summary": "Depo alaninda forklift devrildi, yakininda yerde hareketsiz personel; yuksek risk.",
  "events": [
    {"time": "00:15", "event": "Forklift devrildi"},
    {"time": "00:20", "event": "Yerde hareketsiz yatan personel"}
  ],
  "risk": "Yuksek",
  "actions": ["Saglik ekibini olay yerine yonlendir", "Forklift cevresini guvenlik seridine al"]
}"""

TR = {"person": "kisi", "dog": "kopek", "cat": "kedi", "car": "araba",
      "truck": "kamyon", "motorcycle": "motosiklet", "bus": "otobus",
      "bicycle": "bisiklet", "horse": "at", "cow": "inek", "sheep": "koyun",
      "knife": "bicak", "backpack": "sirt cantasi", "handbag": "el cantasi"}


def build_prompt(vl_frames, cues, seen_classes):
    frame_lines = "\n".join(f"- Kare {i+1} ~= {t}" for i, (_, t, _) in enumerate(vl_frames))

    cls_block = ""
    if seen_classes:
        lines = []
        for t in sorted(seen_classes.keys()):
            names = ", ".join(sorted(TR.get(c, c) for c in seen_classes[t]))
            if names:
                lines.append(f"- {t}: {names}")
        if lines:
            cls_block = ("\nNesne dedektoru (YOLO) su nesneleri tespit etti "
                         "(siniflari dogru kullan, tahmin etme):\n" + "\n".join(lines) + "\n")

    cue_block = ""
    if cues:
        cue_block = ("\nAlgilayici ipuclari (dogrula, tutarli olanlari ekle):\n"
                     + "\n".join(f"- {c}" for c in cues) + "\n")

    return f"""Sen bir guvenlik operasyon merkezi video analiz asistanisin. Sana guvenlik
kamerasindan alinmis, zaman damgali kareler veriliyor. Gorevin SADECE bu karelerde
GERCEKTEN gordugun olaylari raporlamak.

Karelerin yaklasik zaman damgalari:
{frame_lines}
{cls_block}{cue_block}
{CATEGORIES}

KESIN KURALLAR:
1. Yalnizca karelerde net gordugun seyleri yaz. Emin degilsen YAZMA.
2. Nesne dedektoru bir sinif bildirdiyse (or. kopek), onu kadin/insan diye DEGISTIRME.
3. Her olay icin SOMUT aciklama yaz. Tek kelimelik etiket ("Yaralanma") YETERSIZ.
4. Her kareye bir olay UYDURMA. Sadece gercek, farkli olaylari yaz.
5. "summary" ASLA bos kalmasin: en az bir cumle Turkce ozet yaz.
6. "actions" ASLA bos kalmasin: en az bir somut operator onerisi yaz.
7. Anomali yoksa "events" bos olabilir ama "risk" yine "Dusuk" yaz.
8. Asagidaki ornek metni ASLA kopyalama.

Cikti SADECE su JSON: summary (dolu, somut Turkce ozet), events ([{{"time":"MM:SS","event":"..."}}]),
risk ("Dusuk"|"Orta"|"Yuksek"), actions (dolu, somut oneriler). Baska metin yazma.

Sadece FORMAT ornegi (kopyalama):
{EXAMPLE_JSON}"""


def call_vl(vl_frames, cues, seen_classes):
    content = []
    for _, _, bgr in vl_frames:
        b64 = _b64_jpeg(bgr)
        if b64:
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    content.append({"type": "text", "text": build_prompt(vl_frames, cues, seen_classes)})
    payload = {"model": MODEL, "messages": [{"role": "user", "content": content}],
               "max_tokens": 768, "temperature": 0.1, "options": {"num_ctx": NUM_CTX}}
    resp = requests.post(f"{BASE_URL}/chat/completions", json=payload, timeout=900)
    if resp.status_code != 200:
        raise RuntimeError(f"Ollama {resp.status_code}: {resp.text[:400]}")
    return resp.json()["choices"][0]["message"]["content"]


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def merge_audio_events(result, audio_cues):
    if not result or not audio_cues:
        return result
    events = result.get("events", [])
    times = {e.get("time") for e in events}
    for c in audio_cues:
        if c["time"] not in times:
            events.append(c)
    result["events"] = sorted(events, key=lambda e: e.get("time", "99:99"))
    return result


# UCTAN UCA
def analyze(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadi: {video_path}")

    seen_classes = {}
    anomalies, frames_store, fps = [], {}, 30.0
    if YOLO_OK:
        print("[1/4] YOLO+ByteTrack hareket + nesne analizi...")
        try:
            anomalies, frames_store, fps, seen_classes = run_yolo(video_path)
        except Exception as e:
            print(f"  YOLO HATASI (orneklemeye dusuluyor): {e}")
            frames_store, fps = uniform_sample(video_path)
    else:
        print("[1/4] YOLO yok -> duzgun ornekleme...")
        frames_store, fps = uniform_sample(video_path)

    audio_cues = []
    if ENABLE_AUDIO:
        print("[2/4] Whisper ses analizi...")
        audio_cues = transcribe_audio_cues(video_path)

    cues = [f"[HAREKET] {a['time']} | {a['anomaly_type']} | {a['description']}" for a in anomalies]
    cues += [f"[SES] {c['time']} | {c['event']}" for c in audio_cues]

    # Quantum entegrasyon
    print("Hizli Olasiliksal Risk analizi baslatiliyor...")
    move_sc = 0.85 if len(anomalies) > 0 else 0.10
    audio_sc = 0.90 if len(audio_cues) > 0 else 0.10
    prox_sc = 0.75 if "yakinsamasi" in "".join(cues).lower() else 0.10

    q_core = DecisionCore()
    q_core.encode_classical_to_quantum(move_sc, audio_sc, prox_sc)
    quantum_risk_index = q_core.evaluate_risk(iterations=200)
    print(f"  [QUANTUM INDEX] Hesaplanan Hibrit Risk Skoru: {quantum_risk_index:.4f}")

    print(f"[3/4] {len(anomalies)} hareket + {len(audio_cues)} ses ipucu; kare seciliyor...")
    vl_frames = pick_vl_frames(anomalies, frames_store, fps)

    print(f"[4/4] 7B'ye {len(vl_frames)} kare gonderiliyor ({MODEL})...")
    raw = call_vl(vl_frames, cues, seen_classes)

    # Kuantum risk indexini cikti JSON'a enjekte et
    result = merge_audio_events(parse_json(raw), audio_cues)
    if result:
        result["quantum_inspired_risk_score"] = round(quantum_risk_index, 4)
    return result, raw


if __name__ == "__main__":
    videos = [
        "/mnt/c/Users/saphi/Downloads/Abuse010_x264.mp4",
        "/mnt/c/Users/saphi/Downloads/Arrest001_x264.mp4",
    ]
    for path in videos:
        print(f"\n=== Isleniyor (TAM): {path} ===")
        try:
            result, raw = analyze(path)
            print("\n--- Sartnameye uygun JSON ---")
            print(json.dumps(result, ensure_ascii=False, indent=2) if result else raw)
        except requests.exceptions.ConnectionError:
            print("HATA: Ollama'ya baglanilamadi. (curl http://localhost:11434)")
        except Exception as exc:
            print(f"HATA: {exc}")
