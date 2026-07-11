"""
MUDU-VLLM — Senaryo 3: Video Karar Destek (MINI / 2B) V1.4 — TASINABILIR
========================================================================
Qwen2-VL-2B (transformers) + opsiyonel YOLO ipucu + Kuantum-esintili risk.

** Ollama GEREKMEZ. Windows / Linux / macOS (Apple Silicon dahil) calisir. **
** Apple Silicon (M1/M2/M3) Mac'te otomatik MPS (GPU) hizlandirmasi kullanir. **

------------------------------------------------------------------------
KURULUM:
    python3 -m venv venv
    # macOS / Linux:  source venv/bin/activate
    # Windows:        venv\\Scripts\\activate
    pip install torch transformers opencv-python pillow numpy

CALISTIRMA (video yolunu komut satirindan ver):
    # macOS:   python3 video_decision_support_mini_V1_4.py /Users/ad/Downloads/video.mp4
    # Windows: python video_decision_support_mini_V1_4.py "C:\\Users\\Ad\\Downloads\\video.mp4"
------------------------------------------------------------------------
Lisans: Apache License 2.0
"""
import os
import re
import sys
import json
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MAX_FRAMES = 8
MAX_NEW_TOKENS = 512


def pick_device():
    """Platforma gore en hizli cihazi sec:
       - Apple Silicon Mac -> MPS (GPU)
       - NVIDIA GPU'lu makine -> CUDA
       - digerleri -> CPU
    """
    if torch.backends.mps.is_available():          # Apple Silicon
        return "mps", torch.float32
    if torch.cuda.is_available():                   # NVIDIA
        return "cuda", torch.float16
    return "cpu", torch.float32                      # herkes


DEVICE, DTYPE = pick_device()
print(f"Cihaz secildi: {DEVICE.upper()}  (dtype={DTYPE})")

print("Model ve islemci yukleniyor...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=DTYPE,
).to(DEVICE)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def sample_frames_with_time(video_path, max_frames=MAX_FRAMES):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadi: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video okunamadi ya da bos.")
    step = max(total // max_frames, 1)
    frames, stamps = [], []
    for idx in range(0, total, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))
        secs = idx / fps
        stamps.append(f"{int(secs // 60):02d}:{int(secs % 60):02d}")
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames, stamps


class DecisionCore:
    """Hafif, Kuantum-esintili olasiliksal risk fuzyonu."""
    def __init__(self, num_features: int = 4):
        self.num_features = num_features
        self.qubits = np.ones((num_features, 2)) / np.sqrt(2)

    def encode_classical_to_quantum(self, movement_score, audio_score, proximity_score):
        tm = movement_score * (np.pi / 2)
        ta = audio_score * (np.pi / 2)
        tp = proximity_score * (np.pi / 2)
        self.qubits[0] = [np.cos(tm), np.sin(tm)]
        self.qubits[1] = [np.cos(ta), np.sin(ta)]
        self.qubits[2] = [np.cos(tp), np.sin(tp)]
        tt = (tm + ta + tp) / 3
        self.qubits[3] = [np.cos(tt), np.sin(tt)]

    def measure_collapse(self):
        return [1 if np.random.rand() < self.qubits[i, 1] ** 2 else 0
                for i in range(self.num_features)]

    def evaluate_risk(self, iterations=200):
        results = [self.measure_collapse() for _ in range(iterations)]
        mean_states = np.mean(results, axis=0)
        return float(np.dot(mean_states, [0.35, 0.35, 0.15, 0.15]))


def build_prompt(stamps, yolo_events=None, seen_classes=None):
    frame_lines = "\n".join(f"- Kare {i+1} ~= {t}" for i, t in enumerate(stamps))
    TR = {"person": "kisi", "dog": "kopek", "cat": "kedi", "car": "araba",
          "truck": "kamyon", "motorcycle": "motosiklet", "bus": "otobus", "knife": "bicak"}
    cls_block = ""
    if seen_classes:
        lines = [f"- {t}: {', '.join(sorted(TR.get(c, c) for c in seen_classes[t]))}"
                 for t in sorted(seen_classes) if seen_classes[t]]
        if lines:
            cls_block = "\nNesne dedektoru (YOLO) tespitleri:\n" + "\n".join(lines) + "\n"
    yolo_block = ""
    if yolo_events:
        ev = "\n".join(f"- {e['time']} | {e['anomaly_type']} | {e.get('description','')}" for e in yolo_events)
        yolo_block = "\nAlgilayici ipuclari (dogrula):\n" + ev + "\n"

    kategoriler = """Su kategorilere dikkat et, YALNIZCA gercekten gordugunu yaz:
- INSAN TEHDIDI: kavga/arbede, saldiri/darp, hirsizlik, silahla yaralanma, saklanan kisi, yerde hareketsiz kisi, dusme
- SILAH: tufek, tabanca, bicak, roketatar/fuze, patlayici
- ARAC/IS KAZASI: arac/forklift devrilmesi, arac-yaya carpma riski
- HAYVAN TEHDIDI: yilan, akrep, yaban domuzu, vahsi hayvan, saldirgan kopek
- HAYVANA ZARAR: bir hayvanin ezilmesi/carpilmasi (5199 sayili kanun ihlali)
- CEVRE: yangin, yogun duman"""

    ornek = """{
  "summary": "Depoda forklift devrildi, yakininda yerde hareketsiz personel; yuksek risk.",
  "events": [{"time": "00:15", "event": "Forklift devrildi"}],
  "risk": "Yuksek",
  "actions": ["Saglik ekibini yonlendir", "Alani guvenlik seridine al"]
}"""

    return f"""Sen bir guvenlik operasyon merkezi video analiz asistanisin. Zaman damgali kareler veriliyor.
Gorevin SADECE bu karelerde GERCEKTEN gordugun olaylari raporlamak.

Karelerin zaman damgalari:
{frame_lines}
{cls_block}{yolo_block}
{kategoriler}

KESIN KURALLAR:
1. Yalnizca net gordugun seyleri yaz. Emin degilsen YAZMA.
2. Nesne dedektoru HATA yapabilir; karede NET bir hayvan goruyorsan YOLO 'kisi' dese
   bile onu dogru tur olarak yaz. Kendi gozlemin onceliklidir.
3. Her olay icin SOMUT aciklama yaz. Tek kelimelik etiket YETERSIZ.
4. Her kareye olay UYDURMA. Sadece gercek, farkli olaylari yaz.
5. "summary" ASLA bos kalmasin; "actions" ASLA bos kalmasin.
6. RISK: kavga/saldiri/hareketsiz kisi -> EN AZ "Orta"; silah/agir yaralanma -> "Yuksek";
   arac-yaya/hayvan yakinsamasi -> EN AZ "Orta"; tehdit yoksa "Dusuk".
7. Hayvana carpma/eziyet -> risk EN AZ "Orta", "olasi hayvan haklari ihlali" not et.
8. Asagidaki ornegi ASLA kopyalama.

Cikti SADECE su JSON: summary, events ([{{"time":"MM:SS","event":"..."}}]),
risk ("Dusuk"|"Orta"|"Yuksek"), actions. Baska metin yazma.

FORMAT ornegi (kopyalama): {ornek}"""


def parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def analyze(video_path, max_frames=MAX_FRAMES, yolo_events=None, seen_classes=None):
    frames, stamps = sample_frames_with_time(video_path, max_frames)
    prompt = build_prompt(stamps, yolo_events, seen_classes)

    print("Kuantum-esintili risk analizi...")
    has_anom = bool(yolo_events)
    move_sc = 0.85 if has_anom else 0.10
    prox_sc = 0.75 if (yolo_events and any("yakinsamasi" in str(e.get("anomaly_type","")).lower() for e in yolo_events)) else 0.10
    q = DecisionCore()
    q.encode_classical_to_quantum(move_sc, 0.10, prox_sc)
    q_index = q.evaluate_risk(200)
    print(f"  [QUANTUM INDEX] Hibrit Risk Skoru: {q_index:.4f}")

    messages = [{"role": "user",
                 "content": [*[{"type": "image"} for _ in frames], {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to(DEVICE)
    print(f"Mini (2B) model ile analiz uretiliyor ({DEVICE.upper()})...")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True,
                                 clean_up_tokenization_spaces=False)[0]
    result = parse_json(raw)
    if result:
        result["quantum_inspired_risk_score"] = round(q_index, 4)
    return result, raw


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ornek.mp4")
        print(f"Not: video yolu verilmedi, varsayilan: {video_path}")

    print(f"\n=== Isleniyor (Mini 2B): {video_path} ===")
    try:
        result, raw = analyze(video_path)
        print("\n--- Sartnameye uygun JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2) if result else raw)
    except Exception as exc:
        print(f"HATA: {exc}")