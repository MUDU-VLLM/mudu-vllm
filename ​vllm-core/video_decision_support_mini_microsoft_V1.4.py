"""
MUDU-VLLM — Senaryo 3: Video Karar Destek (MINI / 2B) V1.4 — TASINABILIR
========================================================================
Qwen2-VL-2B (transformers, CPU) + opsiyonel YOLO ipucu + Kuantum-esintili risk.

** Bu MINI surumdur: Ollama GEREKMEZ, dusuk kuruluma sahiptir.       **
** Windows / Linux / macOS hepsinde calisir. Juri cikti ASIL 7B'den. **

------------------------------------------------------------------------
KURULUM (her platform):
    python -m venv venv
    # Windows:  venv\\Scripts\\activate
    # Linux/Mac: source venv/bin/activate
    pip install torch transformers opencv-python pillow numpy

CALISTIRMA (video yolunu komut satirindan ver):
    python video_decision_support_mini_V1_4.py "C:\\Users\\Ad\\Downloads\\video.mp4"
    python video_decision_support_mini_V1_4.py /home/kullanici/video.mp4
    # Yol vermezsen ayni klasordeki 'ornek.mp4' aranir.
------------------------------------------------------------------------
Lisans: Apache License 2.0
"""
import os
import re
import sys
import json
import tempfile
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
MAX_FRAMES = 8
MAX_NEW_TOKENS = 512

print("Hafifletilmis model ve islemci yukleniyor (lokal CPU modu)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def sample_frames_with_time(video_path, max_frames=MAX_FRAMES):
    """Videoyu esit araliklarla ornekler; (PIL_kareler, ['MM:SS', ...]) doner."""
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


# [KUANTUM] Quantum inspired hizli risk fuzyonu
class DecisionCore:
    """Hafif, ultra hizli Kuantum-esintili Karar Motoru (olasiliksal Qubit fuzyonu)."""
    def __init__(self, num_features: int = 4):
        self.num_features = num_features
        self.qubits = np.ones((num_features, 2)) / np.sqrt(2)

    def encode_classical_to_quantum(self, movement_score, audio_score, proximity_score):
        theta_move = movement_score * (np.pi / 2)
        theta_audio = audio_score * (np.pi / 2)
        theta_prox = proximity_score * (np.pi / 2)
        self.qubits[0] = [np.cos(theta_move), np.sin(theta_move)]
        self.qubits[1] = [np.cos(theta_audio), np.sin(theta_audio)]
        self.qubits[2] = [np.cos(theta_prox), np.sin(theta_prox)]
        total_theta = (theta_move + theta_audio + theta_prox) / 3
        self.qubits[3] = [np.cos(total_theta), np.sin(total_theta)]

    def measure_collapse(self):
        out = []
        for i in range(self.num_features):
            prob_one = self.qubits[i, 1] ** 2
            out.append(1 if np.random.rand() < prob_one else 0)
        return out

    def evaluate_risk(self, iterations=200):
        results = [self.measure_collapse() for _ in range(iterations)]
        mean_states = np.mean(results, axis=0)
        return float(np.dot(mean_states, [0.35, 0.35, 0.15, 0.15]))


def build_prompt(stamps, yolo_events=None, seen_classes=None):
    frame_lines = "\n".join(f"- Kare {i+1} ~= {t}" for i, t in enumerate(stamps))

    TR = {"person": "kisi", "dog": "kopek", "cat": "kedi", "car": "araba",
          "truck": "kamyon", "motorcycle": "motosiklet", "bus": "otobus",
          "bicycle": "bisiklet", "horse": "at", "knife": "bicak"}
    cls_block = ""
    if seen_classes:
        lines = []
        for t in sorted(seen_classes.keys()):
            names = ", ".join(sorted(TR.get(c, c) for c in seen_classes[t]))
            if names:
                lines.append(f"- {t}: {names}")
        if lines:
            cls_block = ("\nNesne dedektoru (YOLO) su nesneleri tespit etti "
                         "(siniflari dogru kullan):\n" + "\n".join(lines) + "\n")

    yolo_block = ""
    if yolo_events:
        ev = "\n".join(
            f"- {e['time']} | {e['anomaly_type']} | {e.get('description', '')}"
            for e in yolo_events
        )
        yolo_block = "\nAlgilayici ipuclari (dogrula, tutarli olanlari ekle):\n" + ev + "\n"

    kategoriler = """Su kategorilere dikkat et ve YALNIZCA gercekten gordugunu isaretle:
- INSAN TEHDIDI: kavga/arbede, saldiri/darp, hirsizlik, silahla yaralanma, saklanan supheli kisi, yerde hareketsiz kisi, dusme
- SILAH: tufek, tabanca, bicak, roketatar/fuze, patlayici
- ARAC / IS KAZASI: arac veya forklift devrilmesi, arac-yaya carpma riski
- HAYVAN TEHDIDI: yilan, akrep, yaban domuzu, vahsi hayvan, saldirgan kopek
- HAYVANA ZARAR: arac/kisi tarafindan bir hayvanin ezilmesi/carpilmasi (5199 sayili kanun ihlali)
- CEVRE: yangin, yogun duman"""

    ornek = """{
  "summary": "Depo alaninda forklift devrildi, yakininda yerde hareketsiz personel; yuksek risk.",
  "events": [{"time": "00:15", "event": "Forklift devrildi"}, {"time": "00:20", "event": "Yerde hareketsiz personel"}],
  "risk": "Yuksek",
  "actions": ["Saglik ekibini yonlendir", "Alani guvenlik seridine al"]
}"""

    return f"""Sen bir guvenlik operasyon merkezi video analiz asistanisin. Sana guvenlik
kamerasindan alinmis, zaman damgali kareler veriliyor. Gorevin SADECE bu karelerde
GERCEKTEN gordugun olaylari raporlamak.

Karelerin yaklasik zaman damgalari:
{frame_lines}
{cls_block}{yolo_block}
{kategoriler}

KESIN KURALLAR:
1. Yalnizca karelerde net gordugun seyleri yaz. Emin degilsen YAZMA.
2. Nesne dedektoru HATA yapabilir; karede NET bir kopek/hayvan goruyorsan YOLO 'kisi'
   dese bile onu dogru tur (kopek/kedi) olarak yaz. Kendi gozlemin onceliklidir.
3. Her olay icin SOMUT aciklama yaz. Tek kelimelik etiket YETERSIZ.
4. Her kareye olay UYDURMA. Sadece gercek, farkli olaylari yaz.
5. "summary" ASLA bos kalmasin: en az bir cumle Turkce ozet yaz.
6. "actions" ASLA bos kalmasin: en az bir somut operator onerisi yaz.
7. RISK SEVIYESI: kavga/saldiri/hareketsiz kisi -> EN AZ "Orta"; silah/agir yaralanma -> "Yuksek";
   arac-yaya/hayvan yakinsamasi -> EN AZ "Orta"; tehdit yoksa "Dusuk".
8. Hayvana carpma/eziyet varsa risk EN AZ "Orta", "olasi hayvan haklari ihlali" not et.
9. Asagidaki ornek metni ASLA kopyalama.

Cikti SADECE su JSON: summary (dolu, somut), events ([{{"time":"MM:SS","event":"..."}}]),
risk ("Dusuk"|"Orta"|"Yuksek"), actions (dolu). Baska metin yazma.

Sadece FORMAT ornegi (kopyalama):
{ornek}"""


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

    print("Kuantum-esintili olasiliksal risk analizi...")
    has_anom = bool(yolo_events)
    move_sc = 0.85 if has_anom else 0.10
    prox_sc = 0.75 if (yolo_events and any("yakinsamasi" in str(e.get("anomaly_type","")).lower() for e in yolo_events)) else 0.10
    q = DecisionCore()
    q.encode_classical_to_quantum(move_sc, 0.10, prox_sc)
    q_index = q.evaluate_risk(200)
    print(f"  [QUANTUM INDEX] Hibrit Risk Skoru: {q_index:.4f}")

    messages = [{
        "role": "user",
        "content": [*[{"type": "image"} for _ in frames], {"type": "text", "text": prompt}],
    }]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=frames, padding=True, return_tensors="pt").to("cpu")
    print("Mini (2B) model ile analiz uretiliyor...")
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
    # Video yolu: komut satirindan (tasinabilir). Yoksa ayni klasorde 'ornek.mp4'.
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ornek.mp4")
        print(f"Not: video yolu verilmedi, varsayilan deneniyor: {video_path}")

    print(f"\n=== Isleniyor (Mini 2B): {video_path} ===")
    try:
        result, raw = analyze(video_path)
        print("\n--- Sartnameye uygun JSON ---")
        print(json.dumps(result, ensure_ascii=False, indent=2) if result else raw)
    except Exception as exc:
        print(f"HATA: {exc}")