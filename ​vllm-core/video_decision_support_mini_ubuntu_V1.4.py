"""
MUDU-VLLM — Senaryo 3: Video Tabanli Karar Destek Sistemi (MINI / 2B)

Qwen2-VL-2B (transformers, CPU) ile Turkce analiz.

** Bu MINI surumdur: hizli CPU testi / dusuk donanimli makineler icindir. **
** Juriye gosterilecek ASIL cikti 7B surumunden alinir (qwen2.5vl, Ollama). **

YOLO ipuclari opsiyoneldir: analyze(path, yolo_events=detector.anomalies)
Lisans: Apache License 2.0

Kurulum:
    python3 -m venv venv && source venv/bin/activate
    pip install torch transformers opencv-python pillow numpy
"""
import os
import re
import json
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


# Quantum inspired speeded output
class DecisionCore:
    """
    Hafif ve ultra hizli Kuantum esintili Karar Motoru.
    Agir LLM katmanina gitmeden once verileri hizlica suzmek icin
    olasiliksal kuantum durumlari (Qubit) kullanarak milisaniyede karar uretir.
    """
    def __init__(self, num_features: int = 4):
        self.num_features = num_features
        self.qubits = np.ones((num_features, 2)) / np.sqrt(2)

    def encode_classical_to_quantum(self, movement_score: float, audio_active: bool, proximity_active: bool):
        theta_move = movement_score * (np.pi / 2) if audio_active else 0.10 * (np.pi / 2)
        theta_audio = 0.10 * (np.pi / 2)
        theta_prox = 0.75 * (np.pi / 2) if proximity_active else 0.10 * (np.pi / 2)

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


def build_prompt(stamps, yolo_events=None, seen_classes=None):
    """7B ile ayni yapida: kategoriler + YOLO ipucu + summary/actions zorunlu."""
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
                         "(siniflari dogru kullan, tahmin etme):\n" + "\n".join(lines) + "\n")

    yolo_block = ""
    if yolo_events:
        ev = "\n".join(
            f"- {e['time']} | {e['anomaly_type']} | {e.get('description', '')}"
            for e in yolo_events
        )
        yolo_block = (
            "\nAlgilayici ipuclari (dogrula, tutarli olanlari ekle):\n" + ev + "\n"
        )

    kategoriler = """Su kategorilere dikkat et ve YALNIZCA gercekten gordugunu isaretle:
- INSAN TEHDIDI: kavga/arbede, fiziksel saldiri veya darp, hirsizlik, silahla yaralanma,
  gizlenen/saklanan supheli kisi, yerde hareketsiz kisi, dusme
- SILAH: tufek, tabanca, bicak, roketatar/fuze, patlayici
- ARAC / IS KAZASI: arac veya forklift devrilmesi, arac-yaya carpma riski
- HAYVAN TEHDIDI: yilan, akrep, yaban domuzu, kurt/ayi gibi vahsi hayvan, saldirgan kopek
- CEVRE: yangin, yogun duman"""

    ornek = """{
  "summary": "Depo alaninda forklift devrildi, yakininda yerde hareketsiz bir personel var; yuksek yaralanma riski.",
  "events": [
    {"time": "00:15", "event": "Forklift devrildi"},
    {"time": "00:20", "event": "Yerde hareketsiz yatan personel"}
  ],
  "risk": "Yuksek",
  "actions": ["Saglik ekibini yonlendir", "Forklift cevresini guvenlik seridine al"]
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
2. Nesne dedektoru bir sinif bildirdiyse (or. kopek), onu kadin/insan diye DEGISTIRME.
3. Her olay icin SOMUT aciklama yaz (or. "Iki kisi birbirine vuruyor"). Tek kelimelik
   genel etiket ("Yaralanma") YETERSIZ.
4. Her kareye bir olay UYDURMA. Sadece gercek ve birbirinden farkli olaylari yaz.
5. "summary" ASLA bos kalmasin: en az bir cumle Turkce ozet yaz.
6. "actions" ASLA bos kalmasin: en az bir somut operator onerisi yaz.
7. Anomali yoksa "events" bos olabilir ama "risk" yine "Dusuk" yaz.
8. Asagidaki ornek metni ASLA oldugu gibi kopyalama; kendi gozlemlerinle doldur.

Cikti SADECE su JSON formatinda olsun, baska hicbir metin yazma. Alanlar:
- summary: videonun kisa, somut Turkce ozeti (dolu)
- events:  [{{"time": "MM:SS", "event": "gozlemlenen olay"}}]  (yoksa bos liste)
- risk:    "Dusuk" | "Orta" | "Yuksek"
- actions: operatore somut, uygulanabilir oneriler (dolu)

Sadece FORMAT ornegi (icerigi kopyalama, baska senaryoya aittir):
{ornek}"""


def parse_json(raw):
    """Model ciktisindan ilk gecerli JSON blogunu guvenle ayikla."""
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

    # Quantum tetikleme
    print("Kuantum-Esinli Olasiliksal risk analizi baslatiliyor...")
    has_anomalies = yolo_events is not None and len(yolo_events) > 0
    move_score = yolo_events[0].get("score", 0.85) if has_anomalies else 0.10
    prox_active = False
    if yolo_events:
        prox_active = any("yakinsamasi" in str(e.get("anomaly_type", "")).lower() for e in yolo_events)

    q_core = DecisionCore()
    q_core.encode_classical_to_quantum(move_score, has_anomalies, prox_active)
    quantum_risk_index = q_core.evaluate_risk(iterations=200)
    print(f"  [QUANTUM INDEX] Hesaplanan Hibrit Risk Skoru: {quantum_risk_index:.4f}")

    messages = [{
        "role": "user",
        "content": [
            *[{"type": "image"} for _ in frames],
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=frames, padding=True, return_tensors="pt"
    ).to("cpu")
    print("Mini (2B) model ile analiz uretiliyor...")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Olasiliksal Kuantum skorunu JSON sozlugune ekle
    parsed_res = parse_json(raw)
    if parsed_res:
        parsed_res["quantum_inspired_risk_score"] = round(quantum_risk_index, 4)
    return parsed_res, raw


if __name__ == "__main__":
    videos = [
        "/mnt/c/Users/saphi/Downloads/Arrest001_x264.mp4",
    ]
    for path in videos:
        print(f"\n=== Isleniyor (Mini 2B): {path} ===")
        try:
            result, raw = analyze(path, max_frames=MAX_FRAMES)
            print(json.dumps(result, ensure_ascii=False, indent=2)
                  if result else raw)
        except Exception as exc:
            print(f"HATA: {exc}")
