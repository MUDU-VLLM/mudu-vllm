"""
MUDU-VLLM — Senaryo 3: Video Tabanlı Karar Destek Sistemi
Qwen2-VL-2B ile Türkçe analiz + YOLO/ByteTrack anomali olaylarını yorumlama.
Lisans: Apache License 2.0
"""
import os
import re
import json
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# Hızlı CPU testleri / canlı demo için 2B. Daha doğru sonuç için:
#   - Qwen/Qwen2.5-VL-3B-Instruct (daha güçlü, sınıf adı Qwen2_5_VLForConditionalGeneration)
#   - ya da jüriye gösterilecek çıktı için Ollama'daki qwen2.5vl:7b pipeline'ı
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
print("Hafifletilmiş model ve işlemci yükleniyor (lokal CPU modu)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="cpu",
)
processor = AutoProcessor.from_pretrained(MODEL_NAME)


def sample_frames_with_time(video_path, max_frames=10):
    """Videoyu eşit aralıklarla örnekler; (PIL_kareler, ['MM:SS', ...]) döner."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise ValueError("Video okunamadı ya da boş.")
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


def build_prompt(stamps, yolo_events=None):
    """Kopyalamayı önleyen, kategorili, boş-olay destekli JSON talimatı."""
    frame_lines = "\n".join(f"- Kare {i+1} ≈ {t}" for i, t in enumerate(stamps))

    yolo_block = ""
    if yolo_events:
        ev = "\n".join(
            f"- {e['time']} | {e['anomaly_type']} | {e.get('description', '')}"
            for e in yolo_events
        )
        yolo_block = (
            "\nNesne takip katmanı (YOLO+ByteTrack) şu ipuçlarını verdi; "
            "doğrula ve yalnızca tutarlı olanları rapora ekle:\n" + ev + "\n"
        )

    kategoriler = """Şu kategorilere dikkat et ve YALNIZCA gerçekten gördüğünü işaretle:
- İNSAN TEHDİDİ: kavga/arbede, fiziksel saldırı veya darp, hırsızlık, silahla yaralanma,
  gizlenen/saklanan şüpheli kişi, yerde hareketsiz kişi, düşme
- SİLAH: tüfek, tabanca, bıçak, roketatar/füze, patlayıcı
- ARAÇ / İŞ KAZASI: araç veya forklift devrilmesi, araç-yaya çarpma riski
- HAYVAN TEHDİDİ: yılan, akrep, yaban domuzu, kurt/ayı gibi vahşi hayvan, saldırgan köpek
- ÇEVRE: yangın, yoğun duman"""

    ornek = """{
  "summary": "Depo alanında forklift devrildi, yakınında yerde hareketsiz bir personel var; yüksek yaralanma riski.",
  "events": [
    {"time": "00:15", "event": "Forklift devrildi"},
    {"time": "00:20", "event": "Yerde hareketsiz yatan personel"}
  ],
  "risk": "Yüksek",
  "actions": ["Sağlık ekibini yönlendir", "Forklift çevresini güvenlik şeridine al"]
}"""

    return f"""Sen bir güvenlik operasyon merkezi video analiz asistanısın. Sana güvenlik
kamerasından alınmış, zaman damgalı kareler veriliyor. Görevin SADECE bu karelerde
GERÇEKTEN gördüğün olayları raporlamak.

Karelerin yaklaşık zaman damgaları:
{frame_lines}
{yolo_block}
{kategoriler}

KESİN KURALLAR:
1. Yalnızca karelerde net gördüğün şeyleri yaz. Emin değilsen YAZMA.
2. Her olay için SOMUT açıklama yaz (ör. "İki kişi birbirine vuruyor"). Tek kelimelik
   genel etiket ("Yaralanma") YETERSİZ.
3. Her kareye bir olay UYDURMA. Sadece gerçek ve birbirinden farklı olayları yaz.
4. Hiçbir anomali görmüyorsan: "events" listesini BOŞ bırak ve "risk": "Düşük" yaz.
5. Aşağıdaki örnek metni ASLA olduğu gibi kopyalama; kendi gözlemlerinle doldur.

Çıktı SADECE şu JSON formatında olsun, başka hiçbir metin yazma. Alanlar:
- summary: videonun kısa, somut Türkçe özeti
- events:  [{{"time": "MM:SS", "event": "gözlemlenen olay"}}]  (yoksa boş liste)
- risk:    "Düşük" | "Orta" | "Yüksek"
- actions: operatöre somut, uygulanabilir öneriler

Sadece FORMAT örneği (içeriği kopyalama, başka senaryoya aittir):
{ornek}"""


def parse_json(raw):
    """Model çıktısından ilk geçerli JSON bloğunu güvenle ayıkla."""
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def analyze(video_path, max_frames=10, yolo_events=None):
    frames, stamps = sample_frames_with_time(video_path, max_frames)
    prompt = build_prompt(stamps, yolo_events)
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
    print("Mini model ile analiz üretiliyor...")
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=512)
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return parse_json(raw), raw


if __name__ == "__main__":
    videos = [
        "/mnt/c/Users/saphi/Downloads/Abuse010_x264.mp4",
        "/mnt/c/Users/saphi/Downloads/Arrest001_x264.mp4",
    ]
    # Berke'nin AnomalyDetector.anomalies listesini şöyle geçireceksin:
    # result, raw = analyze(path, yolo_events=detector.anomalies)
    for path in videos:
        print(f"\n=== İşleniyor (Mini v1.3): {path} ===")
        try:
            result, raw = analyze(path, max_frames=10)
            print(json.dumps(result, ensure_ascii=False, indent=2)
                  if result else raw)
        except Exception as exc:
            print(f"HATA: {exc}")