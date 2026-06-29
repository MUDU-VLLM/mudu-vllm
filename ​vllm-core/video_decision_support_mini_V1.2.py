"""
MUDU-VLLM — Senaryo 3: Video Tabanlı Karar Destek Sistemi
Qwen2-VL-2B ile Türkçe analiz + YOLO/ByteTrack anomali olaylarını yorumlama.
Akış (şartnameye uygun mimari):
    YOLO+ByteTrack  ->  zaman damgalı anomaliler  ->  LLM yorumlar
    ->  {summary, events[time], risk, actions} şeklinde JSON
Lisans: Apache License 2.0
"""
import os
import re
import json
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
# Geliştirme ekibi, canlı demolar ve hızlı CPU testleri için kararlı 2B Mini Sürümü
MODEL_NAME = "Qwen/Qwen2-VL-2B-Instruct"
print("Hafifletilmiş model ve işlemci yükleniyor (lokal CPU modu)...")
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,   # CPU üzerinde kararlı float32 çalışır.
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
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Qwen renk doğruluğu için
        frames.append(Image.fromarray(rgb))
        secs = idx / fps
        stamps.append(f"{int(secs // 60):02d}:{int(secs % 60):02d}")
        if len(frames) >= max_frames:
            break
    cap.release()
    return frames, stamps
def build_prompt(stamps, yolo_events=None):
    """Şartnameye uygun JSON çıktı için Türkçe talimat üretir."""
    frame_lines = "\n".join(f"- Kare {i+1} ≈ {t}" for i, t in enumerate(stamps))
    # YOLO+ByteTrack katmanından gelen zaman damgalı anomaliler (köprü noktası)
    yolo_block = ""
    if yolo_events:
        ev = "\n".join(
            f"- {e['time']} | {e['anomaly_type']} | {e.get('description', '')}"
            for e in yolo_events
        )
        yolo_block = (
            "\nNesne takip katmanı (YOLO+ByteTrack) şu anomalileri zaman "
            "damgalarıyla bildirdi; bunları doğrula ve yorumla:\n" + ev + "\n"
        )
    return f"""Aşağıdaki güvenlik kamerası karelerini SIRAYLA incele.
Her karenin yaklaşık zaman damgası:
{frame_lines}
{yolo_block}
Görüntülerde hırsızlık, şiddet/arbede, düşme, yerde hareketsiz kişi,
araç-yaya yakınsaması gibi anomali ve riskleri ara. Olayları uygun
zaman damgasıyla eşleştir.
YALNIZCA aşağıdaki JSON şemasını döndür, başka hiçbir metin yazma:
{{
  "summary": "Videonun kısa Türkçe operasyonel özeti",
  "events": [
    {{"time": "MM:SS", "event": "Tespit edilen olay"}}
  ],
  "risk": "Düşük | Orta | Yüksek",
  "actions": ["Operatör aksiyon önerisi 1", "Operatör aksiyon önerisi 2"]
}}"""
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
            *[{"type": "image"} for _ in frames],   # <-- virgül ÖNEMLİ
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[text], images=frames, padding=True, return_tensors="pt"
    ).to("cpu")
    print("Mini model ile hızlı analiz üretiliyor...")
    with torch.no_grad():   # Bellek optimizasyonu
        gen = model.generate(**inputs, max_new_tokens=512) # 2B için 512 token fazlasıyla yeterli
    trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
    raw = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return parse_json(raw), raw
if __name__ == "__main__":
    # Test klasöründeki videoların yolları
    videos = [
        "/mnt/c/Users/saphi/Downloads/Abuse010_x264.mp4",
        "/mnt/c/Users/saphi/Downloads/Arrest001_x264.mp4",
    ]
    # Canlı entegrasyonda Berke'nin AnomalyDetector.anomalies listesini buraya geçireceksin:
    # result, raw = analyze(path, yolo_events=detector.anomalies)
    for path in videos:
        print(f"\n=== İşleniyor (Mini v1.2): {path} ===")
        try:
            result, raw = analyze(path, max_frames=10)
            print(json.dumps(result, ensure_ascii=False, indent=2)
                  if result else raw)
        except Exception as exc:
            print(f"HATA: {exc}")