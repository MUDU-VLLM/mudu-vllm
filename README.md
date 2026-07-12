# MUDU-VLLM

> **TEKNOFEST 2026 · Türkçe Doğal Dil İşleme Yarışması · Senaryo 3**
> Video Tabanlı Karar Destek Sistemi
>
> **Etiket:** `BilisimVadisi2026` · **Lisans:** [Apache License 2.0](LICENSE) · **Sürüm:** V1.4

Güvenlik kamerası videolarını **tamamen yerel / offline** ortamda analiz edip, operatöre
yapılandırılmış **Türkçe JSON** karar çıktısı üreten multimodal karar destek sistemi.
Harici bulut API'sine zorunlu bağımlılık yoktur.

---

## Ne Yapar?

**Girdi:** video (`.mp4`, `.avi`, `.mov`, `.mkv`) → **Çıktı:** Türkçe karar JSON'u

```json
{
  "summary": "Türkçe olay özeti",
  "events": [{ "time": "MM:SS", "event": "..." }],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": ["Operatör aksiyon önerileri"],
  "quantum_inspired_risk_score": 0.0
}
```

---

## Mimari

```
Video
  │
  ├─[1] Algı        YOLOv8 + ByteTrack        → nesne tespiti, takip, hareket anomalileri
  ├─[2] Ses         Whisper (faster-whisper)  → "imdat", "yardım", "silah" vb. (7B yolu)
  ├─[3] Füzyon      Quantum-Inspired          → hareket / ses / yakınsama skor füzyonu
  │                 DecisionCore                (quantum_inspired_risk_score)
  └─[4] Anlam       Qwen2.5-VL-7B (Ollama)    → kare + ipuçlarından Türkçe JSON
                    (mini: Qwen2-VL-2B)
```

İki çalışma yolu vardır:

| Yol | Katmanlar | Kullanım | Konum |
|-----|-----------|----------|-------|
| **Tam pipeline (7B / mini)** | YOLO + Whisper + VL + DecisionCore | Jüri / asıl doğruluk | `vllm-core/` |
| **API / Web servisi** | YOLO + DecisionCore → JSON | Hızlı demo / servis | `api-service/` + `web-ui/` |

---

## Hızlı Başlangıç

Ayrıntılı kurulum ve platform notları için **[CALISTIRMA.md](CALISTIRMA.md)**.

**Mini sürüm (Ollama gerekmez — ilk deneme için önerilir):**

```bash
cd vllm-core
python video_decision_support_mini_ubuntu_V1.4.py video.mp4
```

**Web arayüzü (yerel):**

```bash
cd web-ui
python app.py          # → http://127.0.0.1:7860
```

**API servisi (yerel):**

```bash
cd api-service
python app.py          # → http://127.0.0.1:8000/docs
```

**Docker (API + Web birlikte):**

```bash
docker compose up -d   # api:8000 + web:7860
```

**Sadece YOLO (VL / Ollama olmadan):**

```bash
cd yolo
python yolo_pipeline.py video.mp4   # → video_yolo.json
```

---

## Dizin Yapısı

```
mudu-vllm/
├── README.md              · Bu dosya
├── ozet.md                · Detaylı proje özeti
├── CALISTIRMA.md          · Kurulum + çalıştırma rehberi
├── LICENSE                · Apache 2.0
├── requirements.txt       · Yerel bağımlılıklar
├── docker-compose.yml     · api (8000) + web (7860)
├── yolov8n.pt             · YOLO ağırlığı
│
├── api-service/           · FastAPI · /v1/analyze · /health · /docs
├── web-ui/                · Tek sayfa arayüz · timeline · PDF · geçmiş
├── yolo/                  · YOLO + ByteTrack + anomali dedektörü
└── vllm-core/             · Tam karar destek betikleri (7B + mini · Win/Linux/Mac)
```

---

## API Uç Noktaları

| Endpoint | Açıklama |
|----------|----------|
| `GET  /health` | Servis sağlık kontrolü |
| `GET  /v1/schema` | Girdi/çıktı sözleşmesi |
| `POST /v1/analyze` | `multipart` · field=`video` → `DecisionResponse` |
| `GET  /docs` | Swagger arayüzü |

```bash
curl -F "video=@demo.mp4" http://127.0.0.1:8000/v1/analyze
```

---

## Algı Katmanı — Anomali Kuralları

| Tip | Mantık | Skor |
|-----|--------|------|
| Hareketsiz nesne | person/cat/dog ~1 sn hareketsiz | 0.85 |
| Ani hız değişimi | Baseline farkı > eşik | 0.75 |
| Araç–varlık yakınsaması | Mesafe < 80 px | 0.90 |

## Karar Füzyonu — Quantum-Inspired Risk

Klasik skorlar (hareket / ses / yakınsama) açısal kodlanır; olasılıksal ölçüm sonrası
ağırlıklı skor (`0.35 · 0.35 · 0.15 · 0.15`) üretilir → `quantum_inspired_risk_score`.
Gerçek kuantum donanımı kullanılmaz; olasılıksal bir füzyon motorudur.

---

## Gereksinimler (özet)

| | 7B (vllm-core) | Mini | API / Web / Docker |
|--|--|--|--|
| Python | 3.10+ | 3.10+ | 3.12 (imaj) |
| RAM | ≥ 16 GB | Daha düşük | ~4–8 GB |
| Ollama | Evet | Hayır | Hayır |
| ffmpeg | Ses için | — | — |

Tam liste: `requirements.txt` · Docker: `api-service/requirements.docker.txt`

---

## Lisans

Bu proje [Apache License 2.0](LICENSE) altında lisanslanmıştır.

**Takım:** MUDU-VLLM · Mudanya Üniversitesi
