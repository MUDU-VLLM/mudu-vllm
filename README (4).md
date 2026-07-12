<div align="center">

# MUDU-VLLM

### Video Tabanlı Karar Destek Sistemi

Güvenlik kamerası videolarını **tamamen yerel / offline** analiz eden, operatöre
yapılandırılmış **Türkçe JSON** karar çıktısı üreten multimodal karar destek sistemi.

`TEKNOFEST 2026` · `Türkçe Doğal Dil İşleme` · `Senaryo 3 — Video Analiz ve Karar Destek`

**Etiket:** `BilisimVadisi2026` — **Lisans:** [Apache 2.0](LICENSE) — **Sürüm:** V1.4

</div>

---

## İçindekiler

- [Ne Yapar?](#ne-yapar)
- [Sürüm ve Test Durumu](#sürüm-ve-test-durumu)
- [Mimari](#mimari)
- [Hızlı Başlangıç](#hızlı-başlangıç)
- [Dizin Yapısı](#dizin-yapısı)
- [Servis Katmanı (API · Web · Docker)](#servis-katmanı-api--web--docker)
- [Algı ve Füzyon Detayları](#algı-ve-füzyon-detayları)
- [Veri Seti](#veri-seti)
- [Dokümanlar](#dokümanlar)

---

## Ne Yapar?

**Girdi:** güvenlik kamerası videosu (`.mp4`, `.avi`, `.mov`, `.mkv`)
**Çıktı:** yapılandırılmış Türkçe karar JSON'u

```json
{
  "summary": "Türkçe olay özeti",
  "events": [{ "time": "MM:SS", "event": "..." }],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": ["Operatör aksiyon önerileri"],
  "quantum_inspired_risk_score": 0.0
}
```

Sistem olayları zaman damgasıyla listeler, Türkçe özet çıkarır, risk seviyesi belirler ve
operatöre somut aksiyon önerir. Harici bulut / API bağımlılığı yoktur; tümüyle yerelde çalışır.

---

## Sürüm ve Test Durumu

Şeffaflık için her bileşenin olgunluk seviyesi ayrı belirtilmiştir.

| Bileşen | Durum |
|---------|-------|
| Ubuntu / Linux — 7B (tam pipeline) | ✅ Test edildi, çalışıyor (GPU üzerinde doğrulandı) |
| Ubuntu / Linux — 2B (mini) | ⚙️ Kurulum hazır, kısmi test |
| Windows — 7B / 2B | ⚙️ Kurulum hazır, uçtan uca test sürüyor |
| macOS (Apple Silicon) — 7B / 2B | ⚙️ Kurulum hazır, test aşamasında |
| FastAPI servis katmanı | ✅ Çalışıyor — `/v1/analyze` · `/health` · `/docs` |
| Web UI (tek sayfa arayüz) | ✅ Çalışıyor — NDJSON ilerleme akışı, timeline, PDF |
| Docker entegrasyonu | ✅ Çalışıyor — tek imaj, `docker compose up` |

Ana referans **Ubuntu 7B** sürümüdür. Diğer platform sürümleri aynı kod tabanının taşınabilir varyantlarıdır.

---

## Mimari

Üç katman + olasılıksal füzyon çekirdeği:

```
        ┌──────────────────────────────────────────────────────────┐
Video → │  [1] ALGI       YOLOv8 + ByteTrack                         │
        │        └─ nesne tespiti · takip · hareket anomalileri      │
        │  [2] SES        Whisper (faster-whisper) · opsiyonel       │
        │        └─ "imdat", "yardım", "silah" anahtar ifadeleri     │
        │  [3] FÜZYON     Quantum-Inspired DecisionCore              │
        │        └─ hareket / ses / yakınsama → risk skoru           │
        │  [4] ANLAM      Qwen2.5-VL-7B (mini: 2B)                    │
        │        └─ kare + ipuçları → Türkçe JSON                    │
        └──────────────────────────────────────────────────────────┘
                                   ↓
                    Yapılandırılmış Türkçe karar JSON'u
```

Algı katmanının yapılandırılmış çıktısı, anlamlandırma katmanına **ipucu** olarak verilir;
böylece düşük seviyeli nesne tespiti ile yüksek seviyeli olay yorumu arasında köprü kurulur.

**Model servisleme:** Geliştirmede yerel model için **Ollama** (OpenAI uyumlu `/v1/chat/completions`).
Final/GPU ortamında yalnızca `BASE_URL` + `MODEL` değiştirilerek **vLLM**'e taşınabilir; uygulama kodu aynı kalır.

---

## Hızlı Başlangıç

Ayrıntılı, platforma özel kurulum için → **[CALISTIRMA.md](CALISTIRMA.md)** ve **[vllm-core/README.md](vllm-core/README.md)**

**1) Mini sürüm — en kolay, Ollama gerekmez**
```bash
pip install torch transformers opencv-python pillow numpy
python3 vllm-core/video_decision_support_mini_ubuntu_V1.4.py video.mp4
```

**2) Tam 7B — asıl doğruluk (Ollama gerekir)**
```bash
ollama pull qwen2.5vl:7b
# Modelfile:  FROM qwen2.5vl:7b  /  PARAMETER num_ctx 16384
ollama create qwen2.5vl-16k -f Modelfile
pip install -r requirements.txt
python3 vllm-core/video_decision_support_ubuntu_V1.4.py video.mp4
```

**3) Web arayüzü**
```bash
cd web-ui && python app.py        # → http://127.0.0.1:7860
```

**4) Docker (API + Web birlikte)**
```bash
docker compose up -d              # api:8000 + web:7860
```

> Video yolu verilmezse betikler kendi klasöründeki `ornek.mp4` dosyasını arar.

---

## Dizin Yapısı

```
mudu-vllm/
├── README.md              · Bu dosya (front-page)
├── CALISTIRMA.md          · Platforma özel kurulum + çalıştırma rehberi
├── ozet.md                · Detaylı proje özeti
├── LICENSE                · Apache 2.0
├── requirements.txt       · Yerel bağımlılıklar (7B + mini + YOLO + API)
├── docker-compose.yml     · api (8000) + web (7860)
├── yolov8n.pt             · YOLO ağırlığı (Docker imajına gömülür)
│
├── vllm-core/             · Tam karar destek betikleri (7B + mini · Win/Linux/Mac)
│   ├── README.md          · vllm-core kullanma kılavuzu
│   ├── video_decision_support_{ubuntu,windows,MAC}_V1.4.py        · 7B · Ollama
│   └── video_decision_support_mini_{ubuntu,microsoft,MAC}_V1.4.py · 2B · Ollama'sız
│
├── yolo/                  · YOLO + ByteTrack + anomali dedektörü
│   └── yolo_pipeline.py   · CLI: python yolo_pipeline.py video.mp4 → *_yolo.json
│
├── api-service/           · FastAPI servisi
│   ├── app.py             · /health · /v1/schema · /v1/analyze · /docs
│   └── Dockerfile         · Tek imaj (api + web)
│
└── web-ui/                · Tek sayfa arayüz
    ├── app.py             · / · /api/analyze (NDJSON stream)
    └── index.html         · timeline · PDF · geçmiş
```

---

## Servis Katmanı (API · Web · Docker)

Tam pipeline betiklerine ek olarak, hızlı demo ve entegrasyon için HTTP servis katmanı ve
Docker paketlemesi gelir. Bu yol **algı (YOLO + ByteTrack) + füzyon (DecisionCore)** çalıştırıp
Türkçe JSON döndürür; tam VL (Qwen) + Whisper doğrulaması asıl olarak `vllm-core` 7B betiğindedir.

### FastAPI — `api-service/app.py`

| Uç Nokta | Açıklama |
|----------|----------|
| `GET  /health` | Servis sağlık kontrolü |
| `GET  /v1/schema` | Girdi/çıktı sözleşmesi (JSON şema) |
| `POST /v1/analyze` | `multipart` · field=`video` → `DecisionResponse` |
| `GET  /docs` | Swagger arayüzü |

```bash
cd api-service && python app.py          # → http://127.0.0.1:8000
curl -F "video=@demo.mp4" http://127.0.0.1:8000/v1/analyze
```

### Web UI — `web-ui/app.py` + `index.html`

Video sürükle-bırak, canlı **NDJSON** ilerleme akışı (`POST /api/analyze`), sonuç panelinde
özet / risk / olaylar / aksiyonlar / quantum skor, zaman çizelgesi, PDF indirme, `localStorage` geçmişi.

### Docker

Tek imaj (`mudu-vllm:1.4`) hem API hem Web kodunu içerir. CPU torch ve yerel `yolov8n.pt` imaja gömülür.

```bash
docker compose up -d        # api:8000 + web:7860
docker compose up -d api    # yalnız API
docker compose up -d web    # yalnız Web
```

---

## Algı ve Füzyon Detayları

**Anomali kuralları** (`yolo/yolo_pipeline.py`):

| Tip | Mantık | Skor |
|-----|--------|------|
| Hareketsiz nesne | person/cat/dog ~1 sn hareketsiz (düşme/kaza) | 0.85 |
| Ani hız değişimi | Baseline farkı > eşik (çarpma anı) | 0.75 |
| Araç–varlık yakınsaması | Mesafe < 80 px (çarpma riski) | 0.90 |

**Quantum-Inspired risk** (`DecisionCore`): klasik skorlar (hareket / ses / yakınsama) açısal
kodlanır, olasılıksal ölçüm sonrası ağırlıklı skor (`0.35 · 0.35 · 0.15 · 0.15`) →
`quantum_inspired_risk_score`. Gerçek kuantum donanımı kullanılmaz; olasılıksal bir füzyon motorudur.

---

## Veri Seti

Test videoları açık kaynaktan alınmıştır. Videolar depoya dahil edilmez.

- **Kaynak:** [dataset adı ve linki eklenecek — örn. UCF-Crime / Hugging Face]
- **Lisans:** [dataset lisansı eklenecek]

---

## Dokümanlar

| Dosya | İçerik |
|-------|--------|
| [CALISTIRMA.md](CALISTIRMA.md) | Platforma özel adım adım kurulum ve çalıştırma |
| [ozet.md](ozet.md) | Detaylı proje özeti ve teknik notlar |
| [vllm-core/README.md](vllm-core/README.md) | vllm-core betikleri kullanma kılavuzu |

---

<div align="center">

**MUDU-VLLM** · Mudanya Üniversitesi · TEKNOFEST 2026
Lisans: [Apache License 2.0](LICENSE)

</div>
