# MUDU-VLLM — Proje Özeti

**TEKNOFEST 2026** · Türkçe Doğal Dil İşleme Yarışması · Senaryo 3 (Video Analiz ve Karar Destek)  
**Etiket:** `BilisimVadisi2026` · **Lisans:** Apache License 2.0 · **Sürüm:** V1.4  
**Çalıştırma rehberi:** `CALISTIRMA.md`

---

## 1. Ne Yapıyor?

Güvenlik kamerası videolarını **yerel / offline** ortamda analiz eden bir karar destek sistemi.

**Girdi:** video (`.mp4`, `.avi`, `.mov`, …)  
**Çıktı:** yapılandırılmış Türkçe JSON:

```json
{
  "summary": "Türkçe olay özeti",
  "events": [{"time": "MM:SS", "event": "..."}],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": ["Operatör aksiyon önerileri"],
  "quantum_inspired_risk_score": 0.0
}
```

İki çalışma yolu vardır:

| Yol | Ne yapar | Nerede |
|-----|----------|--------|
| **Tam pipeline (7B / mini)** | YOLO + Whisper + VL (Qwen) + DecisionCore | `vllm-core/` betikleri |
| **API / Web (hızlı servis)** | YOLO + DecisionCore → JSON | `api-service/` + `web-ui/` (+ Docker) |

Harici bulut API zorunlu değildir. Tam 7B yolunda geliştirmede **Ollama**; final/GPU’da OpenAI uyumlu uç nokta ile **vLLM**’e geçilebilir (`BASE_URL` + `MODEL`).

---

## 2. Mimari

### 2.1 Tam pipeline (vllm-core) — 3 + 1 katman

| Katman | Teknoloji | Görev |
|--------|-----------|--------|
| **Algı** | YOLOv8 + ByteTrack | Nesne tespiti, takip, hareket anomalileri |
| **Füzyon** | Quantum-inspired DecisionCore | Hareket / ses / yakınsama skor füzyonu |
| **Anlamlandırma** | Qwen2.5-VL-7B (veya mini 2B) | Kare + ipuçlarından Türkçe JSON |
| **Ses** | Whisper (`faster-whisper`) | “imdat”, “yardım”, “silah” vb. |

Akış: YOLO → Whisper → DecisionCore → VL kareleri → Türkçe JSON.

### 2.2 Servis mimarisi (api-service + web-ui)

```
Video upload
    → Algı (yolo/yolo_pipeline.py · YOLO + ByteTrack)
    → Füzyon (DecisionCore)
    → DecisionResponse JSON
```

- **API:** `POST /v1/analyze` · `GET /health` · `GET /v1/schema` · Swagger `/docs`  
- **Web:** tek sayfa UI · `POST /api/analyze` (NDJSON progress stream)

> Not: API/Web yolu şu an VL ve Whisper çalıştırmaz; jüri/asıl doğruluk için `vllm-core` 7B betiği kullanılır.

---

## 3. Dizin Yapısı

```
mudu-vllm-main/
├── README.md
├── LICENSE                          # Apache 2.0
├── .gitignore / .dockerignore
├── ozet.md                          # Bu dosya
├── CALISTIRMA.md                    # Kurulum + çalıştırma
├── requirements.txt                 # Yerel venv bağımlılıkları
├── docker-compose.yml               # api (8000) + web (7860)
├── yolov8n.pt                       # YOLO ağırlığı (Docker’a kopyalanır)
│
├── api-service/
│   ├── app.py                       # FastAPI · girdi→algı→füzyon→çıktı
│   ├── Dockerfile                   # Tek imaj (api + web kodu)
│   └── requirements.docker.txt
│
├── web-ui/
│   ├── index.html                   # Arayüz (timeline, PDF, geçmiş)
│   └── app.py                       # FastAPI · UI + streaming analiz
│
├── yolo/
│   ├── yolo_pipeline.py             # YOLO + ByteTrack + progress callback
│   └── requirements.txt
│
└── vllm-core/                       # Tam karar destek betikleri V1.4
    ├── README.md
    ├── video_decision_support_*_V1.4.py          # 7B (Ollama)
    └── video_decision_support_mini_*_V1.4.py     # 2B (transformers)
```

---

## 4. Web UI Özellikleri

Adres: **http://127.0.0.1:7860**

- Video sürükle-bırak / seç · canlı önizleme  
- Analiz sırasında **console** + **yüzde** (NDJSON stream)  
- Sonuç: özet, risk, olaylar, aksiyonlar, quantum skor  
- **Zaman çizelgesi** (olaylar zamana göre yatay)  
- **PDF indir** (jsPDF) · **Yazdır / PDF**  
- **Önceki özetler** — `localStorage` geçmişi (en fazla 30 kayıt)

---

## 5. API Service (FastAPI)

Adres: **http://127.0.0.1:8000**

| Endpoint | Açıklama |
|----------|----------|
| `GET /health` | Sağlık |
| `GET /v1/schema` | Girdi/çıktı sözleşmesi |
| `POST /v1/analyze` | `multipart` · field=`video` → `DecisionResponse` |
| `GET /docs` | Swagger |

Örnek:
```bash
curl.exe -F "video=@demo.mp4" http://127.0.0.1:8000/v1/analyze
```

---

## 6. Docker

Tek imaj `mudu-vllm:1.4` · CPU torch · yerel `yolov8n.pt` build’e gömülür.

```bash
docker compose up -d          # api:8000 + web:7860
docker compose up -d api      # yalnız API
```

**Not:** Docker günlük geliştirme için şart değildir; `venv` ile yerel çalışmak daha hızlıdır. Ağır paketler yalnızca ilk başarılı `docker compose build` sırasında iner.

Detay: `CALISTIRMA.md`

---

## 7. vllm-core Sürümleri

| Dosya | Model | Servis | Platform |
|-------|-------|--------|----------|
| `*_ubuntu_V1.4.py` | Qwen2.5-VL-7B | Ollama | Linux |
| `*_windows_V1.4.py` | Qwen2.5-VL-7B | Ollama | Windows |
| `*_MAC_V1.4.py` | Qwen2.5-VL-7B | Ollama | macOS |
| `*_mini_*_V1.4.py` | Qwen2-VL-2B | transformers | Win/Linux/Mac |

- **7B:** Tam doğruluk (YOLO + Whisper + VL).  
- **Mini:** Ollama’sız hızlı test; jüri çıktısı asıl 7B’den beklenir.

---

## 8. Algı — Anomaliler

`AnomalyDetector` (`yolo/yolo_pipeline.py` ve `vllm-core`):

| Tip | Mantık | Skor |
|-----|--------|------|
| Hareketsiz nesne | person/cat/dog ~1–2 sn | 0.85 |
| Ani hız değişimi | Baseline farkı > eşik | 0.75 |
| Araç–varlık yakınsaması | Mesafe < 80 px | 0.90 |

CLI: `python yolo/yolo_pipeline.py video.mp4` → `*_yolo.json`

---

## 9. Quantum-Inspired Risk (`DecisionCore`)

Klasik skorlar (hareket / ses / yakınsama) açısal kodlanır; ~200 ölçüm sonrası ağırlıklı skor:

- Ağırlıklar: 0.35 · 0.35 · 0.15 · 0.15  
- Alan: `quantum_inspired_risk_score`  

Gerçek kuantum donanımı kullanılmaz; olasılıksal füzyon motorudur.

---

## 10. VL Prompt (yalnızca vllm-core)

Kategoriler: insan tehdidi, silah, araç/iş kazası, hayvan tehdidi, hayvana zarar (5199), çevre.  
Kurallar: yalnızca görülen olay; YOLO ipucu hata yapabilir; `summary`/`actions` boş kalmaz.

Ses anahtarları: `imdat`, `yardim`, `help`, `ates`, `silah`, `yangin`, `saldiri` …

---

## 11. Gereksinimler

| | 7B (vllm-core) | Mini | API / Web / Docker |
|--|----------------|------|---------------------|
| Python | 3.10+ | 3.10+ | 3.12 (image) |
| RAM | ≥ 16 GB | Daha düşük | ~4–8 GB+ |
| Ollama | Evet | Hayır | Hayır |
| ffmpeg | Whisper için | — | — |
| Ana paketler | ultralytics, faster-whisper, … | torch, transformers | ultralytics, fastapi, …

Yerel: `requirements.txt` + `venv`  
Docker: `api-service/requirements.docker.txt`

---

## 12. Hızlı Çalıştırma

```powershell
# Yerel Web UI
.\venv\Scripts\Activate.ps1
cd web-ui
python app.py
# → http://127.0.0.1:7860

# Yerel API
cd api-service
python app.py
# → http://127.0.0.1:8000

# Docker
docker compose up -d

# Tam 7B (Windows örneği)
cd vllm-core
python video_decision_support_windows_V1.4.py "C:\...\video.mp4"
```

---

## 13. Durum

| Bileşen | Durum |
|---------|--------|
| 7B pipeline (Win/Linux/Mac) | Hazır (`vllm-core`) |
| Mini 2B pipeline | Hazır |
| YOLO + ByteTrack | Hazır (`yolo/`) |
| FastAPI API | Hazır · Docker’da çalışır |
| Web UI | Hazır · timeline, PDF, geçmiş, console % |
| Docker Compose | Hazır · api:8000 + web:7860 |
| CALISTIRMA.md | Güncel |
| Veri seti linki | `vllm-core/README` içinde placeholder |
| Kök README.md | Kısa; asıl dokümanlar `ozet.md` + `CALISTIRMA.md` + `vllm-core/README.md` |

---

## 14. Teknik Notlar

- API/Web ile `vllm-core` DecisionCore mantığı aynı fikirdedir; kod iki yerde kopyadır.  
- Docker imajında uygulama dosyaları `api_app.py` / `web_app.py` adıyla kopyalanır.  
- Web geçmişi tarayıcı `localStorage`’dadır (sunucuya yazılmaz).  
- 7B: OpenAI uyumlu `/v1/chat/completions`; Ollama ↔ vLLM geçişi `BASE_URL`+`MODEL`.  
- Windows 7B: FFmpeg PATH taraması · macOS: YOLO cihazı `mps`/CUDA/CPU.
)
