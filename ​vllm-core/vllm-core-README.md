# MUDU-VLLM — Video Tabanlı Karar Destek Sistemi

TEKNOFEST 2026 Türkçe Doğal Dil İşleme Yarışması — Senaryo 3 (Video Analiz ve Karar Destek)

Etiket: `BilisimVadisi2026` · Lisans: Apache License 2.0

---

## 1. Proje Tanımı ve Mimari

Sistem, güvenlik kamerası videolarını yerel ortamda analiz ederek olayları zaman damgasıyla listeler, Türkçe özet çıkarır, risk değerlendirmesi yapar ve operatöre aksiyon önerileri üretir. Tüm çıktı yapılandırılmış JSON formatındadır. Sistem tamamen çevrimdışı (offline) ve yerel çalışır; harici API veya bulut bağımlılığı yoktur.

Mimari üç katmandan oluşur:

- **Algı Katmanı (YOLO + ByteTrack):** Video kare kare işlenir. Nesneler tespit edilir, ByteTrack ile takip edilir ve hareket anomalileri (hareketsizlik, ani hız değişimi, araç-yaya yakınsaması) zaman damgasıyla üretilir.
- **Karar Füzyon Katmanı (Quantum-Inspired):** Algı katmanından gelen sinyaller, olasılıksal bir füzyon çekirdeği ile hızlıca birleştirilerek hibrit bir risk skoru üretilir.
- **Anlamlandırma Katmanı (Yerel LLM):** Seçilen kareler ve algı ipuçları, yerel olarak servis edilen çok-modlu dil modeline (Qwen2.5-VL-7B) gönderilir. Model olayları yorumlar ve nihai JSON çıktısını üretir.

Algı katmanından çıkan yapılandırılmış veriler, anlamlandırma katmanına ipucu olarak aktarılır. Böylece düşük seviyeli nesne tespiti ile yüksek seviyeli olay yorumu arasında köprü kurulur. Nihai risk, özet ve aksiyon kararı model tabanlı verilir.

**Model servisleme:** Geliştirme ortamında yerel model servisi için Ollama kullanılır. Kod OpenAI uyumlu bir uç noktaya (`/v1/chat/completions`) bağlanır; bu sayede final/GPU ortamında yalnızca `BASE_URL` ve model adı değiştirilerek vLLM'e taşınabilir. Servisleme katmanı değiştirilebilir, uygulama kodu aynı kalır.

---

## 2. Gereksinimler (Prerequisites)

- **Python:** 3.10 veya üzeri
- **RAM:** en az 16 GB (7B sürüm için)
- **GPU (opsiyonel, önerilir):** NVIDIA (CUDA 12.x) veya Apple Silicon (MPS). GPU yoksa CPU üzerinde çalışır (daha yavaş).
- **Yerel model servisi:** Ollama (7B sürüm için). 2B "mini" sürüm Ollama gerektirmez.
- **ffmpeg:** ses analizi (Whisper) için.

Python bağımlılıkları `requirements.txt` içinde listelenmiştir.

Sürümler:

| Dosya | Model | Servis | Kullanım |
|-------|-------|--------|----------|
| `video_decision_support_ubuntu_V1.4.py` | Qwen2.5-VL-7B | Ollama | Linux/Ubuntu, ana sürüm |
| `video_decision_support_windows_V1.4.py` | Qwen2.5-VL-7B | Ollama | Windows |
| `video_decision_support_MAC_V1.4.py` | Qwen2.5-VL-7B | Ollama | macOS (Apple Silicon) |
| `video_decision_support_mini_ubuntu_V1.4.py` | Qwen2-VL-2B | — (Ollama'sız) | Düşük donanım / hızlı test |
| `video_decision_support_mini_microsoft_V1.4.py` | Qwen2-VL-2B | — (Ollama'sız) | Windows, hızlı test |
| `video_decision_support_mini_MAC_V1.4.py` | Qwen2-VL-2B | — (Ollama'sız) | macOS, hızlı test |

7B sürümler tam doğruluk içindir. 2B "mini" sürümler yalnızca kurulumu düşük tutmak ve sistemi hızlıca doğrulamak içindir.

---

## 3. Adım Adım Kurulum ve Çalıştırma Kılavuzu

### A) Ubuntu / Linux Kurulumu

**Adım 1 — Sistem paketleri:**
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-full ffmpeg
```

**Adım 2 — Ollama kurulumu ve model (yalnızca 7B sürüm için):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5vl:7b
```

**Adım 3 — 16K bağlam sürümü oluşturma:**
`Modelfile` adında bir dosya oluşturun:
```
FROM qwen2.5vl:7b
PARAMETER num_ctx 16384
```
Ardından:
```bash
ollama create qwen2.5vl-16k -f Modelfile
```

**Adım 4 — Sanal ortam ve bağımlılıklar:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Adım 5 — Çalıştırma:**
```bash
python3 video_decision_support_ubuntu_V1.4.py /yol/video.mp4
```

### B) Windows Kurulumu (PowerShell)

**Adım 1 — Gereksinimler:**
- Python 3.10+ : https://www.python.org/downloads/ (kurulumda "Add Python to PATH" seçin)
- ffmpeg : `winget install ffmpeg`
- Ollama : https://ollama.com/download (yalnızca 7B sürüm için)

**Adım 2 — Model (yalnızca 7B sürüm için):**
```powershell
ollama pull qwen2.5vl:7b
```
`Modelfile` (uzantısız) oluşturun:
```
FROM qwen2.5vl:7b
PARAMETER num_ctx 16384
```
```powershell
ollama create qwen2.5vl-16k -f Modelfile
```

**Adım 3 — Sanal ortam ve bağımlılıklar:**
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Adım 4 — Çalıştırma:**
```powershell
python video_decision_support_windows_V1.4.py "C:\Users\Ad\Downloads\video.mp4"
```

### C) macOS (Apple Silicon / M Serisi) Kurulumu

**Adım 1 — Gereksinimler (Homebrew ile):**
```bash
brew install python ffmpeg
```
Ollama (yalnızca 7B sürüm için): https://ollama.com/download

**Adım 2 — Model (yalnızca 7B sürüm için):**
```bash
ollama pull qwen2.5vl:7b
```
`Modelfile` oluşturun:
```
FROM qwen2.5vl:7b
PARAMETER num_ctx 16384
```
```bash
ollama create qwen2.5vl-16k -f Modelfile
```

**Adım 3 — Sanal ortam ve bağımlılıklar:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Adım 4 — Çalıştırma:**
```bash
python3 video_decision_support_MAC_V1.4.py /Users/ad/Downloads/video.mp4
```

macOS sürümü, Apple Silicon üzerinde YOLO için MPS (Metal GPU) hızlandırmasını otomatik seçer. GPU yoksa CPU'ya düşer.

---

## 4. Örnek Demo Video Çalıştırma

Örnek test videosu ile hızlı doğrulama:

```bash
# Ubuntu
python3 video_decision_support_ubuntu_V1.4.py ornekler/demo.mp4

# Windows
python video_decision_support_windows_V1.4.py "ornekler\demo.mp4"

# macOS
python3 video_decision_support_MAC_V1.4.py ornekler/demo.mp4
```

Video yolu verilmezse betik, kendi klasöründeki `ornek.mp4` dosyasını arar.

Çıktı, terminale JSON formatında yazılır:
```json
{
  "summary": "...",
  "events": [{"time": "MM:SS", "event": "..."}],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": ["..."],
  "quantum_inspired_risk_score": 0.0
}
```

Ollama sunucusunun çalıştığı, şu komutla doğrulanabilir:
```bash
curl http://localhost:11434
```

---

## 5. Veri Seti

Test videoları aşağıdaki açık kaynaktan alınmıştır:

- Kaynak: [buraya dataset adı ve linki — örn. UCF-Crime / Hugging Face linki]
- Lisans: [dataset lisansı]

Videolar depoya dahil edilmez; yukarıdaki bağlantıdan indirilebilir.

---

## 6. Servis Katmanı ve Docker

Tam pipeline (`vllm-core`) betiklerine ek olarak, sistem hızlı demo ve entegrasyon için bir HTTP servis katmanı ve Docker paketlemesiyle birlikte gelir. Bu dosyalar depo kökünde ve `api-service/` · `web-ui/` klasörlerindedir. Servis yolu, algı (YOLO + ByteTrack) ve füzyon (DecisionCore) katmanlarını çalıştırıp yapılandırılmış Türkçe JSON döndürür; VL (Qwen) ve Whisper doğrulaması asıl olarak bu klasördeki (`vllm-core`) 7B betiğinde yapılır.

### 6.1 FastAPI Servis Katmanı

Uygulama: `api-service/app.py`

| Uç Nokta | Açıklama |
|----------|----------|
| `GET  /health` | Servis sağlık kontrolü |
| `GET  /v1/schema` | Girdi/çıktı sözleşmesi (JSON şema) |
| `POST /v1/analyze` | `multipart/form-data` · field=`video` → `DecisionResponse` |
| `GET  /docs` | Swagger arayüzü |

Akış: `video → Algı (YOLO+ByteTrack) → Füzyon (DecisionCore) → DecisionResponse JSON`

Yerel çalıştırma:
```bash
cd api-service
python app.py            # → http://127.0.0.1:8000  (Swagger: /docs)
```

Örnek istek:
```bash
curl -F "video=@demo.mp4" http://127.0.0.1:8000/v1/analyze
```

### 6.2 Web UI

Uygulama: `web-ui/app.py` + `web-ui/index.html`

Tek sayfa arayüz; video sürükle-bırak, analiz sırasında canlı **NDJSON** ilerleme akışı (`POST /api/analyze`), sonuç panelinde özet / risk / olaylar / aksiyonlar / quantum skor, zaman çizelgesi, PDF indirme ve `localStorage` tabanlı geçmiş.

```bash
cd web-ui
python app.py            # → http://127.0.0.1:7860
```

### 6.3 Docker

Tek imaj (`mudu-vllm:1.4`) hem API hem Web kodunu içerir. Depo kökündeki `docker-compose.yml` iki servisi kaldırır; imaj `api-service/Dockerfile` ile derlenir. CPU sürümü torch ve yerel `yolov8n.pt` ağırlığı imaja gömülür (build sırasında internetten YOLO indirmez).

```bash
docker compose up -d           # api:8000 + web:7860
docker compose up -d api       # yalnız API
docker compose up -d web       # yalnız Web
```

| Servis | Adres |
|--------|-------|
| API (FastAPI) | http://127.0.0.1:8000 · `/health` · `/v1/analyze` · `/docs` |
| Web UI | http://127.0.0.1:7860 |

**Not (vLLM'e geçiş):** API/Web yolu YOLO + DecisionCore çalıştırır. Final/GPU ortamında tam VL doğruluğu istenirse, OpenAI uyumlu bir vLLM uç noktası ayağa kaldırılıp `vllm-core` betiklerinde yalnızca `BASE_URL` + `MODEL` değiştirilir; uygulama kodu aynı kalır.

> Docker günlük geliştirme için zorunlu değildir; `venv` ile yerel çalışmak daha hızlıdır. Ağır paketler yalnızca ilk başarılı `docker compose build` sırasında iner.

---

## Lisans

Apache License 2.0. Ayrıntılar için `LICENSE` dosyasına bakınız.
