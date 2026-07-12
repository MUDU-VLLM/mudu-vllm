# MUDU-VLLM — Video Tabanlı Karar Destek Sistemi

TEKNOFEST 2026 Türkçe Doğal Dil İşleme Yarışması — Senaryo 3 (Video Analiz ve Karar Destek)

Etiket: `BilisimVadisi2026` · Lisans: Apache License 2.0

---

## Sürüm ve Test Durumu

Aşağıdaki tablo, her bileşenin mevcut olgunluk seviyesini gösterir. Şeffaflık için test edilmiş bileşenler ile geliştirme aşamasındaki bileşenler ayrılmıştır.

| Bileşen | Durum |
|---------|-------|
| Ubuntu / Linux — 7B (tam pipeline) | ✅ Test edildi, çalışıyor (GPU üzerinde doğrulandı) |
| Ubuntu / Linux — 2B (mini) | ✅ Test edildi, çalışıyor |
| Windows — 7B / 2B | ⚙️ Kurulum hazır, uçtan uca test sürüyor |
| macOS (Apple Silicon) — 7B / 2B | ⚙️ Kurulum hazır, test aşamasında |
| FastAPI servis katmanı | ✅ Çalışıyor — `/v1/analyze` · `/health` · `/docs` |
| Web UI (tek sayfa arayüz) | ✅ Çalışıyor — NDJSON ilerleme akışı, timeline, PDF |
| Docker entegrasyonu | ✅ Çalışıyor — tek imaj, `docker compose up` ile api + web |

Ana referans, doğrulanmış Ubuntu 7B sürümüdür. Diğer platform sürümleri aynı kod tabanının taşınabilir varyantlarıdır ve test edilmektedir.

---

## 1. Proje Tanımı ve Mimari

Sistem, güvenlik kamerası videolarını yerel ortamda analiz ederek olayları zaman damgasıyla listeler, Türkçe özet çıkarır, risk değerlendirmesi yapar ve operatöre aksiyon önerileri üretir. Tüm çıktı yapılandırılmış JSON formatındadır. Sistem tamamen çevrimdışı (offline) ve yerel çalışır; harici API veya bulut bağımlılığı yoktur.

Mimari üç katmandan oluşur:

- **Algı Katmanı (YOLO + ByteTrack):** Video kare kare işlenir. Nesneler tespit edilir, ByteTrack ile takip edilir ve hareket anomalileri (hareketsizlik, ani hız değişimi, araç-yaya yakınsaması) zaman damgasıyla üretilir.
- **Karar Füzyon Katmanı (Quantum-Inspired):** Algı katmanından gelen sinyaller, olasılıksal bir füzyon çekirdeği ile birleştirilerek hibrit bir risk skoru üretilir. (Not: gerçek kuantum donanımı değil, kuantum-esintili olasılıksal bir yaklaşımdır.)
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

### GPU Notu (NVIDIA)
NVIDIA GPU üzerinde YOLO hızlandırması için, sürücü CUDA sürümüne uygun PyTorch derlemesi gerekir. CUDA 12.x için:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Doğrulama:
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

Sürüm dosyaları:

| Dosya | Model | Servis |
|-------|-------|--------|
| `video_decision_support_ubuntu_V1.4.py` | Qwen2.5-VL-7B | Ollama |
| `video_decision_support_windows_V1.4.py` | Qwen2.5-VL-7B | Ollama |
| `video_decision_support_MAC_V1.4.py` | Qwen2.5-VL-7B | Ollama |
| `video_decision_support_mini_ubuntu_V1.4.py` | Qwen2-VL-2B | — (Ollama'sız) |
| `video_decision_support_mini_microsoft_V1.4.py` | Qwen2-VL-2B | — (Ollama'sız) |
| `video_decision_support_mini_MAC_V1.4.py` | Qwen2-VL-2B | — (Ollama'sız) |

7B sürümler tam doğruluk içindir. 2B "mini" sürümler kurulumu düşük tutmak ve sistemi hızlıca doğrulamak içindir.

---

## 3. Adım Adım Kurulum ve Çalıştırma Kılavuzu

### A) Ubuntu / Linux Kurulumu  ✅ (test edildi)

**Adım 1 — Sistem paketleri:**
```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-full ffmpeg
```

**Adım 2 — Ollama kurulumu ve model (7B için):**
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5vl:7b
```

**Adım 3 — 16K bağlam sürümü.** `Modelfile` oluşturun:
```
FROM qwen2.5vl:7b
PARAMETER num_ctx 16384
```
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

### B) Windows Kurulumu (PowerShell)  ⚙️ (kurulum hazır, test sürüyor)

**Adım 1 — Gereksinimler:**
- Python 3.10+ : https://www.python.org/downloads/ ("Add Python to PATH" seçin)
- ffmpeg : `winget install ffmpeg`
- Ollama : https://ollama.com/download (7B için)

**Adım 2 — Model (7B için):**
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

### C) macOS (Apple Silicon / M Serisi)  ⚙️ (kurulum hazır, test aşamasında)

**Adım 1 — Gereksinimler (Homebrew):**
```bash
brew install python ffmpeg
```
Ollama (7B için): https://ollama.com/download

**Adım 2 — Model (7B için):**
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

### 2B Mini Sürüm (Ollama'sız, hızlı test)  ⚙️

Ollama gerektirmez; yalnızca Python paketleri yeterlidir:
```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install torch transformers opencv-python pillow numpy
python3 video_decision_support_mini_ubuntu_V1.4.py /yol/video.mp4
```

---

## 4. Örnek Demo Video Çalıştırma

```bash
# Ubuntu
python3 video_decision_support_ubuntu_V1.4.py ornekler/demo.mp4

# Windows
python video_decision_support_windows_V1.4.py "ornekler\demo.mp4"

# macOS
python3 video_decision_support_MAC_V1.4.py ornekler/demo.mp4
```

Video yolu verilmezse betik, kendi klasöründeki `ornek.mp4` dosyasını arar.

Örnek çıktı:
```json
{
  "summary": "...",
  "events": [{"time": "MM:SS", "event": "..."}],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": ["..."],
  "quantum_inspired_risk_score": 0.0
}
```

Ollama sunucusu doğrulama:
```bash
curl http://localhost:11434
```

---

## 5. Veri Seti

Test videoları aşağıdaki açık kaynaktan alınmıştır:

- Kaynak: [buraya dataset adı ve linki eklenecek]
- Lisans: [dataset lisansı eklenecek]

Videolar depoya dahil edilmez; yukarıdaki bağlantıdan indirilebilir.

---

## 6. Servis Katmanı ve Docker

Tam pipeline (`vllm-core`) betiklerine ek olarak, sistem hızlı demo ve entegrasyon için bir HTTP servis katmanı ve Docker paketlemesiyle birlikte gelir. Bu yol, algı (YOLO + ByteTrack) ve füzyon (DecisionCore) katmanlarını çalıştırıp yapılandırılmış Türkçe JSON döndürür; VL (Qwen) ve Whisper doğrulaması asıl olarak `vllm-core` 7B betiğinde yapılır.

### 6.1 FastAPI Servis Katmanı  ✅

Uygulama: `api-service/app.py`

| Uç Nokta | Açıklama |
|----------|----------|
| `GET  /health` | Servis sağlık kontrolü |
| `GET  /v1/schema` | Girdi/çıktı sözleşmesi (JSON şema) |
| `POST /v1/analyze` | `multipart/form-data` · field=`video` → `DecisionResponse` |
| `GET  /docs` | Swagger arayüzü |

Akış: `video → Algı (YOLO+ByteTrack) → Füzyon (DecisionCore) → DecisionResponse JSON`

**Yerel çalıştırma:**
```bash
cd api-service
python app.py            # → http://127.0.0.1:8000  (Swagger: /docs)
```

**Örnek istek:**
```bash
curl -F "video=@demo.mp4" http://127.0.0.1:8000/v1/analyze
```

### 6.2 Web UI  ✅

Uygulama: `web-ui/app.py` + `web-ui/index.html`

Tek sayfa arayüz; video sürükle-bırak, analiz sırasında canlı **NDJSON** ilerleme akışı (`POST /api/analyze`), sonuç panelinde özet / risk / olaylar / aksiyonlar / quantum skor, zaman çizelgesi, PDF indirme ve `localStorage` tabanlı geçmiş.

```bash
cd web-ui
python app.py            # → http://127.0.0.1:7860
```

### 6.3 Docker  ✅

Tek imaj (`mudu-vllm:1.4`) hem API hem Web kodunu içerir. Kök `docker-compose.yml` iki servis kaldırır; imaj `api-service/Dockerfile` ile derlenir. CPU sürümü torch ve yerel `yolov8n.pt` ağırlığı imaja gömülür (build sırasında internetten YOLO indirmez).

```bash
docker compose up -d           # api:8000 + web:7860
docker compose up -d api       # yalnız API
docker compose up -d web       # yalnız Web
```

| Servis | Adres |
|--------|-------|
| API (FastAPI) | http://127.0.0.1:8000 · `/health` · `/v1/analyze` · `/docs` |
| Web UI | http://127.0.0.1:7860 |

**Not (vLLM'e geçiş):** Docker imajı geliştirme için Ollama tabanlı 7B yolunu hedeflemez; API/Web yolu YOLO + DecisionCore çalıştırır. Final/GPU ortamında tam VL doğruluğu istenirse, OpenAI uyumlu bir vLLM uç noktası ayağa kaldırılıp `vllm-core` betiklerinde yalnızca `BASE_URL` + `MODEL` değiştirilir; uygulama kodu aynı kalır.

> Docker günlük geliştirme için zorunlu değildir; `venv` ile yerel çalışmak daha hızlıdır. Ağır paketler yalnızca ilk başarılı `docker compose build` sırasında iner.

---

## Lisans

Apache License 2.0. Ayrıntılar için `LICENSE` dosyasına bakınız.
