# Nasıl Çalıştırılır

Bu rehber **Windows (PowerShell)** odaklıdır. Linux/macOS için platforma özel adımlar
`vllm-core/README.md` içindedir. Python paketleri `venv` içine yüklenir.

---

## 0) Her seferinde: sanal ortamı aç

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
```

Execution policy hatası alırsan:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

---

## Seçenek A — Mini sürüm (önerilen ilk deneme)

**Ollama gerekmez.** İlk çalıştırmada Hugging Face'ten ~2B model iner (internet gerekir).
GPU varsa otomatik CUDA, yoksa CPU kullanılır.

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main\vllm-core
python video_decision_support_mini_microsoft_V1.5.py "C:\yol\video.mp4"
```

Video yolu vermezsen betik aynı klasörde `ornek.mp4` arar.

---

## Seçenek B — Tam 7B sürüm (asıl çıktı, Ollama)

### 1) Ollama kur

1. https://ollama.com/download indirip kur
2. PowerShell:

```powershell
ollama pull qwen2.5vl:7b
```

`vllm-core\Modelfile` oluştur (uzantısız):

```
FROM qwen2.5vl:7b
PARAMETER num_ctx 16384
```

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main\vllm-core
ollama create qwen2.5vl-16k -f Modelfile
curl http://localhost:11434        # ayakta mı?
```

### 2) ffmpeg kur (ses analizi için; opsiyonel)

```powershell
winget install ffmpeg
```

Kurulumdan sonra yeni terminal aç. ffmpeg/faster-whisper yoksa betik sesi atlar, YOLO+VL yine çalışır.

### 3) Çalıştır

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd vllm-core
python video_decision_support_windows_V1.5.py "C:\Users\emrem\Videos\demo.mp4"
```

Video yolu **komut satırı argümanı** olarak verilir. Argüman vermezsen betik koddaki
varsayılan yolu, o da yoksa aynı klasördeki `ornek.mp4`'ü dener.

---

## Seçenek C — vLLM ile 7B (final / GPU ortamı)

Şartnamenin tercih ettiği yüksek performanslı servisleme. **vLLM pratikte Linux + NVIDIA GPU
ister**; ayrıntılı adımlar `vllm-core/README.md` ve `vllm-core/run_vllm_ubuntu.sh` içindedir. Özet:

```bash
# 1) vLLM sunucusu (ayri terminal)
vllm serve Qwen/Qwen2.5-VL-7B-Instruct --host 0.0.0.0 --port 8000 \
  --dtype bfloat16 --max-model-len 16384 --limit-mm-per-prompt image=8

# 2) Betigi vLLM'e yonlendir (env-var; kod degismez)
MUDU_BASE_URL=http://localhost:8000/v1 MUDU_MODEL=Qwen/Qwen2.5-VL-7B-Instruct \
  python3 video_decision_support_ubuntu_V1.5.py /yol/video.mp4
```

Env vermezsen betikler varsayılan olarak **Ollama**'ya bağlanır. Aynı `MUDU_BASE_URL`/`MUDU_MODEL`
mantığı windows ve MAC 7B betiklerinde de vardır.

---

## Sadece YOLO (VL / Ollama yok)

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd yolo
python yolo_pipeline.py "C:\yol\video.mp4"       # -> videoadi_yolo.json
```

## Sadece Ses (Whisper modülü)

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd audio
python audio_cues.py "C:\yol\video.mp4"          # -> sesli tehdit/yardim ifadeleri (JSON)
```

GPU varsa otomatik CUDA. ffmpeg gerekir; yoksa boş liste döner.

---

## Beklenen çıktı (özet)

Terminalde JSON:

```json
{
  "summary": "...",
  "events": [{"time": "00:15", "event": "..."}],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": ["..."],
  "quantum_inspired_risk_score": 0.12
}
```

---

## Docker (ŞART DEĞİL)

Docker, projeyi başka makinede / jüri ortamında tek komutla çalıştırmak içindir. Günlük
kullanımın için gerekmez; `venv` zaten kurulu. Ağır paketler **yalnızca ilk başarılı build'de** iner.

```powershell
# İlk (ve tek ağır) build — 1 kez, sabırlı ol
docker compose build api

# API + Web UI birlikte
docker compose up

# tek tek:
docker compose up api          # sadece API  -> http://127.0.0.1:8000
docker compose up web          # sadece Web  -> http://127.0.0.1:7860
curl.exe -F "video=@C:\yol\demo.mp4" http://127.0.0.1:8000/v1/analyze
```

Build yarıda kesilirse cache bozulur ve tekrar indirir → build'i bölme, bitene kadar bekle.

| Servis | Adres |
|--------|-------|
| API (FastAPI) | http://127.0.0.1:8000 · `/health` · `/v1/analyze` · `/docs` |
| Web UI | http://127.0.0.1:7860 |

**Yerel (Docker'sız):**
```powershell
.\venv\Scripts\Activate.ps1
cd api-service ; python app.py      # API  -> :8000
cd web-ui      ; python app.py      # Web  -> :7860
```

Web arayüzü: video bırak → **Analiz et** (YOLO + quantum risk → Türkçe JSON paneli).
API/Web yolu: Girdi → Algı (YOLO) → Füzyon (DecisionCore) → JSON.

---

## Test videosu

Depoda video yoktur. **UCF-Crime** setinden indir:
https://www.crcv.ucf.edu/projects/real-world/ (ör. `Arrest001_x264.mp4`).

---

## Durum (bu makine)

| Bileşen | Durum |
|---------|-------|
| `venv` + Python paketleri | Kurulu |
| Ollama | 7B için gerekli (kur) |
| ffmpeg | Ses için opsiyonel |
| vLLM | Final/GPU (Linux) için opsiyonel |
| Test videosu | Depoda yok → UCF-Crime'dan indir |