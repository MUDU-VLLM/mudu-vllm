# Nasıl Çalıştırılır (Windows)

Python paketleri `venv` içine yüklendi. Aşağıdaki adımları PowerShell’de çalıştırın.

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

**Ollama gerekmez.** İlk çalıştırmada Hugging Face’ten ~2B model iner (internet gerekir).

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main\vllm-core
python video_decision_support_mini_microsoft_V1.4.py "C:\yol\video.mp4"
```

Video yolu vermezsen betik aynı klasörde `ornek.mp4` arar.

---

## Seçenek B — Tam 7B sürüm (asıl çıktı)

### 1) Ollama kur (sistemde yoktu)

1. https://ollama.com/download indirip kur  
2. PowerShell:

```powershell
ollama pull qwen2.5vl:7b
```

`C:\Users\emrem\Desktop\mudu-vllm-main\vllm-core\Modelfile` oluştur (uzantısız):

```
FROM qwen2.5vl:7b
PARAMETER num_ctx 16384
```

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main\vllm-core
ollama create qwen2.5vl-16k -f Modelfile
```

Ollama’nın ayakta olduğunu kontrol et:

```powershell
curl http://localhost:11434
```

### 2) ffmpeg kur (ses analizi için; yoktu)

```powershell
winget install ffmpeg
```

Kurulumdan sonra yeni bir terminal aç. Ses olmadan da YOLO+VL çalışır; ffmpeg yoksa betik sesi atlar.

### 3) Çalıştır

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd vllm-core
python video_decision_support_windows_V1.4.py
```

Betik video yolu ister; örnek:

```
C:\Users\emrem\Videos\demo.mp4
```

veya kodda `videos = [...]` listesine kendi yolunu yaz.

---

## Sadece YOLO (VL / Ollama yok)

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd yolo
python yolo_pipeline.py "C:\yol\video.mp4"
```

Çıktı: `videoadi_yolo.json`

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

## Docker ne işe yarar? (ŞART DEĞİL)

Docker, projeyi **başka makinede / jüri ortamında** tek komutla çalıştırmak içindir:
Python, pip, YOLO bağımlılıklarını senin bilgisayarına tek tek kurmadan kutuda taşır.

**Senin günlük kullanımın için gerekmez.** Zaten `venv` kurulu; en hızlı yol:

```powershell
.\venv\Scripts\Activate.ps1
cd web-ui
python app.py
```

Docker’da paketler **yalnızca ilk başarılı build’de** iner. Sonraki seferlerde kod değişse bile torch yeniden inmemeli (Dockerfile buna göre düzeltildi).

```powershell
# İlk (ve tek ağır) build — sabırlı ol, 1 kez
docker compose build api

# Sonra ayağa kaldır (indirme yok, imaj hazırsa)
docker compose up
```

Build yarıda kesilirse cache bozulur ve tekrar indirir → build’i bölme, bitene kadar bekle.

---

## API Service (FastAPI + Docker)

**Docker (API + Web UI birlikte):**
```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
docker compose up --build
```

| Servis | Adres |
|--------|--------|
| API FastAPI | http://127.0.0.1:8000 · `/health` · `/v1/analyze` · `/docs` |
| Web UI FastAPI | http://127.0.0.1:7860 |

```powershell
docker compose up --build api   # sadece API
docker compose up --build web   # sadece Web
curl.exe -F "video=@C:\yol\demo.mp4" http://127.0.0.1:8000/v1/analyze
```

**Yerel (Docker’sız):**
```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd api-service
python app.py
```

Girdi → Algı (YOLO) → Füzyon (DecisionCore) → JSON çıktı

---

## Web UI

```powershell
cd C:\Users\emrem\Desktop\mudu-vllm-main
.\venv\Scripts\Activate.ps1
cd web-ui
python app.py
```

Tarayıcı: http://127.0.0.1:7860  
Video bırak → **Analiz et** (YOLO + quantum risk → Türkçe JSON paneli).

---

## Durum (bu makine)

| Bileşen | Durum |
|---------|--------|
| `venv` + Python paketleri | Kurulu |
| Ollama | Yok → 7B için kur |
| ffmpeg | Yok → ses için kur (opsiyonel) |
| Test videosu | Depoda yok → kendi `.mp4` yolunu ver |
