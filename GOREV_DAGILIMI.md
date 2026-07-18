# Görev Dağılımı ve Modül Sahipliği

MUDU-VLLM · TEKNOFEST 2026 · Senaryo 3

Bu belge, modüllerin sorumlularını ve açık işleri takip eder. Bir iş tamamlandığında ilgili satır güncellenir.

---

## Modül Sahipliği

| Modül | Sorumlu | Kapsam |
|-------|---------|--------|
| `yolo/` | Berke | Nesne tespiti, ByteTrack takip, hareket anomalileri, araç-hayvan yakınsaması |
| `vllm-core/` | Elif (kaptan) | Prompt mühendisliği, risk mantığı, DecisionCore, LLM servisleme |
| `api-service/` | Abdulhamit | FastAPI servis katmanı, tam pipeline entegrasyonu, Docker |
| `audio/` | Mehmet Emre | Whisper ses analizi modülü |
| `web-ui/` | *(atanacak)* | Video yükleme arayüzü, sonuç görselleştirme |
| Ölçümleme / KPI | *(atanacak)* | Sistematik test, doğruluk ve süre metrikleri |
| Test veri seti + dokümantasyon | Elif | Test videolarının toplanması, lisans takibi, README veri seti bölümü |

---

## Modül Detayları

### `yolo/` — Berke

**Değişmez arayüz sözleşmesi.** `AnomalyDetector` çıktısındaki her kayıt şu alanları içermelidir:
`time`, `track_id`, `class`, `anomaly_type`, `description`, `frame_idx`

`api-service` şu import'u kullanır: `from yolo_pipeline import YoloPipeline` — sınıf adı ve `run()` imzası korunmalıdır.

Açık işler:
- [ ] GPU üzerinde çalıştığının doğrulanması (`torch.cuda.is_available()`)
- [ ] Eşiklerin (hareketsizlik `spread<4`, yakınsama `dist<80`, hız farkı `>18`) en az 3 videoda test edilip belgelenmesi
- [ ] Düşük çözünürlükte hayvan/insan sınıf karışması: `yolov8m` ve `yolov8x` karşılaştırması

### `vllm-core/` — Elif

Açık işler:
- [ ] Platform sürümlerinin tekilleştirilmesi (6 dosya → otomatik cihaz seçimli tek dosya)
- [ ] Kalan sabit video yollarının temizlenmesi, tümünün komut satırı argümanı ile çalışması
- [ ] Ollama → vLLM servisleme geçiş denemesi
- [ ] Prompt ve risk kurallarının sahipliği (tek elde kalır, çakışma önlenir)

### `api-service/` — Abdulhamit

**Mevcut durum:** API yalnızca YOLO + DecisionCore çağırıyor; 7B (VL) ve Whisper katmanları bağlı değil.

Açık işler:
- [ ] `analyze()` fonksiyonunun API'ye bağlanması (dosya adında nokta olduğu için `importlib.util` ile yoldan yükleme)
- [ ] `BASE_URL`'in ortam değişkeninden okunması (Docker içinde `http://llm:11434/v1`)
- [ ] `/health` uç noktasının genişletilmesi: pipeline durumu, LLM erişilebilirliği, aktif model
- [ ] Uzun işlem süresi yönetimi (timeout artırımı veya asenkron job yapısı)
- [ ] `docker-compose`'a Ollama servisinin eklenmesi

### `audio/` — Mehmet Emre

**Mevcut durum:** `transcribe_audio_cues()` fonksiyonu `vllm-core` içinde gömülü.

Açık işler:
- [ ] Fonksiyonun ayrı modüle çıkarılması
- [ ] `device="cpu"` → GPU (`cuda`) geçişi — mevcut performans darboğazı
- [ ] Anahtar kelime listesinin genişletilmesi (şu an 13 kelime)
- [ ] Sesli içerik barındıran gerçek videoda Türkçe transkripsiyon testi

### `web-ui/` — *(atanacak)*

Açık işler:
- [ ] Video yükleme → JSON sonuç görüntüleme akışı
- [ ] Risk seviyesinin görsel olarak ayrıştırılması (Düşük / Orta / Yüksek)
- [ ] Demo sunumunda kullanılacak ekranın hazırlanması

---

## Ortak ve Sahipsiz İşler

### Ölçümleme / KPI — *(atanacak, öncelikli)*
Şartname ölçülebilir performans kriterleri istemektedir. Şu an sistematik veri toplanmamaktadır.
- [ ] En az 5 videoda test, sonuçların tabloya işlenmesi
- [ ] Metrikler: tespit doğruluğu, işlem süresi, kaçırılan olay sayısı, yanlış alarm oranı

### Test Video Seti — Elif
- [ ] 4-6 farklı senaryo toplanması: düşme, kavga, araç kazası, hayvan olayı, tehditsiz kontrol videosu
- [ ] Her videonun kaynağı ve lisansının not edilmesi
- [ ] Seçilen videoların ekibe duyurulması (KPI ölçümü aynı set üzerinden yapılacak)

### Dokümantasyon — Elif
- [ ] README — veri seti indirme bağlantısı ve lisansı (şartname zorunluluğu)
- [ ] README — örnek demo videosu

### Repo Bakımı
- [ ] `__pycache__` ve `.pyc` dosyalarının temizlenmesi, `.gitignore` düzenlenmesi
- [ ] Dosya adlarındaki görünmez karakterlerin (zero-width space) temizlenmesi

### Demo Videosu — *(atanacak)*
- [ ] Sistemin çalışmasını gösteren kayıt (final aşaması için)

### Sunum — Elif
- [ ] MİMARİ, ALTYAPI, YENİLİKÇİLİK, ÖLÇÜMLEME slaytlarının V1.4'e göre güncellenmesi

---

## Kaptan Sorumlulukları (Elif)

- Haftalık GitHub güncellemesi (şartname zorunluluğu, kesintisiz olmalı)
- KYS üzerinden teslimler
- Modüller arası arayüz sözleşmesinin korunması

---

## Notlar

**İki çalışma yolu vardır, karıştırılmamalıdır:**

| Yol | İçerik | Kullanım |
|-----|--------|----------|
| Tam pipeline (`vllm-core/`) | YOLO + Whisper + 7B (VL) + DecisionCore | Ana sistem, tam doğruluk |
| API / Docker (`api-service/`) | YOLO + DecisionCore | Hafif servis (7B entegrasyonu sürüyor) |
