<div align="center">MUDU-VLLM

Yapay Zekâ Destekli Video Karar Destek Sistemi

AI-Powered Video Decision Support System

Güvenlik kamerası videolarını tamamen yerel ve çevrimdışı analiz eden,
görüntü, hareket ve ses verilerini birleştirerek operatöre yapılandırılmış
Türkçe karar çıktısı sunan multimodal karar destek sistemi.

<br>"Python" (https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
"YOLOv8" (https://img.shields.io/badge/YOLO-v8-00FFFF)
"FastAPI" (https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
"Docker" (https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
"License" (https://img.shields.io/badge/License-Apache%202.0-D22128)
"Version" (https://img.shields.io/badge/Version-V1.5-blueviolet)

<br>"TEKNOFEST 2026" · "Türkçe Doğal Dil İşleme" · "Senaryo 3 — Video Analiz ve Karar Destek"

Etiket: "BilisimVadisi2026"
Lisans: "Apache License 2.0" (LICENSE)
Sürüm: "V1.5"

</div>---

İçindekiler

- "Proje Hakkında" (#proje-hakkında)
- "Ne Yapar?" (#ne-yapar)
- "Temel Özellikler" (#temel-özellikler)
- "Teknoloji Yığını" (#teknoloji-yığını)
- "Sürüm ve Test Durumu" (#sürüm-ve-test-durumı)
- "Sistem Mimarisi" (#sistem-mimarisi)
- "Hızlı Başlangıç" (#hızlı-başlangıç)
- "Dizin Yapısı" (#dizin-yapısı)
- "Servis Katmanı" (#servis-katmanı)
- "Algı ve Füzyon Detayları" (#algı-ve-füzyon-detayları)
- "Veri Seti" (#veri-seti)
- "Sınırlamalar" (#sınırlamalar)
- "Yol Haritası" (#yol-haritası)
- "Ekip" (#ekip)
- "Dokümanlar" (#dokümanlar)
- "Lisans" (#lisans)

---

Proje Hakkında

MUDU-VLLM, güvenlik kamerası videolarını analiz ederek olayları tespit eden, risk seviyesini hesaplayan ve operatöre Türkçe karar desteği sunan multimodal bir yapay zekâ sistemidir.

Sistem; görüntü analizi, nesne takibi, hareket anomalisi tespiti, ses analizi, olasılıksal risk füzyonu ve görsel dil modeli tabanlı olay yorumlama bileşenlerini tek bir analiz akışında birleştirir.

MUDU-VLLM dört temel katmandan oluşur:

1. YOLOv8 + ByteTrack ile nesne tespiti ve takibi
2. Whisper ile opsiyonel ses analizi
3. Quantum-Inspired DecisionCore ile risk füzyonu
4. Qwen2.5-VL ile Türkçe olay yorumlama ve karar üretimi

Sistem, videoları harici bir bulut servisine göndermeden yerel ortamda çalışacak şekilde tasarlanmıştır. Bu sayede veri gizliliğinin önemli olduğu güvenlik ve gözetim senaryolarında kullanılabilir.

«[!IMPORTANT]
MUDU-VLLM bir karar destek sistemidir. Üretilen sonuçlar insan operatörün değerlendirmesinin yerine geçmez ve tek başına otomatik müdahale kararı için kullanılmamalıdır.»

---

Ne Yapar?

Girdi

Sistem aşağıdaki video formatlarını destekler:

- ".mp4"
- ".avi"
- ".mov"
- ".mkv"

Çıktı

Analiz sonucunda yapılandırılmış Türkçe bir JSON çıktısı üretilir.

{
  "summary": "Türkçe olay özeti",
  "events": [
    {
      "time": "MM:SS",
      "event": "Tespit edilen olay"
    }
  ],
  "risk": "Dusuk | Orta | Yuksek",
  "actions": [
    "Operatör için önerilen aksiyon"
  ],
  "quantum_inspired_risk_score": 0.0
}

Çıktı aşağıdaki bilgileri içerir:

- Videonun Türkçe olay özeti
- Zaman damgalı olay listesi
- Genel risk seviyesi
- Operatör için aksiyon önerileri
- Quantum-inspired risk skoru

Tüm analiz süreci desteklenen modeller yerel olarak kurulduğunda çevrimdışı çalışabilir.

---

Temel Özellikler

- Yerel ve çevrimdışı video analizi
- YOLOv8 tabanlı nesne tespiti
- ByteTrack tabanlı çoklu nesne takibi
- Hareket anomalisi tespiti
- Ani hız değişimi analizi
- Hareketsiz nesne tespiti
- Araç–insan ve araç–varlık yakınsama analizi
- Whisper tabanlı opsiyonel ses analizi
- Türkçe ve İngilizce kritik anahtar ifade tespiti
- Quantum-inspired risk füzyonu
- Qwen2.5-VL ile semantik olay yorumlama
- Yapılandırılmış Türkçe JSON çıktısı
- FastAPI servis katmanı
- Web tabanlı kullanıcı arayüzü
- NDJSON tabanlı canlı ilerleme akışı
- Zaman çizelgesi görünümü
- PDF rapor çıktısı
- Docker Compose desteği
- Linux, Windows ve macOS için platform sürümleri

---

Teknoloji Yığını

Katman| Teknoloji
Programlama dili| Python
Görüntü işleme| OpenCV
Nesne tespiti| YOLOv8
Nesne takibi| ByteTrack
Görsel dil modeli| Qwen2.5-VL
Ses analizi| faster-whisper
Yerel model çalıştırma| Ollama
GPU model servisleme| vLLM
API| FastAPI
Web arayüzü| HTML, CSS, JavaScript, Python
Konteynerleştirme| Docker, Docker Compose
Veri formatı| JSON, NDJSON
Risk füzyonu| Quantum-Inspired DecisionCore

---

Sürüm ve Test Durumu

Her bileşenin mevcut olgunluk seviyesi aşağıda ayrı olarak belirtilmiştir.

Bileşen| Durum
Ubuntu / Linux — 7B tam pipeline| ✅ Test edildi, GPU üzerinde doğrulandı
Ubuntu / Linux — 2B mini pipeline| ⚙️ Kurulum hazır, kısmi test tamamlandı
Windows — 7B / 2B| ⚙️ Kurulum hazır, uçtan uca test devam ediyor
macOS Apple Silicon — 7B / 2B| ⚙️ Kurulum hazır, test aşamasında
FastAPI servis katmanı| ✅ Çalışıyor
Web UI| ✅ Çalışıyor
Docker entegrasyonu| ✅ Çalışıyor
Whisper ses analizi| ⚙️ Opsiyonel modül olarak kullanılabilir
vLLM servisleme| 🧪 Final GPU ortamı için entegrasyon aşamasında

FastAPI servisinde mevcut uç noktalar:

GET  /health
GET  /v1/schema
POST /v1/analyze
GET  /docs

Ana referans sürüm Ubuntu/Linux 7B pipeline sürümüdür.

Windows ve macOS sürümleri aynı sistem mimarisini izleyen taşınabilir platform varyantlarıdır.

---

Sistem Mimarisi

MUDU-VLLM; algı, ses, risk füzyonu ve semantik anlamlandırma katmanlarından oluşur.

┌─────────────────────────────────────────────────────────────────────┐
│                        GÜVENLİK KAMERASI VİDEOSU                    │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. ALGI KATMANI                                                   │
│                                                                     │
│  YOLOv8 + ByteTrack                                                │
│  ├─ Nesne tespiti                                                  │
│  ├─ Nesne sınıflandırma                                            │
│  ├─ Çoklu nesne takibi                                             │
│  ├─ Hareket analizi                                                │
│  └─ Yakınsama ve anomali tespiti                                   │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
┌───────────────────────────────┐  ┌──────────────────────────────────┐
│  2. SES KATMANI              │  │  Görsel ve hareket ipuçları     │
│                               │  │                                  │
│  faster-whisper               │  │  Nesne sınıfları                │
│  ├─ Konuşma çözümleme         │  │  Takip kimlikleri               │
│  ├─ Zaman damgaları           │  │  Hareket anomalileri            │
│  └─ Anahtar ifade tespiti     │  │  Yakınsama olayları             │
└───────────────┬───────────────┘  └────────────────┬─────────────────┘
                │                                   │
                └─────────────────┬─────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. FÜZYON KATMANI                                                │
│                                                                     │
│  Quantum-Inspired DecisionCore                                     │
│  ├─ Hareket riski                                                  │
│  ├─ Ses riski                                                      │
│  ├─ Yakınsama riski                                                │
│  ├─ Semantik risk                                                  │
│  └─ Olasılıksal birleşik risk skoru                                │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. ANLAMLANDIRMA KATMANI                                         │
│                                                                     │
│  Qwen2.5-VL                                                        │
│  ├─ Seçilmiş video kareleri                                       │
│  ├─ YOLO ve ByteTrack ipuçları                                    │
│  ├─ Ses analizi sonuçları                                         │
│  ├─ DecisionCore risk skoru                                       │
│  └─ Türkçe olay yorumu ve aksiyon önerisi                          │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│               YAPILANDIRILMIŞ TÜRKÇE KARAR JSON'U                  │
│                                                                     │
│       Özet · Olaylar · Risk · Aksiyonlar · Quantum Skoru           │
└─────────────────────────────────────────────────────────────────────┘

Algı katmanının yapılandırılmış çıktısı, görsel dil modeline ipucu olarak verilir. Böylece düşük seviyeli nesne tespiti ile yüksek seviyeli olay yorumlama arasında bir bağlantı kurulur.

Model Servisleme

Geliştirme ortamında görsel dil modeli Ollama üzerinden çalıştırılır.

Ollama, OpenAI uyumlu aşağıdaki servis yolunu sağlar:

/v1/chat/completions

Final veya yüksek performanslı GPU ortamında servis ayarları değiştirilerek vLLM kullanılabilir.

BASE_URL = "http://localhost:8000/v1"
MODEL = "qwen2.5-vl"

Model servis sağlayıcısı değiştirilse bile ana uygulama ve karar akışının aynı kalması hedeflenmiştir.

---

Hızlı Başlangıç

Ayrıntılı platform kurulumları için:

- "CALISTIRMA.md" (CALISTIRMA.md)
- "vllm-core/README.md" (vllm-core/README.md)

1. Depoyu Klonlayın

git clone https://github.com/MUDU-VLLM/mudu-vllm.git
cd mudu-vllm

2. Sanal Ortam Oluşturun

Linux / macOS

python3 -m venv .venv
source .venv/bin/activate

Windows PowerShell

python -m venv .venv
.venv\Scripts\Activate.ps1

3. Bağımlılıkları Kurun

python -m pip install --upgrade pip
pip install -r requirements.txt

---

Seçenek A — Mini 2B Sürümü

Mini sürüm, Ollama servisine ihtiyaç duymadan Transformers üzerinden çalıştırılabilir.

Ubuntu / Linux

pip install torch transformers opencv-python pillow numpy

python3 vllm-core/video_decision_support_mini_ubuntu_V1.5.py video.mp4

Windows

python vllm-core/video_decision_support_mini_microsoft_V1.5.py video.mp4

macOS

python3 vllm-core/video_decision_support_mini_MAC_V1.5.py video.mp4

---

Seçenek B — Tam 7B Sürümü

Tam sürüm, Qwen2.5-VL-7B modelini kullanır ve daha yüksek doğruluk hedefler.

Öncelikle Ollama modelini indirin:

ollama pull qwen2.5vl:7b

Proje kök dizininde bir "Modelfile" oluşturun:

FROM qwen2.5vl:7b

PARAMETER num_ctx 16384

Model varyantını oluşturun:

ollama create qwen2.5vl-16k -f Modelfile

Modelin kullanılabilir olduğunu doğrulayın:

ollama list

Ubuntu / Linux

python3 vllm-core/video_decision_support_ubuntu_V1.5.py video.mp4

Windows

python vllm-core/video_decision_support_windows_V1.5.py video.mp4

macOS

python3 vllm-core/video_decision_support_MAC_V1.5.py video.mp4

«[!NOTE]
Kullanılan dosya adları yeni sürümlerde değişebilir. Güncel komutlar için "vllm-core/" dizinini ve "vllm-core/README.md" (vllm-core/README.md) dosyasını kontrol edin.»

---

Seçenek C — YOLO Pipeline

Yalnızca nesne tespiti, takip ve anomali analizini çalıştırmak için:

python yolo/yolo_pipeline.py video.mp4

Örnek çıktı dosyası:

video_yolo.json

Python içerisinden örnek kullanım:

from yolo.yolo_pipeline import YoloPipeline


pipeline = YoloPipeline()
result = pipeline.process("video.mp4")

print(result)

---

Seçenek D — FastAPI

API servisini başlatmak için:

cd api-service
python app.py

API adresi:

http://127.0.0.1:8000

Swagger arayüzü:

http://127.0.0.1:8000/docs

Sağlık kontrolü:

curl http://127.0.0.1:8000/health

Video analizi:

curl -X POST \
  -F "video=@demo.mp4" \
  http://127.0.0.1:8000/v1/analyze

---

Seçenek E — Web Arayüzü

Web arayüzünü başlatmak için:

cd web-ui
python app.py

Arayüz adresi:

http://127.0.0.1:7860

---

Seçenek F — Docker Compose

API ve Web arayüzünü birlikte başlatmak için:

docker compose up -d

Yalnızca API servisini başlatmak için:

docker compose up -d api

Yalnızca Web servisini başlatmak için:

docker compose up -d web

Çalışan servisleri görüntülemek için:

docker compose ps

Logları takip etmek için:

docker compose logs -f

Servisleri durdurmak için:

docker compose down

Varsayılan servis adresleri:

Servis| Adres
FastAPI| "http://127.0.0.1:8000"
Swagger| "http://127.0.0.1:8000/docs"
Web UI| "http://127.0.0.1:7860"

«[!NOTE]
Video yolu komut satırında verilmezse bazı pipeline betikleri kendi klasörlerinde "ornek.mp4" dosyasını arayabilir. Tekrarlanabilir analizler için video yolunun açık şekilde verilmesi önerilir.»

---

Dizin Yapısı

mudu-vllm/
├── README.md
│   └── Projenin ana tanıtım ve kullanım dokümanı
│
├── CALISTIRMA.md
│   └── Platforma özel kurulum ve çalıştırma rehberi
│
├── ozet.md
│   └── Detaylı proje özeti ve teknik notlar
│
├── LICENSE
│   └── Apache License 2.0
│
├── requirements.txt
│   └── Python bağımlılıkları
│
├── docker-compose.yml
│   └── API ve Web servislerinin Docker Compose yapılandırması
│
├── yolov8n.pt
│   └── YOLO model ağırlığı
│
├── vllm-core/
│   ├── README.md
│   ├── video_decision_support_ubuntu_V1.5.py
│   ├── video_decision_support_windows_V1.5.py
│   ├── video_decision_support_MAC_V1.5.py
│   ├── video_decision_support_mini_ubuntu_V1.5.py
│   ├── video_decision_support_mini_microsoft_V1.5.py
│   └── video_decision_support_mini_MAC_V1.5.py
│
├── yolo/
│   └── yolo_pipeline.py
│
├── audio/
│   └── audio_cues.py
│
├── api-service/
│   ├── app.py
│   └── Dockerfile
│
└── web-ui/
    ├── app.py
    └── index.html

Modül Sorumlulukları

Dizin| Sorumluluk
"vllm-core/"| Ana video karar destek pipeline'ları, prompt sistemi, DecisionCore ve model servisleme
"yolo/"| Nesne tespiti, ByteTrack takibi, hareket ve yakınsama anomalileri
"audio/"| Whisper tabanlı ses çözümleme ve sesli risk ipuçları
"api-service/"| FastAPI tabanlı HTTP servis katmanı
"web-ui/"| Video yükleme, canlı analiz akışı, zaman çizelgesi ve raporlama arayüzü

---

Servis Katmanı

Tam pipeline betiklerinin yanında hızlı demo ve sistem entegrasyonu için FastAPI, Web UI ve Docker tabanlı servis katmanı bulunur.

Servis katmanının temel analiz akışı:

Video
  │
  ▼
YOLOv8 + ByteTrack
  │
  ▼
Anomali Analizi
  │
  ▼
DecisionCore
  │
  ▼
Türkçe JSON

Tam görsel dil modeli ve Whisper doğrulaması ağırlıklı olarak "vllm-core/" içerisindeki 7B pipeline tarafından gerçekleştirilir.

---

FastAPI

Ana dosya:

api-service/app.py

Metot| Uç Nokta| Açıklama
"GET"| "/health"| Servis sağlık kontrolü
"GET"| "/v1/schema"| Girdi ve çıktı JSON sözleşmesi
"POST"| "/v1/analyze"| Video yükleyerek analiz başlatma
"GET"| "/docs"| Swagger API arayüzü

Örnek istek:

curl -F "video=@demo.mp4" \
  http://127.0.0.1:8000/v1/analyze

Örnek Python istemcisi:

from pathlib import Path

import requests


API_URL = "http://127.0.0.1:8000/v1/analyze"
VIDEO_PATH = Path("demo.mp4")


def analyze_video(video_path: Path) -> dict:
    if not video_path.exists():
        raise FileNotFoundError(f"Video bulunamadı: {video_path}")

    with video_path.open("rb") as video_file:
        response = requests.post(
            API_URL,
            files={
                "video": (
                    video_path.name,
                    video_file,
                    "video/mp4",
                )
            },
            timeout=600,
        )

    response.raise_for_status()
    return response.json()


if __name__ == "__main__":
    analysis_result = analyze_video(VIDEO_PATH)
    print(analysis_result)

---

Web UI

Web arayüzü aşağıdaki özellikleri sunar:

- Sürükle-bırak video yükleme
- "POST /api/analyze" üzerinden analiz başlatma
- Canlı NDJSON ilerleme akışı
- Analiz durumu göstergesi
- Türkçe olay özeti
- Risk seviyesi
- Quantum-inspired risk skoru
- Zaman damgalı olay çizelgesi
- Operatör aksiyon önerileri
- PDF rapor çıktısı
- "localStorage" tabanlı analiz geçmişi

Ana dosyalar:

web-ui/app.py
web-ui/index.html

---

Docker

Docker yapılandırması, API ve Web servislerinin aynı proje ortamında çalıştırılmasını sağlar.

docker compose build
docker compose up -d

Konteyner durumunu görüntülemek için:

docker compose ps

API loglarını takip etmek için:

docker compose logs -f api

Web loglarını takip etmek için:

docker compose logs -f web

---

Algı ve Füzyon Detayları

YOLO ve ByteTrack Çıktısı

"yolo/yolo_pipeline.py" tarafından üretilen anomali kayıtları aşağıdaki alanları içerir:

{
  "time": "00:07",
  "track_id": 12,
  "class": "person",
  "anomaly_type": "sudden_speed_change",
  "description": "Takip edilen nesnede ani hız değişimi tespit edildi.",
  "frame_idx": 184
}

Bu yapı, YOLO modülü ile API ve karar katmanları arasındaki veri sözleşmesini oluşturur.

---

Anomali Kuralları

Anomali Türü| Mantık| Risk Skoru
Hareketsiz nesne| İnsan, kedi veya köpeğin yaklaşık 1 saniye hareketsiz kalması| "0.85"
Ani hız değişimi| Nesne hızının hareket taban çizgisinden belirgin şekilde ayrılması| "0.75"
Araç–varlık yakınsaması| Araç ile insan veya başka bir varlık arasındaki mesafenin 80 pikselin altına düşmesi| "0.90"

«[!NOTE]
Piksel tabanlı eşikler kamera açısına, çözünürlüğe ve görüntü ölçeğine göre yeniden kalibre edilmelidir. Piksel mesafesi fiziksel mesafe olarak doğrudan yorumlanmamalıdır.»

---

Quantum-Inspired DecisionCore

DecisionCore, klasik analiz katmanlarından gelen risk değerlerini olasılıksal bir füzyon yaklaşımıyla birleştirir.

Örnek giriş sinyalleri:

Hareket riski
Ses riski
Yakınsama riski
Semantik risk

Örnek ağırlıklandırma:

0.35 × hareket riski
0.35 × yakınsama riski
0.15 × ses riski
0.15 × semantik risk

Örnek çıktı:

{
  "quantum_inspired_risk_score": 0.78,
  "risk": "Yuksek"
}

«[!IMPORTANT]
Quantum-inspired katman gerçek bir kuantum bilgisayar veya kuantum donanımı kullanmaz. Bu ifade, klasik donanım üzerinde çalışan olasılıksal ve açısal kodlama yaklaşımını belirtir.»

---

Whisper Ses Analizi

Ana modül:

audio/audio_cues.py

Örnek kullanım:

from audio.audio_cues import transcribe_audio_cues


audio_result = transcribe_audio_cues("video.mp4")
print(audio_result)

Hedeflenen kritik sesli ifadeler:

imdat
yardım
yardım edin
silah
ateş
dur
kaç
help

Ses analizi opsiyoneldir. Videoda ses kanalı bulunmadığında veya Whisper modülü devre dışı bırakıldığında sistem yalnızca görsel sinyallerle çalışabilir.

---

Veri Seti

Test videolarında UCF-Crime veri setinden seçilmiş örnekler kullanılmaktadır.

UCF-Crime, gerçek dünya güvenlik kamerası videolarında anomali tespiti araştırmaları için hazırlanmış akademik bir veri setidir.

Akademik Referans

Real-world Anomaly Detection in Surveillance Videos

Waqas Sultani, Chen Chen ve Mubarak Shah
IEEE Conference on Computer Vision and Pattern Recognition — CVPR 2018

Veri Seti İçeriği

- Yaklaşık 1.900 uzun ve kesintisiz gözetim videosu
- 13 anomali kategorisi
- Normal güvenlik kamerası videoları
- Farklı kamera açıları ve görüntü kaliteleri
- Gerçek dünya güvenlik senaryoları

Örnek anomali sınıfları:

- Abuse
- Arrest
- Arson
- Assault
- Burglary
- Explosion
- Fighting
- Road Accidents
- Robbery
- Shooting
- Shoplifting
- Stealing
- Vandalism

Bağlantılar

Resmî proje sayfası:

https://www.crcv.ucf.edu/projects/real-world/

Veri seti arşivi:

https://www.crcv.ucf.edu/data1/chenchen/UCF_Crimes.zip

Videolar, dosya boyutları ve kullanım koşulları nedeniyle bu depoya dahil edilmemektedir.

Projede veri setinden seçilmiş sınırlı bir alt küme test amacıyla kullanılabilir.

Arrest001_x264.mp4
RoadAccidents...
Fighting...
Normal_Videos...

«[!IMPORTANT]
Veri setini kullanmadan önce resmî kullanım koşullarını inceleyin. Akademik çalışmalarda UCF-Crime veri seti ve ilgili CVPR 2018 makalesi için atıf yapılmalıdır.»

---

Sınırlamalar

Mevcut sürüm aşağıdaki sınırlamalara sahiptir:

- Algılama doğruluğu kamera açısına ve görüntü kalitesine bağlıdır.
- Düşük ışık koşulları nesne tespit performansını düşürebilir.
- Yoğun nesne örtüşmesi ByteTrack takip performansını etkileyebilir.
- Hızlı kamera hareketleri yanlış hareket anomalileri oluşturabilir.
- Piksel tabanlı yakınlık eşikleri her kamera için yeniden ayarlanmalıdır.
- Ses analizi arka plan gürültüsünden etkilenebilir.
- Görsel dil modeli zaman zaman eksik veya hatalı yorum üretebilir.
- CPU üzerinde tam 7B analiz süresi yüksek olabilir.
- Windows ve macOS sürümlerinin uçtan uca test süreci devam etmektedir.
- Sistem hukuki, güvenlik veya operasyonel kararların tek kaynağı olarak kullanılmamalıdır.

---

Yol Haritası

Tamamlanan Özellikler

- [x] YOLOv8 nesne tespiti
- [x] ByteTrack nesne takibi
- [x] Temel hareket anomalileri
- [x] Ani hız değişimi tespiti
- [x] Hareketsiz nesne tespiti
- [x] Araç–varlık yakınsama analizi
- [x] Quantum-inspired risk füzyonu
- [x] Qwen2.5-VL tabanlı Türkçe JSON çıktısı
- [x] Whisper ses analizi modülü
- [x] FastAPI servis katmanı
- [x] Web kullanıcı arayüzü
- [x] Docker Compose desteği
- [x] Linux 7B pipeline doğrulaması

Planlanan İyileştirmeler

- [ ] Platform dosyalarını tek bir ortak pipeline altında birleştirme
- [ ] CUDA, MPS ve CPU için otomatik cihaz seçimini merkezîleştirme
- [ ] Sabit video yollarını tamamen kaldırma
- [ ] Merkezi yapılandırma dosyası oluşturma
- [ ] Ortam değişkenleriyle model ve servis ayarı
- [ ] Birim testleri ekleme
- [ ] Entegrasyon testleri ekleme
- [ ] GitHub Actions CI pipeline oluşturma
- [ ] Performans benchmark tablosu hazırlama
- [ ] GPU bellek kullanım ölçümleri
- [ ] Çoklu kamera desteği
- [ ] Gerçek zamanlı RTSP akışı
- [ ] Dağıtık vLLM inference
- [ ] Kamera perspektifine göre dinamik eşik kalibrasyonu
- [ ] Olay bazlı akıllı kare seçimi
- [ ] Gelişmiş PDF raporlama
- [ ] Kullanıcı ve yetkilendirme sistemi

---

Ekip

MUDU-VLLM — Mudanya Üniversitesi

Üye| Rol| Sorumlu Modül
Elif Kübra Sağlam| Takım Kaptanı| Prompt ve risk mantığı, DecisionCore, LLM servisleme, "vllm-core/", dokümantasyon
Berke Baran Tozkoparan| Bilgisayarlı Görü| YOLO nesne tespiti, ByteTrack takibi, hareket ve yakınsama anomalileri, "yolo/"
Abdulhamit Hazine| Sistem Mimarisi ve Gömülü Sistemler| FastAPI servis katmanı, Docker, "api-service/"
Mehmet Emre Macırmemet| Uygulama ve Servis Katmanı| Web arayüzü, servis entegrasyonu, ses analizi, "web-ui/", "audio/"
Dr. Nergis Erdem| Akademik Danışman| Akademik danışmanlık ve proje yönlendirmesi

---

Dokümanlar

Dosya| İçerik
"README.md" (README.md)| Projenin ana tanıtım ve kullanım dokümanı
"CALISTIRMA.md" (CALISTIRMA.md)| Platforma özel kurulum ve çalıştırma rehberi
"ozet.md" (ozet.md)| Detaylı proje özeti ve teknik notlar
"vllm-core/README.md" (vllm-core/README.md)| Video karar destek betiklerinin kullanım kılavuzu
"audio/audio_cues.py" (audio/audio_cues.py)| Whisper tabanlı ses analizi modülü
"yolo/yolo_pipeline.py" (yolo/yolo_pipeline.py)| YOLO ve ByteTrack analiz pipeline'ı
"api-service/app.py" (api-service/app.py)| FastAPI servis uygulaması
"web-ui/app.py" (web-ui/app.py)| Web arayüzünün Python servis katmanı

---

Güvenlik ve Gizlilik

MUDU-VLLM, video verilerinin harici bir bulut servisine gönderilmesini zorunlu kılmadan yerel analiz gerçekleştirebilir.

Projeyi kullanan kişilerin:

- Video kayıtları için gerekli hukuki izinleri alması,
- Kişisel verilerin korunması mevzuatına uyması,
- Analiz sonuçlarını yetkisiz kişilerle paylaşmaması,
- Model çıktılarını insan denetimi olmadan kesin karar olarak kullanmaması

gerekir.

---

Atıf

Bu projeyi akademik bir çalışmada kullanıyorsanız aşağıdaki örnek atıf formatını kullanabilirsiniz:

@software{mudu_vllm_2026,
  author       = {MUDU-VLLM Team},
  title        = {MUDU-VLLM: AI-Powered Video Decision Support System},
  year         = {2026},
  institution  = {Mudanya University},
  url          = {https://github.com/MUDU-VLLM/mudu-vllm},
  version      = {1.5},
  license      = {Apache-2.0}
}

---

Lisans

Bu proje "Apache License 2.0" (LICENSE) kapsamında lisanslanmıştır.

Lisans koşulları çerçevesinde kaynak kod:

- Kullanılabilir
- Değiştirilebilir
- Dağıtılabilir
- Akademik projelerde kullanılabilir
- Ticari projelerde değerlendirilebilir

Lisans ve telif bildirimlerinin korunması gerekir.

---

<div align="center">MUDU-VLLM

Yerel analiz · Multimodal yapay zekâ · Türkçe karar desteği

Mudanya Üniversitesi · TEKNOFEST 2026

MUDU-VLLM Team

"Apache License 2.0" (LICENSE)

</div>