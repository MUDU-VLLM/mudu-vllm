"""
MUDU-VLLM — Senaryo 3: Uçtan Uca Video Karar Destek Pipeline'i (7B) V1.5
===============================================================================
[HAREKET] YOLO + ByteTrack  -> nesne takibi, hareket ve yakınlaşma anomalileri
[SES]     Faster-Whisper    -> yardım/tehdit sözcükleri
[KUANTUM] Quantum-inspired  -> hızlı ve deterministik risk füzyonu
[GÖRSEL]  Qwen2.5-VL-7B     -> semantik doğrulama ve Türkçe yapılandırılmış JSON

Varsayılan servis: Ollama (yerel/offline)
vLLM geçişi: yalnızca BASE_URL, MODEL ve gerekirse API_KEY değerlerini değiştirin.

WINDOWS KURULUM
---------------
python -m venv venv
venv\Scripts\activate
pip install ultralytics opencv-python requests numpy faster-whisper

FFmpeg:
  https://www.gyan.dev/ffmpeg/builds/
  ffmpeg.exe dosyasını PATH'e ekleyin veya C:\ffmpeg\bin altına koyun.

Ollama:
  ollama pull qwen2.5vl:7b

İsteğe bağlı Modelfile:
  FROM qwen2.5vl:7b
  PARAMETER num_ctx 16384

  ollama create qwen2.5vl-16k -f Modelfile

ÇALIŞTIRMA
----------
python video_decision_support_7b_V1.5.py "C:\Users\Ad\Downloads\video.mp4"

Örnek vLLM ayarları:
  set MUDU_BASE_URL=http://localhost:8000/v1
  set MUDU_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
  set MUDU_API_KEY=EMPTY

Lisans: Apache License 2.0
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


# =============================================================================
# YAPILANDIRMA
# =============================================================================

@dataclass(frozen=True)
class Config:
    base_url: str = os.getenv(
        "MUDU_BASE_URL",
        "http://localhost:11434/v1"
    ).rstrip("/")

    model: str = os.getenv(
        "MUDU_MODEL",
        "qwen2.5vl-16k"
    )

    api_key: str = os.getenv(
        "MUDU_API_KEY",
        "ollama"
    )

    yolo_weights: str = os.getenv(
        "MUDU_YOLO_WEIGHTS",
        "yolov8m.pt"
    )

    tracker: str = os.getenv(
        "MUDU_TRACKER",
        "bytetrack.yaml"
    )

    max_vl_frames: int = int(
        os.getenv("MUDU_MAX_VL_FRAMES", "8")
    )

    send_width: int = int(
        os.getenv("MUDU_SEND_WIDTH", "768")
    )

    keep_every_sec: float = float(
        os.getenv("MUDU_KEEP_EVERY_SEC", "0.4")
    )

    vid_stride: int = int(
        os.getenv("MUDU_VID_STRIDE", "6")
    )

    num_ctx: int = int(
        os.getenv("MUDU_NUM_CTX", "16384")
    )

    enable_audio: bool = (
        os.getenv("MUDU_ENABLE_AUDIO", "1")
        not in {"0", "false", "False"}
    )

    whisper_size: str = os.getenv(
        "MUDU_WHISPER_SIZE",
        "small"
    )

    whisper_device: str = os.getenv(
        "MUDU_WHISPER_DEVICE",
        "cpu"
    )

    whisper_compute_type: str = os.getenv(
        "MUDU_WHISPER_COMPUTE",
        "int8"
    )

    request_timeout: int = int(
        os.getenv("MUDU_TIMEOUT", "900")
    )

    max_tokens: int = int(
        os.getenv("MUDU_MAX_TOKENS", "768")
    )

    jpeg_quality: int = int(
        os.getenv("MUDU_JPEG_QUALITY", "80")
    )

    # Yakınlık eşiğini çözünürlükten bağımsızlaştırmak için
    # görüntü köşegeninin oranı kullanılır.
    proximity_ratio: float = float(
        os.getenv("MUDU_PROXIMITY_RATIO", "0.10")
    )


CFG = Config()


if sys.platform == "win32":
    TEMP_DIR = Path(tempfile.gettempdir()) / "mudu_vllm"
else:
    TEMP_DIR = Path("/tmp/mudu_vllm")

TEMP_DIR.mkdir(parents=True, exist_ok=True)


try:
    from ultralytics import YOLO

    YOLO_OK = True

except ImportError:
    YOLO_OK = False


# =============================================================================
# SABİTLER
# =============================================================================

VEHICLE_CLASSES = {
    "car",
    "truck",
    "motorcycle",
    "bus",
}

VULNERABLE_CLASSES = {
    "cat",
    "dog",
    "person",
}

AUDIO_KEYWORDS = {
    "imdat",
    "imdad",
    "yardım",
    "yardim",
    "help",
    "yetişin",
    "yetisin",
    "kurtar",
    "ateş",
    "ates",
    "silah",
    "kaçın",
    "kacin",
    "dikkat",
    "yangın",
    "yangin",
    "vuruldu",
    "saldırı",
    "saldiri",
}

TR = {
    "person": "kişi",
    "dog": "köpek",
    "cat": "kedi",
    "car": "araba",
    "truck": "kamyon",
    "motorcycle": "motosiklet",
    "bus": "otobüs",
    "bicycle": "bisiklet",
    "horse": "at",
    "cow": "inek",
    "sheep": "koyun",
    "knife": "bıçak",
    "backpack": "sırt çantası",
    "handbag": "el çantası",
}

RISK_ORDER = {
    "Dusuk": 0,
    "Orta": 1,
    "Yuksek": 2,
}

VALID_RISKS = set(RISK_ORDER)


CATEGORIES = """
Şu kategorilere dikkat et ve YALNIZCA gerçekten gördüğünü işaretle:

- İNSAN TEHDİDİ:
  kavga/arbede, fiziksel saldırı veya darp, hırsızlık,
  silahla yaralanma, gizlenen şüpheli kişi,
  yerde hareketsiz kişi, düşme

- SİLAH:
  tüfek, tabanca, bıçak, roketatar/füze, patlayıcı

- ARAÇ / İŞ KAZASI:
  araç veya forklift devrilmesi,
  araç-yaya çarpma riski

- HAYVAN TEHDİDİ:
  yılan, akrep, yaban domuzu,
  kurt/ayı gibi vahşi hayvan,
  saldırgan köpek

- HAYVANA ZARAR:
  araç veya kişi tarafından hayvanın ezilmesi,
  çarpılması veya darp edilmesi

- ÇEVRE:
  yangın, yoğun duman
"""


EXAMPLE_JSON = """
{
  "summary": "Depo alanında forklift devrilmiş ve yakınında yerde hareketsiz bir personel görülüyor.",
  "events": [
    {
      "time": "00:15",
      "event": "Forkliftin yan yatmış olduğu görülüyor."
    },
    {
      "time": "00:20",
      "event": "Forklift yakınında yerde hareketsiz bir personel bulunuyor."
    }
  ],
  "risk": "Yuksek",
  "actions": [
    "Sağlık ve iş güvenliği ekiplerini olay yerine yönlendir.",
    "Alanı güvenlik şeridiyle kapat."
  ]
}
"""


# =============================================================================
# GENEL YARDIMCI FONKSİYONLAR
# =============================================================================

def timestamp(
    frame_idx: int,
    fps: float,
) -> str:
    """
    Kare numarasını MM:SS zaman damgasına dönüştürür.
    """

    seconds = max(frame_idx, 0) / max(fps, 1e-6)

    minutes = int(seconds // 60)
    remaining_seconds = int(seconds % 60)

    return f"{minutes:02d}:{remaining_seconds:02d}"


def downscale(
    bgr: np.ndarray,
) -> np.ndarray:
    """
    Görüntü genişliği SEND_WIDTH değerinden büyükse
    oranı bozmadan küçültür.
    """

    height, width = bgr.shape[:2]

    if width <= CFG.send_width:
        return bgr

    new_height = max(
        1,
        round(height * CFG.send_width / width)
    )

    return cv2.resize(
        bgr,
        (CFG.send_width, new_height),
        interpolation=cv2.INTER_AREA,
    )


def b64_jpeg(
    bgr: np.ndarray,
) -> str | None:
    """
    OpenCV görüntüsünü Base64 JPEG metnine dönüştürür.
    """

    ok, buffer = cv2.imencode(
        ".jpg",
        bgr,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            CFG.jpeg_quality,
        ],
    )

    if not ok:
        return None

    return base64.b64encode(
        buffer
    ).decode("ascii")


def normalize_time(
    value: Any,
) -> str:
    """
    Modelden veya sensörden gelen zamanı MM:SS biçimine getirir.
    """

    text = str(
        value or "00:00"
    ).strip()

    match = re.fullmatch(
        r"(\d{1,3}):([0-5]\d)",
        text,
    )

    if not match:
        return "00:00"

    minutes = int(match.group(1))
    seconds = int(match.group(2))

    return f"{minutes:02d}:{seconds:02d}"


def max_risk(
    *risks: str,
) -> str:
    """
    Verilen risk seviyeleri arasından en yüksek olanı döndürür.
    """

    valid = [
        risk
        for risk in risks
        if risk in VALID_RISKS
    ]

    if not valid:
        return "Dusuk"

    return max(
        valid,
        key=lambda risk: RISK_ORDER[risk],
    )


def locate_ffmpeg() -> str | None:
    """
    FFmpeg'i önce PATH üzerinde, ardından yaygın Windows
    klasörlerinde arar.
    """

    found = shutil.which("ffmpeg")

    if found:
        return found

    if sys.platform == "win32":
        candidates = [
            Path(r"C:\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\Program Files\ffmpeg\bin\ffmpeg.exe"),
            Path(r"C:\ffmpeg\ffmpeg.exe"),
        ]

        for candidate in candidates:
            if candidate.exists():
                os.environ["PATH"] = (
                    str(candidate.parent)
                    + os.pathsep
                    + os.environ.get("PATH", "")
                )

                return str(candidate)

    return None


# =============================================================================
# [HAREKET] YOLO + ByteTrack
# =============================================================================

class AnomalyDetector:
    """
    ByteTrack tarafından oluşturulan nesne izlerini kullanarak
    hareket anomalilerini ve araç-varlık yakınlaşmalarını tespit eder.
    """

    def __init__(
        self,
        fps: float,
    ):
        self.fps = max(
            fps,
            1.0,
        )

        self.history: dict[
            int,
            list[dict[str, Any]]
        ] = defaultdict(list)

        self.anomalies: list[
            dict[str, Any]
        ] = []

        self._still_marked: set[int] = set()


    def update(
        self,
        track_id: int,
        cls_name: str,
        cx: float,
        cy: float,
        frame_idx: int,
    ) -> None:
        """
        Bir nesnenin yeni merkez noktasını takip geçmişine ekler.
        """

        ts = timestamp(
            frame_idx,
            self.fps,
        )

        history = self.history[track_id]

        history.append(
            {
                "cx": cx,
                "cy": cy,
                "frame": frame_idx,
                "time": ts,
                "cls": cls_name,
            }
        )

        # Takip geçmişinin bellekte sınırsız büyümesini engeller.
        if len(history) > 300:
            del history[:-300]

        if len(history) >= 2:
            self._check_motion(
                track_id,
                frame_idx,
                ts,
            )


    def check_proximity(
        self,
        detections: list[dict[str, Any]],
        frame_idx: int,
        frame_shape: tuple[int, ...] | None,
    ) -> None:
        """
        Araçlarla kişi veya hayvanlar arasındaki merkez uzaklığını kontrol eder.
        Sabit piksel yerine görüntü köşegenine göre eşik kullanır.
        """

        if not frame_shape:
            return

        height, width = frame_shape[:2]

        distance_threshold = (
            np.hypot(width, height)
            * CFG.proximity_ratio
        )

        ts = timestamp(
            frame_idx,
            self.fps,
        )

        vehicles = [
            detection
            for detection in detections
            if detection["cls"] in VEHICLE_CLASSES
        ]

        vulnerable = [
            detection
            for detection in detections
            if detection["cls"] in VULNERABLE_CLASSES
        ]

        for vehicle in vehicles:
            for subject in vulnerable:
                center_distance = float(
                    np.hypot(
                        vehicle["cx"] - subject["cx"],
                        vehicle["cy"] - subject["cy"],
                    )
                )

                if center_distance < distance_threshold:
                    self._add(
                        ts=ts,
                        tid=subject.get(
                            "track_id",
                            -1,
                        ),
                        cls=subject["cls"],
                        anomaly_type="Araç-varlık yakınlaşması",
                        description=(
                            f"{TR.get(subject['cls'], subject['cls'])} ile "
                            f"{TR.get(vehicle['cls'], vehicle['cls'])} "
                            "birbirine çok yakın; "
                            "olası çarpma riski."
                        ),
                        score=0.90,
                        frame_idx=frame_idx,
                    )


    def _check_motion(
        self,
        track_id: int,
        frame_idx: int,
        ts: str,
    ) -> None:
        """
        Hareketsizlik ve ani hız değişimi anomalilerini kontrol eder.
        """

        history = self.history[track_id]
        cls_name = history[-1]["cls"]

        processed_fps = max(
            self.fps / max(CFG.vid_stride, 1),
            1.0,
        )

        still_window = max(
            round(processed_fps * 2.0),
            3,
        )

        if (
            len(history) >= still_window
            and track_id not in self._still_marked
        ):
            recent = history[-still_window:]

            x_spread = float(
                np.std(
                    [
                        item["cx"]
                        for item in recent
                    ]
                )
            )

            y_spread = float(
                np.std(
                    [
                        item["cy"]
                        for item in recent
                    ]
                )
            )

            if (
                x_spread + y_spread < 4.0
                and cls_name in VULNERABLE_CLASSES
            ):
                self._still_marked.add(
                    track_id
                )

                self._add(
                    ts=ts,
                    tid=track_id,
                    cls=cls_name,
                    anomaly_type="Hareketsiz nesne",
                    description=(
                        f"{TR.get(cls_name, cls_name)} "
                        "yaklaşık iki saniyedir hareketsiz; "
                        "düşme veya bilinç kaybı ihtimali "
                        "doğrulanmalıdır."
                    ),
                    score=0.85,
                    frame_idx=frame_idx,
                )

        if len(history) >= 6:
            speeds = []

            for index in range(-5, 0):
                current = history[index]
                previous = history[index - 1]

                speed = float(
                    np.hypot(
                        current["cx"] - previous["cx"],
                        current["cy"] - previous["cy"],
                    )
                )

                speeds.append(
                    speed
                )

            baseline = float(
                np.mean(
                    speeds[:-1]
                )
            )

            if abs(
                speeds[-1] - baseline
            ) > 18.0:
                self._add(
                    ts=ts,
                    tid=track_id,
                    cls=cls_name,
                    anomaly_type="Ani hız değişimi",
                    description=(
                        f"{TR.get(cls_name, cls_name)} "
                        "nesnesinde ani hız değişimi "
                        "tespit edildi; çarpma veya düşme "
                        "ihtimali doğrulanmalıdır."
                    ),
                    score=0.75,
                    frame_idx=frame_idx,
                )


    def _add(
        self,
        ts: str,
        tid: int,
        cls: str,
        anomaly_type: str,
        description: str,
        score: float,
        frame_idx: int,
    ) -> None:
        """
        Aynı anomalinin kısa süre içinde tekrar eklenmesini önler.
        """

        cooldown_frames = (
            self.fps * 2.0
        )

        for anomaly in self.anomalies:
            same_event = (
                anomaly["track_id"] == tid
                and anomaly["anomaly_type"] == anomaly_type
            )

            close_in_time = (
                abs(
                    frame_idx
                    - anomaly["frame_idx"]
                )
                < cooldown_frames
            )

            if same_event and close_in_time:
                return

        self.anomalies.append(
            {
                "time": ts,
                "track_id": tid,
                "class": cls,
                "anomaly_type": anomaly_type,
                "description": description,
                "score": round(
                    float(score),
                    2,
                ),
                "frame_idx": int(frame_idx),
            }
        )

        print(
            f"  [HAREKET] {ts} | "
            f"{anomaly_type} | "
            f"{TR.get(cls, cls)}"
        )


def run_yolo(
    video_path: str,
) -> tuple[
    list[dict[str, Any]],
    dict[int, np.ndarray],
    float,
    dict[str, set[str]],
]:
    """
    YOLO ve ByteTrack kullanarak videoyu analiz eder.

    Dönen değerler:
        anomalies
        frames_store
        fps
        seen_classes
    """

    cap = cv2.VideoCapture(
        video_path
    )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    cap.release()

    detector = AnomalyDetector(
        fps
    )

    frames_store: dict[
        int,
        np.ndarray
    ] = {}

    seen_classes: dict[
        str,
        set[str]
    ] = defaultdict(set)

    processed_keep_stride = max(
        round(
            CFG.keep_every_sec
            * fps
            / max(
                CFG.vid_stride,
                1,
            )
        ),
        1,
    )

    model = YOLO(
        CFG.yolo_weights
    )

    results = model.track(
        source=video_path,
        persist=True,
        stream=True,
        tracker=CFG.tracker,
        verbose=False,
        vid_stride=CFG.vid_stride,
    )

    for processed_idx, result in enumerate(
        results
    ):
        frame_idx = (
            processed_idx
            * CFG.vid_stride
        )

        detections: list[
            dict[str, Any]
        ] = []

        if (
            result.boxes is not None
            and result.boxes.id is not None
        ):
            boxes = (
                result.boxes.xyxy
                .cpu()
                .numpy()
            )

            track_ids = (
                result.boxes.id
                .cpu()
                .numpy()
                .astype(int)
            )

            classes = (
                result.boxes.cls
                .cpu()
                .numpy()
                .astype(int)
            )

            for (
                x1,
                y1,
                x2,
                y2,
            ), track_id, class_id in zip(
                boxes,
                track_ids,
                classes,
            ):
                cx = float(
                    (x1 + x2) / 2.0
                )

                cy = float(
                    (y1 + y2) / 2.0
                )

                class_name = str(
                    result.names[
                        int(class_id)
                    ]
                )

                detector.update(
                    track_id=track_id,
                    cls_name=class_name,
                    cx=cx,
                    cy=cy,
                    frame_idx=frame_idx,
                )

                detections.append(
                    {
                        "cls": class_name,
                        "cx": cx,
                        "cy": cy,
                        "track_id": int(
                            track_id
                        ),
                    }
                )

                seen_classes[
                    timestamp(
                        frame_idx,
                        fps,
                    )
                ].add(
                    class_name
                )

        detector.check_proximity(
            detections=detections,
            frame_idx=frame_idx,
            frame_shape=(
                result.orig_img.shape
                if result.orig_img is not None
                else None
            ),
        )

        if (
            processed_idx
            % processed_keep_stride
            == 0
            and result.orig_img is not None
        ):
            frames_store[
                frame_idx
            ] = downscale(
                result.orig_img
            )

    return (
        detector.anomalies,
        frames_store,
        fps,
        seen_classes,
    )


# =============================================================================
# KARE ÖRNEKLEME
# =============================================================================

def uniform_sample(
    video_path: str,
    count: int | None = None,
) -> tuple[
    dict[int, np.ndarray],
    float,
]:
    """
    Videodan eşit aralıklarla kare örnekler.
    """

    count = (
        count
        or CFG.max_vl_frames
    )

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise ValueError(
            "Video OpenCV tarafından açılamadı."
        )

    fps = (
        cap.get(
            cv2.CAP_PROP_FPS
        )
        or 30.0
    )

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    store: dict[
        int,
        np.ndarray
    ] = {}

    if total <= 0:
        cap.release()

        raise ValueError(
            "Video boş veya kare sayısı okunamıyor."
        )

    indices = np.linspace(
        0,
        total - 1,
        min(
            count,
            total,
        ),
        dtype=int,
    )

    for index in indices:
        cap.set(
            cv2.CAP_PROP_POS_FRAMES,
            int(index),
        )

        ok, frame = cap.read()

        if ok:
            store[
                int(index)
            ] = downscale(
                frame
            )

    cap.release()

    if not store:
        raise ValueError(
            "Videodan hiçbir kare okunamadı."
        )

    return (
        store,
        fps,
    )


def pick_vl_frames(
    anomalies: list[dict[str, Any]],
    frames_store: dict[int, np.ndarray],
    fps: float,
    count: int | None = None,
) -> list[
    tuple[
        int,
        str,
        np.ndarray,
    ]
]:
    """
    Önce anomalilere yakın kareleri, ardından video geneline
    dağıtılmış kareleri seçer.
    """

    count = (
        count
        or CFG.max_vl_frames
    )

    available = sorted(
        frames_store
    )

    if not available:
        return []

    selected: set[int] = set()

    sorted_anomalies = sorted(
        anomalies,
        key=lambda item: item.get(
            "score",
            0,
        ),
        reverse=True,
    )

    for anomaly in sorted_anomalies:
        nearest = min(
            available,
            key=lambda frame: abs(
                frame
                - int(
                    anomaly["frame_idx"]
                )
            ),
        )

        selected.add(
            nearest
        )

        if len(selected) >= count:
            break

    if len(selected) < count:
        targets = np.linspace(
            available[0],
            available[-1],
            count,
            dtype=int,
        )

        for target in targets:
            nearest = min(
                available,
                key=lambda frame: abs(
                    frame
                    - int(target)
                ),
            )

            selected.add(
                nearest
            )

            if len(selected) >= count:
                break

    ordered = sorted(
        selected
    )[:count]

    return [
        (
            frame_idx,
            timestamp(
                frame_idx,
                fps,
            ),
            frames_store[frame_idx],
        )
        for frame_idx in ordered
    ]


# =============================================================================
# [SES] FASTER-WHISPER
# =============================================================================

def transcribe_audio_cues(
    video_path: str,
) -> list[
    dict[str, str]
]:
    """
    Videonun sesini WAV biçimine dönüştürür ve Faster-Whisper
    kullanarak tehdit veya yardım ifadelerini arar.
    """

    try:
        from faster_whisper import WhisperModel

    except ImportError:
        print(
            "  UYARI: faster-whisper kurulu değil; "
            "ses analizi atlandı."
        )

        return []

    ffmpeg = locate_ffmpeg()

    if not ffmpeg:
        print(
            "  UYARI: FFmpeg bulunamadı; "
            "ses analizi atlandı."
        )

        return []

    file_descriptor, wav_name = tempfile.mkstemp(
        prefix="mudu_audio_",
        suffix=".wav",
        dir=TEMP_DIR,
    )

    os.close(
        file_descriptor
    )

    wav_path = Path(
        wav_name
    )

    try:
        command = [
            ffmpeg,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
            "-loglevel",
            "error",
        ]

        subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )

        model = WhisperModel(
            CFG.whisper_size,
            device=CFG.whisper_device,
            compute_type=CFG.whisper_compute_type,
        )

        segments, _ = model.transcribe(
            str(wav_path),
            language="tr",
            vad_filter=True,
        )

        cues: list[
            dict[str, str]
        ] = []

        for segment in segments:
            text = segment.text.strip()

            lowered = text.casefold()

            hit = next(
                (
                    word
                    for word in AUDIO_KEYWORDS
                    if word in lowered
                ),
                None,
            )

            if not hit:
                continue

            ts = timestamp(
                round(
                    segment.start
                    * 1000
                ),
                1000.0,
            )

            cues.append(
                {
                    "time": ts,
                    "event": (
                        "Ses kaydında tehdit veya yardım "
                        f'ifadesi duyuluyor: "{text}"'
                    ),
                }
            )

            print(
                f"  [SES] {ts} | "
                f"{hit} -> {text}"
            )

        return cues

    except (
        subprocess.CalledProcessError,
        OSError,
    ) as exc:
        print(
            "  UYARI: FFmpeg ses çıkarma hatası: "
            f"{exc}"
        )

        return []

    except Exception as exc:
        print(
            "  UYARI: Whisper analizi başarısız: "
            f"{exc}"
        )

        return []

    finally:
        try:
            wav_path.unlink(
                missing_ok=True
            )

        except OSError:
            pass


# =============================================================================
# [KUANTUM] DETERMINİSTİK QUANTUM-INSPIRED RİSK FÜZYONU
# =============================================================================

class DecisionCore:
    """
    Kuantum-esintili olasılık kodlaması.

    Önceki sürümde Monte Carlo ölçümü kullanıldığı için aynı girdiler
    farklı sonuçlar üretebiliyordu.

    Bu sürümde qubitlerin |1> olasılıkları doğrudan kullanılır.
    Böylece aynı sensör girdileri aynı risk skorunu üretir.
    """

    def __init__(
        self,
        num_features: int = 4,
    ):
        self.num_features = num_features

        self.qubits = (
            np.ones(
                (
                    num_features,
                    2,
                ),
                dtype=np.float64,
            )
            / np.sqrt(2)
        )


    @staticmethod
    def _clip(
        score: float,
    ) -> float:
        """
        Girdiyi 0 ile 1 arasında sınırlar.
        """

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )


    def encode_classical_to_quantum(
        self,
        movement_score: float,
        audio_score: float,
        proximity_score: float,
    ) -> None:
        """
        Klasik sensör skorlarını qubit genliklerine kodlar.
        """

        scores = [
            self._clip(
                movement_score
            ),
            self._clip(
                audio_score
            ),
            self._clip(
                proximity_score
            ),
        ]

        angles = [
            score
            * (
                np.pi / 2.0
            )
            for score in scores
        ]

        for index, angle in enumerate(
            angles
        ):
            self.qubits[index] = [
                np.cos(angle),
                np.sin(angle),
            ]

        fused_angle = float(
            np.mean(
                angles
            )
        )

        self.qubits[3] = [
            np.cos(
                fused_angle
            ),
            np.sin(
                fused_angle
            ),
        ]


    def evaluate_risk(
        self,
    ) -> float:
        """
        Her qubitin |1> olasılığını hesaplar ve ağırlıklı
        birleşik risk skoru üretir.
        """

        probabilities_one = np.square(
            self.qubits[:, 1]
        )

        weights = np.array(
            [
                0.35,
                0.35,
                0.15,
                0.15,
            ],
            dtype=np.float64,
        )

        return float(
            np.dot(
                probabilities_one,
                weights,
            )
        )


def sensor_scores(
    anomalies: list[dict[str, Any]],
    audio_cues: list[dict[str, str]],
) -> tuple[
    float,
    float,
    float,
]:
    """
    Hareket, ses ve yakınlık sensörleri için klasik skorları oluşturur.
    """

    movement_score = max(
        [
            float(
                item.get(
                    "score",
                    0.0,
                )
            )
            for item in anomalies
        ],
        default=0.10,
    )

    audio_score = (
        0.90
        if audio_cues
        else 0.10
    )

    proximity_score = max(
        [
            float(
                item.get(
                    "score",
                    0.0,
                )
            )
            for item in anomalies
            if (
                "yakınlaş"
                in str(
                    item.get(
                        "anomaly_type",
                        "",
                    )
                ).casefold()
                or
                "yaklas"
                in str(
                    item.get(
                        "anomaly_type",
                        "",
                    )
                ).casefold()
            )
        ],
        default=0.10,
    )

    return (
        movement_score,
        audio_score,
        proximity_score,
    )


# =============================================================================
# [GÖRSEL] PROMPT
# =============================================================================

def build_prompt(
    vl_frames: list[
        tuple[
            int,
            str,
            np.ndarray,
        ]
    ],
    movement_cues: list[str],
    audio_cues: list[dict[str, str]],
    seen_classes: dict[str, set[str]],
) -> str:
    """
    Qwen2.5-VL modeli için Türkçe güvenlik analiz promptu üretir.
    """

    frame_lines = "\n".join(
        (
            f"- Kare {index + 1} "
            f"yaklaşık {stamp}"
        )
        for index, (
            _,
            stamp,
            _,
        ) in enumerate(
            vl_frames
        )
    )

    class_block = ""

    if seen_classes:
        lines = []

        for stamp in sorted(
            seen_classes
        ):
            translated = ", ".join(
                sorted(
                    TR.get(
                        name,
                        name,
                    )
                    for name in seen_classes[
                        stamp
                    ]
                )
            )

            if translated:
                lines.append(
                    f"- {stamp}: {translated}"
                )

        if lines:
            class_block = (
                "\nYOLO nesne ipuçları "
                "(hatalı olabilir; görsel olarak doğrula):\n"
                + "\n".join(lines)
                + "\n"
            )

    cue_lines = list(
        movement_cues
    )

    cue_lines.extend(
        (
            f"[SES] {cue['time']} | "
            f"{cue['event']}"
        )
        for cue in audio_cues
    )

    cue_block = ""

    if cue_lines:
        cue_block = (
            "\nAlgılayıcı ipuçları "
            "(yalnızca karelerle tutarlıysa kullan):\n"
            + "\n".join(
                f"- {line}"
                for line in cue_lines
            )
            + "\n"
        )

    return f"""
Sen bir güvenlik operasyon merkezi video analiz asistanısın.

Sana güvenlik kamerasından alınmış zaman damgalı kareler veriliyor.
Görevin yalnızca karelerde gerçekten görülen olayları raporlamaktır.

Karelerin yaklaşık zamanları:
{frame_lines}

{class_block}
{cue_block}

{CATEGORIES}

KESİN KURALLAR:

1. Yalnızca karelerde net gördüğün olayları yaz.
   Emin değilsen yazma.

2. YOLO ve diğer sensörler yalnızca ipucudur ve hata yapabilir.
   Görsel kanıt önceliklidir.

3. Bir hayvan açıkça görünüyorsa yanlış YOLO sınıfını kullanma.

4. Her olay için somut bir cümle yaz.
   Tek kelimelik etiket kullanma.

5. Aynı olayı farklı karelerde tekrar tekrar yazma.

6. "summary" boş olamaz.

7. "actions" boş olamaz ve en az bir uygulanabilir
   operatör eylemi içermelidir.

8. Tehdit görünmüyorsa "events" boş liste olabilir
   ve risk "Dusuk" olabilir.

9. Kavga, saldırı, düşme veya yerde hareketsiz kişi
   için risk en az "Orta" olmalıdır.

10. Silah, ağır yaralanma, gerçekleşmiş çarpma
    veya aktif yangın için risk "Yuksek" olmalıdır.

11. Araç-yaya veya araç-hayvan yakınlaşması için
    risk en az "Orta" olmalıdır.

12. Hayvana çarpma veya eziyet görülürse
    "olası hayvan hakları ihlali" ifadesini kullan.

    Actions içine:
    - olayın kayıt altına alınmasını
    - araç veya plaka bilgisinin not edilmesini
    ekle.

13. Aşağıdaki örnek içeriği kopyalama.

14. Çıktıda Markdown veya açıklama kullanma.
    Yalnızca geçerli JSON döndür.

JSON alanları:

- summary:
  dolu Türkçe metin

- events:
  {{
    "time": "MM:SS",
    "event": "somut olay"
  }}
  nesnelerinden oluşan liste

- risk:
  yalnızca "Dusuk", "Orta" veya "Yuksek"

- actions:
  dolu Türkçe eylem listesi

Yalnızca biçim örneği:

{EXAMPLE_JSON}
""".strip()


def response_schema() -> dict[str, Any]:
    """
    Model sunucusuna gönderilecek yapılandırılmış JSON şeması.
    """

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "video_security_assessment",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {
                        "type": "string",
                        "minLength": 1,
                    },
                    "events": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "time": {
                                    "type": "string",
                                    "pattern": (
                                        r"^\d{2,3}:[0-5]\d$"
                                    ),
                                },
                                "event": {
                                    "type": "string",
                                    "minLength": 1,
                                },
                            },
                            "required": [
                                "time",
                                "event",
                            ],
                        },
                    },
                    "risk": {
                        "type": "string",
                        "enum": [
                            "Dusuk",
                            "Orta",
                            "Yuksek",
                        ],
                    },
                    "actions": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "string",
                            "minLength": 1,
                        },
                    },
                },
                "required": [
                    "summary",
                    "events",
                    "risk",
                    "actions",
                ],
            },
        },
    }


def call_vl(
    vl_frames: list[
        tuple[
            int,
            str,
            np.ndarray,
        ]
    ],
    movement_cues: list[str],
    audio_cues: list[dict[str, str]],
    seen_classes: dict[str, set[str]],
) -> str:
    """
    Kareleri ve promptu OpenAI-uyumlu Ollama veya vLLM
    endpoint'ine gönderir.
    """

    if not vl_frames:
        raise ValueError(
            "Görsel modele gönderilecek kare bulunamadı."
        )

    content: list[
        dict[str, Any]
    ] = []

    for _, _, frame in vl_frames:
        encoded = b64_jpeg(
            frame
        )

        if encoded:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            "data:image/jpeg;base64,"
                            f"{encoded}"
                        ),
                    },
                }
            )

    has_image = any(
        item.get("type") == "image_url"
        for item in content
    )

    if not has_image:
        raise ValueError(
            "Kareler JPEG biçimine dönüştürülemedi."
        )

    prompt = build_prompt(
        vl_frames=vl_frames,
        movement_cues=movement_cues,
        audio_cues=audio_cues,
        seen_classes=seen_classes,
    )

    content.append(
        {
            "type": "text",
            "text": prompt,
        }
    )

    payload: dict[
        str,
        Any,
    ] = {
        "model": CFG.model,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": CFG.max_tokens,
        "temperature": 0,
        "response_format": response_schema(),
    }

    # Ollama için context ayarı.
    # vLLM options alanını tanımayabilir.
    if "11434" in CFG.base_url:
        payload["options"] = {
            "num_ctx": CFG.num_ctx,
        }

    headers = {
        "Content-Type": "application/json",
        "Authorization": (
            f"Bearer {CFG.api_key}"
        ),
    }

    endpoint = (
        f"{CFG.base_url}/chat/completions"
    )

    response = requests.post(
        endpoint,
        json=payload,
        headers=headers,
        timeout=CFG.request_timeout,
    )

    # Bazı OpenAI-uyumlu sunucular json_schema veya
    # options alanını desteklemeyebilir.
    if response.status_code == 400:
        fallback_payload = dict(
            payload
        )

        fallback_payload.pop(
            "options",
            None,
        )

        fallback_payload[
            "response_format"
        ] = {
            "type": "json_object"
        }

        response = requests.post(
            endpoint,
            json=fallback_payload,
            headers=headers,
            timeout=CFG.request_timeout,
        )

    if response.status_code != 200:
        raise RuntimeError(
            "Model sunucusu HTTP "
            f"{response.status_code}: "
            f"{response.text[:800]}"
        )

    body = response.json()

    try:
        return str(
            body["choices"][0]["message"]["content"]
        )

    except (
        KeyError,
        IndexError,
        TypeError,
    ) as exc:
        raise RuntimeError(
            f"Beklenmeyen API yanıtı: {body}"
        ) from exc


# =============================================================================
# JSON AYIKLAMA VE DOĞRULAMA
# =============================================================================

def parse_json(
    raw: str,
) -> dict[str, Any] | None:
    """
    Model yanıtından JSON nesnesini güvenli biçimde ayıklar.
    """

    raw = raw.strip()

    # Önce tüm çıktıyı doğrudan JSON olarak dene.
    try:
        value = json.loads(
            raw
        )

        if isinstance(
            value,
            dict,
        ):
            return value

        return None

    except json.JSONDecodeError:
        pass

    # Markdown kod çitlerini temizle.
    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned,
    )

    try:
        value = json.loads(
            cleaned
        )

        if isinstance(
            value,
            dict,
        ):
            return value

        return None

    except json.JSONDecodeError:
        pass

    # Son çare olarak ilk dengeli JSON nesnesini bul.
    start = raw.find(
        "{"
    )

    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(
        start,
        len(raw),
    ):
        char = raw[index]

        if in_string:
            if escaped:
                escaped = False

            elif char == "\\":
                escaped = True

            elif char == '"':
                in_string = False

            continue

        if char == '"':
            in_string = True

        elif char == "{":
            depth += 1

        elif char == "}":
            depth -= 1

            if depth == 0:
                candidate = raw[
                    start:index + 1
                ]

                try:
                    value = json.loads(
                        candidate
                    )

                    if isinstance(
                        value,
                        dict,
                    ):
                        return value

                    return None

                except json.JSONDecodeError:
                    return None

    return None


def normalize_result(
    result: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Model çıktısını şartnameye uygun, güvenli bir JSON yapısına getirir.
    """

    if not isinstance(
        result,
        dict,
    ):
        result = {}

    summary = str(
        result.get(
            "summary"
        )
        or ""
    ).strip()

    if not summary:
        summary = (
            "Seçilen video karelerinde kesin bir "
            "tehdit olayı doğrulanamadı."
        )

    risk = str(
        result.get(
            "risk"
        )
        or "Dusuk"
    ).strip()

    if risk not in VALID_RISKS:
        risk = "Dusuk"

    events: list[
        dict[str, str]
    ] = []

    raw_events = result.get(
        "events",
        [],
    )

    if isinstance(
        raw_events,
        list,
    ):
        seen = set()

        for event in raw_events:
            if not isinstance(
                event,
                dict,
            ):
                continue

            text = str(
                event.get(
                    "event"
                )
                or ""
            ).strip()

            if not text:
                continue

            stamp = normalize_time(
                event.get(
                    "time"
                )
            )

            key = (
                stamp,
                text.casefold(),
            )

            if key in seen:
                continue

            seen.add(
                key
            )

            events.append(
                {
                    "time": stamp,
                    "event": text,
                }
            )

    actions: list[str] = []

    raw_actions = result.get(
        "actions",
        [],
    )

    if isinstance(
        raw_actions,
        list,
    ):
        for action in raw_actions:
            text = str(
                action or ""
            ).strip()

            if (
                text
                and text not in actions
            ):
                actions.append(
                    text
                )

    elif (
        isinstance(
            raw_actions,
            str,
        )
        and raw_actions.strip()
    ):
        actions.append(
            raw_actions.strip()
        )

    if not actions:
        actions = [
            "Video kaydını operatör tarafından gözden geçir."
        ]

    events.sort(
        key=lambda item: item["time"]
    )

    return {
        "summary": summary,
        "events": events,
        "risk": risk,
        "actions": actions,
    }


def infer_sensor_minimum_risk(
    anomalies: list[dict[str, Any]],
    audio_cues: list[dict[str, str]],
) -> str:
    """
    Sensör kanıtlarına göre minimum risk seviyesini hesaplar.
    """

    minimum = "Dusuk"

    if audio_cues:
        minimum = max_risk(
            minimum,
            "Orta",
        )

    for anomaly in anomalies:
        anomaly_type = str(
            anomaly.get(
                "anomaly_type",
                "",
            )
        ).casefold()

        description = str(
            anomaly.get(
                "description",
                "",
            )
        ).casefold()

        joined = (
            f"{anomaly_type} "
            f"{description}"
        )

        if (
            "yakınlaş" in joined
            or "yaklas" in joined
        ):
            minimum = max_risk(
                minimum,
                "Orta",
            )

        elif "hareketsiz" in joined:
            minimum = max_risk(
                minimum,
                "