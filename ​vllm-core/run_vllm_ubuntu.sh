#!/usr/bin/env bash
# MUDU-VLLM — Ubuntu'da vLLM ile calistirma yardimcisi
#   ./run_vllm_ubuntu.sh serve            # vLLM sunucusu
#   ./run_vllm_ubuntu.sh run /yol/video.mp4
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen2.5-VL-7B-Instruct}"
PORT="${PORT:-8000}"
MAX_LEN="${MAX_LEN:-16384}"
IMG_LIMIT="${IMG_LIMIT:-8}"          # betikteki MAX_VL_FRAMES ile ayni olmali
GPU_UTIL="${GPU_UTIL:-0.90}"

case "${1:-}" in
  serve)
    echo ">> vLLM sunucusu baslatiliyor: $MODEL_ID (port $PORT)"
    exec vllm serve "$MODEL_ID" \
      --host 0.0.0.0 --port "$PORT" \
      --dtype bfloat16 \
      --max-model-len "$MAX_LEN" \
      --limit-mm-per-prompt "image=$IMG_LIMIT" \
      --gpu-memory-utilization "$GPU_UTIL"
    ;;
  run)
    VIDEO="${2:?Kullanim: ./run_vllm_ubuntu.sh run /yol/video.mp4}"
    export MUDU_BASE_URL="http://localhost:${PORT}/v1"
    export MUDU_MODEL="$MODEL_ID"
    echo ">> vLLM'e baglaniliyor: $MUDU_BASE_URL  ($MUDU_MODEL)"
    exec python3 "$(dirname "$0")/video_decision_support_ubuntu_V1.4.py" "$VIDEO"
    ;;
  *)
    echo "Kullanim: $0 {serve|run <video>}"
    exit 1
    ;;
esac