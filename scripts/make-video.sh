#!/bin/bash
# =============================================================================
# make-video.sh - CORRE NO PC LOCAL. Orquestra a geracao do video do coelho:
#   1. (opcional) gera N frames em banana via ./bunnyMIP --frames N
#   2. corre o worker bunnyMIP/viz/make-video.sh EM BANANA (codifica c/ ffmpeg)
#   3. traz so o bunny-spin.mp4 de volta para ./output/
#
# A codificacao e o ffmpeg vivem em banana (bunnyMIP/viz/); aqui so coordenamos.
#
# Uso:
#   bash scripts/make-video.sh --frames 120 --fps 30   # gera frames + video
#   bash scripts/make-video.sh --fps 30                # usa frames existentes
# =============================================================================
set -e
cd "$(dirname "$0")/.."

FRAMES=0          # 0 = nao (re)gerar; usa frames ja em banana
FPS=30
AXIS_ARG="--yaw 1"   # eixo de rotacao p/ o multi-MIP (yaw por omissao)

while [ $# -gt 0 ]; do
  case "$1" in
    --frames) FRAMES="$2"; shift 2 ;;
    --fps)    FPS="$2";    shift 2 ;;
    *) echo "uso: $0 [--frames N] [--fps F]"; exit 1 ;;
  esac
done

PATH_CUDA='export PATH=/usr/local/cuda/bin:$PATH'

# --- 1. (opcional) gerar os frames em banana -------------------------------
if [ "$FRAMES" -gt 0 ]; then
  echo "==> [1] a gerar $FRAMES frames em banana (multi-MIP, 1 kernel call)"
  ssh banana "$PATH_CUDA; cd ~/bunnyMIP && mkdir -p output && \
    ./bunnyMIP --frames $FRAMES $AXIS_ARG" | tail -3
else
  echo "==> [1] a usar frames ja existentes em banana"
fi

# --- 2. codificar o video em banana (worker em viz/) -----------------------
echo "==> [2] a codificar o video em banana (viz/make-video.sh)"
ssh banana "cd ~/bunnyMIP && bash viz/make-video.sh $FPS"

# --- 3. trazer so o .mp4 de volta ------------------------------------------
echo "==> [3] a trazer bunny-spin.mp4 para ./output/"
rsync -az banana:bunnyMIP/output/bunny-spin.mp4 ./output/

echo "==> pronto: output/bunny-spin.mp4"
