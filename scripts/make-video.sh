#!/bin/bash
# =============================================================================
# make-video.sh - CORRE NO PC LOCAL. Orquestra a geracao do video do coelho:
#   1. (opcional) gera N frames em banana via ./bunnyMIP --frames N ...
#   2. corre o worker bunnyMIP/viz/make-video.sh EM BANANA (codifica c/ ffmpeg)
#   3. traz so o bunny-spin.mp4 de volta para ./output/
#
# A codificacao e o ffmpeg vivem em banana (bunnyMIP/viz/); aqui so coordenamos.
#
# Uso:
#   bash scripts/make-video.sh --frames 120 --fps 30 [opcoes de render]
#   bash scripts/make-video.sh --fps 30                 # usa frames existentes
#
# Opcoes de render (passadas ao ./bunnyMIP em multi-MIP):
#   --axis yaw|pitch|roll   eixo que roda 0..360 (default yaw)
#   --tilt-pitch P          inclinacao fixa do eixo (orbita), idem yaw/roll
#   --tilt-yaw Y
#   --tilt-roll R
#   --threshold N           piso de intensidade (por frame)
#   --sigma S               desvio-padrao do blur (por frame)
# Opcao de video:
#   --fps F                 framerate de saida (default 30)
# =============================================================================
set -e
cd "$(dirname "$0")/.."

FRAMES=0          # 0 = nao (re)gerar; usa frames ja em banana
FPS=30
AXIS=yaw
THRESH=""; SIGMA=""           # vazios -> usa defaults do binario
TILT_PITCH=0; TILT_YAW=0; TILT_ROLL=0

while [ $# -gt 0 ]; do
  case "$1" in
    --frames)      FRAMES="$2";     shift 2 ;;
    --fps)         FPS="$2";        shift 2 ;;
    --axis)        AXIS="$2";       shift 2 ;;
    --threshold)   THRESH="$2";     shift 2 ;;
    --sigma)       SIGMA="$2";      shift 2 ;;
    --tilt-pitch)  TILT_PITCH="$2"; shift 2 ;;
    --tilt-yaw)    TILT_YAW="$2";   shift 2 ;;
    --tilt-roll)   TILT_ROLL="$2";  shift 2 ;;
    *) echo "uso: $0 [--frames N] [--fps F] [--axis yaw|pitch|roll]" \
            "[--tilt-pitch P] [--tilt-yaw Y] [--tilt-roll R] [--threshold N] [--sigma S]"; exit 1 ;;
  esac
done

PATH_CUDA='export PATH=/usr/local/cuda/bin:$PATH'

# --- 1. (opcional) gerar os frames em banana -------------------------------
if [ "$FRAMES" -gt 0 ]; then
  # O binario usa --axis p/ escolher o eixo de rotacao; os valores roll/pitch/yaw
  # nos OUTROS eixos definem a inclinacao (orbita). Mapeamos os --tilt-* p/ esses.
  ARGS="--frames $FRAMES --axis $AXIS"
  [ "$TILT_ROLL"  != 0 ] && ARGS="$ARGS --roll $TILT_ROLL"
  [ "$TILT_PITCH" != 0 ] && ARGS="$ARGS --pitch $TILT_PITCH"
  [ "$TILT_YAW"   != 0 ] && ARGS="$ARGS --yaw $TILT_YAW"
  [ -n "$THRESH" ]       && ARGS="$ARGS --threshold $THRESH"
  [ -n "$SIGMA" ]        && ARGS="$ARGS --sigma $SIGMA"

  echo "==> [1] a gerar $FRAMES frames em banana (multi-MIP, 1 kernel call)"
  echo "        ./bunnyMIP $ARGS"
  ssh banana "$PATH_CUDA; cd ~/bunnyMIP && mkdir -p output && ./bunnyMIP $ARGS" | tail -3
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
