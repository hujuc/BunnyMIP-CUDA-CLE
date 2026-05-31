#!/bin/bash
# =============================================================================
# viz/make-video.sh - CORRE NO BANANA. Codifica os frames MIP ja existentes
# num video (bunny-spin.mp4) usando um ffmpeg estatico guardado nesta pasta.
#
# So faz video: assume que os frames ja foram gerados (../bunnyMIP --frames N).
# A geracao dos frames e o orquestrador local (scripts/make-video.sh) tratam
# disso. Aqui so garantimos o ffmpeg e codificamos.
#
# Uso (em banana):  bash viz/make-video.sh [FPS]   (FPS default = 30)
# =============================================================================
set -e
FPS="${1:-30}"

# Corre relativamente a raiz do projecto (pasta-mae de viz/).
cd "$(dirname "$0")/.."
VIZ="viz"
FFMPEG="$VIZ/ffmpeg"
FRAMES="output/frame_%03d.pgm"
VIDEO="output/bunny-spin.mp4"

# Build estatico (linux64, gpl -> tem libx264, sem root). Cacheado em viz/.
FFMPEG_URL="https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linux64-gpl.tar.xz"

# --- 1. Garantir o ffmpeg estatico em viz/ (so na 1a vez) -------------------
if [ ! -x "$FFMPEG" ]; then
  echo "==> ffmpeg nao encontrado em $FFMPEG; a descarregar build estatico..."
  # NB: banana tem /tmp como tmpfs de 512 MB (cheio). Trabalhamos dentro de viz/
  # (particao /home, 1.7 TB) e extraimos so o binario ffmpeg, nada de docs.
  tmp="$VIZ/.dl"
  rm -rf "$tmp"; mkdir -p "$tmp"
  wget -q -O "$tmp/ffmpeg.tar.xz" "$FFMPEG_URL"
  tar -xf "$tmp/ffmpeg.tar.xz" -C "$tmp" \
    --wildcards --no-anchored 'bin/ffmpeg'
  cp "$tmp"/ffmpeg-*-linux64-gpl/bin/ffmpeg "$FFMPEG"
  chmod +x "$FFMPEG"
  rm -rf "$tmp"
  echo "==> ffmpeg instalado em $FFMPEG"
else
  echo "==> a usar ffmpeg em cache: $FFMPEG"
fi

# --- 2. Verificar que ha frames p/ codificar -------------------------------
n=$(ls output/frame_[0-9][0-9][0-9].pgm 2>/dev/null | wc -l)
if [ "$n" -eq 0 ]; then
  echo "ERRO: nao ha frames em output/ (corre ./bunnyMIP --frames N primeiro)." >&2
  exit 1
fi
echo "==> $n frames encontrados; a codificar a $FPS fps"

# --- 3. Codificar o video ---------------------------------------------------
# normalize: estica 16-bit -> 8-bit (frames sao visiveis); yuv420p p/ players.
"$FFMPEG" -y -hide_banner -loglevel warning \
  -framerate "$FPS" -i "$FRAMES" \
  -vf "normalize,format=yuv420p" \
  -c:v libx264 -preset medium -crf 18 \
  "$VIDEO"

echo "==> video criado: $VIDEO"
