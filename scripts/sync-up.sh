#!/bin/bash
# Envia o codigo local -> banana (NAO envia data/: os 361 slices ficam la).
# Uso: bash scripts/sync-up.sh
set -e
cd "$(dirname "$0")/.."
rsync -az --delete \
  --exclude 'data/' --exclude 'output/*' --exclude 'obj/' --exclude '*.d' --exclude 'bunnyMIP' \
  ./bunnyMIP/ banana:bunnyMIP/
echo "codigo sincronizado -> banana:bunnyMIP/ (data/ preservado)"
