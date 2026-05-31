#!/bin/bash
# Traz os resultados banana -> local (output/*.pgm). NAO envia codigo.
# Uso: bash scripts/sync-down.sh
set -e
cd "$(dirname "$0")/.."
rsync -az --delete banana:bunnyMIP/output/ ./output/
echo "resultados sincronizados <- banana:bunnyMIP/output/ (output/ local atualizado)"
