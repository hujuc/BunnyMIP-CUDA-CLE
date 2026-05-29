#!/bin/bash
# Compila e corre o bunnyMIP no banana (nvcc no PATH). Aceita args p/ o binario.
# Uso: bash scripts/build-run.sh [--threshold N --sigma S --roll R --pitch P --yaw Y]
set -e
ARGS="$*"
ssh banana "export PATH=/usr/local/cuda/bin:\$PATH; cd ~/bunnyMIP && make && ./bunnyMIP $ARGS"
