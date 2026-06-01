#!/bin/bash
# =============================================================================
# bench-blur.sh - Compara as 3 variantes do blur em banana (--repeat p/ media)
# e gera tabelas Markdown -> output/perf-blur.txt.
#
#   default        -> blur_kernel_tiled (shared-memory, optimizado)
#   BUNNY_BLUR_NAIVE=1 -> blur_kernel     (global memory, baseline)
#   BUNNY_SEPARABLE=1  -> blur1d_kernel x3 (separavel, experimento)
#
# Como o --repeat reporta o tempo TOTAL dos kernels (threshold+blur+MIP), a
# diferenca entre variantes aparece em "GPU (kernels)". Threshold e MIP sao
# constantes, por isso a variacao = o efeito do blur.
#
# Uso: bash scripts/bench-blur.sh [REPEAT]   (REPEAT default = 10)
# =============================================================================
set -e
REPEAT="${1:-10}"
OUT="output/perf-blur.txt"
RAW="$(mktemp)"
ROWS="$(mktemp)"
trap 'rm -f "$RAW" "$ROWS"' EXIT

# run <label> <ENV> <args...>
#   - corre o binario em banana com a env-flag do blur escolhido
#   - mostra progresso ao vivo (stderr) e captura p/ o awk
#   - extrai cpu / gpu-e2e / gpu-kernels / speedups -> linha markdown
run() {
  local label="$1" env="$2"; shift 2
  local out
  echo ">>> running: $label  (env='$env' $*  | repeat=$REPEAT)" >&2
  out="$(ssh banana "export PATH=/usr/local/cuda/bin:\$PATH; cd ~/bunnyMIP && \
    $env ./bunnyMIP $* --repeat $REPEAT" 2>&1 | tee /dev/stderr)"

  {
    echo "### $label"
    echo "env: $env   args: $*   repeat=$REPEAT"
    echo "$out"
    echo
  } >> "$RAW"

  printf '%s\n' "$out" | awk -v label="$label" '
    function extract(   i) { for (i=1;i<=NF;i++) if ($i+0==$i && $i!="") return $i; return "" }
    /CPU  \(sequential\)/ { cpu = extract() }
    /GPU  \(end-to-end\)/ { gpu = extract() }
    /GPU  \(kernels\)/    { ker = extract() }
    /Speedup end-to-end/  { sp_e = $NF; sub(/x$/, "", sp_e) }
    /Speedup kernels/     { sp_k = $NF; sub(/x$/, "", sp_k) }
    /result:/             { valid = $2 }
    END {
      printf("| %-22s | %10.1f | %8.2f | %8.2f | %8.2fx | %8.2fx | %s |\n",
             label, cpu, gpu, ker, sp_e, sp_k, valid)
    }' >> "$ROWS"
}

HEADER='| Blur variant           | CPU (ms)   | GPU e2e  | GPU ker  | Speedup e2e | Speedup ker | Valid |'
SEP='|------------------------|------------|----------|----------|-------------|-------------|-------|'

# Cada variante, ao mesmo sigma, p/ comparacao justa (kernels-only e o que muda).
run "tiled (default)"   ""                    --sigma 1.5
run "naive (baseline)"  "BUNNY_BLUR_NAIVE=1"  --sigma 1.5
run "separable (3x1D)"  "BUNNY_SEPARABLE=1"   --sigma 1.5

{
  echo "# BunnyMIP - Blur variants benchmark"
  echo
  echo "- host: \`banana.ua.pt\` (GeForce GTX 1660 Ti, sm_75)"
  echo "- date: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- repeat: $REPEAT (avg; first run discarded as CUDA warm-up)"
  echo "- sigma: 1.5 (same for all variants)"
  echo "- NB: GPU(kernels) = threshold + blur + MIP; threshold/MIP are constant,"
  echo "  so the variation across rows is the blur. Use BUNNY_PERF=1 for the"
  echo "  isolated per-kernel blur time."
  echo
  echo "$HEADER"; echo "$SEP"; cat "$ROWS"
  echo
  echo "<details><summary>Raw program output</summary>"
  echo
  echo '```'
  cat "$RAW"
  echo '```'
  echo "</details>"
} | tee "$OUT"

echo
echo "resultados (markdown) gravados em $OUT"
