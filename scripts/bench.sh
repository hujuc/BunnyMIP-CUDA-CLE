#!/bin/bash
# =============================================================================
# bench.sh - Speedup analysis on banana: corre o bunnyMIP em varias
# configuracoes (--repeat p/ media) e grava em output/perf-results.txt.
#
# Compila uma vez, varre sigma e rotacoes, e produz tabelas em Markdown (prontas
# p/ o README e legiveis no terminal) + um apendice com a saida completa. Cada
# corrida descarta a 1a iteracao (warm-up CUDA) e reporta avg de CPU, GPU(e2e),
# GPU(kernels) e os speedups.
#
# Uso: bash scripts/bench.sh [REPEAT]   (REPEAT default = 10)
# =============================================================================
set -e
REPEAT="${1:-10}"
OUT="output/perf-results.txt"
RAW="$(mktemp)"          # apendice: saida completa de cada corrida
SIGMA_ROWS="$(mktemp)"   # linhas markdown da varredura de sigma
ROT_ROWS="$(mktemp)"     # linhas markdown da varredura de rotacao
trap 'rm -f "$RAW" "$SIGMA_ROWS" "$ROT_ROWS"' EXIT

# Build once on banana (nvcc no PATH).
ssh banana "export PATH=/usr/local/cuda/bin:\$PATH; cd ~/bunnyMIP && make" >/dev/null

# run <tabela: sigma|rot> <label> <args...>
#   - corre o binario em banana e guarda a saida em $RAW (apendice)
#   - extrai cpu/gpu/kernels/speedups do bloco ">> Performance" via awk
#   - escreve uma linha markdown na tabela escolhida
run() {
  local table="$1" label="$2"; shift 2
  local out
  # Mostra progresso ao vivo no terminal (stderr) e ao mesmo tempo captura a
  # saida para $out, para o awk poder extrair os numeros. O 'tee /dev/stderr'
  # ecoa cada linha enquanto a substituicao de comando preenche $out.
  echo ">>> running: $label  ($*  | repeat=$REPEAT)" >&2
  out="$(ssh banana "export PATH=/usr/local/cuda/bin:\$PATH; cd ~/bunnyMIP && \
    ./bunnyMIP $* --repeat $REPEAT" 2>&1 | tee /dev/stderr)"

  {
    echo "### $label"
    echo "args: $*  | repeat=$REPEAT"
    echo "$out"
    echo
  } >> "$RAW"

  # Parse os campos 'avg' (2a coluna numerica) e os dois speedups.
  local row
  # avg = primeiro campo numerico (a 2a coluna de tempos e o min, ignorado).
  # extract(): apanha o 1o token que parece um numero, robusto ao n.o de campos.
  row="$(printf '%s\n' "$out" | awk -v label="$label" '
    function extract(   i) { for (i=1;i<=NF;i++) if ($i+0==$i && $i!="") return $i; return "" }
    /CPU  \(sequential\)/      { cpu = extract() }
    /GPU  \(end-to-end\)/      { gpu = extract() }
    /GPU  \(kernels\)/         { ker = extract() }
    /Speedup end-to-end/       { sp_e = $NF; sub(/x$/, "", sp_e) }
    /Speedup kernels/          { sp_k = $NF; sub(/x$/, "", sp_k) }
    /result:/                  { valid = $2 }
    END {
      printf("| %-26s | %10.1f | %8.2f | %8.2f | %8.2fx | %8.2fx | %s |\n",
             label, cpu, gpu, ker, sp_e, sp_k, valid)
    }')"

  if [ "$table" = "sigma" ]; then echo "$row" >> "$SIGMA_ROWS"
  else                            echo "$row" >> "$ROT_ROWS"; fi
}

# Cabecalho comum das tabelas markdown.
HEADER='| Configuration              | CPU (ms)   | GPU e2e  | GPU ker  | Speedup e2e | Speedup ker | Valid |'
SEP='|----------------------------|------------|----------|----------|-------------|-------------|-------|'

# --- Varredura -------------------------------------------------------------
run sigma "sigma=0.5"            --sigma 0.5
run sigma "sigma=1.0"            --sigma 1.0
run sigma "sigma=1.5"            --sigma 1.5
run sigma "sigma=2.0"            --sigma 2.0
run rot   "yaw=45"               --sigma 1.0 --yaw 45
run rot   "pitch=90"             --sigma 1.0 --pitch 90
run rot   "roll=30 pitch=60 yaw=120" --sigma 1.0 --roll 30 --pitch 60 --yaw 120

# --- Saida final (markdown, via tee p/ terminal + ficheiro) ----------------
{
  echo "# BunnyMIP - Performance / Speedup Analysis"
  echo
  echo "- host: \`banana.ua.pt\` (GeForce GTX 1660 Ti, sm_75)"
  echo "- date: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "- repeat: $REPEAT (avg reported; first run discarded as CUDA warm-up)"
  echo "- speedup = CPU_avg / GPU_avg"
  echo
  echo "## Gaussian blur cost (sigma sweep)"
  echo
  echo "$HEADER"; echo "$SEP"; cat "$SIGMA_ROWS"
  echo
  echo "## Rotation sweep (sigma=1.0)"
  echo
  echo "$HEADER"; echo "$SEP"; cat "$ROT_ROWS"
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
