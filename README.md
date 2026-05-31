# Practical Assignment 3 — Volumetric MIP with CUDA

Porting the *Maximum Intensity Projection* pipeline of the **Stanford Bunny CT scan**
(512×512×361 `uint16` volume) to the GPU with CUDA: **thresholding**, **3×3×3 Gaussian
blur**, and **MIP** with *backward mapping*. Group **cle08**.

> **The full deliverable documentation — build/run, kernel design, validation,
> speedup tables, the innovation feature and references — is in
> [bunnyMIP/README.md](bunnyMIP/README.md)** (this is the README that ships in the
> group's `~/bunnyMIP` home on banana). This top-level file is the repository /
> development-workflow guide.

- 📦 Deliverable README: [bunnyMIP/README.md](bunnyMIP/README.md)
- 📋 Work plan / status: [docs/TODO.md](docs/TODO.md)
- 📄 Assignment brief: [docs/03-practical-assignment.txt](docs/03-practical-assignment.txt)
- 🧠 Base-code analysis: [docs/BASE-CODE-ANALYSIS.md](docs/BASE-CODE-ANALYSIS.md)
- 🔌 Server access (VPN + SSH): [docs/VPN-SETUP.md](docs/VPN-SETUP.md)

## Highlights

- Full GPU pipeline (threshold → blur → MIP) **bit-stable vs the CPU gold standard**
  (0.00 % differing pixels).
- **~93–165× end-to-end** speedup over the sequential reference (**~143–287×**
  kernels-only) on the banana GTX 1660 Ti.
- **Innovation:** N rotated MIPs in a **single CUDA kernel call** (`--frames N`),
  ready to assemble into a spin video with `ffmpeg`.

## Where it runs

Everything is compiled and executed on the **`banana.ua.pt`** server (Ubuntu 18.04,
CUDA 10.2, GeForce GT 710 + GTX 1660 GPUs), reachable **via the UA VPN**. Do not use
VSCode Remote-SSH (shared server) — develop locally and sync.

## Structure

```
bunnyMIP/        local copy of the code (without data/, which stays on banana)
  src/           main.cpp, bunnyMIP.cpp (gold standard), device-bunnyMIP.cu (CUDA kernels)
  include/       headers
  output/        generated rasters / frames / perf-results (git-ignored)
  makefile       build with nvcc (-arch=sm_75)
  README.md      ← deliverable documentation
docs/            assignment brief, TODO, base-code analysis, VPN guide
scripts/         helpers (see below)
```

## Workflow

```bash
# 1) connect to the server (see docs/VPN-SETUP.md)
#    browser: https://go.ua.pt   ->   sudo bash scripts/vpn-up.sh   ->   ssh banana

# 2) edit locally in bunnyMIP/ and sync to banana (does not send data/)
bash scripts/sync-up.sh

# 3) build and run on banana (accepts CLI args)
bash scripts/build-run.sh [--threshold N --sigma S --roll R --pitch P --yaw Y]

# 4) (optional) full speedup sweep, then pull the results back locally
bash scripts/bench.sh 10
bash scripts/sync-down.sh
```

See [bunnyMIP/README.md](bunnyMIP/README.md) for the complete CLI (incl. `--repeat`
and the `--frames` innovation mode) and the local `ffmpeg` step for the spin video.

| Script | Purpose |
|---|---|
| [scripts/vpn-up.sh](scripts/vpn-up.sh) | Fixes the VPN after connecting (split tunnel + DNS + NAT). Run on **every** connection. |
| [scripts/sync-up.sh](scripts/sync-up.sh) | `rsync` the local code → banana (excludes the dataset). |
| [scripts/sync-down.sh](scripts/sync-down.sh) | `rsync` the results (`output/`) banana → local. |
| [scripts/build-run.sh](scripts/build-run.sh) | Build with `nvcc` and run on banana. |
| [scripts/bench.sh](scripts/bench.sh) | Speedup sweep on banana → `output/perf-results.txt`. |

## Status

**Core pipeline, CLI, validation, speedup analysis, the single-kernel multi-MIP
innovation and documentation are complete.** Remaining: optional polish, the
presentation, and the final delivery to `~` on banana. Item-by-item state in
[docs/TODO.md](docs/TODO.md).
