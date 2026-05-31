# Practical Assignment 3 — Volumetric MIP with CUDA

Porting the *Maximum Intensity Projection* pipeline of the **Stanford Bunny CT scan**
(512×512×361 `uint16` volume) to the GPU with CUDA: **thresholding**, **3×3×3 Gaussian
blur**, and **MIP** with *backward mapping*. Group **cle08**.

- 📋 Work plan: [docs/TODO.md](docs/TODO.md)
- 📄 Assignment brief: [docs/03-practical-assignment.txt](docs/03-practical-assignment.txt)
- 🔌 Server access (VPN + SSH): [docs/VPN-SETUP.md](docs/VPN-SETUP.md)

## Where it runs

Everything is compiled and executed on the **`banana.ua.pt`** server (Ubuntu 18.04,
CUDA 10.2, GeForce GT 710 + GTX 1660 GPUs), reachable **via the UA VPN**. Do not use
VSCode Remote-SSH (shared server) — develop locally and sync.

## Structure

```
bunnyMIP/        local copy of the code (without data/, which stays on banana)
  src/           main.cpp, bunnyMIP.cpp (gold standard), device-bunnyMIP.cu (CUDA kernels)
  include/       headers
  makefile       build with nvcc
docs/            assignment brief, TODO, VPN guide
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

# 4) bring the result images back and view them (output/bunnyMIP_{cpu,gpu}.pgm)
bash scripts/sync-down.sh

# 5) speedup analysis: sweeps sigma/rotations, writes output/perf-results.txt
bash scripts/bench.sh [REPEAT]            # REPEAT default = 10

# 6) spinning-bunny video: N MIPs in one kernel call -> bunny-spin.mp4
bash scripts/make-video.sh --frames 120 --fps 30
```

| Script | Purpose |
|---|---|
| [scripts/vpn-up.sh](scripts/vpn-up.sh) | Fixes the VPN after connecting (split tunnel + DNS + NAT). Run on **every** connection. |
| [scripts/sync-up.sh](scripts/sync-up.sh) | `rsync` the local code → banana (excludes the dataset). |
| [scripts/sync-down.sh](scripts/sync-down.sh) | `rsync` the results (`output/`) banana → local. |
| [scripts/build-run.sh](scripts/build-run.sh) | Build with `nvcc` and run on banana. |
| [scripts/bench.sh](scripts/bench.sh) | Speedup analysis sweep → Markdown tables in `output/perf-results.txt`. |
| [scripts/make-video.sh](scripts/make-video.sh) | Orchestrates the rotation video: generate N frames on banana, encode via `bunnyMIP/viz/`, fetch `bunny-spin.mp4` back. |

## Status

Next steps in [docs/TODO.md](docs/TODO.md).
