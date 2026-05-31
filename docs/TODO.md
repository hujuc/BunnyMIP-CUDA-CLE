# TODO — Practical Assignment 3: Volumetric MIP with CUDA

> Based on the assignment brief in [docs/03-practical-assignment.txt](docs/03-practical-assignment.txt).
> Goal: port the sequential volumetric processing pipeline (Stanford Bunny CT scan) to the GPU with CUDA.
> **Deadline: June 1, 09:00.** Delivered in the group's home directory on `banana.ua.pt`.

---

## 0. Setup and environment — ✅ DONE (2026-05-29)

- [x] `banana.ua.pt` server credentials: group **cle08**.
- [x] UA VPN access established (Check Point/SNX via ChrootVPN) + working **split tunnel** — see section 8 and [scripts/vpn-up.sh](../scripts/vpn-up.sh).
- [x] SSH access confirmed (key installed, `ssh banana` shortcut). `banana.ua.pt` = `192.168.8.210`.
- [x] Base code obtained: lives in `~/bunnyMIP` on banana; local copy in [bunnyMIP/](../bunnyMIP/) (without the dataset).
- [x] Dataset confirmed: **361 slices** in `~/bunnyMIP/data` (files `1`..`361`, 524288 B each = 512×512×`uint16`). Loads 180 MiB. Stays **on banana only**.
- [x] Toolchain validated: **CUDA 10.2** (`/usr/local/cuda/bin/nvcc`), builds and runs (current diff 55.97% — empty device, as expected).
- [x] Workflow defined (no VSCode Remote-SSH): edit locally → [scripts/sync-up.sh](../scripts/sync-up.sh) (rsync, excludes `data/`) → [scripts/build-run.sh](../scripts/build-run.sh) (build+run on banana).

**banana environment:** Ubuntu 18.04, 2 GPUs — **GeForce GT 710** (2 GB, sm_35) and **GeForce GTX 1660** (6 GB, sm_75); driver 440.33; CUDA 10.2. **No `ffmpeg`** (affects the optional video feature — see §6).

**Base-code findings (relevant to §2/§3):**
- `device-bunnyMIP.cu` is an **empty stub** (`device_bunny_mip` only prints; it never fills `d_raster`).
- `bunnyMIP.cpp` holds the **complete** sequential implementation (gold standard): `apply_threshold`, `gaussian_blur` (3×3×3, clamped boundaries), `rotated_mip` (backward mapping).
- ⚠️ **There is no CLI parsing** — `main.cpp` hardcodes angles/threshold/sigma and `host_bunny_mip` ignores its args (uses fixed `1<<15` and `1.0`). The §3 CLI must be built from scratch.
- Validation already present in `main.cpp`: counts pixels with `|cpu-gpu| > 2` and prints the difference %.

---

## 1. Understand the sequential base code (`src/bunnyMIP`) — ✅ DONE (2026-05-30)

Full write-up: [docs/BASE-CODE-ANALYSIS.md](BASE-CODE-ANALYSIS.md).

- [x] Study how the volume is loaded into memory (1D `uint16_t` array, `(x,y,z)` indexing).
  → 512×512×361 `uint16` (~180 MiB), `idx = z*N*N + y*N + x` (x contiguous); slices read from `data/1..361`.
- [x] Study the sequential implementation of:
  - [x] Thresholding (point operation) — in-place `if (v < threshold) v = 0`.
  - [x] 3D Gaussian blur (3×3×3 stencil) — `exp(-d²/2σ²)` weights, normalized, clamped boundaries, out-of-place.
  - [x] Maximum Intensity Projection (MIP) with backward mapping — ray per pixel, `R·(u,v,w)+offsets`, nearest-neighbor max over ~810 steps.
- [x] Identify how the host "Gold Standard" image is generated (for validation).
  → `host_bunny_mip` (threshold→blur→MIP); `main` counts pixels with `|cpu-gpu|>2`; PGM16 big-endian output.

**Key gotchas found** (detail in the analysis): `host_bunny_mip` ignores its `threshold`/`sigma` args (hardcodes `1<<15`, `1.0`); `d2r` uses `π/360` not `π/180`; sampling is nearest-neighbor with truncation; device+host must use identical params for the diff check to be meaningful.

---

## 2. CUDA implementation — `src/device-bunnyMIP.cu`

> **All implementation work must be contained in `src/device-bunnyMIP.cu`.**
> Write the kernel definitions, device (VRAM) memory allocation, and manage Host↔Device data transfers.

### 2.1. Memory management and transfers — ✅ DONE (2026-05-30)
- [x] Allocate the volume in VRAM (device) — `d_volume`, `d_blurred`, `d_raster`.
- [x] Transfer the volume Host → Device.
- [x] Transfer the result (MIP raster) Device → Host.
- [x] Free device memory (avoid leaving large allocations active when not debugging — server etiquette).
- [x] Extras: `CUDA_CHECK` error macro; `pick_best_device()` selects the GTX 1660 Ti (sm_75) over the GT 710. Validated on banana (compiles + runs clean; raster zero-filled until kernels exist).

### 2.2. Kernel 1 — Thresholding (Point Operation) — ✅ DONE (2026-05-30)
- [x] Implement a kernel that sets `v = 0` for every voxel `v < threshold` (`threshold_kernel`, grid-stride).
- [x] Parameterize via the `threshold` argument (CLI wiring is §3).

### 2.3. Kernel 2 — 3D Gaussian Blur (Stencil Operation) — ✅ DONE (2026-05-30)
- [x] Implement the spatial convolution: each voxel = weighted average of its 3×3×3 neighborhood (`blur_kernel`, 8×8×8 blocks, out-of-place).
- [x] Compute the Gaussian weights from `sigma` (same math as the host).
- [x] Handle the volume boundaries correctly (clamp, like the host).
- [x] Use **constant memory** for the 27 weights (`c_weights`). Shared-memory tiling left as a §5 optimization.

### 2.4. Kernel 3 — Maximum Intensity Projection (MIP) — ✅ DONE (2026-05-30)
- [x] Implement the projection: for each screen pixel, max voxel intensity along the ray (`mip_kernel`, 16×16 blocks, `__ldg`).
- [x] Build the 3×3 rotation matrix `R` from `--roll`, `--pitch`, `--yaw` (degrees) — done via the CLI (§3); kernel consumes `R`.
- [x] Use `R` (constant memory `c_R`) for backward mapping (map each pixel/ray back into the volume).
- [x] Kernels parameterized at runtime by `threshold`/`sigma`/`R` (driven by the CLI in §3).

**Result:** full GPU pipeline matches the host gold standard — **`Output difference: 0.00%`**. Required `-arch=sm_75` in the makefile (for `__ldg` / Turing codegen).

---

## 3. Command Line Interface (CLI) — ✅ DONE (2026-05-30)

- [x] Parse and accept the parameters (in `main.cpp`):
  - [x] `--threshold` — intensity floor (voxel `< threshold` → 0).
  - [x] `--sigma` — standard deviation (σ) for the Gaussian weights.
  - [x] `--roll` — rotation in degrees.
  - [x] `--pitch` — rotation in degrees.
  - [x] `--yaw` — rotation in degrees.
- [x] Ensure these parameters actually drive the final render at runtime (defaults match the old hardcoded values; `--help`; invalid-option handling).
- [x] Fixed `host_bunny_mip` to use its `threshold`/`sigma` args (was hardcoded) so host and device stay consistent → diff stays **0.00%** for all params.
- [x] Side fixes: `d2r` corrected to `π/180` (was `π/360`, half-angle); diff metric in `main.cpp` corrected to `abs(h - d)` (was `abs(h) - d`).

---

## 4. Validation and correctness — ✅ DONE (2026-05-31, branch `dev/luis`)

- [x] Implement the validation check: compare the raster produced by the CUDA kernels against the host "Gold Standard" image (in [main.cpp](../bunnyMIP/src/main.cpp)).
- [x] Ensure the device MIP **does not differ significantly** from the host-computed version → **PASS** in every benchmarked config (see §5).
- [x] Define the comparison metric/tolerance and report the validation result.
  → metric: % of pixels with `|cpu-gpu| > PIXEL_TOL` (`PIXEL_TOL = 2`); also tracks the max abs diff. Passes if that % `≤ ACCEPT_PCT` (`1.0%`). Prints `PASS/FAIL`; `main` returns non-zero on FAIL (scriptable).

**Verified correct (port reviewed kernel-by-kernel against the host reference):**
- Threshold, blur (weights/indexing/clamp/accumulation order/`(uint16_t)` truncation) and MIP (`ray_range`=405, `R`, offsets, nearest-neighbor truncation) are a **faithful, bit-stable port** → observed diff **0.00%**, max abs diff 0.
- ⚠️ **To document in the README (transparency / academic integrity):** (a) work is not 100% confined to `device-bunnyMIP.cu` — `main.cpp` was necessarily edited for the CLI (§3) and the benchmark/validation harness; (b) the team fixed real bugs in the *reference* code (host ignored its `threshold`/`sigma`; `d2r` was π/360; broken diff metric) so the "Gold Standard" is the corrected sequential version. Both deviations are defensible but must be stated.

---

## 5. Performance evaluation — Speedup Analysis — ✅ DONE (2026-05-31, branch `dev/luis`)

- [x] Measure the execution time of the sequential version (CPU reference) — `std::chrono` wall-clock in [main.cpp](../bunnyMIP/src/main.cpp).
- [x] Measure the execution time of the CUDA version (GPU) — two figures: **end-to-end** (`std::chrono`, incl. malloc + H2D/D2H + free) and **kernels-only** (`cudaEvent`, via the new `kernels_ms` out-param in `device_bunny_mip`).
- [x] Compute the speedup and demonstrate a measurable performance gain → **~93–165× end-to-end**, **~143–287× kernels-only**.
- [x] Collect profiling results obtained **on the `banana.ua.pt` server** (GeForce GTX 1660 Ti, sm_75) via [scripts/bench.sh](../scripts/bench.sh) (sigma sweep + rotation sweep; `--repeat`, first run discarded as CUDA warm-up; avg+min reported).
- [x] Save the performance-analysis results (time comparisons + speedup calculations) for the report/README → [output/perf-results.txt](../output/perf-results.txt) (Markdown tables ready for the README).
- [x] Added [scripts/sync-down.sh](../scripts/sync-down.sh) to pull results (`output/*.pgm`) banana → local.

---

## 6. Additional features (Innovation — 15%) [OPTIONAL but graded] — ✅ N-MIP DONE (2026-05-31)

- [x] **N MIP raster images in a single kernel call:** `device_bunny_mip_multi` + `mip_multi_kernel` ([device-bunnyMIP.cu](../bunnyMIP/src/device-bunnyMIP.cu)). Threshold+blur run once; the blurred volume is reused; the **3D grid `(x,y,frame)`** renders all N MIPs in **one launch**, each frame using `c_Rmulti[f]` (N matrices in constant memory, `MAX_FRAMES=512`). CLI: `--frames N` sweeps one axis 0–360° (default yaw) → `output/frame_000.pgm…`. Per-frame math identical to the validated `mip_kernel`.
  - Verified: single kernel call, different rotations, on-device — meets the brief. Caveats (documented in README §6): `frames` clamped to 512 (constant-mem bound); multi-MIP mode skips CPU validation (shares the validated projection math).
- [~] **Video generation:** frames are written sequentially as `frame_%03d.pgm`; **`ffmpeg` is NOT on banana**, so the video is assembled **locally** after `sync-down.sh` (command documented in README §6). Optional: actually produce/commit a sample `bunny_spin.mp4`.

---

## 7. Documentation — README.md (15%) — ✅ DONE (2026-05-31)

> The README.md must live in the group's home directory on `banana.ua.pt` and document the architectural decisions and concurrency strategies.
> Written in [bunnyMIP/README.md](../bunnyMIP/README.md) (this is the file that syncs to `~/bunnyMIP` on banana = the deliverable README). The root [README.md](../README.md) stays as the dev-workflow doc.

- [x] **Build and run instructions** for the application (§1: requirements, `make`, run from `~/bunnyMIP`, CLI table + examples).
- [x] **Explanation of the developed CUDA kernels** (§3: threshold grid-stride, blur 8×8×8/constant weights, MIP 16×16/`c_R`/`__ldg`).
- [x] **Speedup analysis** of the code (§5: σ sweep + rotation sweep tables, ~93–165× e2e / ~143–287× kernels, from `output/perf-results.txt`).
- [x] Architectural decisions and concurrency strategies employed (§3 "concurrency strategy" + §6 design decisions + memory model).
- [x] Usage details and a summary of the approach (§1 CLI + §2 pipeline overview).
- [x] Any other information the group considers relevant (§7 transparency note on reference-code fixes).
- [x] Technical documentation quality (5%) — structured, cross-linked, data-backed.
- [x] **IEEE-format citations** for external code/concepts (§9: CUDA Guide, Stanford Bunny/Turk&Levoy, MIP/Wallis et al.).

> ⚠️ Pending when the §6 innovation feature lands: add an N-MIP / video section to the README and cite `ffmpeg` if used.

---

## 8. Shared-server etiquette (banana.ua.pt) — ✅ DONE (ongoing practice)

- [x] Responsible use of VRAM and CPU; do not leave large allocations active outside debugging.
  → every device run frees all `cudaMalloc` buffers before returning (`device_bunny_mip` / `device_bunny_mip_multi`).
- [x] **Do NOT use the VSCode Remote-SSH extension** (forbidden — Node.js zombie processes). → never used; we develop locally.
- [x] Edit with lightweight tools (vim/nano) **or** develop locally and sync via `scp`/`rsync`/Git/SFTP.
  → local edit + [scripts/sync-up.sh](../scripts/sync-up.sh) (rsync, excludes `data/`) + Git.

---

## 9. Presentation (20%) — MANDATORY — ⏳ TODO (group task, not in repo)

- [ ] Prepare the presentation (5 min): overview of the implemented CUDA kernels.
  → source material ready: README §3 (kernels + concurrency) and §6 (innovation).
- [ ] Prepare the live demo + Q&A (10 min): software demonstration and answers to technical questions.
  → demo script: `./bunnyMIP --repeat 10` (speedup + validation), then `./bunnyMIP --frames 60` (single-kernel multi-MIP) + local `ffmpeg` spin video. Q&A prep: grid-stride, constant memory, `__ldg`, out-of-place blur, 3D grid for multi-MIP.
- [ ] Total duration strictly limited to 15 minutes.

---

## 10. Deliverable — ⏳ MOSTLY READY (final sync + confirmation pending)

- [x] Include a `README.md` with setup, usage, and a summary of the approach → [bunnyMIP/README.md](../bunnyMIP/README.md) (ships in `~/bunnyMIP`).
- [x] Include the performance-analysis results (times + speedup) → [output/perf-results.txt](../output/perf-results.txt) + README §5 tables.
- [x] Ensure the codebase is the group's unique, original creation (academic integrity) → original implementation; transparency note (README §8) on the reference-code fixes and `main.cpp` scope; IEEE citations (README §10).
- [ ] Ensure **all source files** are in the group's home directory on `banana.ua.pt` → run `bash scripts/sync-up.sh` once more. It mirrors `bunnyMIP/` → `~/bunnyMIP` (excludes `data/`, `output/*`, `obj/`, the binary), so `README.md` and all sources are included. ⚠️ `--delete` mirrors: make sure nothing needed lives only on banana.
- [ ] Confirm delivery **before June 1, 09:00** (no late submissions accepted) → **final check pending**.

---

## Grading weights summary (reference)

| Component | Weight |
|---|---|
| Functionality (all core features working) | 55% |
| Code Quality (correct CUDA usage 10% + readability/comments 5%) | 15% |
| Documentation (README 10% + technical quality 5%) | 15% |
| Innovation (additional features — section 6) | 15% |
| **Technical implementation (total)** | **80%** |
| **Presentation** | **20%** |
