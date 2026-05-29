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

## 1. Understand the sequential base code (`src/bunnyMIP`)

- [ ] Study how the volume is loaded into memory (1D `uint16_t` array, `(x,y,z)` indexing).
- [ ] Study the sequential implementation of:
  - [ ] Thresholding (point operation).
  - [ ] 3D Gaussian blur (3×3×3 stencil).
  - [ ] Maximum Intensity Projection (MIP) with backward mapping.
- [ ] Identify how the host "Gold Standard" image is generated (for validation).

---

## 2. CUDA implementation — `src/device-bunnyMIP.cu`

> **All implementation work must be contained in `src/device-bunnyMIP.cu`.**
> Write the kernel definitions, device (VRAM) memory allocation, and manage Host↔Device data transfers.

### 2.1. Memory management and transfers
- [ ] Allocate the volume in VRAM (device).
- [ ] Transfer the volume Host → Device.
- [ ] Transfer the result (MIP raster) Device → Host.
- [ ] Free device memory (avoid leaving large allocations active when not debugging — server etiquette).

### 2.2. Kernel 1 — Thresholding (Point Operation)
- [ ] Implement a kernel that sets `v = 0` for every voxel `v < threshold`.
- [ ] Parameterize via the `--threshold` argument.

### 2.3. Kernel 2 — 3D Gaussian Blur (Stencil Operation)
- [ ] Implement the spatial convolution: each voxel = weighted average of its 3×3×3 neighborhood.
- [ ] Compute the Gaussian weights from `--sigma`.
- [ ] Handle the volume boundaries correctly (boundary handling).
- [ ] (Consider using shared memory / constant memory for the weights — see optimizations.)

### 2.4. Kernel 3 — Maximum Intensity Projection (MIP)
- [ ] Implement the projection: for each screen pixel, find the maximum voxel intensity along the corresponding parallel ray.
- [ ] Build the 3×3 rotation matrix `R` from `--roll`, `--pitch`, `--yaw` (degrees).
- [ ] Use `R` for backward mapping (map each pixel/ray back into the volume).
- [ ] Update the kernels at runtime according to the received parameters.

---

## 3. Command Line Interface (CLI)

- [ ] Parse and accept the parameters:
  - [ ] `--threshold` — intensity floor (voxel `< threshold` → 0).
  - [ ] `--sigma` — standard deviation (σ) for the Gaussian weights.
  - [ ] `--roll` — rotation in degrees.
  - [ ] `--pitch` — rotation in degrees.
  - [ ] `--yaw` — rotation in degrees.
- [ ] Ensure these parameters actually drive the final render at runtime.

---

## 4. Validation and correctness

- [ ] Implement the validation check: compare the raster produced by the CUDA kernels against the host "Gold Standard" image.
- [ ] Ensure the device MIP **does not differ significantly** from the host-computed version.
- [ ] Define the comparison metric/tolerance and report the validation result.

---

## 5. Performance evaluation — Speedup Analysis

- [ ] Measure the execution time of the sequential version (CPU reference).
- [ ] Measure the execution time of the CUDA version (GPU).
- [ ] Compute the speedup and demonstrate a measurable performance gain.
- [ ] Collect profiling results obtained **on the `banana.ua.pt` server**.
- [ ] Save the performance-analysis results (time comparisons + speedup calculations) for the report/README.

---

## 6. Additional features (Innovation — 15%) [OPTIONAL but graded]

- [ ] **N MIP raster images in a single kernel call:** compute N MIP images with different rotations on the device with a **single** CUDA kernel call.
- [ ] **Video generation:** generate a video from the N generated raster images using an external tool (e.g. `ffmpeg`).

---

## 7. Documentation — README.md (15%)

> The README.md must live in the group's home directory on `banana.ua.pt` and document the architectural decisions and concurrency strategies.

- [ ] **Build and run instructions** for the application.
- [ ] **Explanation of the developed CUDA kernels** (threshold, blur, MIP).
- [ ] **Speedup analysis** of the code (execution-time comparisons).
- [ ] Architectural decisions and concurrency strategies employed.
- [ ] Usage details and a summary of the approach.
- [ ] Any other information the group considers relevant.
- [ ] Technical documentation quality (5%).
- [ ] **IEEE-format citations** for any external code/concepts (zero-tolerance plagiarism policy).

---

## 8. Shared-server etiquette (banana.ua.pt)

- [ ] Responsible use of VRAM and CPU; do not leave large allocations active outside debugging.
- [ ] **Do NOT use the VSCode Remote-SSH extension** (forbidden — Node.js zombie processes).
- [ ] Edit with lightweight tools (vim/nano) **or** develop locally and sync via `scp`/`rsync`/Git/SFTP.

---

## 9. Presentation (20%) — MANDATORY

- [ ] Prepare the presentation (5 min): overview of the implemented CUDA kernels.
- [ ] Prepare the live demo + Q&A (10 min): software demonstration and answers to technical questions.
- [ ] Total duration strictly limited to 15 minutes.

---

## 10. Deliverable

- [ ] Ensure **all source files** are in the group's home directory on `banana.ua.pt`.
- [ ] Include a `README.md` with setup, usage, and a summary of the approach.
- [ ] Include the performance-analysis results (times + speedup).
- [ ] Confirm delivery **before June 1, 09:00** (no late submissions accepted).
- [ ] Ensure the codebase is the group's unique, original creation (academic integrity).

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
