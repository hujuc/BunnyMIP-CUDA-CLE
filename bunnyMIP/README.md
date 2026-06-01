# BunnyMIP — Volumetric Maximum Intensity Projection with CUDA

**Large Scale Computing (CLE) — Practical Assignment 3 — Group cle08**

A GPU port of a volumetric rendering pipeline that turns the **Stanford Bunny CT
scan** (a 512×512×361 `uint16` density volume, ~180 MiB) into a 2D image through
three stages: **thresholding**, a **3×3×3 Gaussian blur**, and a rotated
**Maximum Intensity Projection (MIP)**. The CPU reference is kept as a "Gold
Standard"; the CUDA implementation is validated against it and benchmarked for
speedup.

> Developed and profiled on the **`banana.ua.pt`** server (NVIDIA GeForce GTX 1660 Ti,
> Turing `sm_75`, CUDA 10.2).

---

## 1. Build and run

### Requirements
- NVIDIA GPU with compute capability ≥ 7.5 (the makefile targets `-arch=sm_75`).
- CUDA toolkit with `nvcc` on the `PATH` (CUDA 10.2 on banana).
- The CT dataset in a `data/` subdirectory: 361 raw slices named `1`..`361`,
  each 512×512 `uint16` (524 288 bytes). On banana this lives in `~/bunnyMIP/data`.

### Compile
```bash
cd ~/bunnyMIP
export PATH=/usr/local/cuda/bin:$PATH   # if nvcc is not already on the PATH
make                                    # produces ./bunnyMIP
make clean                              # remove objects and the binary
```

### Run
The program loads the volume from the relative path `data/` and writes the two
rasters to `output/`, so **run it from the project root** (`~/bunnyMIP`) where both
directories exist:
```bash
./bunnyMIP [options]
```
It runs the CPU pipeline and the GPU pipeline, prints a performance report and a
validation result, and saves:
- `output/bunnyMIP_cpu.pgm` — host "Gold Standard" raster (16-bit PGM, big-endian).
- `output/bunnyMIP_gpu.pgm` — GPU raster.

### Command-line interface
| Option | Meaning | Default |
|---|---|---|
| `--threshold N` | Intensity floor; every voxel `< N` is set to 0. | `32768` (`1<<15`) |
| `--sigma S` | Standard deviation σ of the Gaussian blur weights. | `1.0` |
| `--roll D` | Roll rotation in **degrees**. | `0` |
| `--pitch D` | Pitch rotation in **degrees**. | `0` |
| `--yaw D` | Yaw rotation in **degrees**. | `0` |
| `--repeat N` | Timed CPU/GPU runs for the speedup analysis. With `N>1` the first run is discarded as CUDA warm-up. | `1` |
| `--frames N` | **Innovation mode:** render N MIPs at evenly-spaced rotations (0–360°) in a **single** kernel call. Writes `output/frame_000.pgm`…. See §6. | `1` |
| `--axis A` | Multi-MIP only: which axis sweeps 0–360° — `roll`\|`pitch`\|`yaw`. The `--roll`/`--pitch`/`--yaw` values on the other two axes set a fixed tilt (orbit). | `yaw` |
| `-h`, `--help` | Show usage and exit. | — |

The rotation angles are combined into a 3×3 rotation matrix `R = Rz·Ry·Rx` that
drives the MIP ray mapping. The same `threshold`, `sigma` and `R` feed **both** the
CPU and GPU pipelines, which is what makes the validation meaningful.

**Examples**
```bash
./bunnyMIP                                  # defaults
./bunnyMIP --threshold 20000 --sigma 1.5    # tune the cleanup / smoothing
./bunnyMIP --yaw 45 --pitch 30              # rotate the view
./bunnyMIP --repeat 10                      # speedup analysis (avg of 9 + warm-up)
```

---

## 2. Pipeline and data layout

The volume is a contiguous 1D array of `uint16_t`, indexed as

```
idx = z * (N*N) + y * N + x        with N = 512 (kBunnySize), M = 361 (kBunnyN)
```

so `x` is the fastest-varying (contiguous) axis. The three stages run in order:

1. **Thresholding** (point operation) — removes low-intensity "air noise":
   `v < threshold ⇒ v = 0`, in place.
2. **3D Gaussian blur** (stencil operation) — each voxel becomes the weighted
   average of its 3×3×3 neighbourhood; weights are `exp(-d²/2σ²)` normalised to
   sum to 1, with **clamped** boundaries. Out-of-place (input ≠ output buffer).
3. **MIP** (reduction along a ray) — for every output pixel, a ray is cast through
   the volume (rotated by `R`, **backward mapping** screen→volume with
   nearest-neighbour sampling) and the **maximum** voxel intensity along it is kept.

---

## 3. The CUDA kernels

All device code lives in [`src/device-bunnyMIP.cu`](src/device-bunnyMIP.cu). The host
entry point `device_bunny_mip()` selects the GPU, allocates VRAM, uploads the
volume and the read-only constants, launches the three kernels, downloads the
raster, and frees everything.

### Memory model
- **Three device buffers:** `d_volume` (volume; thresholded in place),
  `d_blurred` (blur output, out-of-place like the host), `d_raster` (512×512 result).
- **Constant memory** holds the two read-only, broadcast-to-all-threads inputs:
  the 27 Gaussian weights (`c_weights`) and the flattened rotation matrix (`c_R`).
  Constant memory is ideal here because every thread reads the *same* values, which
  the constant cache broadcasts in a single transaction.
- The 27 weights are computed **on the host** (identical math to the reference) and
  uploaded with `cudaMemcpyToSymbol`, so the expensive `exp`/normalisation is done
  once instead of per thread.

### Kernel 1 — `threshold_kernel` (point operation)
A **grid-stride loop** over the `512·512·361 ≈ 9.5×10⁷` voxels: each thread clamps
`volume[i] < threshold ⇒ 0`. The grid-stride pattern decouples the launch
configuration from the data size and keeps memory accesses fully coalesced.
Launched with 256 threads/block.

### Kernel 2 — Gaussian blur (3×3×3 stencil)
One thread per voxel, organised as **8×8×8 blocks** over the 3D grid. All variants
clamp boundaries like the host (`clampi` ↔ `max/min`), are **out-of-place**
(input ≠ output, no read-after-write hazard) and mark inputs `__restrict__`. Three
implementations are present; the **tiled** one is the default and the other two are
selectable at runtime for a before/after comparison (see §5):

- **`blur_kernel_tiled` — shared-memory tiled (default).** Each block cooperatively
  loads its 8³ voxels **plus a 1-voxel halo** into a `10×10×10` `__shared__` tile
  (clamped at the volume edge), `__syncthreads()`, then every thread reads its 27
  neighbours from **fast shared memory** instead of re-fetching from global. With an
  8³ block each voxel is loaded ~once instead of ~27×, which roughly **halves the blur
  cost**. Same accumulation order, weights and `(uint16_t)` truncation as the host →
  **bit-identical** output.
- **`blur_kernel` — naïve global memory (`BUNNY_BLUR_NAIVE=1`).** The straightforward
  27-tap version: each thread reads its 27 neighbours straight from global memory
  (relying on the L1 cache for reuse). The **pre-optimisation baseline**.
- **`blur1d_kernel` — separable 3×(1-D 3-tap) (`BUNNY_SEPARABLE=1`).** Three 1-D
  passes (X→Y→Z), 9 taps instead of 27. Faster in theory, but it truncates to
  `uint16` **between** passes, so it is **not** bit-identical (`max |diff| = 2`, still
  PASS) — kept as an experiment, not the default.

(`BUNNY_PERF=1` prints per-kernel times and occupancy for any variant.)

### Kernel 3 — `mip_kernel` (ray-cast reduction)
One thread per **output pixel**, organised as **16×16 blocks**. Each thread walks
its ray from `-ray_range` to `+ray_range` (`ray_range = 405`, the volume half-diagonal;
**810 steps** span the full diagonal so the ray crosses the volume at any rotation),
applies the rotation `c_R` for backward mapping, bounds-checks the sample, and keeps
the running maximum. Volume reads use
`__ldg` (read-only data cache) since the volume is never written by this kernel.
Per-thread local maxima mean **no inter-thread synchronisation or atomics** are
needed — the reduction is private to each ray.

### Concurrency strategy (summary)
- **Threshold:** embarrassingly parallel, 1 thread ↔ many voxels (grid-stride).
- **Blur:** 1 thread ↔ 1 voxel, independent reads from a single input buffer.
- **MIP:** 1 thread ↔ 1 output pixel, independent ray reductions, no shared state.
- Read-only inputs in **constant memory**; volume reads through the **read-only
  cache** (`__ldg`). No locks, no atomics, no race conditions by construction.

---

## 4. Validation and correctness

After both pipelines run, `main.cpp` compares the GPU raster against the CPU "Gold
Standard" pixel by pixel:

- **Metric:** percentage of pixels whose absolute intensity difference exceeds
  `PIXEL_TOL = 2` (16-bit units); the maximum absolute difference is also reported.
- **Tolerance:** validation **PASSES** if that percentage is `≤ ACCEPT_PCT = 1.0 %`.
- The process exit code is non-zero on FAIL, so the check is scriptable.

**Result:** every benchmarked configuration **PASSES** — fewer than `0.01 %` of pixels
ever exceed the tolerance (well under the `1.0 %` bar). For **axis-aligned** views
(no rotation; the σ and threshold sweeps) the agreement is essentially exact:
`0.0000 %` differing pixels and a **maximum absolute difference of 1**. Under
**rotation** a *handful* of pixels (`< 0.001 %`, e.g. `0.0008 %` at yaw 45° pitch 30°)
can differ by more — up to a few thousand intensity units on those isolated pixels.

This is **not** a port defect: the kernels reproduce the host's arithmetic (same weight
math, boundary clamping, `(uint16_t)` truncation and `ray_range`). The residual comes
from **nearest-neighbour ray sampling** — a sub-ULP floating-point difference in the
rotated coordinate (e.g. GPU fused multiply-add vs the host's separate mul/add) can
cross an integer boundary and pick an *adjacent* voxel of very different intensity.
The effect is inherent to nearest-neighbour MIP, and the percentage-based metric is
exactly what makes the validation robust to it.

---

## 5. Speedup analysis

Measured on `banana.ua.pt` (GeForce GTX 1660 Ti, `sm_75`) over `--repeat 10` runs
(the first discarded as CUDA warm-up; the average of the remaining 9 is reported).
Two GPU figures are given:

- **GPU end-to-end** — wall-clock of `device_bunny_mip`, *including* allocation and
  Host↔Device transfers (`std::chrono`).
- **GPU kernels-only** — the three kernels bracketed by `cudaEvent` timers,
  *excluding* allocation and transfers.

All GPU runs below use the default **shared-memory tiled blur**.

**Speedup (σ sweep, no rotation)**

| Configuration | CPU (ms) | GPU e2e (ms) | GPU kernels (ms) | Speedup e2e | Speedup kernels |
|---|---|---|---|---|---|
| σ = 0.5 | 6615.7 | 33.19 | 15.66 | 199.35× | 422.43× |
| σ = 1.0 | 6859.9 | 33.27 | 15.67 | 206.16× | 437.75× |
| σ = 1.5 | 6868.5 | 33.20 | 15.65 | 206.86× | 438.76× |
| σ = 2.0 | 6585.2 | 33.18 | 15.66 | 198.49× | 420.43× |

**Rotation sweep (σ = 1.0)**

| Configuration | CPU (ms) | GPU e2e (ms) | GPU kernels (ms) | Speedup e2e | Speedup kernels |
|---|---|---|---|---|---|
| yaw = 45 | 6184.7 | 39.20 | 21.52 | 157.79× | 287.33× |
| pitch = 90 | 4708.0 | 33.46 | 15.90 | 140.70× | 296.17× |
| roll=30 pitch=60 yaw=120 | 4913.6 | 43.07 | 25.44 | 114.08× | 193.15× |

**Blur variants (σ = 1.5, no rotation)** — the tiled blur is the default; the other two
are runtime-selectable. Threshold and MIP are constant, so the change in *GPU kernels*
isolates the blur:

| Blur variant | GPU kernels (ms) | Speedup kernels | Δ kernels vs naïve | Bit-exact? |
|---|---|---|---|---|
| **tiled** (default) | **15.66** | **425.41×** | **−8.3 ms** | yes (max \|diff\| = 1) |
| naïve (`BUNNY_BLUR_NAIVE=1`) | 23.94 | 285.39× | — | yes (max \|diff\| = 1) |
| separable (`BUNNY_SEPARABLE=1`) | 23.01 | 289.86× | −0.9 ms | no (max \|diff\| = 2) |

**Reading the numbers**
- With the tiled blur the GPU delivers **~114–207× end-to-end** and **~193–439×
  kernels-only**.
- **Shared-memory tiling is the headline optimisation:** it removes **~8.3 ms** of
  kernel time (the blur stage) — ~35 % of the total — by fetching each voxel ~once from
  the shared tile instead of ~27× from global memory. The separable variant saves
  almost nothing here (still global-bound) *and* is not bit-exact, so it stays an
  experiment.
- The blur is essentially **σ-independent** (the 3×3×3 stencil is the same work for any
  σ; only the precomputed weights change), so its cost is flat across the sweep.
- Rotations cost a little more because oblique rays traverse more in-bounds voxels.
- The gap between kernels-only and end-to-end (~17.5 ms) is the fixed overhead of
  allocating the VRAM buffers and the 180 MiB Host→Device upload — one-shot costs that
  amortise across multiple renders.

Reproduce these numbers with `./bunnyMIP --repeat 10` (one `--sigma`/rotation at a
time); the blur variants are selected with the `BUNNY_BLUR_NAIVE` / `BUNNY_SEPARABLE`
environment toggles.

---

## 6. Innovation: N MIPs in a single kernel call (+ video)

The graded innovation feature computes **N MIP images at different rotations on the
device with one CUDA kernel call**, then turns them into a rotating animation.

### How it works

- A second device entry point, `device_bunny_mip_multi()`, runs the threshold and blur once and reuses the single blurred volume for every frame.
- The N rotation matrices are uploaded together to constant memory (`c_Rmulti[MAX_FRAMES*9]`, with `MAX_FRAMES = 512`; 512·9 floats ≈ 18 KB, well within the 64 KB constant space).
- A single launch of `mip_multi_kernel` uses a 3D grid `(x, y, frame)`: the block z-dimension indexes the frame, so the existing 16×16 pixel tiling is reused and the frame axis is just extra grid layers. Thread `(x,y,f)` casts pixel `(x,y)` of frame `f` through the shared volume using `c_Rmulti[f]` and writes into that frame's slice of a `frames·N·N` output buffer. Each ray is independent, so, no atomics, no synchronisation.
- Per-frame MIP arithmetic is identical to the validated single-frame `mip_kernel` (same ray range, offsets, `__ldg`, nearest-neighbour truncation), so every frame matches what the standard pipeline would produce for that angle.

The `main.cpp` driver (mode entered when `--frames N > 1`) builds the N matrices on the host. `--axis {roll|pitch|yaw}` selects the axis swept over 0-360° (default yaw); the `--roll`/`--pitch`/`--yaw` values on the other two axes define a fixed tilt that orients the spin axis, so the result can be a tilted orbit rather than a flat turntable. Each frame's matrix is the composition `R_frame(f) = R_tilt · R_spin(f)` (a 3×3 `matmul3`), and the stack is saved as `output/frame_000.pgm` … `frame_{N-1}.pgm`. Only the multi-MIP kernel is timed (CUDA events), reported as total and per-frame.

```bash
# 60-frame yaw spin, all rendered in ONE kernel call:
./bunnyMIP --frames 60 --axis yaw
# spin about a different axis (e.g. pitch):
./bunnyMIP --frames 60 --axis pitch
# tilted orbit: spin yaw, but tilt the axis 30° in pitch:
./bunnyMIP --frames 60 --axis yaw --pitch 30
```

### Video generation
The N frames are encoded into a rotating animation with FFmpeg [6]. Because FFmpeg is not installed on `banana`, a self-contained static FFmpeg build is kept in [`viz/`](viz/) on the server (downloaded once, not redistributed with this project).

```bash
# 1) render the N frames (single kernel call) -> output/frame_000.pgm ...
./bunnyMIP --frames 120 --axis yaw
./bunnyMIP --frames 120 --axis yaw --pitch 30   # tilted orbit (optional)

# 2) encode the frames into output/bunny-spin.mp4 (FPS default 30)
bash viz/make-video.sh 30
```
`viz/make-video.sh` uses the static `viz/ffmpeg` (downloading it once if absent), contrast-normalises the 16-bit frames to 8-bit during encoding (`-vf normalize`) so the bunny is visible, and sets `-pix_fmt yuv420p` to keep the H.264 output broadly playable. The result is `output/bunny-spin.mp4`.

### Notes / limits
- `frames` is **clamped to `MAX_FRAMES = 512`** on the device(constant-memory bound). Requesting more renders only the first 512; keep `--frames ≤ 512`.
- Multi-MIP mode does not run the CPU validation path (it shares the already-validated projection math); the standard single-frame mode remains the validated reference.

---

## 7. Architectural decisions

- **`-arch=sm_75`** — targets the GTX 1660 Ti (Turing) on banana and enables `__ldg`
  / current codegen. Change this flag to build for another GPU.
- **Automatic device selection** — `pick_best_device()` enumerates the GPUs and
  picks the most capable one (higher compute capability, then more SMs), so the code
  uses the GTX 1660 Ti rather than the server's secondary GT 710.
- **Constant memory for weights and `R`** — uniform, read-only, tiny: a textbook fit
  for the broadcast constant cache.
- **Out-of-place blur** — a separate output buffer avoids read-after-write hazards
  between neighbouring voxels.
- **Server etiquette** — VRAM is freed at the end of each device run; no allocations
  are left resident.
- **Robust error handling** — every CUDA call is wrapped in a `CUDA_CHECK` macro that
  reports `file:line` and the CUDA error string and aborts on failure;
  `cudaGetLastError()` is checked after each kernel launch.

---

## 8. Notes on the reference code (transparency)

For full academic honesty, two deviations from the untouched base code are recorded:

1. **Scope of changes.** The brief asks for the implementation to live in
   `device-bunnyMIP.cu`; all kernels and device memory management do. However,
   `main.cpp` was necessarily edited to add the required **CLI** (the base code had
   none and hardcoded its parameters) and the benchmark/validation harness.
2. **Bug fixes in the sequential reference**, without which the CLI and the
   validation would be meaningless:
   - `host_bunny_mip` ignored its `threshold`/`sigma` arguments (it hardcoded
     `1<<15` and `1.0`) → fixed to honour the CLI, so host and device stay consistent.
   - `d2r` converted degrees with `π/360` (a half-angle) → corrected to `π/180`.
   - the difference metric computed `abs(h) - d` → corrected to `abs(h - d)`.

   The "Gold Standard" used for validation is therefore the **corrected** sequential
   version, identical in arithmetic to the GPU pipeline.

---

## 9. Project layout

```
bunnyMIP/
├── makefile                 # nvcc build (-arch=sm_75)
├── README.md                # this file
├── data/                    # 361 raw CT slices (on banana only; not in git)
├── output/                  # bunnyMIP_cpu.pgm, bunnyMIP_gpu.pgm, frame_*.pgm
├── viz/                     # make-video.sh (+ static ffmpeg, fetched on first use)
├── include/                 # headers (bunnyIO, imageIO, print, bunnyMIP, device-bunnyMIP)
└── src/
    ├── main.cpp             # CLI, pipeline driver, benchmark loop, validation
    ├── bunnyMIP.cpp         # sequential reference (Gold Standard)
    ├── device-bunnyMIP.cu   # CUDA kernels + device memory management  ← core work
    ├── bunnyIO.cpp          # volume loader
    ├── imageIO.cpp          # 16-bit PGM writer
    └── print.cpp            # printf helpers
```

---

## 10. References

[1] NVIDIA Corporation, *CUDA C++ Programming Guide*, NVIDIA, 2020. [Online].
    Available: https://docs.nvidia.com/cuda/cuda-c-programming-guide/

[2] G. Turk and M. Levoy, "Zippered polygon meshes from range images," in *Proc.
    21st Annu. Conf. Computer Graphics and Interactive Techniques (SIGGRAPH '94)*,
    1994, pp. 311–318. (Origin of the Stanford Bunny model; *The Stanford 3D
    Scanning Repository*, https://graphics.stanford.edu/data/3Dscanrep/.)

[3] J. W. Wallis, T. R. Miller, C. A. Lerner, and E. C. Kleerup, "Three-dimensional
    display in nuclear medicine," *IEEE Trans. Med. Imaging*, vol. 8, no. 4,
    pp. 297–303, Dec. 1989. (Maximum Intensity Projection.)

[4] M. Harris, "CUDA Pro Tip: Write Flexible Kernels with Grid-Stride Loops,"
    NVIDIA Developer Blog, 2013. [Online]. Available:
    https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
    (Grid-stride loop pattern in `threshold_kernel`.)

[5] NVIDIA Corporation, *CUDA Samples* — `Common/helper_cuda.h`
    (`gpuGetMaxGflopsDeviceId`), NVIDIA. [Online]. Available:
    https://github.com/NVIDIA/cuda-samples (Inspiration for the "pick the best
    GPU" heuristic in `pick_best_device`.)

[6] FFmpeg Project, *FFmpeg: A complete, cross-platform solution to record, convert
    and stream audio and video*. [Online]. Available: https://ffmpeg.org/
    Static Linux build (GPL) by J. Van Sickle, https://johnvansickle.com/ffmpeg/

