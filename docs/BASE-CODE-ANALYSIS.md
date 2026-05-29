# Base-code analysis (sequential reference)

Study of the provided sequential implementation in [bunnyMIP/](../bunnyMIP/), which is
the **gold standard** the CUDA port must reproduce. This document feeds directly into
the CUDA implementation (see [TODO §2](TODO.md)).

## File map

| File | Role |
|---|---|
| `src/main.cpp` | Entry point: load volume, build `R`, run host + device, compare, save PGMs. |
| `src/bunnyMIP.cpp` | **Sequential reference** of the 3 stages + `host_bunny_mip`. |
| `src/device-bunnyMIP.cu` | **Empty stub** — where the CUDA kernels go (`device_bunny_mip`). |
| `src/bunnyIO.cpp` | `loadBunnyCT` — reads the 361 slices into one array. |
| `src/imageIO.cpp` | `savePGM16` — writes a 16-bit binary PGM (big-endian). |
| `src/print.cpp` | `print` / `format` helpers (printf-style). |
| `include/bunnyIO.h` | Constants `kBunnySize = 512`, `kBunnyN = 361`. |

## 1. Data layout and loading

- Volume dimensions: `N = kBunnySize = 512` (x and y), `M = kBunnyN = 361` (z / slices).
- Stored as a single contiguous 1-D `uint16_t` array of `512 × 512 × 361 = 94,633,984`
  elements ≈ **180 MiB**.
- **Indexing convention** (used everywhere):
  ```
  idx = (size_t) z * N * N  +  y * N  +  x
  ```
  i.e. `x` is contiguous (fastest), then `y`, then `z` (slice). One slice = `N*N` elements.
- `loadBunnyCT("data")` reads files `data/1` … `data/361` (1-indexed). File `i+1`
  is slice `i`, written at byte offset `i * (N*N*2)`. Each file is exactly
  `512*512*2 = 524288` bytes.

## 2. Stage 1 — Thresholding (`apply_threshold`)

```cpp
for i in [0, N*N*M):  if (volume[i] < threshold) volume[i] = 0;
```
- Pure **point operation**, in-place. Removes low-intensity "air noise".
- CUDA: trivially parallel — 1 thread per voxel, `grid = ceil(N*N*M / blockSize)`.

## 3. Stage 2 — 3D Gaussian blur (`gaussian_blur`)

1. Build a `3×3×3` kernel: `weight = exp(-(dx²+dy²+dz²) / (2σ²))` for `dx,dy,dz ∈ {-1,0,1}`.
2. Normalize by the sum of all 27 weights.
3. Convolve: each output voxel = weighted sum of its 3×3×3 neighborhood.
   - **Boundary handling: clamp** (`max(0, min(dim-1, idx))`), i.e. edge voxels are repeated.
   - **Out-of-place**: reads `input`, writes a separate `output` buffer (no read/write race).
   - Result cast to `uint16_t` (truncation).
- CUDA: **stencil operation** — 1 thread per voxel reading 27 neighbors.
  Optimization candidates: the 27 weights depend only on σ → compute once and store in
  **constant memory**; tile the volume into **shared memory** to reuse neighbor reads.

## 4. Stage 3 — Maximum Intensity Projection (`rotated_mip`)

```cpp
ray_range = (int)(sqrt(N*N + N*N + M*M) / 2.0f) + 1;   // = 405 for 512×512×361
for y in [0,N):  for x in [0,N):
    maxIntensity = 0
    u = x - N/2;  v = y - N/2;                 // centered screen coords
    for step in [-ray_range, ray_range):        // ~810 samples along the ray
        w = step
        rotX = R[0]*u + R[1]*v + R[2]*w + N/2
        rotY = R[3]*u + R[4]*v + R[5]*w + N/2
        rotZ = R[6]*u + R[7]*v + R[8]*w + M/2
        if point inside [0,N)×[0,N)×[0,M):
            val = volume[(int)rotZ * N*N + (int)rotY * N + (int)rotX]   // nearest-neighbor
            maxIntensity = max(maxIntensity, val)
    image[y*N + x] = maxIntensity
```
- Simulates parallel rays through the volume; each output pixel keeps the **max** voxel
  along its ray. Output image is `N×N = 512×512`.
- **Backward mapping**: the rotation matrix `R` maps screen-space `(u,v,w)` into
  volume space. Sampling is **nearest-neighbor** via integer truncation (no interpolation).
- CUDA: **embarrassingly parallel** over the 262,144 output pixels — 1 thread per pixel,
  inner loop over ~810 steps, one write per pixel (no atomics needed). The volume is
  read-only → texture / `__ldg` read-only cache are good fits.

## 5. Rotation matrix (`generate_rotation_matrix` in `main.cpp`)

- Builds `R = Rz·Ry·Rx` (yaw·pitch·roll combined), flattened row-major into `R[9]`.
- Has an `inverse` flag that stores the **transpose** (for backward mapping); **`main`
  currently calls it with `inverse=false`** (unused). With 0° angles this is irrelevant,
  but it matters once real rotations are introduced.

## 6. Gold standard and validation (`host_bunny_mip` + `main`)

Host pipeline (the reference image): `copy input → threshold → blur → MIP → h_raster`.

`main` compares device vs host:
```cpp
if (abs(h_raster[i] - d_raster[i]) > 2) diff++;
print("Output difference: %.2f%%", diff/raster_size*100);
```
Both rasters are saved via `savePGM16` (format **P5**, maxval **65535**, **big-endian**
byte order). Current run prints **55.97%** because the device stub leaves `d_raster`
uninitialized.

## 7. ⚠️ Gotchas for the CUDA port

1. **`host_bunny_mip` ignores its `threshold`/`sigma` arguments** — it hardcodes
   `1<<15` (32768) and `1.0`. The validation compares device vs host, so **both must use
   the same values**. Plan: when wiring the CLI (§3), fix the host to use its args too,
   and feed the device the identical values — otherwise the diff check is meaningless.
2. **`d2r` uses `π/360`, not `π/180`** (`angle * M_PI / 360.0`) — effectively **half**
   the usual degrees→radians. It's a base-code quirk; to match the gold standard the
   device path must use the **same** conversion. Flag it; don't silently "fix" it in
   isolation.
3. **Memory budget on device**: volume (180 MiB) + blurred copy (180 MiB) + raster
   (512 KiB) ≈ 360 MiB. Fine on the GTX 1660 (6 GB); threshold can stay in-place to save
   one buffer. Free everything at the end (shared-server etiquette).
4. **Sampling is nearest-neighbor with truncation** and **blur clamps** at boundaries —
   reproduce exactly to keep the diff within the `> 2` tolerance.
5. **Indexing** `z*N*N + y*N + x` and the **centering offsets** (`N/2`, `M/2`) must match
   the reference one-for-one.

## 8. Suggested device data flow (for §2)

```
host input ──H2D──► d_volume ──(threshold, in-place)──► d_volume
                              ──(blur, out-of-place)───► d_blurred
                              ──(MIP, R)──────────────► d_raster ──D2H──► output
```
Recommended target GPU: **GTX 1660 (sm_75)** via `nvcc -arch=sm_75` (also works on the
GT 710 at sm_35). One kernel per stage to start; fuse/optimize later.
