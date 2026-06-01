// =============================================================================
// device-bunnyMIP.cu - CUDA port of the volumetric MIP pipeline (Group cle08).
//
// All code in this file is the group's original work. It implements the four
// device kernels (threshold, 3x3x3 blur, MIP, multi-MIP), the read-only constant
// uploads, GPU selection, and the Host<->Device memory management.
//
// External concepts/idioms used (cited per the academic-integrity policy; see the
// numbered references in ../README.md, section "References"):
//   - Grid-stride loop pattern in threshold_kernel ............... [4]
//   - "Pick the best GPU" heuristic in pick_best_device() ........ [5] (CUDA Samples)
//   - Constant memory + __ldg read-only cache + CUDA-event timing  [1] (CUDA Guide)
// The Euler rotation matrix consumed by the MIP kernels is built by the provided
// base code (generate_rotation_matrix in main.cpp), not in this file.
// =============================================================================

#include <cstdint>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

#include "print.h"
#include "bunnyIO.h"
#include "device-bunnyMIP.h"

// =============================================================================
// Performance-debug instrumentation (per-kernel + transfer timings, occupancy).
// Off by default; enable at runtime with  BUNNY_PERF=1 ./bunnyMIP ...
// Kept out of the normal/validation path so the reported speedup numbers (which
// use the coarse kernels-only event window) are unaffected.
// =============================================================================
static bool perf_enabled()
{
    static int v = -1;                       // cache the env lookup
    if (v < 0) { const char* e = getenv("BUNNY_PERF"); v = (e && *e && *e != '0'); }
    return v == 1;
}

// Separable-blur experiment toggle: BUNNY_SEPARABLE=1 swaps the 27-tap tiled
// blur for three 1D passes (X->Y->Z). Off by default (keeps bit-exact blur).
static bool separable_enabled()
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("BUNNY_SEPARABLE"); v = (e && *e && *e != '0'); }
    return v == 1;
}

// Naive-blur toggle: BUNNY_BLUR_NAIVE=1 uses the original global-memory 27-tap
// blur_kernel (no shared-memory tiling) — the pre-optimization baseline, kept
// for before/after comparison. Off by default (tiled blur is the default).
static bool blur_naive_enabled()
{
    static int v = -1;
    if (v < 0) { const char* e = getenv("BUNNY_BLUR_NAIVE"); v = (e && *e && *e != '0'); }
    return v == 1;
}

// Scoped CUDA-event stopwatch: times the GPU work between start() and stop()
// and prints "[perf] <label>: X.XXX ms" when perf is enabled. A no-op otherwise.
struct GpuTimer {
    cudaEvent_t a{}, b{};
    bool on;
    GpuTimer() : on(perf_enabled()) {
        if (on) { cudaEventCreate(&a); cudaEventCreate(&b); }
    }
    ~GpuTimer() { if (on) { cudaEventDestroy(a); cudaEventDestroy(b); } }
    void start() { if (on) cudaEventRecord(a); }
    void stop(const char* label) {
        if (!on) return;
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0.0f; cudaEventElapsedTime(&ms, a, b);
        print("  [perf] %-12s %8.3f ms\n", label, ms);
    }
};

// Print a kernel's launch config and theoretical occupancy (warps/SM).
static void perf_launch(const char* label, const void* kernel,
                        dim3 grid, dim3 block)
{
    if (!perf_enabled()) return;
    int threadsPerBlock = block.x * block.y * block.z;
    int maxBlocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks, kernel,
                                                  threadsPerBlock, 0);
    cudaDeviceProp p; int dev; cudaGetDevice(&dev); cudaGetDeviceProperties(&p, dev);
    int warpsPerBlock  = (threadsPerBlock + p.warpSize - 1) / p.warpSize;
    int activeWarps    = maxBlocks * warpsPerBlock;
    int maxWarps       = p.maxThreadsPerMultiProcessor / p.warpSize;
    float occ = maxWarps ? (100.0f * activeWarps / maxWarps) : 0.0f;
    print("  [perf] %-12s grid=(%u,%u,%u) block=(%u,%u,%u) occ=%.0f%% (%d/%d warps)\n",
          label, grid.x, grid.y, grid.z, block.x, block.y, block.z,
          occ, activeWarps, maxWarps);
}

// =============================================================================
// Error-checking wrapper for CUDA Runtime API calls.
// Aborts with file:line and the CUDA error string on failure.
// =============================================================================
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            print("CUDA error at %s:%d -> %s\n", __FILE__, __LINE__,           \
                  cudaGetErrorString(err_));                                   \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

// Read-only data broadcast to every thread (cheap, cached):
//   c_weights - the 27 normalized 3x3x3 Gaussian weights
//   c_R       - the flattened 3x3 rotation matrix
__constant__ float c_weights[27];
__constant__ float c_R[9];

// Separable-blur experiment (BUNNY_SEPARABLE=1): the 3-tap normalized 1D Gaussian
// applied independently along X, then Y, then Z. NB: a separable pass truncates
// to uint16 between axes, so the result is NOT bit-identical to the 27-tap blur.
__constant__ float c_weights1d[3];

// Innovation #1: up to MAX_FRAMES rotation matrices for the multi-MIP kernel,
// stored back-to-back (matrix f at c_Rmulti + f*9). Constant memory is 64 KB;
// 512 * 9 floats = 18 KB, leaving room for c_weights/c_R.
#define MAX_FRAMES 512
__constant__ float c_Rmulti[MAX_FRAMES * 9];

// Clamp helper for the blur boundary handling (mirrors the host's max/min).
__device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// =============================================================================
// Kernel 1 - Thresholding (point operation).
// Sets every voxel below `threshold` to 0. Embarrassingly parallel: each voxel
// is independent. Uses a grid-stride loop [4] so any launch configuration covers
// the whole `n`-element volume (each thread handles indices i, i+stride, ...),
// keeping global-memory accesses coalesced regardless of grid size.
// =============================================================================
__global__ void threshold_kernel(uint16_t* volume, size_t n, uint16_t threshold)
{
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        if (volume[i] < threshold)
            volume[i] = 0;
    }
}

// =============================================================================
// Kernel 2 - 3D Gaussian blur (3x3x3 stencil).
// One thread per voxel, indexed from a 3D grid. Each thread computes the weighted
// average of its 27-voxel (3x3x3) neighborhood and writes one output voxel.
//   - Out-of-place (input != output): neighbours are read from the unmodified
//     input, so there is no read-after-write hazard between threads.
//   - Boundary voxels clamp their neighbour coordinates (clampi), matching the
//     host reference exactly so the result is bit-identical.
//   - The 27 weights live in constant memory (c_weights): every thread reads the
//     same values, broadcast from the constant cache in a single transaction.
//   - `__restrict__` promises input/output do not alias, enabling the compiler to
//     keep the running accumulator in a register.
// The (uint16_t) store truncates toward zero, identical to the host's cast.
// =============================================================================
__global__ void blur_kernel(const uint16_t* __restrict__ input,
                            uint16_t* __restrict__ output, int N, int M)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= N || y >= N || z >= M) return;   // guard the ragged edge of the grid

    float acc = 0.0f;
    for (int dz = -1; dz <= 1; ++dz) {
        int nz = clampi(z + dz, 0, M - 1);
        for (int dy = -1; dy <= 1; ++dy) {
            int ny = clampi(y + dy, 0, N - 1);
            for (int dx = -1; dx <= 1; ++dx) {
                int nx = clampi(x + dx, 0, N - 1);
                uint16_t voxel = input[(size_t)nz * N * N + ny * N + nx];
                acc += (float)voxel * c_weights[(dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)];
            }
        }
    }
    output[(size_t)z * N * N + y * N + x] = (uint16_t)acc;
}

// =============================================================================
// Kernel 2b - 3D Gaussian blur, SHARED-MEMORY TILED (optimization).
// Each (BX*BY*BZ) block cooperatively loads its voxels plus a 1-voxel halo into
// a (BX+2)(BY+2)(BZ+2) shared tile (clamped boundaries), then every thread reads
// its 27 neighbours from fast shared memory instead of re-fetching from global.
// Produces bit-identical output to blur_kernel (same accumulation order, same
// weights, same (uint16_t) truncation). Launch with an 8x8x8 block.
// =============================================================================
#define BLUR_BX 8
#define BLUR_BY 8
#define BLUR_BZ 8
__global__ void blur_kernel_tiled(const uint16_t* __restrict__ input,
                                  uint16_t* __restrict__ output, int N, int M)
{
    // Shared tile = block volume + 1-voxel halo on every side.
    const int TX = BLUR_BX + 2, TY = BLUR_BY + 2, TZ = BLUR_BZ + 2;
    __shared__ uint16_t tile[TZ][TY][TX];

    // This block's origin in volume space (top-left-front corner, halo offset -1).
    const int bx0 = blockIdx.x * BLUR_BX - 1;
    const int by0 = blockIdx.y * BLUR_BY - 1;
    const int bz0 = blockIdx.z * BLUR_BZ - 1;

    // Cooperative load: TX*TY*TZ cells filled by BX*BY*BZ threads (grid-stride).
    const int tid     = (threadIdx.z * BLUR_BY + threadIdx.y) * BLUR_BX + threadIdx.x;
    const int nThread = BLUR_BX * BLUR_BY * BLUR_BZ;
    const int nCells  = TX * TY * TZ;
    for (int c = tid; c < nCells; c += nThread) {
        int lx =  c % TX;
        int ly = (c / TX) % TY;
        int lz =  c / (TX * TY);
        // Global coords with clamping (mirrors clampi in the reference).
        int gx = clampi(bx0 + lx, 0, N - 1);
        int gy = clampi(by0 + ly, 0, N - 1);
        int gz = clampi(bz0 + lz, 0, M - 1);
        tile[lz][ly][lx] = input[(size_t)gz * N * N + gy * N + gx];
    }
    __syncthreads();

    int x = blockIdx.x * BLUR_BX + threadIdx.x;
    int y = blockIdx.y * BLUR_BY + threadIdx.y;
    int z = blockIdx.z * BLUR_BZ + threadIdx.z;
    if (x >= N || y >= N || z >= M) return;

    // This thread's centre within the shared tile (+1 for the halo offset).
    const int lx = threadIdx.x + 1;
    const int ly = threadIdx.y + 1;
    const int lz = threadIdx.z + 1;

    // Same accumulation order / weights / truncation as blur_kernel.
    float acc = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                uint16_t voxel = tile[lz + dz][ly + dy][lx + dx];
                acc += (float)voxel * c_weights[(dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)];
            }
    output[(size_t)z * N * N + y * N + x] = (uint16_t)acc;
}

// =============================================================================
// Kernel 2c - Separable Gaussian: one 1D 3-tap pass along `axis` (0=x,1=y,2=z).
// Applied X->Y->Z replaces the 27-tap blur with 3x3-tap passes (9 ops vs 27).
// Each pass truncates to uint16, so the result differs slightly from the 27-tap
// blur (per-pass rounding) — measured as an experiment, not the default.
// =============================================================================
__global__ void blur1d_kernel(const uint16_t* __restrict__ input,
                              uint16_t* __restrict__ output, int N, int M, int axis)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= N || y >= N || z >= M) return;

    float acc = 0.0f;
    for (int d = -1; d <= 1; ++d) {
        int nx = x, ny = y, nz = z;
        if      (axis == 0) nx = clampi(x + d, 0, N - 1);
        else if (axis == 1) ny = clampi(y + d, 0, N - 1);
        else                nz = clampi(z + d, 0, M - 1);
        uint16_t voxel = __ldg(&input[(size_t)nz * N * N + ny * N + nx]);
        acc += (float)voxel * c_weights1d[d + 1];
    }
    output[(size_t)z * N * N + y * N + x] = (uint16_t)acc;
}

// =============================================================================
// Kernel 3 - Maximum Intensity Projection (MIP).
// One thread per OUTPUT pixel (x,y). The thread walks a single ray through the
// volume and keeps the largest voxel intensity it crosses (the "maximum intensity
// projection"). The reduction is private to the thread, so no atomics or
// synchronisation are needed. Backward mapping: the screen point (u,v,w) is
// rotated by c_R into volume space, sampled with nearest-neighbor (the (int)
// truncation), and bounds-checked. Volume reads go through the read-only data
// cache via __ldg, since the volume is never written by this kernel.
// =============================================================================
__global__ void mip_kernel(const uint16_t* __restrict__ volume,
                           uint16_t* __restrict__ image, int N, int M, int ray_range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Center the screen coordinates on the volume's mid-plane.
    float u = x - N / 2.0f;
    float v = y - N / 2.0f;

    // Step the ray across the full half-diagonal so it traverses the volume at
    // any rotation; keep the running maximum of the sampled voxels.
    uint16_t maxIntensity = 0;
    for (int step = -ray_range; step < ray_range; ++step) {
        float w = (float)step;

        float rotX = c_R[0] * u + c_R[1] * v + c_R[2] * w + N / 2.0f;
        float rotY = c_R[3] * u + c_R[4] * v + c_R[5] * w + N / 2.0f;
        float rotZ = c_R[6] * u + c_R[7] * v + c_R[8] * w + M / 2.0f;

        if (rotX >= 0 && rotX < N &&
            rotY >= 0 && rotY < N &&
            rotZ >= 0 && rotZ < M) {
            size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
            uint16_t val = __ldg(&volume[idx]);
            if (val > maxIntensity)
                maxIntensity = val;
        }
    }
    image[y * N + x] = maxIntensity;
}

// =============================================================================
// Kernel 3b - Multi-MIP: N projections in ONE launch (Innovation #1).
// The grid is 3D: (x, y, frame). Each thread renders pixel (x,y) of frame f
// using rotation matrix c_Rmulti[f] and writes into that frame's slice of a
// frames*N*N output buffer. The volume is shared by all frames.
// =============================================================================
__global__ void mip_multi_kernel(const uint16_t* __restrict__ volume,
                                 uint16_t* __restrict__ images,
                                 int N, int M, int ray_range, int frames)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int f = blockIdx.z * blockDim.z + threadIdx.z;   // frame index
    if (x >= N || y >= N || f >= frames) return;

    // This frame's rotation matrix (9 floats at offset f*9).
    const float* R = c_Rmulti + f * 9;

    float u = x - N / 2.0f;
    float v = y - N / 2.0f;

    uint16_t maxIntensity = 0;
    for (int step = -ray_range; step < ray_range; ++step) {
        float w = (float)step;

        float rotX = R[0] * u + R[1] * v + R[2] * w + N / 2.0f;
        float rotY = R[3] * u + R[4] * v + R[5] * w + N / 2.0f;
        float rotZ = R[6] * u + R[7] * v + R[8] * w + M / 2.0f;

        if (rotX >= 0 && rotX < N &&
            rotY >= 0 && rotY < N &&
            rotZ >= 0 && rotZ < M) {
            size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
            uint16_t val = __ldg(&volume[idx]);
            if (val > maxIntensity)
                maxIntensity = val;
        }
    }
    // Write into frame f's slice of the output buffer.
    images[(size_t)f * N * N + y * N + x] = maxIntensity;
}

// Build the normalized 3x3x3 Gaussian weights on the host (identical math to the
// reference `gaussian_blur`) and upload them to constant memory.
static void upload_gaussian_weights(float sigma)
{
    float weights[27];
    float sum = 0.0f;
    for (int dz = -1; dz <= 1; ++dz)
        for (int dy = -1; dy <= 1; ++dy)
            for (int dx = -1; dx <= 1; ++dx) {
                float distance_sq = (float)(dx * dx + dy * dy + dz * dz);
                float weight = std::exp(-distance_sq / (2.0f * sigma * sigma));
                weights[(dz + 1) * 9 + (dy + 1) * 3 + (dx + 1)] = weight;
                sum += weight;
            }
    for (int i = 0; i < 27; ++i) weights[i] /= sum;

    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, weights, sizeof(weights)));

    // 1D Gaussian (3-tap) for the separable experiment: g(d)=exp(-d^2/2s^2),
    // normalized so the three taps sum to 1.
    float w1[3], s1 = 0.0f;
    for (int d = -1; d <= 1; ++d) { w1[d + 1] = std::exp(-(float)(d * d) / (2.0f * sigma * sigma)); s1 += w1[d + 1]; }
    for (int i = 0; i < 3; ++i) w1[i] /= s1;
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights1d, w1, sizeof(w1)));
}

// =============================================================================
// Pick the most capable GPU (banana has two: GT 710 sm_35 and GTX 1660 sm_75).
// Heuristic: prefer higher compute capability, then more streaming
// multiprocessors. Same idea as the gpuGetMaxGflopsDeviceId helper shipped with
// the CUDA Samples [5]; here simplified to (major, minor, #SM) so we always land
// on the GTX 1660 Ti rather than the weak secondary GPU.
// =============================================================================
static int pick_best_device()
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) {
        print("no CUDA-capable device found\n");
        std::exit(EXIT_FAILURE);
    }

    int best = 0;
    cudaDeviceProp bestProp;
    CUDA_CHECK(cudaGetDeviceProperties(&bestProp, best));
    for (int d = 1; d < count; ++d) {
        cudaDeviceProp p;
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        bool better = (p.major != bestProp.major) ? (p.major > bestProp.major)
                    : (p.minor != bestProp.minor) ? (p.minor > bestProp.minor)
                    : (p.multiProcessorCount > bestProp.multiProcessorCount);
        if (better) { best = d; bestProp = p; }
    }

    print("  gpu: using device %d (%s, sm_%d%d, %d SMs, %.0f MiB)\n",
          best, bestProp.name, bestProp.major, bestProp.minor,
          bestProp.multiProcessorCount,
          bestProp.totalGlobalMem / (1024.0 * 1024.0));
    return best;
}

// =============================================================================
// Device entry point: allocate VRAM, upload data, run the 3 kernels, download
// the resulting raster, and free everything.
// =============================================================================
void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output,
        float* kernels_ms)
{
    print("Running functions on the GPU\n");

    const int    N           = kBunnySize;
    const int    M           = kBunnyN;
    const size_t volCount    = (size_t)N * N * M;
    const size_t rasterCount = (size_t)N * N;
    const size_t volBytes    = volCount    * sizeof(uint16_t);
    const size_t rasterBytes = rasterCount * sizeof(uint16_t);

    // Select the most capable GPU available.
    CUDA_CHECK(cudaSetDevice(pick_best_device()));

    // Upload the read-only constants (Gaussian weights + rotation matrix).
    upload_gaussian_weights(sigma);
    CUDA_CHECK(cudaMemcpyToSymbol(c_R, R, 9 * sizeof(float)));

    // Device buffers:
    //   d_volume  - CT volume (threshold is applied in-place here)
    //   d_blurred - blurred volume (blur is out-of-place, like the host)
    //   d_raster  - 512x512 MIP output
    uint16_t *d_volume = nullptr, *d_blurred = nullptr, *d_raster = nullptr;
    CUDA_CHECK(cudaMalloc(&d_volume,  volBytes));
    CUDA_CHECK(cudaMalloc(&d_blurred, volBytes));
    CUDA_CHECK(cudaMalloc(&d_raster,  rasterBytes));

    // Host -> Device: upload the CT volume (perf: time the H2D transfer).
    GpuTimer t;
    t.start();
    CUDA_CHECK(cudaMemcpy(d_volume, input, volBytes, cudaMemcpyHostToDevice));
    t.stop("H2D upload");

    // Kernels-only timing: bracket the 3 kernels with CUDA events (excludes the
    // allocations and the H2D/D2H transfers above/below).
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    CUDA_CHECK(cudaEventRecord(ev_start));

    // Step 1: Threshold (in-place on d_volume).
    print("  gpu: applying threshold\n");
    {
        const int threads = 256;
        const int blocks  = (int)((volCount + threads - 1) / threads);
        perf_launch("threshold", (const void*)threshold_kernel,
                    dim3(blocks), dim3(threads));
        t.start();
        threshold_kernel<<<blocks, threads>>>(d_volume, volCount, threshold);
        CUDA_CHECK(cudaGetLastError());
        t.stop("threshold");
    }

    // Step 2: Gaussian Blur (d_volume -> d_blurred).
    print("  gpu: applying filter\n");
    {
        dim3 block(8, 8, 8);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y,
                  (M + block.z - 1) / block.z);
        if (separable_enabled()) {
            // 3 passes X->Y->Z, ping-pong ending in d_blurred (same as tiled).
            perf_launch("blur(sep)", (const void*)blur1d_kernel, grid, block);
            t.start();
            blur1d_kernel<<<grid, block>>>(d_volume,  d_blurred, N, M, 0); // X
            blur1d_kernel<<<grid, block>>>(d_blurred, d_volume,  N, M, 1); // Y
            blur1d_kernel<<<grid, block>>>(d_volume,  d_blurred, N, M, 2); // Z
            CUDA_CHECK(cudaGetLastError());
            t.stop("blur(sep)");
        } else if (blur_naive_enabled()) {
            perf_launch("blur(naive)", (const void*)blur_kernel, grid, block);
            t.start();
            blur_kernel<<<grid, block>>>(d_volume, d_blurred, N, M);
            CUDA_CHECK(cudaGetLastError());
            t.stop("blur(naive)");
        } else {
            perf_launch("blur", (const void*)blur_kernel_tiled, grid, block);
            t.start();
            blur_kernel_tiled<<<grid, block>>>(d_volume, d_blurred, N, M);
            CUDA_CHECK(cudaGetLastError());
            t.stop("blur");
        }
    }

    // Step 3: MIP Projection (d_blurred -> d_raster).
    print("  gpu: generating MIP\n");
    {
        // Maximum ray length so a ray crosses the volume at any rotation.
        int ray_range = (int)(sqrt((double)N * N + (double)N * N + (double)M * M) / 2.0) + 1;
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        perf_launch("mip", (const void*)mip_kernel, grid, block);
        t.start();
        mip_kernel<<<grid, block>>>(d_blurred, d_raster, N, M, ray_range);
        CUDA_CHECK(cudaGetLastError());
        t.stop("mip");
    }

    // Close the kernels-only timing window (after the last kernel, before D2H).
    CUDA_CHECK(cudaEventRecord(ev_stop));

    // Wait for the GPU and surface any asynchronous kernel error.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Report the kernels-only elapsed time to the caller, if requested.
    if (kernels_ms) {
        CUDA_CHECK(cudaEventElapsedTime(kernels_ms, ev_start, ev_stop));
    }
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Device -> Host: download the resulting raster (perf: time the D2H copy).
    t.start();
    CUDA_CHECK(cudaMemcpy(output, d_raster, rasterBytes, cudaMemcpyDeviceToHost));
    t.stop("D2H copy");

    // Release VRAM (shared-server etiquette: free large allocations promptly).
    CUDA_CHECK(cudaFree(d_volume));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_raster));
}

// =============================================================================
// Innovation #1 entry point: threshold + blur once, then render `frames` MIP
// images with different rotations in a SINGLE mip_multi_kernel launch.
// =============================================================================
void device_bunny_mip_multi(const uint16_t* input, uint16_t threshold,
        float sigma, const float* Rs, int frames, uint16_t* output,
        float* mip_ms)
{
    print("Running multi-MIP on the GPU (%d frames, single kernel call)\n", frames);

    if (frames < 1) frames = 1;
    if (frames > MAX_FRAMES) {
        print("  gpu: frames clamped to MAX_FRAMES=%d\n", MAX_FRAMES);
        frames = MAX_FRAMES;
    }

    const int    N           = kBunnySize;
    const int    M           = kBunnyN;
    const size_t volCount    = (size_t)N * N * M;
    const size_t rasterCount = (size_t)N * N;
    const size_t volBytes    = volCount    * sizeof(uint16_t);
    const size_t outBytes    = rasterCount * (size_t)frames * sizeof(uint16_t);

    CUDA_CHECK(cudaSetDevice(pick_best_device()));

    // Upload read-only constants: Gaussian weights + all `frames` matrices.
    upload_gaussian_weights(sigma);
    CUDA_CHECK(cudaMemcpyToSymbol(c_Rmulti, Rs, (size_t)frames * 9 * sizeof(float)));

    // Device buffers: volume, blurred volume, and the frames*N*N output stack.
    uint16_t *d_volume = nullptr, *d_blurred = nullptr, *d_images = nullptr;
    CUDA_CHECK(cudaMalloc(&d_volume,  volBytes));
    CUDA_CHECK(cudaMalloc(&d_blurred, volBytes));
    CUDA_CHECK(cudaMalloc(&d_images,  outBytes));

    GpuTimer t;
    t.start();
    CUDA_CHECK(cudaMemcpy(d_volume, input, volBytes, cudaMemcpyHostToDevice));
    t.stop("H2D upload");

    // Step 1: Threshold (in-place).
    print("  gpu: applying threshold\n");
    {
        const int threads = 256;
        const int blocks  = (int)((volCount + threads - 1) / threads);
        perf_launch("threshold", (const void*)threshold_kernel,
                    dim3(blocks), dim3(threads));
        t.start();
        threshold_kernel<<<blocks, threads>>>(d_volume, volCount, threshold);
        CUDA_CHECK(cudaGetLastError());
        t.stop("threshold");
    }

    // Step 2: Gaussian Blur (d_volume -> d_blurred), shared by all frames.
    print("  gpu: applying filter\n");
    {
        dim3 block(8, 8, 8);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y,
                  (M + block.z - 1) / block.z);
        if (separable_enabled()) {
            perf_launch("blur(sep)", (const void*)blur1d_kernel, grid, block);
            t.start();
            blur1d_kernel<<<grid, block>>>(d_volume,  d_blurred, N, M, 0); // X
            blur1d_kernel<<<grid, block>>>(d_blurred, d_volume,  N, M, 1); // Y
            blur1d_kernel<<<grid, block>>>(d_volume,  d_blurred, N, M, 2); // Z
            CUDA_CHECK(cudaGetLastError());
            t.stop("blur(sep)");
        } else if (blur_naive_enabled()) {
            perf_launch("blur(naive)", (const void*)blur_kernel, grid, block);
            t.start();
            blur_kernel<<<grid, block>>>(d_volume, d_blurred, N, M);
            CUDA_CHECK(cudaGetLastError());
            t.stop("blur(naive)");
        } else {
            perf_launch("blur", (const void*)blur_kernel_tiled, grid, block);
            t.start();
            blur_kernel_tiled<<<grid, block>>>(d_volume, d_blurred, N, M);
            CUDA_CHECK(cudaGetLastError());
            t.stop("blur");
        }
    }

    // Step 3: Multi-MIP in ONE launch (3D grid: x, y, frame). Timed via events.
    print("  gpu: generating %d MIPs in a single kernel call\n", frames);
    cudaEvent_t ev_start, ev_stop;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_stop));
    {
        int ray_range = (int)(sqrt((double)N * N + (double)N * N + (double)M * M) / 2.0) + 1;
        dim3 block(16, 16, 1);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y,
                  (frames + block.z - 1) / block.z);   // one grid layer per frame
        perf_launch("mip_multi", (const void*)mip_multi_kernel, grid, block);
        CUDA_CHECK(cudaEventRecord(ev_start));
        mip_multi_kernel<<<grid, block>>>(d_blurred, d_images, N, M, ray_range, frames);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(ev_stop));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    if (mip_ms) CUDA_CHECK(cudaEventElapsedTime(mip_ms, ev_start, ev_stop));
    if (perf_enabled() && mip_ms)
        print("  [perf] %-12s %8.3f ms (%.3f ms/frame)\n", "mip_multi", *mip_ms,
              *mip_ms / frames);
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_stop));

    // Download all frames at once (perf: time the D2H copy).
    t.start();
    CUDA_CHECK(cudaMemcpy(output, d_images, outBytes, cudaMemcpyDeviceToHost));
    t.stop("D2H copy");

    CUDA_CHECK(cudaFree(d_volume));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_images));
}
