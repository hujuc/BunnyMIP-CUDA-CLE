#include <cstdint>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

#include "print.h"
#include "bunnyIO.h"
#include "device-bunnyMIP.h"

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

// Clamp helper for the blur boundary handling (mirrors the host's max/min).
__device__ __forceinline__ int clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// =============================================================================
// Kernel 1 - Thresholding (point operation).
// Sets every voxel below `threshold` to 0. Grid-stride loop over the volume.
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
// One thread per voxel: weighted average of the 3x3x3 neighborhood (clamped
// boundaries), out-of-place. Weights come from constant memory.
// =============================================================================
__global__ void blur_kernel(const uint16_t* __restrict__ input,
                            uint16_t* __restrict__ output, int N, int M)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    if (x >= N || y >= N || z >= M) return;

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
// Kernel 3 - Maximum Intensity Projection (MIP).
// One thread per output pixel: cast a ray (rotated by c_R) through the volume
// and keep the maximum voxel intensity (nearest-neighbor sampling).
// =============================================================================
__global__ void mip_kernel(const uint16_t* __restrict__ volume,
                           uint16_t* __restrict__ image, int N, int M, int ray_range)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Centered screen coordinates.
    float u = x - N / 2.0f;
    float v = y - N / 2.0f;

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
}

// =============================================================================
// Pick the most capable GPU (the server has two: GT 710 and GTX 1660).
// Prefers higher compute capability, then more streaming multiprocessors.
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
        float sigma, const float* R, uint16_t* output)
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

    // Host -> Device: upload the CT volume.
    CUDA_CHECK(cudaMemcpy(d_volume, input, volBytes, cudaMemcpyHostToDevice));

    // Step 1: Threshold (in-place on d_volume).
    print("  gpu: applying threshold\n");
    {
        const int threads = 256;
        const int blocks  = (int)((volCount + threads - 1) / threads);
        threshold_kernel<<<blocks, threads>>>(d_volume, volCount, threshold);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 2: Gaussian Blur (d_volume -> d_blurred).
    print("  gpu: applying filter\n");
    {
        dim3 block(8, 8, 8);
        dim3 grid((N + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y,
                  (M + block.z - 1) / block.z);
        blur_kernel<<<grid, block>>>(d_volume, d_blurred, N, M);
        CUDA_CHECK(cudaGetLastError());
    }

    // Step 3: MIP Projection (d_blurred -> d_raster).
    print("  gpu: generating MIP\n");
    {
        // Maximum ray length so a ray crosses the volume at any rotation.
        int ray_range = (int)(sqrt((double)N * N + (double)N * N + (double)M * M) / 2.0) + 1;
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        mip_kernel<<<grid, block>>>(d_blurred, d_raster, N, M, ray_range);
        CUDA_CHECK(cudaGetLastError());
    }

    // Wait for the GPU and surface any asynchronous kernel error.
    CUDA_CHECK(cudaDeviceSynchronize());

    // Device -> Host: download the resulting raster.
    CUDA_CHECK(cudaMemcpy(output, d_raster, rasterBytes, cudaMemcpyDeviceToHost));

    // Release VRAM (shared-server etiquette: free large allocations promptly).
    CUDA_CHECK(cudaFree(d_volume));
    CUDA_CHECK(cudaFree(d_blurred));
    CUDA_CHECK(cudaFree(d_raster));
}
