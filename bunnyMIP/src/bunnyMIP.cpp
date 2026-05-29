
#include <cmath>
#include <cstring>

#include "print.h"
#include "bunnyIO.h"

// =============================================================================
// 1. Thresholding
// Removes low-intensity "air noise" around the bunny.
// =============================================================================
void apply_threshold(uint16_t* volume, uint16_t threshold) {

    int sz = kBunnySize * kBunnySize * kBunnyN;
    for (int i = 0; i < sz; ++i) {
        if (volume[i] < threshold) {
            volume[i] = 0;
        }
    }// end for
}

// =============================================================================
// 2. 3D Gaussian Blur
// Smooths the data to prevent single-pixel noise from dominating the MIP.
// =============================================================================
void gaussian_blur(uint16_t* input, uint16_t* output, float sigma) {
    const auto N = kBunnySize;
    const auto M = kBunnyN;

    float kernel[3][3][3];
    float kernel_sum = 0.0f;

    // 1. Generate the dynamic 3x3x3 Gaussian kernel
    // We iterate from -1 to 1 to represent the distance from the center voxel
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                float distance_sq = static_cast<float>(dx*dx + dy*dy + dz*dz);
                float weight = std::exp(-distance_sq / (2.0f * sigma * sigma));

                kernel[dz + 1][dy + 1][dx + 1] = weight;
                kernel_sum += weight;
            }
        }
    }

    // 2. Normalize the kernel
    // This prevents the image from becoming brighter or darker
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            for (int k = 0; k < 3; ++k) {
                kernel[i][j][k] /= kernel_sum;
            }
        }
    }

    // 3. Apply the convolution
    for (int z = 0; z < M; ++z) {
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                float blurredValue = 0.0f;

                for (int dz = -1; dz <= 1; ++dz) {
                    for (int dy = -1; dy <= 1; ++dy) {
                        for (int dx = -1; dx <= 1; ++dx) {
                            // Clamp boundary conditions
                            int nz = std::max(0, std::min(M - 1, z + dz));
                            int ny = std::max(0, std::min(N - 1, y + dy));
                            int nx = std::max(0, std::min(N - 1, x + dx));

                            uint16_t voxel = input[(size_t)nz * N * N + ny * N + nx];
                            blurredValue += static_cast<float>(voxel) * kernel[dz + 1][dy + 1][dx + 1];
                        }
                    }
                }

                output[(size_t)z * N * N + y * N + x] = static_cast<uint16_t>(blurredValue);
            }
        }
    }
}

// =============================================================================
// 3. Maximum Intensity Projection (MIP)
// Reduces the 3D volume into a 2D image for visualization.
// =============================================================================
void rotated_mip(const uint16_t* volume, uint16_t* image, int N, int M, const float* R) {
    // Calculate the maximum possible diagonal to ensure the ray
    // travels completely through the volume regardless of angle.
    int ray_range = (int)(sqrt(N*N + N*N + M*M) / 2.0f) + 1;

    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            uint16_t maxIntensity = 0;

            // 1. Center the 2D screen coordinates
            float u = x - N / 2.0f;
            float v = y - N / 2.0f;

            // 2. Step the ray through the volume
            for (int step = -ray_range; step < ray_range; step++) {
                float w = (float)step;

                // 3. Transform point (u, v, w) to volume space
                // Notice the offset for rotZ uses M/2.0f
                float rotX = R[0] * u + R[1] * v + R[2] * w + N / 2.0f;
                float rotY = R[3] * u + R[4] * v + R[5] * w + N / 2.0f;
                float rotZ = R[6] * u + R[7] * v + R[8] * w + M / 2.0f;

                // 4. Boundary Check: rotZ is compared against M
                if (rotX >= 0 && rotX < N &&
                    rotY >= 0 && rotY < N &&
                    rotZ >= 0 && rotZ < M) {
                    // 5. Indexing: The stride for Z is still N*N
                    // because each slice is N pixels wide and N pixels high.
                    size_t idx = (size_t)((int)rotZ * N * N + (int)rotY * N + (int)rotX);
                    uint16_t val = volume[idx];
                    if (val > maxIntensity) {
                        maxIntensity = val;
                    }
                }
            }
            image[y * N + x] = maxIntensity;
        }
    }
}


// =============================================================================
// Host entry point.
// =============================================================================
void host_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output)
{

    //==========================================================================
    print("Running functions on the CPU\n");

    uint16_t* h_volume = new uint16_t[kBunnySize*kBunnySize*kBunnyN];
    uint16_t* h_blured = new uint16_t[kBunnySize*kBunnySize*kBunnyN];

    // Step 1: Threshold
    print("  cpu: applying threshold\n");
    memcpy(h_volume, input, kBunnySize*kBunnySize*kBunnyN*sizeof(uint16_t));
    apply_threshold(h_volume, threshold);

    // Step 2: Gaussian Blur
    print("  cpu: applying filter\n");
    gaussian_blur(h_volume, h_blured, sigma);

    // Step 3: MIP Projection
    print("  cpu: generating MIP\n");
    rotated_mip(h_blured, output, kBunnySize, kBunnyN, R);

    delete [] h_volume;
    delete [] h_blured;
}
