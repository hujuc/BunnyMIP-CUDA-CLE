
#pragma once

#include <cstdint>

// If kernels_ms != nullptr, the GPU kernels-only elapsed time (threshold +
// blur + MIP, excluding allocations and host<->device transfers) is written
// there in milliseconds, measured with CUDA events.
void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output,
        float* kernels_ms = nullptr);

// Innovation #1: compute `frames` MIP images with different rotations on the
// device using a SINGLE mip kernel launch (frame index = grid Z dimension).
// Thresholding and the blur run once; the blurred volume is reused for all
// frames. `Rs` is `frames` flattened 3x3 matrices (frames*9 floats); `output`
// must hold frames * 512 * 512 uint16_t (frame f at offset f*512*512).
// If mip_ms != nullptr, the multi-MIP kernel-only time (ms) is written there.
void device_bunny_mip_multi(const uint16_t* input, uint16_t threshold,
        float sigma, const float* Rs, int frames, uint16_t* output,
        float* mip_ms = nullptr);

