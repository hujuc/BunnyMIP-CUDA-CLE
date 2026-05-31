
#pragma once

#include <cstdint>

// If kernels_ms != nullptr, the GPU kernels-only elapsed time (threshold +
// blur + MIP, excluding allocations and host<->device transfers) is written
// there in milliseconds, measured with CUDA events.
void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output,
        float* kernels_ms = nullptr);

