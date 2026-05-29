
#include "print.h"
#include "bunnyIO.h"
#include "device-bunnyMIP.h"


// TODO: Define your kernels here


// =============================================================================
// Device entry point.
// =============================================================================
void device_bunny_mip(const uint16_t* input, uint16_t threshold,
        float sigma, const float* R, uint16_t* output)
{
    print("Running functions on the GPU\n");

    // Step 1: Threshold
    print("  gpu: applying threshold\n");


    // Step 2: Gaussian Blur
    print("  gpu: applying filter\n");

    // Step 3: MIP Projection
    print("  gpu: generating MIP\n");
}
