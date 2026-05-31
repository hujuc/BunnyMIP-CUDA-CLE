// =============================================================================
// main.cpp - CLI, pipeline driver, speedup benchmark and validation (cle08).
//
// Provenance:
//   - generate_rotation_matrix() and d2r() are from the provided base code
//     (instructor-supplied). The group only fixed d2r (pi/360 -> pi/180, see
//     below) and the loop bounds; the Euler-matrix math itself is unchanged.
//   - The CLI parsing, the timed benchmark loop, the validation block and the
//     multi-MIP (Innovation #1) driver are the group's original work.
// =============================================================================

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>

#include "print.h"
#include "bunnyIO.h"
#include "imageIO.h"

#include "bunnyMIP.h"
#include "device-bunnyMIP.h"

// Degrees -> radians (standard pi/180). NOTE: the original base code used
// pi/360 (half the angle); corrected here so the CLI angles mean real degrees.
inline float d2r(float angle) { return angle * M_PI / 180.0f; }

// [BASE CODE] Build the flattened 3x3 rotation matrix R = Rz*Ry*Rx from the
// pitch/yaw/roll angles (radians). `inverse` transposes it for backward mapping.
// Provided with the assignment; consumed by the MIP kernels.
void generate_rotation_matrix(float pitch, float yaw, float roll, float* R, bool inverse = false) {
    float cp = std::cos(pitch); float sp = std::sin(pitch);
    float cy = std::cos(yaw);   float sy = std::sin(yaw);
    float cr = std::cos(roll);  float sr = std::sin(roll);

    // Combined rotation R = Rz * Ry * Rx
    float m[3][3];
    m[0][0] = cy * cr;
    m[0][1] = cy * sr;
    m[0][2] = -sy;

    m[1][0] = sp * sy * cr - cp * sr;
    m[1][1] = sp * sy * sr + cp * cr;
    m[1][2] = sp * cy;

    m[2][0] = cp * sy * cr + sp * sr;
    m[2][1] = cp * sy * sr - sp * cr;
    m[2][2] = cp * cy;

    // Fill the flattened array
    if (!inverse) {
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[i * 3 + j] = m[i][j];
    } else {
        // Transpose for backward mapping (Inverse rotation)
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                R[j * 3 + i] = m[i][j];
    }
}

// C = A * B for flattened 3x3 matrices (row-major). Used by the multi-MIP
// turntable to compose a fixed tilt with the per-frame spin (R = R_tilt*R_spin).
static void matmul3(const float* A, const float* B, float* C) {
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            C[i * 3 + j] = A[i * 3 + 0] * B[0 * 3 + j]
                         + A[i * 3 + 1] * B[1 * 3 + j]
                         + A[i * 3 + 2] * B[2 * 3 + j];
}

static void usage(const char* prog) {
    print("Usage: %s [options]\n", prog);
    print("  --threshold N   intensity floor; voxels < N are set to 0 (default 32768)\n");
    print("  --sigma S       standard deviation for the Gaussian blur (default 1.0)\n");
    print("  --roll D        roll rotation in degrees  (default 0)\n");
    print("  --pitch D       pitch rotation in degrees (default 0)\n");
    print("  --yaw D         yaw rotation in degrees   (default 0)\n");
    print("  --repeat N      timed CPU/GPU runs for the speedup analysis (default 1;\n");
    print("                  the first run is discarded as warm-up when N > 1)\n");
    print("  --frames N      render N MIPs with different rotations in ONE CUDA kernel\n");
    print("                  call, sweeping 0..360 deg about --axis. The --roll/--pitch/\n");
    print("                  --yaw values on the OTHER two axes set a fixed tilt (orbit).\n");
    print("                  Writes output/frame_XXX.pgm\n");
    print("  --axis A        multi-MIP spin axis: roll|pitch|yaw (default yaw)\n");
    print("  -h, --help      show this help and exit\n");
}

int main(int argc, char* argv[]) {
    print("CLE2026 - BunnyMIP\n");

    // ---- Defaults (match the original hardcoded values) --------------------
    uint16_t threshold = 1 << 15;   // 32768
    float    sigma     = 1.0f;
    float    roll      = 0.0f;
    float    pitch     = 0.0f;
    float    yaw       = 0.0f;
    int      repeat    = 1;          // timed runs for the speedup analysis
    int      frames    = 1;          // >1 -> multi-MIP: N rotations, 1 kernel call
    std::string spin_axis = "yaw";   // multi-MIP: which axis sweeps 0..360

    // ---- Parse the command line --------------------------------------------
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        auto value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                print("error: missing value for %s\n", name);
                usage(argv[0]);
                std::exit(EXIT_FAILURE);
            }
            return argv[++i];
        };

        if (arg == "--threshold") {
            long t = std::atol(value("--threshold"));
            if (t < 0 || t > 65535) {
                print("warning: --threshold %ld out of [0,65535]; clamping\n", t);
                t = (t < 0) ? 0 : 65535;
            }
            threshold = (uint16_t)t;
        }
        else if (arg == "--sigma") {
            sigma = (float)std::atof(value("--sigma"));
            if (sigma <= 0.0f)
                print("warning: --sigma should be > 0 (got %.3f); blur weights may be invalid\n", sigma);
        }
        else if (arg == "--roll")      roll      = (float)std::atof(value("--roll"));
        else if (arg == "--pitch")     pitch     = (float)std::atof(value("--pitch"));
        else if (arg == "--yaw")       yaw       = (float)std::atof(value("--yaw"));
        else if (arg == "--repeat") {
            repeat = std::atoi(value("--repeat"));
            if (repeat < 1) { print("warning: --repeat must be >= 1; using 1\n"); repeat = 1; }
        }
        else if (arg == "--frames") {
            frames = std::atoi(value("--frames"));
            if (frames < 1) { print("warning: --frames must be >= 1; using 1\n"); frames = 1; }
        }
        else if (arg == "--axis") {
            spin_axis = value("--axis");
            if (spin_axis != "roll" && spin_axis != "pitch" && spin_axis != "yaw") {
                print("warning: --axis must be roll|pitch|yaw (got '%s'); using yaw\n",
                      spin_axis.c_str());
                spin_axis = "yaw";
            }
        }
        else if (arg == "-h" || arg == "--help") { usage(argv[0]); return 0; }
        else {
            print("error: unknown option '%s'\n", arg.c_str());
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    print("params: threshold=%u sigma=%.3f roll=%.1f pitch=%.1f yaw=%.1f (degrees)\n",
          (unsigned)threshold, sigma, roll, pitch, yaw);

    uint16_t* volume = loadBunnyCT("data");

    // ---- Multi-MIP mode (Innovation #1): N rotations, single kernel call -----
    if (frames > 1) {
        // --axis selects which axis sweeps 0..360 (the "spin"). The roll/pitch/yaw
        // CLI values on the OTHER two axes define a fixed tilt that orients the
        // spin axis, giving a tilted orbit (not just a flat turntable):
        //   R_frame(f) = R_tilt * R_spin(f),  R_spin sweeps 0..360 about the axis.
        const char axis = spin_axis[0];   // 'r', 'p', or 'y'

        // Fixed tilt from the two non-spin angles (the spin axis contributes 0).
        float tilt_roll  = (axis == 'r') ? 0.0f : roll;
        float tilt_pitch = (axis == 'p') ? 0.0f : pitch;
        float tilt_yaw   = (axis == 'y') ? 0.0f : yaw;
        float R_tilt[9];
        generate_rotation_matrix(d2r(tilt_pitch), d2r(tilt_yaw), d2r(tilt_roll), R_tilt);

        print("multi-MIP: %d frames, spin %s 0..360 deg; tilt(r=%.1f p=%.1f y=%.1f), 1 kernel call\n",
              frames, spin_axis.c_str(), tilt_roll, tilt_pitch, tilt_yaw);

        // Build N matrices: R_frame = R_tilt * R_spin(f*360/N about the spin axis).
        const int raster = kBunnySize * kBunnySize;
        float* Rs = new float[(size_t)frames * 9];
        for (int f = 0; f < frames; ++f) {
            float ang = f * (360.0f / frames);
            float rr = (axis == 'r') ? ang : 0.0f;
            float pp = (axis == 'p') ? ang : 0.0f;
            float yy = (axis == 'y') ? ang : 0.0f;
            float R_spin[9];
            generate_rotation_matrix(d2r(pp), d2r(yy), d2r(rr), R_spin);
            matmul3(R_tilt, R_spin, Rs + (size_t)f * 9);
        }

        uint16_t* allframes = new uint16_t[(size_t)raster * frames];
        float mip_ms = 0.0f;
        device_bunny_mip_multi(volume, threshold, sigma, Rs, frames, allframes, &mip_ms);

        // Save each frame as output/frame_XXX.pgm (input to the ffmpeg video).
        for (int f = 0; f < frames; ++f) {
            char name[64];
            snprintf(name, sizeof(name), "output/frame_%03d.pgm", f);
            savePGM16(name, allframes + (size_t)f * raster, kBunnySize, kBunnySize);
        }
        print("\n>> Multi-MIP done: %d frames, single kernel call: %.3f ms (%.3f ms/frame)\n",
              frames, mip_ms, mip_ms / frames);
        print("   wrote output/frame_000.pgm .. frame_%03d.pgm\n", frames - 1);

        delete [] Rs;
        delete [] allframes;
        delete [] volume;
        return 0;
    }

    // Raster output when running on the host
    uint16_t* h_raster = new uint16_t[kBunnySize*kBunnySize];
    // Raster output when running on the GPU
    uint16_t* d_raster = new uint16_t[kBunnySize*kBunnySize];

    // Rotation matrix from the CLI angles (used by the MIP backward mapping).
    float R[3*3];
    generate_rotation_matrix(d2r(pitch), d2r(yaw), d2r(roll), R);

    // --- Timed benchmark loop (speedup analysis) -----------------------------
    // Run the CPU and GPU pipelines `repeat` times. When repeat > 1 the first
    // iteration is discarded as warm-up (it pays the one-time CUDA context
    // initialization cost). We accumulate avg and track min over the rest.
    //   T_cpu        - host pipeline wall-clock (std::chrono)
    //   T_gpu_e2e    - device pipeline wall-clock, incl. malloc + H2D/D2H + free
    //   T_gpu_kernel - device kernels only (CUDA events, no transfers)
    using clk = std::chrono::high_resolution_clock;
    const int warmup  = (repeat > 1) ? 1 : 0;
    const int counted = repeat - warmup;

    double cpu_sum = 0.0, gpu_sum = 0.0, ker_sum = 0.0;
    double cpu_min = std::numeric_limits<double>::infinity();
    double gpu_min = std::numeric_limits<double>::infinity();
    double ker_min = std::numeric_limits<double>::infinity();

    for (int r = 0; r < repeat; ++r) {
        // CPU (sequential reference / Gold Standard)
        auto c0 = clk::now();
        host_bunny_mip(volume, threshold, sigma, R, h_raster);
        auto c1 = clk::now();

        // GPU (CUDA); kernel_ms receives the kernels-only time via CUDA events
        float kernel_ms = 0.0f;
        auto g0 = clk::now();
        device_bunny_mip(volume, threshold, sigma, R, d_raster, &kernel_ms);
        auto g1 = clk::now();

        double cpu_ms = std::chrono::duration<double, std::milli>(c1 - c0).count();
        double gpu_ms = std::chrono::duration<double, std::milli>(g1 - g0).count();

        if (r >= warmup) {
            cpu_sum += cpu_ms; gpu_sum += gpu_ms; ker_sum += kernel_ms;
            if (cpu_ms    < cpu_min) cpu_min = cpu_ms;
            if (gpu_ms    < gpu_min) gpu_min = gpu_ms;
            if (kernel_ms < ker_min) ker_min = kernel_ms;
        }
    }

    const double cpu_avg = cpu_sum / counted;
    const double gpu_avg = gpu_sum / counted;
    const double ker_avg = ker_sum / counted;

    // --- Performance report (speedup analysis) -------------------------------
    print("\n>> Performance (%d run(s)%s)\n", repeat,
          warmup ? ", first discarded as warm-up" : "");
    print("                          avg            min\n");
    print("   CPU  (sequential): %10.3f ms  %10.3f ms\n", cpu_avg, cpu_min);
    print("   GPU  (end-to-end): %10.3f ms  %10.3f ms\n", gpu_avg, gpu_min);
    print("   GPU  (kernels)   : %10.3f ms  %10.3f ms\n", ker_avg, ker_min);
    print("   Speedup end-to-end (CPU/GPU): %8.2fx\n", cpu_avg / gpu_avg);
    print("   Speedup kernels    (CPU/GPU): %8.2fx\n", cpu_avg / ker_avg);

    // --- Validation: compare GPU raster against the CPU "Gold Standard" ------
    // Metric: percentage of pixels whose absolute intensity difference exceeds
    // PIXEL_TOL. Validation PASSES if that percentage stays below ACCEPT_PCT.
    const int   PIXEL_TOL  = 2;     // per-pixel intensity tolerance (16-bit units)
    const float ACCEPT_PCT = 1.0f;  // max % of differing pixels allowed

    int raster_size = kBunnySize * kBunnySize;
    int diff = 0;
    int max_abs = 0;
    for (int i = 0; i < raster_size; i++) {
        int local_diff = abs((int)h_raster[i] - (int)d_raster[i]);
        if (local_diff > max_abs) max_abs = local_diff;
        if (local_diff > PIXEL_TOL)
            diff = diff + 1;
    }
    float pct = (diff / (float)raster_size) * 100.0f;
    bool pass = (pct <= ACCEPT_PCT);

    print("\n>> Validation (GPU vs CPU Gold Standard)\n");
    print("   metric: %% of pixels with |diff| > %d  (max |diff| = %d)\n", PIXEL_TOL, max_abs);
    print("   differing pixels: %.4f%%   tolerance: %.2f%%\n", pct, ACCEPT_PCT);
    print("   result: %s\n", pass ? "PASS" : "FAIL");

    savePGM16("output/bunnyMIP_cpu.pgm", h_raster, kBunnySize, kBunnySize);
    savePGM16("output/bunnyMIP_gpu.pgm", d_raster, kBunnySize, kBunnySize);

    delete [] volume;
    delete [] h_raster;
    delete [] d_raster;
    return pass ? 0 : 1;
}
