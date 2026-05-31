
#include <chrono>
#include <cmath>
#include <cstdint>
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

static void usage(const char* prog) {
    print("Usage: %s [options]\n", prog);
    print("  --threshold N   intensity floor; voxels < N are set to 0 (default 32768)\n");
    print("  --sigma S       standard deviation for the Gaussian blur (default 1.0)\n");
    print("  --roll D        roll rotation in degrees  (default 0)\n");
    print("  --pitch D       pitch rotation in degrees (default 0)\n");
    print("  --yaw D         yaw rotation in degrees   (default 0)\n");
    print("  --repeat N      timed CPU/GPU runs for the speedup analysis (default 1;\n");
    print("                  the first run is discarded as warm-up when N > 1)\n");
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
