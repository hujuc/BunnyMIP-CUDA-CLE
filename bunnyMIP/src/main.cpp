
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

        if      (arg == "--threshold") threshold = (uint16_t)std::atoi(value("--threshold"));
        else if (arg == "--sigma")     sigma     = (float)std::atof(value("--sigma"));
        else if (arg == "--roll")      roll      = (float)std::atof(value("--roll"));
        else if (arg == "--pitch")     pitch     = (float)std::atof(value("--pitch"));
        else if (arg == "--yaw")       yaw       = (float)std::atof(value("--yaw"));
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

    // CPU
    host_bunny_mip(volume, threshold, sigma, R, h_raster);
    // GPU (CUDA)
    device_bunny_mip(volume, threshold, sigma , R, d_raster);

    int raster_size = kBunnySize * kBunnySize;
    int diff = 0;
    for (int i = 0; i < raster_size; i++) {
        int local_diff = abs((int)h_raster[i] - (int)d_raster[i]);
        if (local_diff > 2)
            diff = diff + 1;
    }

    print("\n>> Output difference: %.2f%%\n", (diff / (float)raster_size) * 100.0);

    savePGM16("output/bunnyMIP_cpu.pgm", h_raster, kBunnySize, kBunnySize);
    savePGM16("output/bunnyMIP_gpu.pgm", d_raster, kBunnySize, kBunnySize);

    delete [] volume;
    return 0;
}
