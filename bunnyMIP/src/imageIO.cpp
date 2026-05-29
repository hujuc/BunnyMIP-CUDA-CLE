
#include <fstream>
#include <memory>

#include "imageIO.h"

bool savePGM16(const std::string& filename, const uint16_t* data, uint32_t width, uint32_t height) {
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) return false;

    // 1. Header: Set Maxval to 65535
    ofs << "P5\n" << width << " " << height << "\n65535\n";

    // 2. Prepare Big-Endian buffer
    size_t totalPixels = static_cast<size_t>(width) * height;
    auto bigEndianData = std::make_unique<uint16_t[]>(totalPixels);

    for (size_t i = 0; i < totalPixels; ++i) {
        uint16_t val = data[i];
        // Swap bytes: 0xAABB -> 0xBBAA
        bigEndianData[i] = (val << 8) | (val >> 8);
    }

    // 3. Write (Each pixel is now 2 bytes)
    ofs.write(reinterpret_cast<const char*>(bigEndianData.get()), totalPixels * sizeof(uint16_t));

    return ofs.good();
}
