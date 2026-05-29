
#include <fstream>

#include "print.h"
#include "bunnyIO.h"

uint16_t* loadBunnyCT(const std::string& base_directory)
{
    // Allocate enough data for the CT volume.
    auto size = kBunnySize * kBunnySize * kBunnyN;
    auto data = new uint16_t[size];


    for (int i = 0; i < kBunnyN; i++) {
        auto fname = format("%s/%d", base_directory.c_str(), i+1);
        std::ifstream file(fname, std::ios::binary);

        if (not file) {
            print("bunny: unable to open file %s\n", fname.c_str());
            exit(1);
        }

        auto slice_sz = kBunnySize * kBunnySize * sizeof(uint16_t);
        file.read((char*)data + i*slice_sz, slice_sz);
    }// end for


    print("bunny loaded: %ld MiB allocated\n", size * sizeof(uint16_t) / 1024 / 1024);
    return data;
}
