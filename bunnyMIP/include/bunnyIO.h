
#pragma once

#include <cstdint>
#include <string>
#include <memory>

// The Stanford Bunny volume is 512x512x361
static const int kBunnySize = 512;
static const int kBunnyN = 361;

uint16_t* loadBunnyCT(const std::string& base_directory);
