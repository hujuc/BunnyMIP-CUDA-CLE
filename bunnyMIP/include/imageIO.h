#pragma once

#include <string>
#include <cstdint>

bool savePGM16(const std::string& filename, const uint16_t* data, uint32_t width, uint32_t height);
