
#pragma once

#include <string>

// Output a formated string to stdout.
void print(const char* format, ...) __attribute__((__format__ (__printf__, 1, 2)));

// Create a string from a formated text.
std::string format(const char* format, ...) __attribute__((__format__ (__printf__, 1, 2)));
