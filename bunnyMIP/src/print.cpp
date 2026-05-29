
#include <cstdarg>
#include <iostream>

#include "print.h"

void fmt_appendV(std::string* dst, const char* format, va_list ap)
{
    char buffer[1024];

    // To prevent possible changes to *ap*, we use a copy.
    va_list cap;
    va_copy(cap, ap);
    int result = vsnprintf(buffer, sizeof(buffer), format, cap);
    va_end(cap);

    if (result < (int)sizeof(buffer)){
        if (result < 0)
            return; // vnsprintf failed :(

        dst->append(buffer, result);
        return;
    }

    // The buffer was too small. Lucky for us vnsprintf
    // tells us how much memory it needs.
    int len = result + 1; // +1 for '\0'
    char* heap_buffer = new char[len];

    va_copy(cap, ap);
    result = vsnprintf(heap_buffer, len, format, cap);
    va_end(cap);

    if (result >=0 && result < len)
        dst->append(heap_buffer, result);

    delete [] heap_buffer;
}

// Output a formated string to stdout.
void print(const char* format, ...)
{
    std::string message;

    va_list  ap;
    va_start(ap, format);
    fmt_appendV(&message, format, ap);
    va_end(ap);

    std::cout << message;
}

// Create a string from a formated text.
std::string format(const char* format, ...)
{
    std::string message;

    va_list  ap;
    va_start(ap, format);
    fmt_appendV(&message, format, ap);
    va_end(ap);

    return message;
}
