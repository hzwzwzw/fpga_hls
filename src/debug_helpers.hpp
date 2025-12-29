#ifndef DEBUG_HELPERS_HPP
#define DEBUG_HELPERS_HPP

#include <cstdio>
#include <numeric>

template<typename T>
void trace_tensor(const char* name, const T* tensor, int size) {
    double sum = 0.0;
    for (int i = 0; i < size; ++i) {
        sum += tensor[i];
    }
    printf("--- %s ---\n", name);
    printf("  - Head: [%.6f, %.6f, %.6f, ...]\n", (float)tensor[0], (float)tensor[1], (float)tensor[2]);
    printf("  - Sum:  %.6f\n\n", sum);
}

#endif // DEBUG_HELPERS_HPP
