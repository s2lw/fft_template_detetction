#include "fft-seq.hpp"
#include "utils.hpp"
#include <vector>
#include <complex>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Computes the bit-reversed index for a given index and number of bits
unsigned int bitReversal(unsigned int index, unsigned int bits) {
    unsigned int reversed = 0;
    for (unsigned int i = 0; i < bits; ++i) {
        if (index & (1u << i))
            reversed |= (1u << (bits - 1 - i));
    }
    return reversed;
}

// Returns a vector with elements in bit-reversed order
std::vector<std::complex<double>> bitReverseOrder(const std::vector<std::complex<double>>& input) {
    size_t n = input.size();
    unsigned int bits = 0;
    while ((1ull << bits) < n) ++bits;

    std::vector<std::complex<double>> output(n);
    for (size_t i = 0; i < n; ++i) {
        unsigned int rev = bitReversal(static_cast<unsigned int>(i), bits);
        output[rev] = input[i];
    }
    return output;
}

/**
 * @brief Sequential 1D FFT implementation (Cooley-Tukey, radix-2, iterative)
 * @param input Vector of complex numbers
 * @return Vector containing the FFT result
 */
std::vector<std::complex<double>> fft1D_seq(const std::vector<std::complex<double>>& input) {
    std::vector<std::complex<double>> output = bitReverseOrder(input);
    std::vector<std::complex<double>> temp_buffer(input.size());

    int log2n = static_cast<int>(std::log2(input.size()));

    std::vector<std::complex<double>> input_buf = output;
    std::vector<std::complex<double>> output_buf = temp_buffer;

    for (int s = 0; s < log2n; s++) {
        int k = 1 << (s+1); // 2^s, bitwise left shift
        int half_k = k /2;
        const double angle = -2.0 * M_PI / k;
        const std::complex<double> w_m = std::exp(std::complex<double>(0, angle));
        
        for (int j = 0; j < input.size() / k; j++){
            int start = j * k;

            std::complex<double> w = 1.0;
            for (int i = 0; i < half_k; i++){
                std::complex<double> u = input_buf[start + i];
                std::complex<double> t = w * input_buf[start + i + half_k];
                output_buf[start + i] = u + t;
                output_buf[start + i + half_k] = u - t;
                w = w * w_m;
            }
        }
        std::swap(input_buf, output_buf);
    }

    if (log2n % 2 != 0)
        return input_buf;
    else
        return output_buf;
}