#pragma once
#include <string>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>

/**
 * @brief Sekwencyjna implementacja FFT 1D dla pojedynczego wiersza (Cooley-Tukey)
 * @param input Wektor liczb zespolonych
 * @return Wektor zawierający wynik FFT
 */
std::vector<std::complex<double>> fft1D_seq(const std::vector<std::complex<double>>& input);

