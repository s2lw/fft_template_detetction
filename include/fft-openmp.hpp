#pragma once
#include <string>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>

/**
 * @brief Implementacja FFT 1D dla pojedynczego wiersza (Cooley-Tukey) z wykorzystaniem OpenMP
 * @param input Wektor liczb zespolonych
 * @return Wektor zawierający wynik FFT
 */
std::vector<std::complex<double>> fft1D_openmp(const std::vector<std::complex<double>>& input);
