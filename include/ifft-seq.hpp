#pragma once
#include <vector>
#include <complex>

/**
 * @brief Odwrotna 1D FFT (IFFT) wykorzystująca OpenCV (cv::dft)
 * @param input Wektor liczb zespolonych
 * @return Wektor liczb zespolonych po IFFT
 */
std::vector<std::complex<double>> ifft1D_seq(const std::vector<std::complex<double>>& input);

std::vector<std::vector<std::complex<double>>> ifft2D_seq(
    const std::vector<std::vector<std::complex<double>>>& input
);