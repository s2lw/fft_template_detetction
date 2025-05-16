#include "ifft-seq.hpp"
#include <opencv2/opencv.hpp>
#include <vector>
#include <complex>

std::vector<std::complex<double>> ifft1D_seq(const std::vector<std::complex<double>>& input)
{
    int n = static_cast<int>(input.size());
    // Zamiana std::vector<std::complex<double>> na cv::Mat (CV_64FC2)
    cv::Mat freqMat(1, n, CV_64FC2);
    for (int i = 0; i < n; ++i) {
        freqMat.at<cv::Vec2d>(0, i)[0] = input[i].real();
        freqMat.at<cv::Vec2d>(0, i)[1] = input[i].imag();
    }

    // Odwrotna DFT
    cv::Mat ifftMat;
    cv::dft(freqMat, ifftMat, cv::DFT_INVERSE | cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);

    // Zamiana wyniku na std::vector<std::complex<double>>
    std::vector<std::complex<double>> output(n);
    for (int i = 0; i < n; ++i) {
        double re = ifftMat.at<cv::Vec2d>(0, i)[0];
        double im = ifftMat.at<cv::Vec2d>(0, i)[1];
        output[i] = std::complex<double>(re, im);
    }
    return output;
}

// Prosta implementacja IFFT 2D
std::vector<std::vector<std::complex<double>>> ifft2D_seq(const std::vector<std::vector<std::complex<double>>>& input) {
    int rows = input.size();
    int cols = input[0].size();

    // IFFT w wierszach
    std::vector<std::vector<std::complex<double>>> rowTransformed(rows, std::vector<std::complex<double>>(cols));
    for (int i = 0; i < rows; ++i) {
        rowTransformed[i] = ifft1D_seq(input[i]);
    }

    // IFFT w kolumnach
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));
    for (int j = 0; j < cols; ++j) {
        std::vector<std::complex<double>> column(rows);
        for (int i = 0; i < rows; ++i)
            column[i] = rowTransformed[i][j];

        std::vector<std::complex<double>> columnResult = ifft1D_seq(column);

        for (int i = 0; i < rows; ++i)
            output[i][j] = columnResult[i];
    }

    return output;
}