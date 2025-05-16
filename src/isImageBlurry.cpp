#include "isImageBlurry.hpp"
#include "ifft-seq.hpp" // ifft1D_seq
#include <opencv2/opencv.hpp>
#include <numeric>
#include <cmath>

// aditional functionality todo for later

std::pair<double, bool> isImageBlurry( 
    const std::vector<std::vector<std::complex<double>>>& fftResult,
    double thresh
) {
    int rows = fftResult.size();
    int cols = fftResult[0].size();

    // Skopiuj FFT i wyzeruj centrum (low frequencies)
    auto modifiedFFT = fftResult;
    int centerY = rows / 2;
    int centerX = cols / 2;
    int radius = std::min(rows, cols) / 10;

    for (int y = centerY - radius; y <= centerY + radius; ++y) {
        for (int x = centerX - radius; x <= centerX + radius; ++x) {
            if (y >= 0 && y < rows && x >= 0 && x < cols)
                modifiedFFT[y][x] = 0.0;
        }
    }

    // Inverse FFT 2D
    auto recon = ifft2D_seq(modifiedFFT);

    // Liczenie widma amplitudy i jego średniej
    std::vector<double> magnitudes;
    for (const auto& row : recon) {
        for (const auto& val : row) {
            double mag = std::abs(val);
            if (mag > 0.0)
                magnitudes.push_back(20.0 * std::log(mag));
        }
    }

    double mean = std::accumulate(magnitudes.begin(), magnitudes.end(), 0.0) / magnitudes.size();
    return {mean, mean <= thresh};
}
