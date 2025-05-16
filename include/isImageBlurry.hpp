#pragma once

#include <vector>
#include <complex>

/**
 * @brief Analizuje, czy obraz jest rozmazany na podstawie widma FFT.
 * 
 * @param fftResult Wynik FFT obrazu (2D)
 * @param thresh Próg rozmycia – im wyższy, tym bardziej "tolerancyjna" detekcja
 * @return std::pair<double, bool> — średnia wartość widma, czy obraz rozmazany
 */
std::pair<double, bool> isImageBlurry(
    const std::vector<std::vector<std::complex<double>>>& fftResult,
    double thresh
);
