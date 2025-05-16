#pragma once
#include <string>
#include <vector>
#include <complex>
#include <opencv2/opencv.hpp>
#include <functional>

// Funkcja zwracająca najbliższą większą lub równą potęgę liczby 2
int nextPowerOf2(int n);


/**
 * @brief Funkcja do pomiaru czasu wykonania FFT
 * @param fftFunction Funkcja FFT do zmierzenia
 * @param input Dane wejściowe
 * @param iterations Liczba iteracji do wykonania (dla dokładniejszego pomiaru)
 * @return Średni czas wykonania w milisekundach
 */
template<typename FFTFunc, typename InputType>
double measureFFTExecutionTime(FFTFunc fftFunction, const InputType& input, int iterations = 10);

/**
 * @brief Funkcja do porównania wyników różnych implementacji FFT
 * @param original Wynik referencyjnej implementacji FFT
 * @param tested Wynik testowanej implementacji FFT
 * @return Średni błąd kwadratowy między implementacjami
 */
double compareFFTResults(
    const std::vector<std::vector<std::complex<double>>>& original,
    const std::vector<std::vector<std::complex<double>>>& tested);

/**
 * @brief Converts a grayscale OpenCV image to a 2D vector of complex numbers.
 * @param input Input grayscale image (cv::Mat)
 * @param applyWindow Whether to apply Hanning window
 * @return 2D vector of complex numbers
 */
std::vector<std::vector<std::complex<double>>> convertImageToComplexVector(
    const cv::Mat& input, bool applyWindow);

/**
 * @brief Applies a 1D FFT function to all rows and then all columns of a 2D complex vector (image).
 * @param input 2D vector of complex numbers (image)
 * @param fft1D_func Function to apply to each row/column (e.g., fft1D_seq)
 * @return 2D vector of complex numbers after 2D FFT
 */
std::vector<std::vector<std::complex<double>>> applyFFT2D(
    const std::vector<std::vector<std::complex<double>>>& input,
    const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)>& fft1D_func);

/**
 * @brief Umieszcza mniejszy obrazek (patch) w większej macierzy o zadanym rozmiarze, resztę wypełnia zerami.
 * @param small Obrazek wejściowy (mniejszy)
 * @param targetRows Liczba wierszy większego obrazka
 * @param targetCols Liczba kolumn większego obrazka
 * @return 2D wektor liczb zespolonych o rozmiarze targetRows x targetCols
 */
std::vector<std::vector<std::complex<double>>> padWithZeros(
    const std::vector<std::vector<std::complex<double>>>& small,
    int targetRows, int targetCols);    


// Loads a grayscale image from file
cv::Mat loadGrayscaleImage(const std::string& path);

// Pads image to next power of 2 in both dimensions, padding symmetrically
cv::Mat padToNextPowerOf2(const cv::Mat& image);

// Computes the correlation map using FFT-based template matching
cv::Mat computeCorrelationMap(
    const cv::Mat& bigImage,
    const cv::Mat& smallImage,
    const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)> &fft1D,
    const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)> &ifft1D
);

// Finds all matches above a threshold (0-1, relative to max)
std::vector<cv::Point> findMatches(const cv::Mat& corrImage, double threshold, cv::Point wzorzecCenter);

// Draws circles at given locations
void drawMatches(cv::Mat& image, const std::vector<cv::Point>& locations, int radius, uchar color);

// Saves image to file
void saveImage(const std::string& filename, const cv::Mat& image);

// Pads image for FFT (to next power of 2)
void padForFFT(const cv::Mat& image, const cv::Mat& templ,
               cv::Mat& paddedImg, cv::Mat& paddedTpl);