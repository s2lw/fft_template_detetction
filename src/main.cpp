#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "fft-seq.hpp"
#include "fft-openmp.hpp"
#include "ifft-seq.hpp"
#include "utils.hpp"
#include <complex>

int main(int argc, char** argv)
{
    // Wczytaj obrazy
    //if (argc < 3) {
    //    std::cerr << "Usage: " << argv[0] << " <path_to_large_image> <path_to_small_image>" << std::endl;
    //    return 1;
    //}

    // Load images from command-line arguments
    //cv::Mat image = loadGrayscaleImage(argv[1]);
    //cv::Mat image_small = loadGrayscaleImage(argv[2]);
    cv::Mat image = loadGrayscaleImage("C:/Users/jedrz/Desktop/rownoloegle/fft-projekt/test_image.png");
    cv::Mat image_small = loadGrayscaleImage("C:/Users/jedrz/Desktop/rownoloegle/fft-projekt/test_tank.png");

    image.convertTo(image, CV_64F);
    image_small.convertTo(image_small, CV_64F);
    
    if (image.empty() || image_small.empty()) {
        std::cout << "Could not read the image. Please check the path." << std::endl;
        return 1;
    }

    std::cout << "Successfully read image: " << image.rows << "x" << image.cols << std::endl;
    std::cout << "Successfully read image: " << image_small.rows << "x" << image_small.cols << std::endl;

    // Opcjonalnie: padding obrazu do następnej potęgi 2
    cv::Mat paddedImage, paddedSmallImage;
    padForFFT(image, image_small, paddedImage, paddedSmallImage);
    saveImage("padded.png", paddedImage);
    saveImage("padded_small.png", paddedSmallImage);


    // Oblicz mapę korelacji (domyślnie używamy niepaddowanego obrazu)
    cv::Mat corrImage = computeCorrelationMap(paddedImage, paddedSmallImage, fft1D_seq, ifft1D_seq);
    
    
    // Oblicz FFT dużego obrazu i zapisz jego wizualizację
    auto complexImage = convertImageToComplexVector(paddedImage, false);
    //auto fftResult = applyFFT2D(complexImage, fft1D_openmp);
    
    // --- Measure Sequential version ---
    auto start_seq = std::chrono::high_resolution_clock::now();
    auto fftResult_seq = applyFFT2D(complexImage, fft1D_seq);
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_seq = end_seq - start_seq;
    std::cout << "Sequential FFT2D time: " << duration_seq.count() << " seconds\n";

    // --- Measure OpenMP version ---
    auto start_omp = std::chrono::high_resolution_clock::now();
    auto fftResult = applyFFT2D(complexImage, fft1D_openmp);
    auto end_omp = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration_omp = end_omp - start_omp;
    std::cout << "OpenMP FFT2D time: " << duration_omp.count() << " seconds\n";
    
    // Wizualizacja FFT (opcjonalna)
    cv::Mat magImage(paddedImage.rows, paddedImage.cols, CV_64F);
    for (int y = 0; y < paddedImage.rows; ++y) {
        for (int x = 0; x < paddedImage.cols; ++x) {
            magImage.at<double>(y, x) = std::abs(fftResult[y][x]);
        }
    }
    
    // Logarytmowanie dla lepszej widoczności
    magImage += 1.0;
    cv::log(magImage, magImage);
    
    // Normalizacja i zapis FFT
    cv::normalize(magImage, magImage, 0, 255, cv::NORM_MINMAX);
    magImage.convertTo(magImage, CV_8U);
    saveImage("fft_magnitude.png", magImage);

    // Normalizacja mapy korelacji
    cv::normalize(corrImage, corrImage, 0, 255, cv::NORM_MINMAX);
    corrImage.convertTo(corrImage, CV_8U);
    
    // Znajdź najlepsze dopasowanie
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(corrImage, &minVal, &maxVal, &minLoc, &maxLoc);
    
    // Oblicz środek wzorca
    cv::Point wzorzecCenter(image_small.cols / 2, image_small.rows / 2);
    cv::Point realMatch = maxLoc + wzorzecCenter;
    
    std::cout << "Najlepsze dopasowanie w punkcie: " << maxLoc << std::endl;
    std::cout << "Najlepsze dopasowanie (środek wzorca): " << realMatch << std::endl;
    
    // Zaznacz najlepsze dopasowanie na mapie korelacji
    cv::circle(corrImage, realMatch, 10, cv::Scalar(100), 2);
    saveImage("correlation_map.png", corrImage);
    
    // Znajdź wszystkie potencjalne dopasowania
    double threshold = 0.9; // próg np. 90% maksimum
    std::vector<cv::Point> locations = findMatches(corrImage, threshold, wzorzecCenter);
    
    std::cout << "Znaleziono " << locations.size() << " potencjalnych dopasowań." << std::endl;
    
    // Narysuj dopasowania na oryginalnym obrazie
    cv::Mat imageGray = image.clone();
    drawMatches(imageGray, locations, 10, 180);
    saveImage("all_matches_on_original.png", imageGray);

    return 0;
}