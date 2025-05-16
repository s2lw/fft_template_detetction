#include "utils.hpp"
#include <complex>
#include <vector>
#include <opencv2/opencv.hpp>
#include <functional>
#include <cmath>

std::vector<std::vector<std::complex<double>>> convertImageToComplexVector(
    const cv::Mat& input, bool applyWindow)
{
    int rows = input.rows;
    int cols = input.cols;
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));

    // Prepare Hanning windows if needed
    cv::Mat winRow, winCol;
    if (applyWindow) {
        cv::createHanningWindow(winRow, cv::Size(cols, 1), CV_64F);
        cv::createHanningWindow(winCol, cv::Size(1, rows), CV_64F);
    }

    for (int y = 0; y < rows; ++y) {
        for (int x = 0; x < cols; ++x) {
            double val;
            if (input.type() == CV_8U)
                val = static_cast<double>(input.at<uchar>(y, x));
            else if (input.type() == CV_64F)
                val = input.at<double>(y, x);
            else
                val = 0.0;  // Domyślnie
                
            if (applyWindow) {
                val *= winRow.at<double>(0, x) * winCol.at<double>(y, 0);
            }
            output[y][x] = std::complex<double>(val, 0.0);
        }
    }
    return output;
}

// Applies 1D FFT function to all rows and then all columns of a 2D complex vector (image)
std::vector<std::vector<std::complex<double>>> applyFFT2D(
    const std::vector<std::vector<std::complex<double>>>& input,
    const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)> &fft1D_func)
{
    int rows = input.size();
    int cols = input[0].size();
    std::vector<std::vector<std::complex<double>>> temp(rows, std::vector<std::complex<double>>(cols));
    std::vector<std::vector<std::complex<double>>> output(rows, std::vector<std::complex<double>>(cols));

    // Apply FFT to each row
    for (int y = 0; y < rows; ++y) {
        temp[y] = fft1D_func(input[y]);
    }

    // Apply FFT to each column
    for (int x = 0; x < cols; ++x) {
        std::vector<std::complex<double>> col(rows);
        for (int y = 0; y < rows; ++y)
            col[y] = temp[y][x];
        col = fft1D_func(col);
        for (int y = 0; y < rows; ++y)
            output[y][x] = col[y];
    }

    return output;
}

std::vector<std::vector<std::complex<double>>> padWithZeros(
    const std::vector<std::vector<std::complex<double>>>& small,
    int targetRows, int targetCols)
{
    std::vector<std::vector<std::complex<double>>> result(
        targetRows, std::vector<std::complex<double>>(targetCols, std::complex<double>(0.0, 0.0)));

    int smallRows = small.size();
    int smallCols = small.empty() ? 0 : small[0].size();

    for (int y = 0; y < smallRows && y < targetRows; ++y) {
        for (int x = 0; x < smallCols && x < targetCols; ++x) {
            result[y][x] = small[y][x];
        }
    }
    return result;
}



// Loads a grayscale image from file
cv::Mat loadGrayscaleImage(const std::string& path) {
    return cv::imread(path, cv::IMREAD_GRAYSCALE);
}

// Helper: next power of 2
static int nextPowerOf2(int n) {
    if (n <= 0) return 1;
    return std::pow(2, std::ceil(std::log2(n)));
}

// Pads image to next power of 2 in both dimensions, padding symmetrically
cv::Mat padToNextPowerOf2(const cv::Mat& image) {
    int newRows = nextPowerOf2(image.rows);
    int newCols = nextPowerOf2(image.cols);
    int top = (newRows - image.rows) / 2;
    int bottom = newRows - image.rows - top;
    int left = (newCols - image.cols) / 2;
    int right = newCols - image.cols - left;
    cv::Mat padded;
    cv::copyMakeBorder(image, padded, top, bottom, left, right, cv::BORDER_CONSTANT, cv::Scalar(0));
    return padded;
}

// Computes the correlation map using FFT-based template matching
cv::Mat computeCorrelationMap(
    const cv::Mat& bigImage,
    const cv::Mat& smallImage,
    const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)> &fft1D,
    const std::function<std::vector<std::complex<double>>(const std::vector<std::complex<double>>&)> &ifft1D
) {
    auto complexBig = convertImageToComplexVector(bigImage, false);
    auto complexSmall = convertImageToComplexVector(smallImage, false);
    auto paddedSmall = padWithZeros(complexSmall, bigImage.rows, bigImage.cols);

    auto fftBig = applyFFT2D(complexBig, fft1D);
    auto fftSmall = applyFFT2D(paddedSmall, fft1D);

    std::vector<std::vector<std::complex<double>>> corrFreq(
        bigImage.rows, std::vector<std::complex<double>>(bigImage.cols));
    for (int y = 0; y < bigImage.rows; ++y)
        for (int x = 0; x < bigImage.cols; ++x)
            corrFreq[y][x] = fftBig[y][x] * std::conj(fftSmall[y][x]);

    auto corrMap = applyFFT2D(corrFreq, ifft1D);

    // Convert to cv::Mat (magnitude)
    cv::Mat corrImage(bigImage.rows, bigImage.cols, CV_64F);
    for (int y = 0; y < bigImage.rows; ++y)
        for (int x = 0; x < bigImage.cols; ++x)
            corrImage.at<double>(y, x) = std::abs(corrMap[y][x]);
    return corrImage;
}

// Finds all matches above a threshold (0-1, relative to max)
std::vector<cv::Point> findMatches(const cv::Mat& corrImage, double threshold, cv::Point wzorzecCenter) {
    double minVal, maxVal;
    cv::minMaxLoc(corrImage, &minVal, &maxVal);
    cv::Mat mask = corrImage > (threshold * maxVal);
    std::vector<cv::Point> locations;
    cv::findNonZero(mask, locations);
    // Adjust by wzorzecCenter if needed
    for (auto& pt : locations)
        pt += wzorzecCenter;
    return locations;
}

// Draws circles at given locations
void drawMatches(cv::Mat& image, const std::vector<cv::Point>& locations, int radius, uchar color) {
    for (const auto& pt : locations) {
        if (pt.x >= 0 && pt.x < image.cols && pt.y >= 0 && pt.y < image.rows)
            cv::circle(image, pt, radius, cv::Scalar(color), 2);
    }
}

// Saves image to file
void saveImage(const std::string& filename, const cv::Mat& image) {
    std::string path = "C:/Users/jedrz/Desktop/rownoloegle/fft-projekt/output/" + filename;
    cv::Mat imageToSave;
    
    if (image.depth() != CV_8U) {
        // Konwertuj do nowej macierzy, nie modyfikując oryginalnej
        image.convertTo(imageToSave, CV_8U, 1.0, 0.0);
    } else {
        // Kopiuj obraz jeśli już jest CV_8U
        imageToSave = image.clone();
    }
    
    bool success = cv::imwrite(path, imageToSave);
    if (!success) {
        std::cout << "Error: Could not save image to " << path << std::endl;
    } else {
        std::cout << "Image saved to " << path << std::endl;
    }
}

void padForFFT(const cv::Mat& image, const cv::Mat& templ,
               cv::Mat& paddedImg, cv::Mat& paddedTpl) {
    // Sprawdź, czy obrazy mają typ CV_64F (czyli double)
    CV_Assert(image.type() == CV_64F && templ.type() == CV_64F);

    int M = image.rows, N = image.cols;
    int P = templ.rows, Q = templ.cols;

    // Symetryczne obramowanie (dla pełnej konwolucji)
    int pad_top    = P-1;
    int pad_bottom = P - 1;
    int pad_left   = Q-1;
    int pad_right  = Q - 1;

    cv::Mat tempImg, tempTpl;
    cv::copyMakeBorder(image, tempImg, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0.0));
    cv::copyMakeBorder(templ, tempTpl, pad_top, pad_bottom, pad_left, pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0.0));

    // Oblicz rozmiar końcowy (najbliższa potęga dwójki)
    int H = nextPowerOf2(tempImg.rows);
    int W = nextPowerOf2(tempImg.cols);

    // Dodatkowe zerowanie do rozmiaru potęgi 2
    cv::copyMakeBorder(tempImg, paddedImg,
                       0, H - tempImg.rows,
                       0, W - tempImg.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0.0));
    cv::copyMakeBorder(tempTpl, paddedTpl,
                       0, H - tempTpl.rows,
                       0, W - tempTpl.cols,
                       cv::BORDER_CONSTANT, cv::Scalar::all(0.0));
}
