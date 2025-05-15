# FFT-Based Template Matching

This project implements template matching using a custom implementation of the Fast Fourier Transform (FFT) in C++.  
It supports both a sequential and an OpenMP-parallelized version of the algorithm, allowing users to compare their performance.  

The program loads two grayscale images: a large image and a smaller template image.  
It performs cross-correlation in the frequency domain using FFT to locate where the template best matches the large image.  
The sequential and parallel implementations are timed independently so that execution speed can be compared.  

### Build Instructions

To use the program, first build it using CMake.  
Clone the repository, then create and enter a `build` directory, run `cmake ..`, and finally run `cmake --build .` to compile.

### Usage

The resulting executable can be run from the command line using:

./fft-template-matching <path_to_large_image> <path_to_template_image>


### Output

The output will include timing results for both FFT implementations,  
and it may optionally display or save the result of the template match.

### Requirements

- OpenCV  
- OpenMP  
- A compiler supporting C++17 or newer
