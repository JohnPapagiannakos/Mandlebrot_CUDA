// Author: Yannis Papagiannakos
#define USE_CUDA 1

#include <string>
#include <iterator>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <array>
#include <cmath>
#include <complex>

#include <chrono>
#include <ctime>
#include <unistd.h>

#include "masterlib.hpp"

#include "OpenGL/plot.hpp"

int main ( void ){
    using namespace std::complex_literals;

    // const size_t dim = 1000;

    // std::array<size_t, 2> Dims = {dim, dim};
    std::array<size_t, 2> Dims = {1920, 1080};

    const int MAX_ITERS = 500;

    // double offset_x = 1.0;
    // double offset_y = 1.0;

    double ratio = Dims[0] / Dims[1];

    double offset_x = ratio;
    double offset_y = 1.0;

    std::array<double, 2> center = {0, 0};

    // double alpha = 3*M_PI_4; // pi/4
    // DoubleComplex tmp_const_c = 0.7885 * std::exp(1i * alpha);
    
    // DoubleComplex tmp_const_c = -1.476;

    DoubleComplex tmp_const_c = -0.79 + 0.15i;

    std::cout << "c=" << real(tmp_const_c);
    if(imag(tmp_const_c)>=0)
        std::cout << "+" << imag(tmp_const_c) << "i" << std::endl;
    else
        std::cout << imag(tmp_const_c) << "i" << std::endl;
    cuDoubleComplex const_c;
    const_c.x = real(tmp_const_c);
    const_c.y = imag(tmp_const_c);

    std::array<double, 2> XLIM = {center[0] - offset_x, center[0] + offset_x};
    std::array<double, 2> YLIM = {center[1] - offset_y, center[1] + offset_y};

    // Create meshgrid
    size_t prod_dims = Dims[0] * Dims[1];

    cuDoubleComplex *z0;
    cudaMallocManaged((void **)&z0, Dims[0] * Dims[1] * sizeof(cuDoubleComplex));
    cudameshgrid(XLIM, YLIM, Dims, z0);

    double *count;
    cudaMallocManaged((void **)&count, prod_dims * sizeof(double)); // unified mem.

    std::fill(&count[0], &count[prod_dims - 1], 1.0);
    std::chrono::time_point<std::chrono::system_clock> start, end;
    
    // Start timers
    cudaDeviceSynchronize();
    start = std::chrono::system_clock::now();
    v2::cuJuliaOp2(z0, const_c, count, prod_dims, MAX_ITERS);
    // v2::cuJuliaOp3(z0, const_c, count, prod_dims, MAX_ITERS);
    cudaDeviceSynchronize();
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    std::vector<double> _data(prod_dims, 1);
    for (size_t idx = 0; idx < prod_dims; idx++)
    {
        _data[idx] = count[idx];
    }

    // Illustrate fractal
    figure<double> fig(Dims);
    fig.newFigure("Fig 1");
    fig.plotRGB(_data);
    // fig.showFigure();
    sleep(5);
    fig.closeFigure();

    cudaFree(z0);
    cudaFree(count);

    return 0;
}
