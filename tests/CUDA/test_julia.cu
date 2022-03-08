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

#include "masterlib.hpp"

int main ( void ){
    using namespace std::complex_literals;

    const size_t dim = 5000;

    std::array<size_t, 2> Dims = {dim, dim};

    const int MAX_ITERS = 500;

    double offset = 1.0;

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

    std::array<double, 2> XLIM = {center[0] - offset, center[0] + offset};
    std::array<double, 2> YLIM = {center[1] - offset, center[1] + offset};

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
              

    // Write resulting fractal to binary file
    Write_to_File<double>(prod_dims, count, "count.bin");
    std::array<double, dim> x_vec;
    std::array<double, dim> y_vec;

    double dx = (XLIM[1] - XLIM[0]) / Dims[0];
    double dy = (YLIM[1] - YLIM[0]) / Dims[1];

    x_vec[0] = XLIM[0];
    y_vec[0] = YLIM[0];
    
    for (int x = 1; x < dim; x++)
    {
        x_vec[x] = x_vec[x-1] + dx;
    }

    for (int y = 1; y < dim; y++)
    {
        y_vec[y] = y_vec[y-1] + dy;
    }

    Write_to_File<double>(dim, &x_vec[0], "x.bin");
    Write_to_File<double>(dim, &y_vec[0], "y.bin");

    cudaFree(z0);
    cudaFree(count);

    return 0 ;
}
