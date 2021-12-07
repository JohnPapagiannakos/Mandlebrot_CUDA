// Author: Yannis Papagiannakos

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "operators.hpp"
#include "IO.hpp"

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


int main ( void ){
    using namespace std::complex_literals;

    const int dim = 10000;

    std::array<int, 2> Dims = {dim, dim};

    const int MAX_ITERS = 100;

    double offset = 0.5;

    std::array<double, 2> center = {0, 0};

    double alpha = M_PI_2; // pi/4
    std::complex<double> tmp_const_c = 0.7885 * std::exp(1i * alpha);
    std::cout << tmp_const_c << std::endl;
    cuDoubleComplex const_c;
    const_c.x = real(tmp_const_c);
    const_c.y = imag(tmp_const_c);

    //
    std::array<double, 2> XLIM = {center[0] - offset, center[0] + offset};
    std::array<double, 2> YLIM = {center[1] - offset, center[1] + offset};

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


    // Create meshgrid
    cuDoubleComplex *z0;
    cudaMallocManaged((void **)&z0, Dims[0] * Dims[1] * sizeof(cuDoubleComplex)); // unified mem.

    double *count;
    cudaMallocManaged((void **)&count, Dims[0] * Dims[1] * sizeof(double)); // unified mem.

    for (int cols=0; cols < dim; cols++)
    {
        for (int rows = 0; rows < dim; rows++)
        {
            long int lin_idx = (cols * dim) + rows;
            z0[lin_idx].x = x_vec[rows];
            z0[lin_idx].y = y_vec[cols];
            count[lin_idx] = 1;
            // std::cout << z0[lin_idx].x << " " << z0[lin_idx].y << "1i" << "\t";
        }
        // std::cout << std::endl;
    }

    cudaDeviceSynchronize();

    //
    std::chrono::time_point<std::chrono::system_clock> start, end;

    start = std::chrono::system_clock::now();
    cuda_vecJuliaOp(z0, const_c, count, dim*dim, MAX_ITERS);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
              
    cudaDeviceSynchronize();

    Write_to_File(dim, dim, count, "count.bin");
    Write_to_File(dim, 1, &x_vec[0], "x.bin");
    Write_to_File(dim, 1, &y_vec[0], "y.bin");

    cudaFree(z0);
    cudaFree(count);

    return 0 ;
}
