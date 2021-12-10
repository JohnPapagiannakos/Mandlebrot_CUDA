// Author: Yannis Papagiannakos

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

    const int dim = 1000;

    std::array<int, 2> Dims = {dim, dim};

    const int MAX_ITERS = 500;

    double offset = 1.5;

    std::array<double, 2> center = {0, 0};

    double alpha = 3*M_PI_4; // pi/4
    DoubleComplex const_c = 0.7885 * std::exp(1i * alpha);

    std::cout << "c=" << real(const_c);
    if(imag(const_c)>=0)
        std::cout << "+" << imag(const_c) << "i" << std::endl;
    else
        std::cout << imag(const_c) << "i" << std::endl;
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
    constexpr size_t prod_dims = dim * dim;
    // std::array<DoubleComplex, prod_dims> z0;

    // std::array<double, prod_dims> count;

    // for (int cols=0; cols < dim; cols++)
    // {
    //     for (int rows = 0; rows < dim; rows++)
    //     {
    //         size_t lin_idx = (cols * dim) + rows;
    //         z0[lin_idx].real(x_vec[rows]);
    //         z0[lin_idx].imag(y_vec[cols]);
    //         count[lin_idx] = 1;
    //     }
    // }

    std::vector<DoubleComplex> z0;

    std::vector<double> count;

    for (int cols = 0; cols < dim; cols++)
    {
        for (int rows = 0; rows < dim; rows++)
        {
            DoubleComplex tmp;
            tmp.real(x_vec[rows]);
            tmp.imag(y_vec[cols]);
            z0.push_back(tmp);
            count.push_back(1);
        }
    }

    std::chrono::time_point<std::chrono::system_clock> start, end;
    
    // Start timers
    start = std::chrono::system_clock::now();
    JuliaOp2(&z0[0], const_c, &count[0], dim * dim, MAX_ITERS);
    // JuliaOp3(&z0[0], const_c, &count[0], dim*dim, MAX_ITERS);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
              

    // Write resulting fractal to binary file
    Write_to_File(dim, dim, &count[0], "count.bin");
    Write_to_File(dim, 1, &x_vec[0], "x.bin");
    Write_to_File(dim, 1, &y_vec[0], "y.bin");

    return 0 ;
}
