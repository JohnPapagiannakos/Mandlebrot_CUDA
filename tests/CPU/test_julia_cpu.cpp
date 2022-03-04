// Author: Yannis Papagiannakos

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

#include "OpenGL/plot.hpp"

int main ( void ){
    using namespace std::complex_literals;

    const int dim = 800;

    std::array<size_t, 2> Dims = {dim, dim};

    const int MAX_ITERS = 500;

    double offset = 1.0;

    std::array<double, 2> center = {0, 0};

    // double alpha = 3*M_PI_4; // pi/4
    // DoubleComplex const_c = 0.7885 * std::exp(1i * alpha);

    // DoubleComplex const_c = -1.476;

    DoubleComplex const_c = -0.79 + 0.15i;

    std::cout << "c=" << real(const_c);
    if(imag(const_c)>=0)
        std::cout << "+" << imag(const_c) << "i" << std::endl;
    else
        std::cout << imag(const_c) << "i" << std::endl;


    std::array<double, 2> XLIM = {center[0] - offset, center[0] + offset};
    std::array<double, 2> YLIM = {center[1] - offset, center[1] + offset};

   
    // Create meshgrid
    size_t prod_dims = Dims[0] * Dims[1];

    std::vector<DoubleComplex> z0 = meshgrid(XLIM, YLIM, Dims);

    std::vector<double> count(prod_dims, 1);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    
    // Start timers
    start = std::chrono::system_clock::now();
    v2::JuliaOp2(&z0[0], const_c, &count[0], prod_dims, MAX_ITERS);
    // v2::JuliaOp3(&z0[0], const_c, &count[0], prod_dims, MAX_ITERS);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
              

    // Write resulting fractal to binary file
    Write_to_File<double>(prod_dims, &count[0], "count.bin");

    std::vector<double> x_vec(Dims[0]);
    std::vector<double> y_vec(Dims[1]);

    double dx = (XLIM[1] - XLIM[0]) / Dims[0];
    double dy = (YLIM[1] - YLIM[0]) / Dims[1];

    x_vec[0] = XLIM[0];
    y_vec[0] = YLIM[0];

    for (int x = 1; x < Dims[0]; x++)
    {
        x_vec[x] = x_vec[x - 1] + dx;
    }

    for (int y = 1; y < Dims[1]; y++)
    {
        y_vec[y] = y_vec[y - 1] + dy;
    }

    Write_to_File<double>(dim, &x_vec[0], "x.bin");
    Write_to_File<double>(dim, &y_vec[0], "y.bin");

    return 0 ;
}
