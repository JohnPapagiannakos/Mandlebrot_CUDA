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
#include <unistd.h>

#include "masterlib.hpp"

#include "OpenGL/plot.hpp"

int main ( void ){
    using namespace std::complex_literals;

    const int dim = 1000;

    std::array<size_t, 2> Dims = {dim, dim};

    const int MAX_ITERS = 500;

    double offset = 1.0;

    std::array<double, 2> center = {0, 0};

    // double alpha = 3*M_PI_4; // pi/4
    // DoubleComplex const_c = 0.7885 * std::exp(1i * alpha);

    DoubleComplex const_c = -1.476;

    // DoubleComplex const_c = -0.79 + 0.15i;

    std::cout << "c=" << real(const_c);
    if(imag(const_c)>=0)
        std::cout << "+" << imag(const_c) << "i" << std::endl;
    else
        std::cout << imag(const_c) << "i" << std::endl;


    std::array<double, 2> XLIM = {center[0] - offset, center[0] + offset};
    std::array<double, 2> YLIM = {center[1] - offset, center[1] + offset};

   
    // Create meshgrid
    size_t prod_dims = Dims[0] * Dims[1];

    std::vector<DoubleComplex> z0(prod_dims);
    meshgrid(XLIM, YLIM, Dims, z0);

    std::vector<double> count(prod_dims, 1);

    std::chrono::time_point<std::chrono::system_clock> start, end;
    
    // Start timers
    start = std::chrono::system_clock::now();
    v2::JuliaOp2(&z0[0], const_c, &count[0], prod_dims, MAX_ITERS);
    // v2::JuliaOp3(&z0[0], const_c, &count[0], prod_dims, MAX_ITERS);
    end = std::chrono::system_clock::now();

    std::chrono::duration<double> elapsed_seconds = end - start;

    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";

    // Illustrate fractal
    figure<double> fig(Dims);
    fig.newFigure("Fig 1");
    fig.plotRGB(count);
    // fig.showFigure();
    sleep(5);
    fig.closeFigure();


    return 0 ;
}
