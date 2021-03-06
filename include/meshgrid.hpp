#include <array>
#include <vector>
#include <cassert>
#include <types.hpp>
#include <type_traits>
#include <iostream>

#if USE_CUDA
    #include "cublas_v2.h"
    #include "cuda_runtime.h"
#endif
void meshgrid(std::array<double, 2> XLIM, std::array<double, 2> YLIM, std::array<size_t, 2> Dims, std::vector<DoubleComplex> &z0)
{
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

    // Create meshgrid
    size_t prod_dims = Dims[0] * Dims[1];

    assert(z0.size() == prod_dims);

    std::vector<double> count(prod_dims, 1);

    for (size_t cols = 0; cols < Dims[1]; cols++)
    {
        size_t lin_idx = (cols * Dims[0]);
        for (size_t rows = 0; rows < Dims[0]; rows++)
        {
            z0[lin_idx].real(x_vec[rows]);
            z0[lin_idx].imag(y_vec[cols]);
            lin_idx++;
        }
    }
}

#if USE_CUDA
void cudameshgrid(std::array<double, 2> XLIM, std::array<double, 2> YLIM, std::array<size_t, 2> Dims, cuDoubleComplex *z0)
{
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

    // Create meshgrid
    size_t prod_dims = Dims[0] * Dims[1];

    // cuDoubleComplex *z0;
    // cudaMallocManaged((void **)&z0, Dims[0] * Dims[1] * sizeof(cuDoubleComplex));

    for (size_t cols = 0; cols < Dims[1]; cols++)
    {
        size_t lin_idx = (cols * Dims[0]);
        for (size_t rows = 0; rows < Dims[0]; rows++)
        {
            z0[lin_idx].x = x_vec[rows];
            z0[lin_idx].y = y_vec[cols];
            lin_idx++;
        }
    }
}
#endif
