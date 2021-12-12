// Author: Yannis Papagiannakos

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuComplex.h>
// #include "device_launch_parameters.h"
#include <types.hpp>

#define CUDA_VEC_BLOCK_SIZE 1024

void cuaddv(double *A, double *B, double *C, int length);

namespace v1
{
    void cuJuliaOp2(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);

    void cuJuliaOp3(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);
}

inline namespace v2
{
    void cuJuliaOp2(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);

    void cuJuliaOp3(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);
}