// Author: Yannis Papagiannakos

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuComplex.h>
// #include "device_launch_parameters.h"
#include <types.hpp>

#define CUDA_VEC_BLOCK_SIZE 1024

extern "C" void cuaddv(double *A, double *B, double *C, int length);

extern "C" void cuJuliaOp2(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);

extern "C" void cuJuliaOp3(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);
