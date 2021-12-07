// Author: Yannis Papagiannakos

#include "cuda.h"
#include "cuda_runtime.h"
#include <cuComplex.h>
#include "device_launch_parameters.h"

#define CUDA_VEC_BLOCK_SIZE 1024

// __global__ void vecAdd(double *a, double *b, double *c, int n);

extern "C" void cuda_vecAdd(double *A, double *B, double *C, int length);

extern "C" void cuda_vecJuliaOp(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS);
