// Author: Yannis Papagiannakos

#include "operators.hpp"

__global__ void vecAdd(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

__global__ void vecJuliaOp(cuDoubleComplex *a, const cuDoubleComplex c, double *count, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    double absval;

    int bool_count;

    // Make sure we do not go out of bounds
    if (id < n)
    {
        double a_x = a[id].x;
        double a_y = a[id].y;

        a_x = (a[id].x * a[id].x) - (a[id].y * a[id].y) + c.x;
        a_y = 2*(a[id].x * a[id].y) + c.y;
        absval = sqrt((a_x * a_x) + (a_y * a_y));
        bool_count = (absval <= 2);
        count[id] = count[id] + bool_count;   
        
        a[id].x = a_x;
        a[id].y = a_y;
    }
}

__global__ void vecLog(double *a, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        a[id] = log(a[id]);
}


extern "C" void cuda_vecAdd(double *A, double *B, double *C, int length)
{
    int blockSize, gridSize;
    blockSize = CUDA_VEC_BLOCK_SIZE;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)length / blockSize);

    vecAdd<<<gridSize, blockSize>>>(A, B, C, length);
}   


extern "C" void cuda_vecJuliaOp(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS)
{
    int blockSize, gridSize;
    blockSize = CUDA_VEC_BLOCK_SIZE;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)length / blockSize);

    for(int iter=0; iter<=MAX_ITERS; iter++)
    {
        vecJuliaOp<<<gridSize, blockSize>>>(z, c, count, length);
    }
    cudaDeviceSynchronize();
    vecLog<<<gridSize, blockSize>>>(count, length);
} 