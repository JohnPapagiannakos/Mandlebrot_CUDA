// Author: Yannis Papagiannakos

#include "cudaoperators.hpp"

__device__ inline double margind(double x, double y)
{
    return sqrt((x * x) + (y * y));
}

__global__ void addv(double *a, double *b, double *c, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        c[id] = a[id] + b[id];
}

__global__ void juliaOp2v(cuDoubleComplex *a, const cuDoubleComplex c, double *count, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    double margin;

    int bool_count;

    // Make sure we do not go out of bounds
    if (id < n)
    {
        double a_x = c.x;
        double a_y = c.y;

        a_x += a[id].x * a[id].x - a[id].y * a[id].y;
        a_y += 2*a[id].x * a[id].y;
        margin = margind(a_x, a_y);
        bool_count = (margin <= 2);
        count[id] = count[id] + bool_count;   
        
        a[id].x = a_x;
        a[id].y = a_y;
    }
}

__global__ void juliaOp3v(cuDoubleComplex *a, const cuDoubleComplex c, double *count, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    double margin;

    int bool_count;

    // Make sure we do not go out of bounds
    if (id < n)
    {
        double a_x = c.x;
        double a_y = c.y;

        a_x += a[id].x * (a[id].x * a[id].x - 3*a[id].y * a[id].y);
        a_y += a[id].y * (3*a[id].x * a[id].x - a[id].y * a[id].y);
        margin = margind(a_x, a_y);
        bool_count = (margin <= 2);
        count[id] = count[id] + bool_count;   
        
        a[id].x = a_x;
        a[id].y = a_y;
    }
}

__global__ void logv(double *a, int n)
{
    // Get our global thread ID
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    // Make sure we do not go out of bounds
    if (id < n)
        a[id] = log(a[id]);
}


extern "C" void cuaddv(double *A, double *B, double *C, int length)
{
    int blockSize, gridSize;
    blockSize = CUDA_VEC_BLOCK_SIZE;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)length / blockSize);

    addv<<<gridSize, blockSize>>>(A, B, C, length);
}   


extern "C" void cuJuliaOp2(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS)
{
    int blockSize, gridSize;
    blockSize = CUDA_VEC_BLOCK_SIZE;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)length / blockSize);

    for(int iter=0; iter<=MAX_ITERS; iter++)
    {
        juliaOp2v<<<gridSize, blockSize>>>(z, c, count, length);
    }
    // cudaDeviceSynchronize();
    logv<<<gridSize, blockSize>>>(count, length);
} 

extern "C" void cuJuliaOp3(cuDoubleComplex *z, const cuDoubleComplex c, double *count, int length, const int MAX_ITERS)
{
    int blockSize, gridSize;
    blockSize = CUDA_VEC_BLOCK_SIZE;

    // Number of thread blocks in grid
    gridSize = (int)ceil((float)length / blockSize);

    for(int iter=0; iter<=MAX_ITERS; iter++)
    {
        juliaOp3v<<<gridSize, blockSize>>>(z, c, count, length);
    }
    // cudaDeviceSynchronize();
    logv<<<gridSize, blockSize>>>(count, length);
} 