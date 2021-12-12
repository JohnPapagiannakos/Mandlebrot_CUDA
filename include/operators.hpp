#include <omp.h>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <types.hpp>
#include <vector>
#include <numeric>
#include <string>
#include <functional>

#define CHUNKSIZE 64

inline double margind(DoubleComplex z)
{
    // double x = real(z), y = imag(z);
    // return sqrt((x * x) + (y * y));
    return sqrt(norm(z));
}

inline void z3(DoubleComplex &z, const DoubleComplex c)
{
    double a_x = real(c);
    double a_y = imag(c);
    double x = real(z), y = imag(z);

    a_x += x * (x * x - 3 * y * y);
    a_y += y * (3 * x * x - y * y);

    z.real(a_x);
    z.imag(a_y);
}

inline void z2(DoubleComplex &z, const DoubleComplex c)
{
    double a_x = real(c);
    double a_y = imag(c);
    double x = real(z), y = imag(z);

    a_x += x * x - y * y;
    a_y += 2 * x * y;

    z.real(a_x);
    z.imag(a_y);
}

inline void logv(double *a, int length)
{
    #pragma omp for schedule(static, CHUNKSIZE)
    for (int idx = 0; idx < length; idx++)
    {
        a[idx] = std::log(a[idx]);
    }
}


namespace v1 //slow
{
    inline void juliaop2(DoubleComplex *z, const DoubleComplex c, double *count, int length)
    {
    #pragma omp for schedule(static, CHUNKSIZE)
        for (int idx = 0; idx < length; idx++)
        {
            z2(z[idx], c);
            double margin = margind(z[idx]);
            count[idx] += static_cast<double>(margin <= 2);
        }
    }

    inline void juliaop3(DoubleComplex *z, const DoubleComplex c, double *count, int length)
    {
    #pragma omp for schedule(static, CHUNKSIZE)
        for (int idx = 0; idx < length; idx++)
        {
            z3(z[idx], c);
            double margin = margind(z[idx]);
            count[idx] += static_cast<double>(margin <= 2);
        }
    }


    void JuliaOp2(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
    {
        #pragma omp parallel
        {
            for (int iter = 0; iter <= MAX_ITERS; iter++)
            {
                juliaop2(z, c, count, length);
            }
            logv(count, length);
        }
    }

    void JuliaOp3(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
    {
    #pragma omp parallel
            {
                for (int iter = 0; iter <= MAX_ITERS; iter++)
                {
                    juliaop3(z, c, count, length);
                }
                logv(count, length);
            }
        }
}

inline namespace v2 // fast
{
    inline void juliaop2(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
    {
        #pragma omp for schedule(static, CHUNKSIZE)
        for (int idx = 0; idx < length; idx++)
        {
            for (int iter = 0; iter <= MAX_ITERS; iter++)
            {
                z2(z[idx], c);
                double margin = margind(z[idx]);
                count[idx] += static_cast<double>(margin <= 2);
            }
        }
    }

    inline void juliaop3(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
    {
        #pragma omp for schedule(static, CHUNKSIZE)
        for (int idx = 0; idx < length; idx++)
        {
            for (int iter = 0; iter <= MAX_ITERS; iter++)
            {
                z3(z[idx], c);
                double margin = margind(z[idx]);
                count[idx] += static_cast<double>(margin <= 2);
            }
        }
    }


    void JuliaOp2(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
    {
        #pragma omp parallel
        {
            juliaop2(z, c, count, length, MAX_ITERS);
            logv(count, length);
        }
    }

    void JuliaOp3(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
    {
        #pragma omp parallel
        {
            juliaop3(z, c, count, length, MAX_ITERS);
            logv(count, length);
        }
    }
}

namespace rfc // if values of matrix count do not change stop algorithm 
{
    int JuliaOp2(DoubleComplex *z, const DoubleComplex c, double *count, int length)
    {
        double f_val_prev = 0, f_val_curr = 0;
        int iter = 0;
        while(1)
        {
            v1::juliaop2(z, c, count, length);
            f_val_prev = f_val_curr;
            f_val_curr = std::accumulate(&count[0], &count[length - 1], 0);
            if (std::abs(f_val_prev - f_val_curr) <= 1)
                break;
            iter++;
        }
        logv(count, length);
        return iter;
    }

    int JuliaOp3(DoubleComplex *z, const DoubleComplex c, double *count, int length)
    {
        double f_val_prev = 0, f_val_curr = 0;
        int iter = 0;
        while (1)
        {
            v1::juliaop3(z, c, count, length);
            f_val_prev = f_val_curr;
            f_val_curr = std::accumulate(&count[0], &count[length - 1], 0);
            if (std::abs(f_val_prev - f_val_curr) <= 1)
                break;
            iter++;
        }
        logv(count, length);
        return iter;
    }
}