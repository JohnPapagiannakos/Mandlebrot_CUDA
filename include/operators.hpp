#include <iostream>
#include <iomanip>
#include <complex>
#include <cmath>

typedef std::complex<double> DoubleComplex;

inline double margind(DoubleComplex z)
{
    double x=real(z), y=imag(z);
    return sqrt((x * x) + (y * y));
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
    for (int idx = 0; idx < length; idx++)
    {
        a[idx] = std::log(a[idx]);
    }
}

inline void juliaop2(DoubleComplex *z, const DoubleComplex c, double *count, int length)
{
    for(int idx=0; idx<length; idx++)
    {
        z2(z[idx], c);
        double margin = margind(z[idx]);
        count[idx] += static_cast<double>(margin <= 2);
    }
}

void JuliaOp2(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
{
    for (int iter = 0; iter <= MAX_ITERS; iter++)
    {
        juliaop2(z, c, count, length);
    }
    logv(count, length);
}

inline void juliaop3(DoubleComplex *z, const DoubleComplex c, double *count, int length)
{
    for (int idx = 0; idx < length; idx++)
    {
        z3(z[idx], c);
        double margin = margind(z[idx]);
        count[idx] += static_cast<double>(margin <= 2);
    }
}

void JuliaOp3(DoubleComplex *z, const DoubleComplex c, double *count, int length, const int MAX_ITERS)
{
    for (int iter = 0; iter <= MAX_ITERS; iter++)
    {
        juliaop3(z, c, count, length);
    }
    logv(count, length);
}