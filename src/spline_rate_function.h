#ifndef SPLINE_RATE_FUNCTION
#define SPLINE_RATE_FUNCTION

#include "common.h"
#include "rate_function.h"
#include "piecewise_polynomial.h"

template <typename T>
class SplineRateFunction : public RateFunction<T>
{
    public:
    SplineRateFunction(const std::vector<std::vector<double>> &params);
    virtual T R(const T &x) const;
    virtual std::vector<T> Rv(const std::vector<T> &v) const;
    virtual T Rinv(const T &y, const T &x) const;

    private:
    PiecewisePolynomial<T, 3> make_inverse();
    PiecewisePolynomial<T, 6> spline;
    PiecewisePolynomial<T, 7> spline_integral;
    PiecewisePolynomial<T, 3> spline_integral_inverse;
};

#endif

