#include "spline_rate_function.h"

template <typename T>
PiecewisePolynomial<T, 6> squared_spline_fit(const std::vector<T> &x, const std::vector<T> &sqrt_y)
{
    PiecewisePolynomial<T, 3> f = cubic_spline_fit(x, sqrt_y);
    return f * f;
}

template <typename T>
SplineRateFunction<T>::SplineRateFunction(const std::vector<std::vector<double>> &params) : 
    RateFunction<T>(params), spline(squared_spline_fit(this->ad_params[0], this->ad_params[1])),
    spline_integral(spline.antiderivative()), spline_integral_inverse(make_inverse())
{
    auto deriv = spline.derivative().derivative();
    T xmax = this->ad_params[0].back() * 1.1;
    this->_reg = (deriv * deriv).antiderivative()(xmax);
}

template <typename T>
PiecewisePolynomial<T, 3> SplineRateFunction<T>::make_inverse()
{
    // Approximate the inverse function
    T x0 = RateFunction<T>::zero;
    T xmax = this->ad_params[0].back() * 1.1;
    T step = xmax / 100.0;
    std::vector<T> x, y;
    while (x0 < xmax)
    {
        x.push_back(x0);
        x0 += step;
    }
    y = Rv(x);
    return cubic_spline_fit(y, x);
}

template <typename T>
T SplineRateFunction<T>::R(const T &x) const { return spline(x); } 

template <typename T>
std::vector<T> SplineRateFunction<T>::Rv(const std::vector<T> &v) const { return spline(v); }

template <typename T>
T SplineRateFunction<T>::Rinv(const T &y, const T &x) const
{ 
    return spline_integral_inverse(y + R(x)) - x;
}

template class SplineRateFunction<double>;
template class SplineRateFunction<adouble>;
