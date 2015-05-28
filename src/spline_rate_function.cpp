#include "spline_rate_function.h"

template <typename T>
PiecewisePolynomial<T, 6> squared_spline_fit(const std::vector<T> &x, const std::vector<T> &sqrt_y)
{
    // For completeness, cover the edge case of a single piece
    if (x.size() == 1)
    {
        T sy = sqrt_y[0];
        std::vector<T> xp = x;
        xp.push_back(INFINITY);
        Matrix<T> coef = Matrix<T>::Zero(7, 1);
        coef(6, 0) = sy * sy;
        return PiecewisePolynomial<T, 6>(xp, coef);
    }
    // Cumulate the time intervals
    std::vector<T> t;
    t.push_back(x[0]);
    for (int i = 1; i < x.size(); ++i)
        t.push_back(t.back() + abs(x[i]));
    PiecewisePolynomial<T, 3> f = cubic_spline_fit(t, sqrt_y, true);
    return f * f;
}

template <typename T>
SplineRateFunction<T>::SplineRateFunction(const std::vector<std::vector<double>> &params) : 
    RateFunction<T>(params), eta(squared_spline_fit(this->ad_params[0], this->ad_params[1])),
    R(eta.antiderivative()), Rinv(make_inverse())
{
    auto deriv = eta.derivative().derivative();
    T xmax = this->ad_params[0].back() * 1.1;
    this->_reg = (deriv * deriv).antiderivative()(xmax);
}

template <typename T>
PiecewisePolynomial<T, 1> SplineRateFunction<T>::make_inverse()
{
    // Approximate the inverse function
    T x0 = this->zero;
    T xmax = this->ad_params[0].back() * 1.1;
    if (xmax == 0.0)
        xmax = this->one;
    T step = xmax / 20.0;
    std::vector<T> x, y;
    while (x0 < xmax)
    {
        x.push_back(x0);
        x0 += step;
    }
    y = R(x);
    return linear_spline_fit(y, x);
}

template class SplineRateFunction<double>;
template class SplineRateFunction<adouble>;
