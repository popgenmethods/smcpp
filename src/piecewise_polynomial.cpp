#include "piecewise_polynomial.h"

template <typename T, int order>
inline T evalpoly(const Eigen::Matrix<T, order + 1, 1> &c, const T &x)
{
    Eigen::Matrix<T, order + 1, 1> xpow(order + 1, 1);
    T x1 = 1.0;
    for (int m = order; m >= 0; --m)
    {
        xpow(m) = x1;
        x1 *= x;
    }
    T ret = xpow.transpose() * c;
    if (isnan(toDouble(ret)))
        throw std::domain_error("nan encountered in evalpoly");
    return ret;
}

template <typename T, int order>
std::vector<T> PiecewisePolynomial<T, order>::operator()(const std::vector<T> &v) const
{
    std::vector<T> ret(v.size());
    int ip = insertion_point(v[0], knots, 0, M + 1);
    ret.push_back(evalpoly<T, order>(coef.col(ip), v[0] - knots[ip]));
    for (typename std::vector<T>::const_iterator it = std::next(v.begin()); it != v.end(); ++it)
    {
        if (*(it - 1) > *it)
            throw std::domain_error("vector must be sorted");
        while (*it > knots[ip + 1]) ip++;
        ret.push_back(evalpoly<T, order>(coef.col(ip), *it - knots[ip]));
    }
    return ret;
}

template <typename T, int order>
T PiecewisePolynomial<T, order>::operator()(const T &x) const
{
    int ip = insertion_point(x, knots, 0, M + 1);
    return evalpoly<T, order>(coef.col(ip), x - knots[ip]);
}

template <typename T, int order>
Eigen::Matrix<T, order + 1, Eigen::Dynamic> PiecewisePolynomial<T, order>::getCoef() const
{
    return coef;
}

template <typename T, int order>
PiecewisePolynomial<T, order - 1> PiecewisePolynomial<T, order>::derivative(void) const
{
    Eigen::Array<T, order, Eigen::Dynamic> new_coef(order, M);
    new_coef = coef.block(0, 0, order, M);
    Eigen::ArrayXd div = Eigen::ArrayXd::LinSpaced(order, 1, order).reverse();
    new_coef.colwise() *= div;
    return PiecewisePolynomial<T, order - 1>(knots, new_coef);
}

template <typename T, int order>
PiecewisePolynomial<T, order + 1> PiecewisePolynomial<T, order>::antiderivative(void) const
{
    Eigen::Array<T, order + 2, Eigen::Dynamic> new_coef(order + 2, M);
    new_coef.block(0, 0, order + 1, M) = coef;
    Eigen::ArrayXd div = Eigen::ArrayXd::LinSpaced(order + 1, 1, order + 1).reverse();
    new_coef.block(0, 0, order + 1, M).colwise() /= div;
    // Enforce continuity
    new_coef(order + 1, 0) = 0.0;
    for (int m = 1; m < M; ++m)
        new_coef(order + 1, m) = evalpoly<T, order + 1>(new_coef.col(m - 1), knots[m] - knots[m - 1]);
    return PiecewisePolynomial<T, order + 1>(knots, new_coef);
}


/*
int main(int argc, char** argv)
{
    Eigen::Matrix<double, 3, 3> a;
    a << 1, 1, 1, 0, 0, 0, 1, 1, 1;
    std::vector<double> knots = {1, 2, 3, INFINITY};
    PiecewisePolynomial<double, 2> p(knots, a);
    PiecewisePolynomial<double, 3> pad = p.antiderivative();
    std::cout << p << std::endl;
    std::cout << pad << std::endl;

    auto cube = cubic_spline_fit<adouble>({0, 1, 2, 4.5, 8.2}, {4, 8, 15, 1, 2.2});
    auto cube2 = (cube * cube).antiderivative();
    std::vector<adouble> pts = {0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, 3.0};
    std::cout << cube2(pts) << std::endl;

    adouble x = 8.2 + 1;
    adouble step = x / 100.0;
    std::vector<adouble> inv_x;
    adouble z = 0.0;
    while (z < x)
    {
        inv_x.push_back(z);
        z += step;
    }
    auto inv_y = cube2(inv_x);
    std::cout << "inv_y " << std::endl << inv_y << std::endl << std::endl;
    auto invcube = cubic_spline_fit<adouble>(inv_y, inv_x);
    for (adouble x = 0; x < 10; x += 0.2)
    {
        std::cout << x << " " << cube2(invcube(x)) << std::endl;
    }

    auto d = invcube.derivative().derivative();
    auto d2 = d * d;
    std::cout << invcube << std::endl << std::endl;
    std::cout << d << std::endl << std::endl;
    std::cout << d2 << std::endl << std::endl;
    for (adouble x = 0; x < 10; x += 0.2)
    {
        std::cout << "(" << x << "," << d2(x) << ") ";
    }
    std::cout << std::endl;
}
*/

template class PiecewisePolynomial<double, 9>;
template class PiecewisePolynomial<double, 8>;
template class PiecewisePolynomial<double, 7>;
template class PiecewisePolynomial<double, 6>;
template class PiecewisePolynomial<double, 5>;
template class PiecewisePolynomial<double, 4>;
template class PiecewisePolynomial<double, 3>;
template class PiecewisePolynomial<double, 1>;
template class PiecewisePolynomial<adouble, 9>;
template class PiecewisePolynomial<adouble, 8>;
template class PiecewisePolynomial<adouble, 7>;
template class PiecewisePolynomial<adouble, 6>;
template class PiecewisePolynomial<adouble, 5>;
template class PiecewisePolynomial<adouble, 4>;
template class PiecewisePolynomial<adouble, 3>;
template class PiecewisePolynomial<adouble, 1>;
