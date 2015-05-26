#include <unsupported/Eigen/Polynomials>
#include "common.h"

template <typename T>
inline T evalpoly(const Eigen::Matrix<T, Eigen::Dynamic, 1> &c, const T &x)
{
    int M = c.rows();
    Eigen::Matrix<T, Eigen::Dynamic, 1> xpow(M, 1);
    T x1 = 1.0;
    for (int m = M - 1; m >= 0; --m)
    {
        xpow(m) = x1;
        x1 *= x;
    }
    return xpow.transpose() * c;
}


template <typename T, int order>
class PiecewisePolynomial
{
    public:
    PiecewisePolynomial(const std::vector<T> &knots, const Eigen::Matrix<T, order + 1, Eigen::Dynamic> &coef) : 
        knots(knots), coef(coef), M(knots.size() - 1) {}
    T operator()(const T &x);
    std::vector<T> operator()(const std::vector<T> &x);
    PiecewisePolynomial<T, order - 1> derivative() const;
    PiecewisePolynomial<T, order + 1> antiderivative() const;
    Eigen::Matrix<T, order + 1, Eigen::Dynamic> getCoef() const;
    template <int other_order>
    PiecewisePolynomial<T, order + other_order> operator*(const PiecewisePolynomial<T, other_order> &other) const;

    friend std::ostream& operator<<(std::ostream& os, const PiecewisePolynomial& poly)
    {
        os << poly.knots << std::endl;
        os << poly.coef.template cast<double>() << std::endl << std::endl;
        return os;
    }

    private:
    const int M;
    std::vector<T> knots;
    Eigen::Matrix<T, order + 1, Eigen::Dynamic> coef;
};

template <typename T>
PiecewisePolynomial<T, 3> cubic_spline_fit(const std::vector<T> &x, const std::vector<T> &y)
{
    int n = x.size();
    if (n != y.size())
        throw std::domain_error("x and y must have same dimensions");
    Eigen::Matrix<T, 4, Eigen::Dynamic> coef(4, n);
    for (int i = 0; i < n; ++i)
        coef(3, i) = y[i];
    Matrix<T> A = Matrix<T>::Zero(n, n);
    Vector<T> rhs = Vector<T>::Zero(n);
    /* This code is borrowed from: http://kluge.in-chemnitz.de/opensource/spline/ */
    for(int i = 1; i < n - 1; i++)
    {
        A(i, i - 1) = 1.0 / 3.0 * (x[i] - x[i - 1]);
        A(i, i) = 2.0 / 3.0 * (x[i + 1] - x[i - 1]);
        A(i, i + 1) = 1.0 / 3.0 * (x[i + 1] -x[i]);
        rhs[i] = (y[i + 1] - y[i])/(x[i + 1] - x[i]) - (y[i] - y[i - 1])/(x[i] - x[i - 1]);
    }
    A(0, 0) = 2.0;
    A(0, 1) = 0.0;
    rhs[0] = 0.0;
    A(n - 1, n - 1) = 2.0;
    A(n - 1, n - 2) = 0.0;
    rhs[n - 1] = 0.0;
    Vector<T> b = A.colPivHouseholderQr().solve(rhs);
    Vector<T> a(n), c(n);
    for(int i = 0; i < n - 1; i++)
    {
        a[i] = 1.0 / 3.0 * (b[i + 1] - b[i]) / (x[i + 1] - x[i]);
        c[i] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - 1.0 / 3.0 * (2.0 * b[i] + b[i + 1]) * (x[i + 1] - x[i]);
    }
    a[n - 1] = 0.0;
    b[n - 1] = 0.0;
    T h = x[n - 1] - x[n - 2];
    c[n - 1] = 3.0 * a[n - 2] * h * h + 2.0 * b[n - 2] * h + c[n - 2];   // = f'_{n-2}(x_{n-1})
    coef.row(0).transpose() = a;
    coef.row(1).transpose() = b;
    coef.row(2).transpose() = c;
    std::vector<T> xp = x;
    xp.push_back(INFINITY);
    return PiecewisePolynomial<T, 3>(xp, coef);
}


template <typename T, int order>
std::vector<T> PiecewisePolynomial<T, order>::operator()(const std::vector<T> &v)
{
    std::vector<T> ret;
    int ip = insertion_point(v[0], knots, 0, M + 1);
    ret.push_back(evalpoly<T>(coef.col(ip), v[0] - knots[ip]));
    for (typename std::vector<T>::const_iterator it = std::next(v.begin()); it != v.end(); ++it)
    {
        if (*(it - 1) > *it)
            throw std::domain_error("vector must be sorted");
        while (*it > knots[ip + 1]) ip++;
        ret.push_back(evalpoly<T>(coef.col(ip), *it - knots[ip]));
    }
    return ret;
}

template <typename T, int order>
T PiecewisePolynomial<T, order>::operator()(const T &x)
{
    int ip = insertion_point(x, knots, 0, M + 1);
    return evalpoly<T>(coef.col(ip), x - knots[ip]);
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
        new_coef(order + 1, m) = evalpoly<T>(new_coef.col(m - 1), knots[m] - knots[m - 1]);
    return PiecewisePolynomial<T, order + 1>(knots, new_coef);
}

template <typename T, int order>
template <int other_order>
PiecewisePolynomial<T, order + other_order> PiecewisePolynomial<T, order>::operator*(const PiecewisePolynomial<T, other_order> &other) const
{
    if (knots != other.knots)
        throw std::domain_error("Can't multiply polynomials with different breakpoints.");
    Eigen::Array<T, order + other_order + 1, Eigen::Dynamic> new_coef(order + other_order + 1, M);
    new_coef.fill(0);
    for (int m = 0; m < M; ++m)
        for (int i = 0; i < order + 1; ++i)
            for (int j = 0; j < other_order + 1; ++j)
                new_coef(i + j, m) += coef(i, m) * other.coef(j, m);
    return PiecewisePolynomial<T, order + other_order>(knots, new_coef);
}

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

