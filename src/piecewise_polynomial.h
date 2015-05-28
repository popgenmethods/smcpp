#ifndef PIECEWISE_POLYNOMIAL_H
#define PIECEWISE_POLYNOMIAL_H

#include <unsupported/Eigen/Polynomials>
#include "common.h"
#include "function_evaluator.h"

template <typename T, int order>
class PiecewisePolynomial : public FunctionEvaluator<T>
{
    public:
    PiecewisePolynomial(const std::vector<T> &knots, const Eigen::Matrix<T, order + 1, Eigen::Dynamic> &coef) : 
        M(knots.size() - 1), knots(knots), coef(coef) {}
    PiecewisePolynomial<T, order - 1> derivative() const;
    PiecewisePolynomial<T, order + 1> antiderivative() const;
    Eigen::Matrix<T, order + 1, Eigen::Dynamic> getCoef() const;
    template <int other_order>
    PiecewisePolynomial<T, order + other_order> operator*(const PiecewisePolynomial<T, other_order> &other) const;
    virtual T operator()(const T &x) const;
    virtual std::vector<T> operator()(const std::vector<T> &x) const;

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
PiecewisePolynomial<T, 1> linear_spline_fit(const std::vector<T> &x, const std::vector<T> &y)
{
    int n = x.size();
    if (n != y.size())
        throw std::domain_error("x and y must have same dimensions");
    Eigen::Matrix<T, 2, Eigen::Dynamic> coef(2, n - 1);
    for (int i = 0; i < n - 1; ++i)
        coef(1, i) = y[i];
    Matrix<T> A = Matrix<T>::Zero(n, n);
    Vector<T> rhs = Vector<T>::Zero(n);
    for(int i = 0; i < n - 1; i++)
        coef(0, i) = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    std::vector<T> xp = x;
    xp.back() = INFINITY;
    return PiecewisePolynomial<T, 1>(xp, coef);
}

template <typename T>
PiecewisePolynomial<T, 3> monotone_cubic_spline_fit(const std::vector<T> &x, const std::vector<T> &y)
{
    int n = x.size();
    if (n != y.size())
        throw std::domain_error("x and y must have same dimensions");
    Eigen::Matrix<T, 4, Eigen::Dynamic> coef(4, n);
    for (int i = 0; i < n; ++i)
        coef(3, i) = y[i];
    Vector<T> a(n), b(n), c(n);
    std::vector<T> dxs, dys, ms;
    for (int i = 1; i < n; ++i)
    {
        dxs.push_back(x[i] - x[i - 1]);
        dys.push_back(y[i] - y[i - 1]);
        ms.push_back(dys.back() / dxs.back());
    }
    c(0) = ms[0];
    c(n - 1) = 0.0;
    for (int i = 1; i < n - 2; ++i)
    {
        T m = ms[i - 1];
        T m_next = ms[i];
        if (m * m_next <= 0)
            c(i) = 0.0;
        else
        {
            T dx = dxs[i - 1];
            T dx_next = dxs[i];
            T common = dx + dx_next;
            c(i) = 3 * common / ((common + dx_next) / m + (common + dx) / m_next);
        }
    }
    for (int i = 0; i < n - 1; ++i)
    {
        T m = ms[i];
        T dxi = 1. / dxs[i];
        T common = c(i) + c(i + 1) - 2 * m;
        b(i) = (m - c(i) - common) * dxi;
        a(i) = common * dxi * dxi;
    }
    a(n - 1) = 0.0;
    b(n - 1) = 0.0;
    coef.row(0).transpose() = a;
    coef.row(1).transpose() = b;
    coef.row(2).transpose() = c;
    std::vector<T> xp = x;
    xp.push_back(INFINITY);
    return PiecewisePolynomial<T, 3>(xp, coef);
}

template <typename T>
PiecewisePolynomial<T, 3> cubic_spline_fit(const std::vector<T> &x, const std::vector<T> &y, bool constant_last_piece)
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
    if (constant_last_piece)
        c[n - 1] = 0.0;
    else
    {
        T h  = x[n - 1] - x[n - 2];
        c[n - 1] = 3.0 * a[n - 2] * h * h + 2.0 * b[n - 2] * h + c[n - 2];   // = f'_{n-2}(x_{n-1})
    }
    coef.row(0).transpose() = a;
    coef.row(1).transpose() = b;
    coef.row(2).transpose() = c;
    std::vector<T> xp = x;
    xp.push_back(INFINITY);
    return PiecewisePolynomial<T, 3>(xp, coef);
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

#endif
