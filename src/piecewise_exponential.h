#ifndef PIECEWISE_EXPONENTIAL_H
#define PIECEWISE_EXPONENTIAL_H

#include <Eigen/Dense>
#include "common.h"

template <typename T>
class PiecewiseExponential
{
    public:
    PiecewiseExponential(std::vector<double> logu, std::vector<double> logv, std::vector<double> logs);
    int num_derivatives(void);
    T R(T t) const;
    T inverse_rate(T y, T t, double coalescence_rate) const;
    // Don't overload this: keeps leading to problems with the derivatives()
    // getting blown away at various points.
    double double_inverse_rate(double y, double t, double coalescence_rate) const;
    double double_R(double t) const;
    void print_debug() const;
    std::vector<std::vector<T>> ad_vars() const;
    const int K;

    private:
    T one;
    const std::vector<double> a, b, s;
    std::vector<T> ada, adb, ads, ts, Rrng;
    void initialize_derivatives();
    void compute_antiderivative();
};

#endif
