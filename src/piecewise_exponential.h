#ifndef PIECEWISE_EXPONENTIAL_H
#define PIECEWISE_EXPONENTIAL_H

#include <Eigen/Dense>
#include "common.h"

class PiecewiseExponential
{
    public:
    PiecewiseExponential(std::vector<double> sqrt_a, std::vector<double> b, 
            std::vector<double> sqrt_s, double T_max);
    int num_derivatives(void);
    int K(void) const;
    adouble R(adouble t) const;
    adouble inverse_rate(double y, adouble t, double coalescence_rate) const;
    // Don't overload this: keeps leading to problems with the derivatives()
    // getting blown away at various points.
    double double_inverse_rate(double y, double t, double coalescence_rate) const;
    double double_R(double t) const;
    void print_debug() const;
    std::vector<std::vector<adouble>> ad_vars() const;

    private:
    const std::vector<double> sqrt_a, b, sqrt_s;
    int _K;
    std::vector<adouble> adsqrt_a, adb, adsqrt_s;
    std::vector<adouble> adasq, ts, Ra, Rb, Rc, Rrng;
    void _compute_antiderivative();
};

#endif
