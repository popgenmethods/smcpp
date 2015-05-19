#ifndef PIECEWISE_EXPONENTIAL_H
#define PIECEWISE_EXPONENTIAL_H

#include <Eigen/Dense>
#include "common.h"

class PiecewiseExponential
{
    public:
    PiecewiseExponential(std::vector<double> logu, std::vector<double> logv, std::vector<double> logs);
    int num_derivatives(void);
    adouble R(adouble t) const;
    adouble inverse_rate(double y, adouble t, double coalescence_rate) const;
    // Don't overload this: keeps leading to problems with the derivatives()
    // getting blown away at various points.
    double double_inverse_rate(double y, double t, double coalescence_rate) const;
    double double_R(double t) const;
    void print_debug() const;
    std::vector<std::vector<adouble>> ad_vars() const;
    const int K;

    private:
    const std::vector<double> logu, logv, logs;
    std::vector<adouble> adlogu, adlogv, adlogs;
    std::vector<adouble> ada, adb, ts, Ra, Rb, Rc, Rrng;
    void compute_antiderivative();
};

#endif
