#ifndef PIECEWISE_CONSTANT_RATE_FUNCTION_H
#define PIECEWISE_CONSTANT_RATE_FUNCTION_H

#include <random>
#include "common.h"
#include "mpq_support.h"

template <typename T>
class PiecewiseConstantRateFunction
{
    public:
    PiecewiseConstantRateFunction(const std::vector<std::vector<adouble>>, const std::vector<double>);
    PiecewiseConstantRateFunction(const PiecewiseConstantRateFunction &other) : 
        PiecewiseConstantRateFunction(other.params, other.hidden_states) {}
    T R(const T) const;
    T Rinv(const T) const;
    T R_integral(const double, const double, const T) const;
    std::vector<T> average_coal_times() const;
    T random_time(const double, const double, const long long) const;
    T random_time(const double, const double, const double, std::mt19937 &) const;
    void print_debug() const;
    
    void tjj_double_integral_above(const int, long, std::vector<Matrix<T> > &) const;
    void tjj_double_integral_below(const int, const int, Matrix<T>&) const;

    friend std::ostream& operator<<(std::ostream& os, const PiecewiseConstantRateFunction& pexp)
    {
        os << pexp.ts << std::endl;
        os << pexp.ada << std::endl;
        return os;
    }
    
    private:
    std::vector<std::vector<adouble>> params;
    const int nder;
    int K;
    std::vector<double> s;
    void compute_antiderivative();

    public:
    std::vector<T> ada;
    std::vector<double> ts;
    std::vector<T> Rrng;
    std::vector<int> hs_indices;
    const std::vector<double> hidden_states;
    const double tmax;
};

#endif
