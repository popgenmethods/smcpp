#ifndef PIECEWISE_CONSTANT_RATE_FUNCTION_H
#define PIECEWISE_CONSTANT_RATE_FUNCTION_H

#include <random>
#include "common.h"

template <typename T>
class PiecewiseConstantRateFunction
{
    public:
    PiecewiseConstantRateFunction(const std::vector<std::vector<adouble>>, const std::vector<double>);
    PiecewiseConstantRateFunction(const PiecewiseConstantRateFunction &other) : 
        PiecewiseConstantRateFunction(other.params, other.hidden_states) {}
    T zero() const;
    T R(const T) const;
    T Rinv(const T) const;
    T R_integral(const double, const double, const T) const;
    std::vector<T> average_coal_times() const;
    T random_time(const double, const double, const long long) const;
    T random_time(const double, const double, const double, std::mt19937 &) const;
    int getNder() const { return nder; }
    void print_debug() const;
    
    void tjj_double_integral_above(const int, long, std::vector<Matrix<T> > &) const;
    void tjj_double_integral_below(const int, const int, Matrix<T>&) const;

    // Getters
    const std::vector<double>& getHiddenStates() const { return hidden_states; }
    const std::vector<double>& getTs() const { return ts; }
    const std::vector<int>& getHsIndices() const { return hs_indices; }
    const std::vector<T>& getRrng() const { return Rrng; }
    const std::vector<T>& getAda() const { return ada; }

    friend std::ostream& operator<<(std::ostream& os, const PiecewiseConstantRateFunction& pexp)
    {
        os << pexp.ts << std::endl;
        os << pexp.ada << std::endl;
        return os;
    }
    
    private:
    const std::vector<std::vector<adouble>> params;
    const int nder;
    int K;
    std::vector<T> ada;
    std::vector<double> s;
    std::vector<double> ts;
    std::vector<T> Rrng;
    const std::vector<double> hidden_states;
    std::vector<int> hs_indices;
    // Methods
    void compute_antiderivative();
};

#endif
