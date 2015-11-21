#ifndef TRANSITION_H
#define TRANSITION_H

#include <random>
#include <map>
#include <unsupported/Eigen/NumericalDiff>

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "simpsons.h"

template <typename T>
class Transition
{
    public:
    Transition(const PiecewiseExponentialRateFunction<T> &eta, const double rho) : 
        eta(&eta), M(eta.hidden_states.size()), Phi(M - 1, M - 1), rho(rho) {}
    Matrix<T>& matrix(void) { return Phi; }

    protected:
    // Variables
    const PiecewiseExponentialRateFunction<T> *eta;
    const int M;
    Matrix<T> Phi;
    const double rho;
};

template <typename T> 
class SMCPrimeTransition : public Transition<T>
{
    public:
    SMCPrimeTransition(const PiecewiseExponentialRateFunction<T> &eta, const double rho) : Transition<T>(eta, rho) { compute(); }

    protected:
    void compute();
    T P_no_recomb(const int);
    T trans(int, int);
};

template <typename T>
class HJTransition : public Transition<T>
{
    public:
    HJTransition(const PiecewiseExponentialRateFunction<T> &eta, const double rho) : Transition<T>(eta, rho) 
    { 
        compute(); 
    }

    protected:
    void compute();
    Matrix<T> expm(int, int);
    std::map<std::pair<int, int>, Matrix<T>> _expm_memo;
    const std::map<std::pair<int, int>, Matrix<T>> & expm_memo(void) { return _expm_memo; }
};

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &, const double, bool);

#endif
