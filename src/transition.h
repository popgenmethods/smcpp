#ifndef TRANSITION_H
#define TRANSITION_H

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "simpsons.h"

template <typename T>
class Transition
{
    public:
        Transition(const PiecewiseExponentialRateFunction<T> &eta, const double);
        void compute(void);
        Matrix<T>& matrix(void);

    private:
        T P_no_recomb(const int);
        T trans(int, int);

        // Variables
        const PiecewiseExponentialRateFunction<T> *eta;
        const int M;
        Matrix<T> Phi;
        const double rho;
};

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &eta, const double rho)
{
    Transition<T> trans(eta, rho);
    return trans.matrix();
}

#endif
