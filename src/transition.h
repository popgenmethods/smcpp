#ifndef TRANSITION_H
#define TRANSITION_H

#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/NumericalDiff>

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "gauss_legendre.h"
#include "cubature/cubature.h"

template <typename T>
class Transition
{
    public:
        Transition(const PiecewiseExponentialRateFunction<T> &eta, const double);
        void compute(void);
        Matrix<T>& matrix(void);

    private:
        T P_no_recomb(const int, const double);
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

/*
void cython_calculate_transition(const std::vector<std::vector<double>> params,
        const std::vector<double> hidden_states, double rho, double* outtrans);
void cython_calculate_transition_jac(const std::vector<std::vector<double>> params,
        const std::vector<double> hidden_states, double rho, double* outtrans, double* outjac);
*/

#endif
