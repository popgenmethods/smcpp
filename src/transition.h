#ifndef TRANSITION_H
#define TRANSITION_H

#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/NumericalDiff>

#include "common.h"
#include "piecewise_exponential_rate_function.h"

template <typename T>
class Transition
{
    public:
        Transition(const PiecewiseExponentialRateFunction<T> &eta, double);
        void compute(void);
        // void store_results(double*, double*);
        Matrix<T>& matrix(void);

    private:
        Matrix<T> expm(int, int);
        const PiecewiseExponentialRateFunction<T> *eta;
        double rho;
        int M;
        Matrix<T> I, Phi;
        std::map<std::pair<int, int>, Matrix<T>> _expm_memo;
};

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &eta, double rho)
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
