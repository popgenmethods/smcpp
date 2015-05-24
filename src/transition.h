#ifndef TRANSITION_H
#define TRANSITION_H

#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/NumericalDiff>

#include "common.h"
#include "piecewise_exponential.h"

template <typename T>
class Transition
{
    public:
        Transition(const PiecewiseExponential<T> &eta, const std::vector<double>&, double);
        void compute(void);
        // void store_results(double*, double*);
        Matrix<T>& matrix(void);

    private:
        Matrix<T> expm(int, int);
        PiecewiseExponential<T> eta;
        const std::vector<double>& _hs;
        double rho;
        int M;
        Matrix<T> I, Phi;
        std::map<std::pair<int, int>, Matrix<T>> _expm_memo;
};

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponential<T> &eta, const std::vector<double> &hidden_states, double rho)
{
    Transition<T> trans(eta, hidden_states, rho);
    return trans.matrix();
}

void cython_calculate_transition(const std::vector<double> a, const std::vector<double> b, const std::vector<double> s, 
        const std::vector<double> hidden_states, double rho, double* outtrans);
void cython_calculate_transition_jac(const std::vector<double> a, const std::vector<double> b, const std::vector<double> s, 
        const std::vector<double> hidden_states, double rho, double* outtrans, double* outjac);

#endif
