#ifndef TRANSITION_H
#define TRANSITION_H

#include <map>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/NumericalDiff>

#include "common.h"
#include "piecewise_exponential_rate_function.h"
#include "gauss_legendre.h"

typedef struct {
    double x, y, w;
} quad_point;
typedef std::vector<std::vector<quad_point> > QuadPoints;

template <typename T>
class Transition
{
    public:
        Transition(const PiecewiseExponentialRateFunction<T> &eta, double);
        void compute(void);
        Matrix<T>& matrix(void);

    private:
        T P_no_recomb(const int, const double);
        T trans(int, int);

        // Variables
        const PiecewiseExponentialRateFunction<T> *eta;
        int M;
        Matrix<T> I, Phi;
        double rho;
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
