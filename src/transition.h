#ifndef TRANSITION_H
#define TRANSITION_H

#include <unsupported/Eigen/MatrixFunctions>
#include <map>
#include "common.h"
#include "piecewise_exponential.h"

class Transition
{
    public:
        Transition(PiecewiseExponential*, const std::vector<double>&, double);
        void compute(void);
        void store_results(double*, double*);
        AdMatrix& matrix(void);

    private:
        AdMatrix expm(int, int);
        PiecewiseExponential *eta;
        const std::vector<double>& _hs;
        double rho;
        int M;
        AdMatrix I, Phi;
        std::map<std::pair<int, int>, AdMatrix> _expm_memo;
};

#endif
