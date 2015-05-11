#ifndef HMM_H
#define HMM_H

#include <cassert>
#include <cfenv>
#include <algorithm>
#include <unsupported/Eigen/MatrixFunctions>

#include "common.h"
#include "conditioned_sfs.h"
#include "piecewise_exponential.h"
#include "transition.h"

typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> npArray;

class HMM
{
    public:
    HMM(PiecewiseExponential *eta, 
        const std::vector<std::vector<ConditionedSFS*>> &csfs,
        const std::vector<double> hidden_states,
        int L, int* obs, double rho, double theta);
    double logp(double*);
    std::vector<int>& viterbi(void);

    private:
    // Methods
    void compute_initial_distribution(void);
    void compute_transition(void);
    void average_sfs(const std::vector<std::vector<ConditionedSFS*>> &csfs);
    void forward(void);
    void add_derivatives(void);
    AdMatrix O0Tpow(int);

    // Instance variables
    int M, L;
    PiecewiseExponential* eta;
    AdVector pi;
    AdMatrix transition;
    std::vector<AdMatrix> emission;
    Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, 2, Eigen::RowMajor>> obs;
    const std::vector<double> hidden_states;
    double rho, theta;
    AdVector c;
    std::vector<int> viterbi_path;
};

//
#endif
