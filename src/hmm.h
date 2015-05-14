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
#include "ThreadPool.h"

typedef Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> npArray;

class HMM
{
    public:
    HMM(const AdMatrix &pi, const AdMatrix &transition,
        const std::vector<AdMatrix> &emission,
        const std::vector<double> hidden_states,
        const int L, const int* obs, double rho);
    adouble logp(void);
    std::vector<int>& viterbi(void);
    void forward(void);
    void printobs(void);

    private:
    // Methods
    void average_sfs(const std::vector<std::vector<ConditionedSFS*>> &csfs);
    AdMatrix matpow(const AdMatrix&, int);
    template <typename T, int s>
    void diag_obs(Eigen::DiagonalMatrix<T, s> &D, int a, int b);

    // Instance variables
    AdVector pi;
    AdMatrix transition;
    std::vector<AdMatrix> emission;
    int M, L;
    Eigen::Map<const Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>> obs;
    const std::vector<double> hidden_states;
    std::vector<adouble> logc;
    double rho;
    std::vector<int> viterbi_path;
};

AdMatrix compute_initial_distribution(const PiecewiseExponential &eta, const std::vector<double> &hidden_states);
AdMatrix compute_transition(const PiecewiseExponential &eta, const std::vector<double> &hidden_states, double rho);
double compute_hmm_likelihood(double*, const PiecewiseExponential &eta,
        const std::vector<AdMatrix>& emission, const int L, const std::vector<int*> obs, 
        const std::vector<double> &hidden_states, const double rho, int numthreads);
//
#endif
