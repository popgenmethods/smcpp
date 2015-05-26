#ifndef LOGLIK_H
#define LOGLIK_H

#include "common.h"
#include "rate_function.h"
#include "spline_rate_function.h"
#include "transition.h"
#include "conditioned_sfs.h"
#include "hmm.h"

template <typename T>
T loglik(
        // Model parameters
        const std::vector<double> &a, const std::vector<double> &b, const std::vector<double> &s,
        // Sample size
        const int n, 
        // Number of iterations for numerical integrals
        const int S, const int M,
        // Times and matrix exponentials, for interpolation
        const std::vector<double> &ts, const std::vector<double*> &expM,
        // Length & obs vector
        const int L, const std::vector<int*> &obs, 
        // The hidden states
        const std::vector<double> &hidden_states, 
        // Model parameters 
        const double rho, const double theta,
        // Number of threads to use for computations
        int numthreads, 
        // Optionally compute viterbi
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths,
        // Regularizers
        double lambda);

void fill_jacobian(const adouble &ll, double* outjac);

#endif
