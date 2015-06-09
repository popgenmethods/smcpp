#ifndef LOGLIK_H
#define LOGLIK_H

#include "common.h"
#include "rate_function.h"
#include "piecewise_exponential_rate_function.h"
#include "spline_rate_function.h"
#include "transition.h"
#include "conditioned_sfs.h"
#include "hmm.h"

/*
template <typename T>
T loglik(
        // Model parameters
        const std::vector<std::vector<double>> &params,
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
        const int block_size,
        // Number of threads to use for computations
        int numthreads, 
        // Optionally compute viterbi
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths,
        // Regularization parameter
        double lambda);
        */

template <typename T>
T compute_Q(
        // Model parameters
        const std::vector<std::vector<double>> &params,
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
        const int block_size,
        // Number of threads to use for computations
        int numthreads, 
        // Regularization parameter
        double lambda,
        std::vector<Vector<double>> &cs,
        std::vector<Matrix<double>> &alpha_hats,
        std::vector<Matrix<double>> &beta_hats,
        std::vector<Matrix<double>> &Bs,
        bool compute_forward_backward);

template <typename T>
T sfs_loglik(
        // Model parameters
        const std::vector<std::vector<double>> &params,
        // Sample size
        const int n, 
        // Number of iterations for numerical integrals
        const int S, const int M,
        // Times and matrix exponentials, for interpolation
        const std::vector<double> &ts, const std::vector<double*> &expM,
        // obs vector
        double* observed_sfs,
        // Number of threads to use for computations
        int numthreads, 
        // Regularization parameter
        double lambda,
        double theta);

void fill_jacobian(const adouble &ll, double* outjac);

#endif
