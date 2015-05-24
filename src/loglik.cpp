#include "loglik.h"

template <typename T>
Vector<T> compute_initial_distribution(const PiecewiseExponential<T> &eta, const std::vector<double> &hidden_states)
{
    int M = hidden_states.size() - 1;
    Vector<T> pi(M);
    T s = 0.0;
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-eta.inverse_rate(hidden_states[m], 0.0, 1.0)) - 
            exp(-eta.inverse_rate(hidden_states[m + 1], 0.0, 1.0));
        assert(pi(m) > 0.0); 
        assert(pi(m) < 1.0); 
    }
    pi(M - 1) = exp(-eta.inverse_rate(hidden_states[M - 1], 0.0, 1.0));
    assert(pi(M - 1) > 0.0);
    assert(pi(M - 1) < 1.0);
    assert(pi.sum() == 1.0);
    return pi;
}

template Vector<double> compute_initial_distribution(const PiecewiseExponential<double>&, const std::vector<double>&);
template Vector<adouble> compute_initial_distribution(const PiecewiseExponential<adouble>&, const std::vector<double>&);

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
        double reg_a, double reg_b, double reg_s)
{
    int K = a.size();
    PiecewiseExponential<T> eta(a, b, s);
    Vector<T> pi = compute_initial_distribution(eta, hidden_states);
    Matrix<T> transition = compute_transition(eta, hidden_states, rho);
    std::vector<Matrix<T>> emission;
    for (int i = 1; i < hidden_states.size(); ++i)
        emission.push_back(ConditionedSFS<T>::calculate_sfs(eta, n, S, M, ts, expM, hidden_states[i - 1], hidden_states[i], numthreads, theta));
    T ll = compute_hmm_likelihood<T>(eta, pi, transition, emission, L, obs, numthreads, viterbi, viterbi_paths);
    std::vector<double> regs = {reg_a, reg_b, reg_s};
    std::vector<std::vector<T>> ad = eta.ad_vars();
    for (int k = 0; k < K; ++k)
        for (int ell = 0; ell < 3; ++ell)
            ll -= regs[ell] * pow(ad[ell][k], 2);
    return ll;
}

template adouble loglik(
        const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
        const int, 
        const int, const int,
        const std::vector<double>&, const std::vector<double*>&,
        const int, const std::vector<int*>&, 
        const std::vector<double>&,
        const double, const double,
        int, 
        bool, std::vector<std::vector<int>>&,
        double, double, double);

template double loglik(
        const std::vector<double>&, const std::vector<double>&, const std::vector<double>&,
        const int, 
        const int, const int,
        const std::vector<double>&, const std::vector<double*>&,
        const int, const std::vector<int*>&, 
        const std::vector<double>&,
        const double, const double,
        int, 
        bool, std::vector<std::vector<int>> &,
        double, double, double);

void fill_jacobian(const adouble &ll, double* outjac)
{
    Eigen::VectorXd d = ll.derivatives();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(outjac, d.rows());
    _jac = d;
}
