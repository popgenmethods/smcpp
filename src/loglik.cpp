#include "loglik.h"

template <typename T>
Vector<T> compute_initial_distribution(const RateFunction<T> &eta, const std::vector<double> &hidden_states)
{
    auto Rinv = eta.getRinv();
    int M = hidden_states.size() - 1;
    Vector<T> pi(M);
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-Rinv->operator()(hidden_states[m])) - exp(-Rinv->operator()(hidden_states[m + 1]));
        assert(pi(m) > 0.0); 
        assert(pi(m) < 1.0); 
    }
    pi(M - 1) = exp(-Rinv->operator()(hidden_states[M - 1]));

    assert(pi(M - 1) > 0.0);
    assert(pi(M - 1) < 1.0);
    assert(pi.sum() == 1.0);
    return pi;
}

template Vector<double> compute_initial_distribution(const RateFunction<double>&, const std::vector<double>&);
template Vector<adouble> compute_initial_distribution(const RateFunction<adouble>&, const std::vector<double>&);

template <typename T>
T sfs_loglik(
        // Model parameters
        const std::vector<std::vector<double>> &params,
        // Sample size
        const int n, 
        // Number of iterations for numerical integrals
        const int num_samples,
        // Times and matrix exponentials, for interpolation
        const MatrixInterpolator &moran_interp,
        // Length & obs vector
        double* observed_sfs,
        // Number of threads to use for computations
        const int numthreads, 
        // Regularization parameter
        double lambda,
        // theta
        double theta)
{
    RATE_FUNCTION<T> eta(params);
    // eta.print_debug();
    Vector<double> emp_sfs = Vector<double>::Map(observed_sfs, n + 2);
    Matrix<T> dist_sfs = ConditionedSFS<T>::calculate_sfs(eta, n, num_samples, moran_interp, 0, INFINITY, numthreads, theta);
    Vector<T> undist_sfs = Vector<T>::Zero(n + 2);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n; ++j)
            if (i + j < n + 2)
                undist_sfs(i + j) += dist_sfs(i, j);
    set_seed(1234);
    std::cout << "empirical sfs: " << std::endl << emp_sfs.transpose() / emp_sfs.sum() << std::endl << std::endl;
    std::cout << "theoretical sfs: " << std::endl << undist_sfs.transpose().template cast<double>() 
        << std::endl << std::endl;
    if (undist_sfs.minCoeff() <= 0)
        return -INFINITY;
    T ll = emp_sfs.template cast<T>().transpose() * undist_sfs.array().log().matrix();
    std::cout << "D(obs || expected) (without reg): " << ll << std::endl;
    ll -= lambda * eta.regularizer();
    std::cout << "D(obs || expected) (with reg): " << ll << std::endl;
    return ll;
}

template double sfs_loglik(
        const std::vector<std::vector<double>>&,
        const int, 
        const int,
        const MatrixInterpolator&,
        double*,
        const int,
        double, double);

template adouble sfs_loglik(
        const std::vector<std::vector<double>>&,
        const int, 
        const int,
        const MatrixInterpolator&,
        double*,
        const int,
        double, double);


template <typename T>
T compute_Q(
        // Model parameters
        const std::vector<std::vector<double>> &params,
        // Sample size
        const int n, 
        // Number of iterations for numerical integrals
        const int num_samples,
        // Times and matrix exponentials, for interpolation
        const MatrixInterpolator &moran_interp,
        // Length & obs vector
        const int L, const std::vector<int*> &obs, 
        // The hidden states
        const std::vector<double> &hidden_states, 
        // Model parameters 
        const double rho, const double theta,
        // blocking parameter for hmm
        int block_size,
        // Number of threads to use for computations
        int numthreads, 
        // Regularization parameter
        double lambda,
        std::vector<Matrix<double>> &gammas,
        std::vector<Matrix<double>> &xisums,
        bool recompute)
{
    RATE_FUNCTION<T> eta(params);
    Vector<T> pi = compute_initial_distribution(eta, hidden_states);
    Matrix<T> transition = compute_transition(eta, hidden_states, rho);
    int M = hidden_states.size() - 1;
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> emission(M, 3 * (n + 1)), tmp;
    for (int i = 0; i < M; ++i)
    {
        tmp = ConditionedSFS<T>::calculate_sfs(eta, n, num_samples, moran_interp, 
                hidden_states[i], hidden_states[i + 1], numthreads, theta);
        emission.row(i) = Matrix<T>::Map(tmp.data(), 1, 3 * (n + 1));
    }
    tmp = ConditionedSFS<T>::calculate_sfs(eta, n, num_samples, moran_interp, 0, INFINITY, numthreads, theta);
    T ll = compute_hmm_Q<T>(pi, transition, emission, n, L, obs, 
            block_size, numthreads, gammas, xisums, recompute);
    ll -= lambda * eta.regularizer();
    return ll;
}

template adouble compute_Q(
        const std::vector<std::vector<double>> &params,
        const int n, 
        const int num_samples,
        const MatrixInterpolator &moran_interp,
        const int L, const std::vector<int*> &obs, 
        const std::vector<double> &hidden_states, 
        const double rho, const double theta,
        const int block_size,
        int numthreads, 
        double lambda,
        std::vector<Matrix<double>> &gammas,
        std::vector<Matrix<double>> &xisums,
        bool recompute);

template double compute_Q(
        const std::vector<std::vector<double>> &params,
        const int n, 
        const int num_samples,
        const MatrixInterpolator &moran_interp,
        const int L, const std::vector<int*> &obs, 
        const std::vector<double> &hidden_states, 
        const double rho, const double theta,
        const int block_size,
        int numthreads, 
        double lambda,
        std::vector<Matrix<double>> &gammas,
        std::vector<Matrix<double>> &xisums,
        bool recompute);

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
        double lambda)
{
    RATE_FUNCTION<T> eta(params);
    // eta.print_debug();
    double duration;
    std::clock_t start;
    std::cout << "pi ... ";
    start = std::clock();
    Vector<T> pi = compute_initial_distribution(eta, hidden_states);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << duration << std::endl;
    std::cout << "transition ... ";
    start = std::clock();
    Matrix<T> transition = compute_transition(eta, hidden_states, rho);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << duration << std::endl;
    std::cout << "emission ... ";
    start = std::clock();
    std::vector<Matrix<T>> emission;
    for (int i = 1; i < hidden_states.size(); ++i)
        emission.push_back(ConditionedSFS<T>::calculate_sfs(eta, n, S, M, ts, expM, 
                    hidden_states[i - 1], hidden_states[i], numthreads, theta));
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << duration << std::endl;
    std::cout << "hmm ... ";
    start = std::clock();
    T ll = compute_hmm_likelihood<T>(pi, transition, emission, L, obs, block_size, numthreads, viterbi, viterbi_paths);
    duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
    std::cout << duration << std::endl;
    std::cout << "loglik: " << ll;
    ll -= lambda * eta.regularizer();
    std::cout << " (with reg): " << ll << std::endl;
    return ll;
}

template adouble loglik(
        const std::vector<std::vector<double>>&,
        const int, 
        const int, const int,
        const std::vector<double>&, const std::vector<double*>&,
        const int, const std::vector<int*>&, 
        const std::vector<double>&,
        const double, const double,
        const int,
        int, 
        bool, std::vector<std::vector<int>>&,
        double);

template double loglik(
        const std::vector<std::vector<double>>&,
        const int, 
        const int, const int,
        const std::vector<double>&, const std::vector<double*>&,
        const int, const std::vector<int*>&, 
        const std::vector<double>&,
        const double, const double,
        const int,
        int, 
        bool, std::vector<std::vector<int>> &,
        double);

*/

void fill_jacobian(const adouble &ll, double* outjac)
{
    Eigen::VectorXd d = ll.derivatives();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(outjac, d.rows());
    _jac = d;
}
