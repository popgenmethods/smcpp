#include "hmm.h"
#include <csignal>

template <typename T>
HMM<T>::HMM(const Vector<T> &pi, const Matrix<T> &transition, 
        const std::vector<Matrix<T>> &emission,
        const int L, const int* obs) :
    pi(pi), transition(transition), emission(emission), M(emission.size()),
    L(L), obs(obs, L, 3)
{ 
    feraiseexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    Eigen::DiagonalMatrix<T, Eigen::Dynamic> D(M);
    D.setZero();
    diag_obs(D, 0, 0);
    O0T = D * transition.transpose();
}


template <typename T>
T HMM<T>::loglik()
{
    return std::accumulate(logc.begin(), logc.end(), (T)0.0);
}

template <typename T>
std::vector<int>& HMM<T>::viterbi(void)
{
    // Compute logs of transition and emission matrices 
    // Do not bother to record derivatives since we don't use them for Viterbi algorithm
    std::vector<Eigen::MatrixXd> log_emission(L);
    Eigen::MatrixXd log_transition = transition.template cast<double>().array().log();
    std::transform(emission.begin(), emission.end(), log_emission.begin(), 
            [](decltype(emission[0]) &x) -> Eigen::MatrixXd {return x.template cast<double>().array().log();});
    std::vector<double> V(M), V1(M);
    std::vector<std::vector<int>> path(M), newpath(M);
    Eigen::MatrixXd pid = pi.template cast<double>();
    std::vector<int> zeros(M, 0);
    double p, p2, lemit;
    int st;
    for (int m = 0; m < M; ++m)
    {
        V[m] = log(pid(m)) + log_emission[m](obs(0,1), obs(0,2));
        path[m] = zeros;
        path[m][m]++;
    }
    for (int ell = 0; ell < L; ++ell)
    {
        int R = obs(ell, 0);
        if (ell == 0)
            R--;
        for (int r = 0; r < R; ++r)
        {
            for (int m = 0; m < M; ++m)
            {
                lemit = log_emission[m](obs(ell, 1), obs(ell, 2));
                p = -INFINITY;
                st = 0;
                for (int m2 = 0; m2 < M; ++m2)
                {
                    p2 = V[m2] + log_transition(m2, m) + lemit;
                    if (p2 > p)
                    {
                        p = p2;
                        st = m2;
                    }
                }
                V1[m] = p;
                newpath[m] = path[st];
                newpath[m][m]++;
            }
            path = newpath;
            V = V1;
        }
    }
    p = -INFINITY;
    st = 0;
    for (int m = 0; m < M; ++m)
    {
        if (V[m] > p)
        {
            p = V[m];
            st = m;
        }
    }
    viterbi_path = path[st];
    return viterbi_path;
}

template <typename T>
Matrix<T> HMM<T>::O0Tpow(int p)
{
    Matrix<T> A;
    if (p == 0)
        return Matrix<T>::Identity(M, M);
    if (p == 1)
        return O0T;
    if (O0Tpow_memo.count(p) == 0)
    {
        if (p % 2 == 0)
        {
            A = O0Tpow(p / 2);
            O0Tpow_memo[p] = A * A;
        }
        else
        {
            A = O0Tpow(p - 1);
            O0Tpow_memo[p] = O0T * A;
        }
    }
    return O0Tpow_memo[p];
}

template <typename T>
Matrix<T> HMM<T>::matpow(const Matrix<T> &A, int pp)
{
    Matrix<T> m_tmp = A;
    Matrix<T> res = Matrix<T>::Identity(M, M);
    while (pp >= 1) {
        if (std::fmod(pp, 2) >= 1)
            res = m_tmp * res;
        m_tmp *= m_tmp;
        pp /= 2;
    }
    return res;
}

template <typename T>
template <int s>
void HMM<T>::diag_obs(Eigen::DiagonalMatrix<T, s> &D, int a, int b)
{
    for (int m = 0; m < M; ++m)
    {
        D.diagonal()(m) = emission[m](a, b);
        assert(emission[m](a,b)>=0);
    }
}

template <typename T>
void HMM<T>::forward(void)
{
    // feenableexcept(FE_DIVBYZERO | FE_INVALID | FE_OVERFLOW);
    T c0;
    Vector<T> alpha_hat(M);
    Eigen::DiagonalMatrix<T, Eigen::Dynamic> D(M);
    D.setZero();
    int p;
    // M x M transition matrix
    // Initialize alpha hat
    diag_obs(D, obs(0,1), obs(0,2));
    alpha_hat = D * pi;
    p = obs(0,0) - 1; 
    alpha_hat = matpow(D * transition.transpose(), p) * alpha_hat;
    c0 = alpha_hat.sum();
    alpha_hat /= c0;
    assert(c0 > 0);
    logc.push_back(log(c0));
    for (int ell = 1; ell < L; ++ell)
    {
        int p = obs(ell, 0);
        if (obs(ell, 1) == 0 && obs(ell, 2) == 0)
            alpha_hat = O0Tpow(p) * alpha_hat;
        else    
        {
            diag_obs(D, obs(ell, 1), obs(ell, 2));
            alpha_hat = matpow(D * transition.transpose(), p) * alpha_hat;
        }
        // std::cout << obs.block(ell, 0, 1, 3) << " :: " << alpha_hat.cast<double>().transpose() << std::endl;
        c0 = alpha_hat.sum();
        if (isnan(toDouble(c0)) || c0 <= 0.0)
            throw std::domain_error("nan encountered in hmm");
        alpha_hat /= c0;
        assert(c0 > 0);
        logc.push_back(log(c0));
    }
}

template <typename T>
T compute_hmm_likelihood(
        const RateFunction<T> &eta, 
        const Vector<T> &pi, const Matrix<T> &transition,
        const std::vector<Matrix<T>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths)
{
    // eta.print_debug();
    ThreadPool tp(numthreads);
    std::vector<HMM<T>> hmms;
    std::vector<std::thread> t;
    std::vector<std::future<T>> results;
    for (auto ob : obs)
        hmms.emplace_back(pi, transition, emission, L, ob);
    for (auto &hmm : hmms)
        results.emplace_back(tp.enqueue([&] { hmm.forward(); return hmm.loglik(); }));
    T ret = 0.0;
    for (auto &&res : results)
        ret += res.get();
    std::vector<std::future<std::vector<int>>> viterbi_results;
    if (viterbi)
    {
        for (auto &hmm : hmms)
            viterbi_results.emplace_back(tp.enqueue([&] { return hmm.viterbi(); }));
        for (auto &&res : viterbi_results)
            viterbi_paths.push_back(res.get());
    }
    return ret;
}

template double compute_hmm_likelihood(
        const RateFunction<double> &eta, 
        const Vector<double> &pi, const Matrix<double> &transition,
        const std::vector<Matrix<double>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

template adouble compute_hmm_likelihood(
        const RateFunction<adouble> &eta, 
        const Vector<adouble> &pi, const Matrix<adouble> &transition,
        const std::vector<Matrix<adouble>>& emission, 
        const int L, const std::vector<int*> obs,
        int numthreads, 
        bool viterbi, std::vector<std::vector<int>> &viterbi_paths);

template class HMM<double>;
template class HMM<adouble>;

