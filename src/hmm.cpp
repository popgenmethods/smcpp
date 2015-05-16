#include "hmm.h"

double mylog(double x) { return log(x); }

HMM::HMM(const AdMatrix &pi, const AdMatrix &transition, const std::vector<AdMatrix> &emission,
        const std::vector<double> hidden_states, const int L, const int* obs, double rho) :
    pi(pi), transition(transition), emission(emission), M(hidden_states.size() - 1), 
    L(L), obs(obs, L, 3), hidden_states(hidden_states), rho(rho)
{ 
    // feenableexcept(FE_INVALID | FE_OVERFLOW);
    Eigen::DiagonalMatrix<adouble, Eigen::Dynamic> D(M);
    D.setZero();
    diag_obs(D, 0, 0);
    O0T = D * transition.transpose();
}

AdMatrix compute_initial_distribution(const PiecewiseExponential &eta, const std::vector<double> &hidden_states)
{
    int M = hidden_states.size() - 1;
    AdVector pi(M);
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-eta.inverse_rate(hidden_states[m], 0.0, 1.0)) - 
            exp(-eta.inverse_rate(hidden_states[m + 1], 0.0, 1.0));
    }
    pi(M - 1) = exp(-eta.inverse_rate(hidden_states[M - 1], 0.0, 1.0));
    return pi;
}

AdMatrix compute_transition(const PiecewiseExponential &eta, const std::vector<double> &hidden_states, double rho)
{
    Transition trans(eta, hidden_states, rho);
    // std::cout << "** Transition" << std::endl << trans.matrix().cast<double>() << std::endl << std::endl;
    return trans.matrix();
}

adouble HMM::logp()
{
    return std::accumulate(logc.begin(), logc.end(), (adouble)0.0);
}

std::vector<int>& HMM::viterbi(void)
{
    // Compute logs of transition and emission matrices 
    // Do not bother to record derivatives since we don't use them for Viterbi algorithm
    std::vector<Eigen::MatrixXd> log_emission(L);
    Eigen::ArrayXd log_transition = transition.cast<double>().array().log();
    std::transform(emission.begin(), emission.end(), log_emission.begin(), [](AdMatrix &x){return x.cast<double>().array().log().matrix();});
    std::vector<double> V(M), V1(M);
    std::vector<std::vector<int>> path(M), newpath(M);
    Eigen::MatrixXd pid = pi.cast<double>();
    double p, p2, lemit;
    int st;
    for (int m = 0; m < M; ++m)
    {
        V[m] = log(pid(m)) + log_emission[m](obs(0,0), obs(0,1));
        path[m] = {m};
    }
    for (int ell = 1; ell < L; ++ell)
    {
        for (int m = 0; m < M; ++m)
        {
            lemit = log_emission[m](obs(ell,0), obs(ell,1));
            p = V[0] + log_transition(0,m) + lemit;
            st = 0;
            for (int m2 = 1; m2 < M; ++m2)
            {
                p2 = V[m2] + log_transition(m2,m) + lemit;
                if (p2 > p)
                {
                    p = p2;
                    st = m2;
                }
            }
            V1[m] = p;
            newpath[m].assign(path[st].begin(), path[st].end());
            newpath[m].push_back(m);
        }
        path = newpath;
        V = V1;
    }
    p = V[0];
    st = 0;
    for (int m = 1; m < M; ++m)
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

AdMatrix HMM::O0Tpow(int p)
{
    AdMatrix A;
    if (p == 0)
        return AdMatrix::Identity(M, M);
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

AdMatrix HMM::matpow(const AdMatrix &T, int pp)
{
    AdMatrix m_tmp = T;
    AdMatrix res = AdMatrix::Identity(M, M);
    while (pp >= 1) {
        if (std::fmod(pp, 2) >= 1)
            res = m_tmp * res;
        m_tmp *= m_tmp;
        pp /= 2;
    }
    return res;
}

template <typename T, int s>
void HMM::diag_obs(Eigen::DiagonalMatrix<T, s> &D, int a, int b)
{
    for (int m = 0; m < M; ++m)
    {
        D.diagonal()(m) = emission[m](a, b);
        assert(emission[m](a,b)>=0);
    }
}

void HMM::forward(void)
{
    adouble c0;
    AdVector alpha_hat(M);
    Eigen::DiagonalMatrix<adouble, Eigen::Dynamic> D(M);
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
    logc.push_back(log(c0));
    for (int ell = 1; ell < L; ++ell)
    {
        p = obs(ell, 0);
        if (obs(ell, 1) == 0 && obs(ell, 2) == 0)
            alpha_hat = O0Tpow(p) * alpha_hat;
        else    
        {
            diag_obs(D, obs(ell, 1), obs(ell, 2));
            alpha_hat = matpow(D * transition.transpose(), p) * alpha_hat;
        }
        // std::cout << obs.block(ell, 0, 1, 3) << " :: " << alpha_hat.cast<double>().transpose() << std::endl;
        c0 = alpha_hat.sum();
        alpha_hat /= c0;
        logc.push_back(log(c0));
    }
}

double compute_hmm_likelihood(double *jac, 
        const PiecewiseExponential &eta, const std::vector<AdMatrix>& emission, 
        const int L, const std::vector<int*> obs, const std::vector<double> &hidden_states, 
        const double rho, int numthreads, std::vector<std::vector<int>> &viterbi_paths, bool do_viterbi,
        double reg_a, double reg_b, double reg_s)
{
    AdMatrix pi = compute_initial_distribution(eta, hidden_states);
    AdMatrix transition = compute_transition(eta, hidden_states, rho);
    ThreadPool tp(numthreads);
    std::vector<HMM> hmms;
    std::vector<std::thread> t;
    std::vector<std::future<adouble>> results;
    for (auto ob : obs)
        hmms.emplace_back(pi, transition, emission, hidden_states, L, ob, rho);
    for (auto &hmm : hmms)
        results.emplace_back(tp.enqueue([&] { hmm.forward(); return hmm.logp(); }));
    adouble ret = 0.0;
    for (auto &&res : results)
        ret += res.get();
    // Add in regularizers
    int K = eta.K();
    std::vector<std::vector<adouble>> ad = eta.ad_vars();
    for (int k = 0; k < K; ++k)
    {
        ret += reg_a * pow(ad[0][k], 2);
        ret += reg_b * pow(ad[1][k], 2);
        ret += reg_s * pow(ad[2][k], 2);
    }
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(jac, 3 * K);
    _jac = ret.derivatives();
    std::vector<std::future<std::vector<int>>> viterbi_results;
    if (do_viterbi)
    {
        for (auto &hmm : hmms)
            viterbi_results.emplace_back(tp.enqueue([&] { return hmm.viterbi(); }));
        for (auto &&res : viterbi_results)
            viterbi_paths.push_back(res.get());
    }
    return ret.value();
}

void HMM::printobs() { std::cout << obs << std::endl; }
