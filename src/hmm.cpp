#include "hmm.h"

double mylog(double x) { return log(x); }

HMM::HMM(PiecewiseExponential *eta, 
        const std::vector<std::vector<ConditionedSFS*>> &csfs,
        const std::vector<double> hidden_states,
        int L, int* obs,
        double rho, double theta) : 
    M(hidden_states.size() - 1), pi(M), eta(eta), 
    L(L), obs(obs, L, 2), hidden_states(hidden_states),
    c(L), rho(rho), theta(theta)
{ 
    feraiseexcept(FE_ALL_EXCEPT & ~FE_UNDERFLOW & ~FE_INEXACT);
    average_sfs(csfs);
    compute_initial_distribution();
    compute_transition();
}

void HMM::compute_initial_distribution(void)
{
    for (int m = 0; m < M - 1; ++m)
        pi(m) = exp(-eta->inverse_rate(hidden_states[m], 0.0, 1.0)) - 
            exp(-eta->inverse_rate(hidden_states[m + 1], 0.0, 1.0));
    pi(M - 1) = exp(-eta->inverse_rate(hidden_states[M - 1], 0.0, 1.0));
}

void HMM::compute_transition(void)
{
    Transition trans(eta, hidden_states, rho);
    transition = trans.matrix();
}

void HMM::average_sfs(const std::vector<std::vector<ConditionedSFS*>> &csfs)
{
    int n = csfs[0][0]->matrix().cols();
    adouble s;
    for (int m = 0; m < csfs.size(); ++m)
    {
        emission.emplace_back(3, n);
        emission[m].setZero();
        for (int ell = 0; ell < csfs[m].size(); ++ell)
        {
            emission[m] += csfs[m][ell]->matrix();
        }
        emission[m] /= csfs[m].size();
        s = emission[m].sum();
        emission[m] *= theta;
        emission[m](0, 0) = (1 - theta) * s;
    }
}

double HMM::logp(double* jac)
{
    int r = std::feraiseexcept(FE_UNDERFLOW);
    // Add jacobian dependence
    adouble lp = 0.0;
    forward();
    for (int ell = 0; ell < L; ++ell)
        lp += log(c[ell]);
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(jac, lp.derivatives().rows());
    _jac = lp.derivatives();
    return lp.value();
}

std::vector<int>& HMM::viterbi(void)
{
    // Compute logs of transition and emission matrices 
    // Do not bother to record derivatives since we don't use them for MAP decoding
    std::vector<Eigen::MatrixXd> log_emission;
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

/*
AdMatrix HMM::O0Tpow(int pp)
{
    AdMatrix m_tmp = O0T;
    AdMatrix res = AdMatrix::Identity(M, M);
    while (pp >= 1) {
        if (std::fmod(pp, 2) >= 1)
            res = m_tmp * res;
        m_tmp *= m_tmp;
        pp /= 2;
    }
}
*/

void HMM::forward(void)
{
    AdVector alpha_hat(M);
    Eigen::DiagonalMatrix<adouble, Eigen::Dynamic> D(M);
    D.setZero();
    /*
    // The unsegregating state will be emitted a lot so store its Schur decomposition
    // for easy powering
    AdMatrix O0 = AdMatrix::Zero(M, M);
    for (int m = 0; m < M; ++m)
        O0.diagonal(m) = emission[m](0, 0);
    O0T = emission * O0;
    */

    // M x M transition matrix
    // Initialize alpha hat
#define DIAG_OBS(a, b) \
    for (int m = 0; m < M; ++m) \
        D.diagonal()(m) = emission[m](a, b);
    DIAG_OBS(obs(0,0), obs(0,1));
    alpha_hat = D * pi;
    c[0] = alpha_hat.sum();
    alpha_hat /= c[0];
    // Now proceed with forward recursion
    for (int ell = 1; ell < L; ++ell)
    {
        DIAG_OBS(obs(ell,0), obs(ell,1));
        alpha_hat = D * transition.transpose() * alpha_hat;
        c[ell] = alpha_hat.sum();
        alpha_hat /= c[ell];
    }
#undef DIAG_OBS
}

