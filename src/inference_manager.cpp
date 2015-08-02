#include "inference_manager.h"

template <typename T>
Vector<T> compute_initial_distribution(const PiecewiseExponentialRateFunction<T> &eta, const std::vector<double> &hidden_states)
{
    auto R = eta.getR();
    int M = hidden_states.size() - 1;
    Vector<T> pi(M);
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = dmax(exp(-(*R)(hidden_states[m])) - exp(-(*R)(hidden_states[m + 1])), 1e-16);
        assert(pi(m) > 0.0); 
        assert(pi(m) < 1.0); 
    }
    pi(M - 1) = dmax(exp(-(*R)(hidden_states[M - 1])), 1e-16);
    pi /= pi.sum();
    return pi;
}

InferenceManager::InferenceManager(
            const MatrixInterpolator &moran_interp,
            const int n, const int L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const int* emask,
            const int mask_freq,
            std::vector<int> mask_offset,
            const double theta, const double rho, 
            const int block_size, const int num_threads, 
            const int num_samples) : 
    debug(false),
    moran_interp(moran_interp), n(n), L(L),
    observations(observations), 
    hidden_states(hidden_states),
    emask(emask, 3, n + 1),
    mask_freq(mask_freq),
    mask_offset(mask_offset),
    theta(theta), rho(rho),
    block_size(block_size), num_threads(num_threads),
    num_samples(num_samples), 
    M(hidden_states.size() - 1), 
    tp(num_threads), seed(1)
{
    pi = Vector<adouble>::Zero(M);
    transition = Matrix<adouble>::Zero(M, M);
    emission = Matrix<adouble>::Zero(M, 3 * (n + 1));
    emission_mask = Matrix<adouble>::Zero(M, 3 * (n + 1));
    Eigen::Matrix<int, Eigen::Dynamic, 2>  ob(L, 2);
    Eigen::Matrix<int, Eigen::Dynamic, 3>  tmp(L, 3);
    for (auto &obs : observations)
    {
        tmp = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>::Map(obs, L, 3);
        ob.col(0) = tmp.col(0);
        ob.col(1) = (n + 1) * tmp.col(1) + tmp.col(2);
        for (int off : mask_offset)
            hmms.emplace_back(ob, block_size, &pi, &transition, &emission, &emission_mask, mask_freq, off);
    }
}

template <typename T>
Matrix<T> matpow(Matrix<T> M, int p)
{
    if (p == 1)
        return M;
    Matrix<T> P = matpow(M, std::floor(p / 2.0));
    if (p % 2 == 0)
        return P * P;
    return M * P * P;
}

template <typename T>
void InferenceManager::setParams(const ParameterVector params, const std::vector<std::pair<int, int>> derivatives)
{
    PiecewiseExponentialRateFunction<T> eta(params, derivatives);
    regularizer = adouble(eta.regularizer());
    pi = compute_initial_distribution<T>(eta, hidden_states).template cast<adouble>();
    Matrix<adouble> ttmp = compute_transition<T>(eta, hidden_states, rho).template cast<adouble>();
    transition = matpow(ttmp, block_size);
    Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor> tmp;
    std::map<int, std::vector<T>> tmask;
    std::map<int, T> tavg;
    for (int m = 0; m < M; ++m)
    {
        PROGRESS("emission (" << m << ")");
        tmask.clear();
        tavg.clear();
        tmp = sfs<T>(eta, hidden_states[m], hidden_states[m + 1]);
        emission.row(m) = Matrix<T>::Map(tmp.data(), 1, 3 * (n + 1)).template cast<adouble>();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < n + 1; ++j)
                tmask[emask(i, j)].push_back(tmp(i, j));
        for (auto p : tmask)
            tavg[p.first] = std::accumulate(p.second.begin(), p.second.end(), (T)0.0);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < n + 1; ++j)
                tmp(i, j) = tavg[emask(i, j)];
        emission_mask.row(m) = Matrix<T>::Map(tmp.data(), 1, 3 * (n + 1)).template cast<adouble>();
        PROGRESS_DONE();
    }
    PROGRESS("compute B");
    parallel_do([] (HMM &hmm) { hmm.recompute_B(); });
    PROGRESS_DONE();
}
template void InferenceManager::setParams<double>(const ParameterVector, const std::vector<std::pair<int, int>>);
template void InferenceManager::setParams<adouble>(const ParameterVector, const std::vector<std::pair<int, int>>);

template <typename T>
Matrix<T> InferenceManager::sfs(PiecewiseExponentialRateFunction<T> eta, double t1, double t2)
{
    static CSFSManager<T> c(n, moran_interp, num_threads, theta);
    c.set_seed(seed);
    return c.compute(eta, num_samples, t1, t2);
}

void InferenceManager::parallel_do(std::function<void(HMM &)> lambda)
{
    std::vector<std::future<void>> results;
    for (auto &hmm : hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, std::ref(hmm))));
    for (auto &res : results) 
        res.wait();
}

template <typename T>
std::vector<T> InferenceManager::parallel_select(std::function<T(HMM &)> lambda)
{
    std::vector<std::future<T>> results;
    for (auto &hmm : hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, std::ref(hmm))));
    std::vector<T> ret; 
    for (auto &res : results) 
        ret.push_back(res.get());
    return ret;
}
template std::vector<double> InferenceManager::parallel_select(std::function<double(HMM &)>);
template std::vector<adouble> InferenceManager::parallel_select(std::function<adouble(HMM &)>);

void InferenceManager::Estep(void)
{
    PROGRESS("E step");
    parallel_do([] (HMM &hmm) { hmm.Estep(); });
    PROGRESS_DONE();
}

std::vector<adouble> InferenceManager::Q(double lambda)
{
    adouble reg = regularizer;
    PROGRESS("Q");
    return parallel_select<adouble>([lambda, reg] (HMM &hmm) { 
            adouble q = hmm.Q();
            adouble rr = reg * lambda;
            adouble ret = q - rr;
            return ret;
            });
    PROGRESS_DONE();
}

std::vector<Matrix<double>*> InferenceManager::getAlphas()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm.alpha_hat);
    }
    return ret;
}

std::vector<Matrix<double>*> InferenceManager::getBetas()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm.beta_hat);
    }
    return ret;
}

std::vector<Matrix<double>*> InferenceManager::getGammas()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm.gamma);
    }
    return ret;
}

std::vector<Matrix<adouble>*> InferenceManager::getBs()
{
    std::vector<Matrix<adouble>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm.B);
    }
    return ret;
}

Matrix<double> InferenceManager::getPi(void)
{
    return hmms[0].pi->cast<double>();
}

Matrix<double> InferenceManager::getTransition(void)
{
    return hmms[0].transition->cast<double>();
}

Matrix<double> InferenceManager::getEmission(void)
{
    return hmms[0].emission->cast<double>();
}

Matrix<double> InferenceManager::getMaskedEmission(void)
{
    return hmms[0].emission_mask->cast<double>();
}

std::vector<double> InferenceManager::loglik(double lambda)
{
    double reg = toDouble(regularizer);
    return parallel_select<double>([lambda, reg] (HMM &hmm) { return hmm.loglik() - lambda * reg; });
}
