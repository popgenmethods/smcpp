#include "inference_manager.h"

template <typename T>
Vector<T> compute_initial_distribution(const RateFunction<T> &eta, const std::vector<double> &hidden_states)
{
    auto Rinv = eta.getRinv();
    int M = hidden_states.size() - 1;
    Vector<T> pi(M);
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = dmax(exp(-Rinv->operator()(hidden_states[m])) - exp(-Rinv->operator()(hidden_states[m + 1])), 1e-16);
        assert(pi(m) > 0.0); 
        assert(pi(m) < 1.0); 
    }
    pi(M - 1) = dmax(exp(-Rinv->operator()(hidden_states[M - 1])), 1e-16);
    pi /= pi.sum();
    return pi;
}

InferenceManager::InferenceManager(
            const MatrixInterpolator &moran_interp,
            const int n, const int L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const double theta, const double rho, 
            const int block_size, const int num_threads, 
            const int num_samples) : 
    debug(false),
    moran_interp(moran_interp), n(n), L(L),
    observations(observations), 
    hidden_states(hidden_states),
    theta(theta), rho(rho),
    block_size(block_size), num_threads(num_threads),
    num_samples(num_samples), M(hidden_states.size() - 1), 
    tp(num_threads) 
{
    pi = Vector<adouble>::Zero(M);
    transition = Matrix<adouble>::Zero(M, M);
    emission = Matrix<adouble>::Zero(M, 3 * (n + 1));
    Eigen::Matrix<int, Eigen::Dynamic, 2>  ob(L, 2);
    Eigen::Matrix<int, Eigen::Dynamic, 3>  tmp(L, 3);
    for (auto &obs : observations)
    {
        tmp = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>::Map(obs, L, 3);
        ob.col(0) = tmp.col(0);
        ob.col(1) = (n + 1) * tmp.col(1) + tmp.col(2);
        hmms.emplace_back(ob, block_size, &pi, &transition, &emission);
    }
}


template <typename T>
void InferenceManager::setParams(const ParameterVector params)
{
    PiecewiseExponentialRateFunction<T> eta(params);
    regularizer = adouble(eta.regularizer());
    pi = compute_initial_distribution<T>(eta, hidden_states).template cast<adouble>();
    transition = compute_transition<T>(eta, hidden_states, rho).template cast<adouble>();
    Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor> tmp;
    for (int m = 0; m < M; ++m)
    {
        tmp = sfs<T>(params, hidden_states[m], hidden_states[m + 1]);
        emission.row(m) = Matrix<T>::Map(tmp.data(), 1, 3 * (n + 1)).template cast<adouble>();
    }
    parallel_do([] (HMM &hmm) { hmm.recompute_B(); });
    if (debug)
    {
        std::cout << emission.template cast<double>() << std::endl << std::endl;
        std::cout << hmms[0].B.leftCols(10).template cast<double>() << std::endl << std::endl;
    }
    
}
template void InferenceManager::setParams<double>(const ParameterVector);
template void InferenceManager::setParams<adouble>(const ParameterVector);

template <typename T>
Matrix<T> InferenceManager::sfs(const ParameterVector params, double t1, double t2)
{
    PiecewiseExponentialRateFunction<T> eta(params);
    return ConditionedSFS<T>::calculate_sfs(eta, n, num_samples, moran_interp, t1, t2, num_threads, theta);
}
template Matrix<double> InferenceManager::sfs<double>(const ParameterVector, double, double);
template Matrix<adouble> InferenceManager::sfs<adouble>(const ParameterVector, double, double);

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
    parallel_do([] (HMM &hmm) { hmm.Estep(); });
}

std::vector<adouble> InferenceManager::Q(double lambda)
{
    adouble reg = regularizer;
    return parallel_select<adouble>([lambda, reg] (HMM &hmm) { 
            adouble q = hmm.Q();
            adouble rr = reg * lambda;
            adouble ret = q - rr;
            return ret;
            });
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

std::vector<double> InferenceManager::loglik(double lambda)
{
    double reg = toDouble(regularizer);
    return parallel_select<double>([lambda, reg] (HMM &hmm) { return hmm.loglik() - lambda * reg; });
}
