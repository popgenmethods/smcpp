#include "inference_manager.h"

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

InferenceManager::InferenceManager(
            const MatrixInterpolator &moran_interp,
            const int n, const int L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const double theta, const double rho, 
            const int block_size, const int num_threads, 
            const int num_samples) : 
    moran_interp(moran_interp), n(n), L(L),
    observations(observations), 
    hidden_states(hidden_states),
    theta(theta), rho(rho),
    block_size(block_size), num_threads(num_threads),
    num_samples(num_samples), M(hidden_states.size() - 1), 
    tp(num_threads) 
{
    d_bundle.pi = Vector<double>::Zero(M);
    ad_bundle.pi = Vector<adouble>::Zero(M);
    d_bundle.transition = Matrix<double>::Zero(M, M);
    ad_bundle.transition = Matrix<adouble>::Zero(M, M);
    d_bundle.emission = Matrix<double>::Zero(M, 3 * (n + 1));
    ad_bundle.emission = Matrix<adouble>::Zero(M, 3 * (n + 1));
    Eigen::Matrix<int, Eigen::Dynamic, 2>  ob(L, 2);
    Eigen::Matrix<int, Eigen::Dynamic, 3>  tmp(L, 3);
    for (auto &obs : observations)
    {
        tmp = Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>::Map(obs, L, 3);
        ob.col(0) = tmp.col(0);
        ob.col(1) = (n + 1) * tmp.col(1) + tmp.col(2);
        d_bundle.hmms.emplace_back(ob, block_size, &d_bundle.pi, &d_bundle.transition, &d_bundle.emission);
        ad_bundle.hmms.emplace_back(ob, block_size, &ad_bundle.pi, &ad_bundle.transition, &ad_bundle.emission);
    }
}

template <typename T>
void InferenceManager::setParams(const ParameterVector params)
{
    PiecewiseExponentialRateFunction<T> eta(params);
    HMMBundle<T> &bundle = getBundle<T>();
    bundle.pi = compute_initial_distribution<T>(eta, hidden_states);
    bundle.transition = compute_transition(eta, hidden_states, rho);
    Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor> tmp;
    for (int m = 0; m < M; ++m)
    {
        tmp = sfs<T>(params, hidden_states[m], hidden_states[m + 1]);
        bundle.emission.row(m) = Matrix<T>::Map(tmp.data(), 1, 3 * (n + 1));
    }
    parallel_do<T>([] (HMM<T> &hmm) { hmm.recompute_B(); });
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
    
template <typename T>
void InferenceManager::parallel_do(std::function<void(HMM<T> &)> lambda)
{
    std::vector<std::future<void>> results;
    for (auto &&hmm : getBundle<T>().hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, hmm)));
    for (auto &res : results) 
        res.wait();
}
template void InferenceManager::parallel_do(std::function<void(HMM<double> &)>);
template void InferenceManager::parallel_do(std::function<void(HMM<adouble> &)>);

template <typename T>
std::vector<T> InferenceManager::parallel_select(std::function<T(HMM<T> &)> lambda)
{
    std::vector<std::future<T>> results;
    for (auto &&hmm : getBundle<T>().hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, hmm)));
    std::vector<T> ret; 
    for (auto &res : results) 
        ret.push_back(res.get());
    return ret;
}
template std::vector<double> InferenceManager::parallel_select(std::function<double(HMM<double> &)>);
template std::vector<adouble> InferenceManager::parallel_select(std::function<adouble(HMM<adouble> &)>);

template <typename T>
void InferenceManager::Estep(void)
{
    parallel_do<T>([] (HMM<T> &hmm) { hmm.Estep(); });
}
template void InferenceManager::Estep<double>(void);
template void InferenceManager::Estep<adouble>(void);

template <typename T>
std::vector<T> InferenceManager::Q(double lambda)
{
    return parallel_select<T>([] (HMM<T> &hmm) { return hmm.Q(); });
}
template std::vector<double> InferenceManager::Q<double>(double lambda);
template std::vector<adouble> InferenceManager::Q<adouble>(double lambda);

template <typename T>
std::vector<T> InferenceManager::loglik(double lambda)
{
    return parallel_select<T>([] (HMM<T> &hmm) { return hmm.loglik(); });
}
template std::vector<double> InferenceManager::loglik<double>(double lambda);
template std::vector<adouble> InferenceManager::loglik<adouble>(double lambda);

template<>
InferenceManager::HMMBundle<double>& InferenceManager::getBundle () { return d_bundle; }

template <>
InferenceManager::HMMBundle<adouble>& InferenceManager::getBundle () { return ad_bundle; }
