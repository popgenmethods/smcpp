#include "inference_manager.h"

template <typename T>
Vector<T> compute_initial_distribution(const PiecewiseExponentialRateFunction<T> &eta)
{
    auto R = eta.getR();
    int M = eta.hidden_states.size() - 1;
    Vector<T> pi(M);
    T minval = eta.zero + 1e-20;
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-(*R)(eta.hidden_states[m])) - exp(-(*R)(eta.hidden_states[m + 1]));
        if (pi(m) < minval) pi(m) = minval;
        assert(pi(m) > 0.0); 
        assert(pi(m) < 1.0); 
    }
    pi(M - 1) = exp(-(*R)(eta.hidden_states[M - 1]));
    if (pi(M - 1) < minval) pi(M - 1) = minval;
    pi /= pi.sum();
    check_nan(pi);
    return pi;
}

InferenceManager::InferenceManager(
            const int n, const std::vector<int> L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const int* _emask,
            const int mask_freq,
            const double theta, const double rho, 
            const int block_size) :
    debug(false), hj(true),
    n(n), L(L),
    observations(observations), 
    hidden_states(hidden_states),
    H(hidden_states.size() - 1),
    emask(Eigen::Map<const Eigen::Matrix<int, 3, Eigen::Dynamic, Eigen::RowMajor>>(_emask, 3, n + 1)),
    mask_freq(mask_freq),
    theta(theta), rho(rho),
    block_size(block_size), 
    M(hidden_states.size() - 1), 
    seed(1), csfs_d(n, H), csfs_ad(n, H)
{
    if (*std::min_element(hidden_states.begin(), hidden_states.end()) != 0.)
        throw std::runtime_error("first hidden interval should be [0, <something>)");
    if (*std::max_element(hidden_states.begin(), hidden_states.end()) > T_MAX)
        throw std::runtime_error("largest hidden state cannot exceed T_MAX=" + std::to_string(T_MAX));
    pi = Vector<adouble>::Zero(M);
    transition = Matrix<adouble>::Zero(M, M);
    transition.setZero();
    emission = Matrix<adouble>::Zero(M, 3 * (n + 1));
    emission.setZero();
    hmms.resize(observations.size());
#pragma omp parallel for
    for (unsigned int i = 0; i < observations.size(); ++i)
    {
        int ell = L[i];
        Eigen::Matrix<int, Eigen::Dynamic, 3> obs = 
            Eigen::Matrix<int, Eigen::Dynamic, 3, Eigen::RowMajor>::Map(observations[i], ell, 3);
        int max_n = obs.rightCols(2).rowwise().sum().maxCoeff();
        if (max_n > n + 2 - 1)
            throw std::runtime_error("An observation has derived allele count greater than n + 1");
        PROGRESS("creating HMM");
        hmmptr h(new HMM(obs, n, block_size, &pi, &transition, &emission, emask, mask_freq));
        hmms[i] = std::move(h);
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
    int ps = params[0].size();
    for (auto &pp : params)
        if (pp.size() != params[0].size())
            throw std::runtime_error("params must have matching sizes");
    PiecewiseExponentialRateFunction<T> eta(params, derivatives, hidden_states);
    regularizer = adouble(eta.regularizer());
    pi = compute_initial_distribution<T>(eta).template cast<adouble>();
    transition = compute_transition<T>(eta, rho, hj).template cast<adouble>();
    check_nan(transition);
    std::vector<Matrix<T> > sfss = sfs<T>(eta);
    Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor> em_tmp(3, n + 1);
    for (int m = 0; m < M; ++m)
    {
        check_nan(sfss[m]);
        em_tmp = sfss[m];
        emission.row(m) = Matrix<T>::Map(em_tmp.data(), 1, 3 * (n + 1)).template cast<adouble>();
    }
    /*
    std::cout << emission.template cast<double>() << std::endl << std::endl;
    std::cout << emission_mask.template cast<double>() << std::endl << std::endl;
    std::cout << emission2.template cast<double>() << std::endl;
    */
    parallel_do([] (hmmptr &hmm) { hmm->recompute_B(); });
}
template void InferenceManager::setParams<double>(const ParameterVector, const std::vector<std::pair<int, int>>);
template void InferenceManager::setParams<adouble>(const ParameterVector, const std::vector<std::pair<int, int>>);

template <>
ConditionedSFS<double>& InferenceManager::getCsfs() { return csfs_d; }
template <>
ConditionedSFS<adouble>& InferenceManager::getCsfs() { return csfs_ad; }

template <typename T>
std::vector<Matrix<T> > InferenceManager::sfs(const PiecewiseExponentialRateFunction<T> &eta)
{
    PROGRESS("sfs");
    return getCsfs<T>().compute(eta, theta);
    PROGRESS("sfs done");
}

template std::vector<Matrix<double> > InferenceManager::sfs(const PiecewiseExponentialRateFunction<double> &);
template std::vector<Matrix<adouble> > InferenceManager::sfs(const PiecewiseExponentialRateFunction<adouble> &);

void InferenceManager::parallel_do(std::function<void(hmmptr&)> lambda)
{
#pragma omp parallel for
    for (auto it = hmms.begin(); it < hmms.end(); ++it)
        lambda(*it);
    /*
    std::vector<std::future<void>> results;
    for (auto &hmmptr : hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, std::ref(hmmptr))));
    for (auto &res : results) 
        res.wait();
        */
}

template <typename T>
std::vector<T> InferenceManager::parallel_select(std::function<T(hmmptr &)> lambda)
{
    std::vector<T> ret(hmms.size());
#pragma omp parallel for
    for (int i = 0; i < hmms.size(); ++i)
        ret[i] = lambda(hmms[i]);
    return ret;
    /*
    std::vector<std::future<T>> results;
    for (auto &hmmptr : hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, std::ref(hmmptr))));
    std::vector<T> ret; 
    for (auto &res : results) 
        ret.push_back(res.get());
    return ret;
    */
}
template std::vector<double> InferenceManager::parallel_select(std::function<double(hmmptr &)>);
template std::vector<adouble> InferenceManager::parallel_select(std::function<adouble(hmmptr &)>);

void InferenceManager::Estep(void)
{
    PROGRESS("E step");
    parallel_do([] (hmmptr &hmm) { hmm->Estep(); });
    PROGRESS_DONE();
}

std::vector<adouble> InferenceManager::Q(double lambda)
{
    adouble reg = regularizer;
    PROGRESS("InferenceManager::Q");
    return parallel_select<adouble>([lambda, reg] (hmmptr &hmm) { 
            adouble q = hmm->Q();
            adouble rr = reg * lambda;
            adouble ret = q - rr;
            return ret;
            });
}

std::vector<Matrix<float>*> InferenceManager::getXisums()
{
    std::vector<Matrix<float>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm->xisum);
    }
    return ret;
}

std::vector<Matrix<adouble>*> InferenceManager::getBs()
{
    std::vector<Matrix<adouble>*> ret;
    for (auto &hmm : hmms)
    {
        hmm->fill_B();
        ret.push_back(&hmm->B);
        // This takes up too much memory so only return one.
        break;
    }
    return ret;
}

std::vector<std::vector<std::pair<bool, decltype(block_key::powers)> > > InferenceManager::getBlockKeys()
{
    std::vector<std::vector<std::pair<bool, decltype(block_key::powers)> > > ret;
    for (auto &hmm : hmms)
        ret.push_back(hmm->block_keys);
    return ret;
}

Matrix<adouble>& InferenceManager::getPi(void)
{
    static Matrix<adouble> mat;
    mat = pi;
    return mat;
}

Matrix<adouble>& InferenceManager::getTransition(void)
{
    return transition;
}

Matrix<adouble>& InferenceManager::getEmission(void)
{
    return emission;
}

Matrix<adouble>& InferenceManager::getMaskedEmission(void)
{
    return emission_mask;
}

std::vector<double> InferenceManager::loglik(double lambda)
{
    double reg = toDouble(regularizer);
    return parallel_select<double>([lambda, reg] (hmmptr &hmm) { return hmm->loglik() - lambda * reg; });
}

/*
int main(int argc, char** argv)
{
    std::vector<int> L = {3000};
    std::vector<int> obs;
    for (int i = 0; i < L[0]; ++i)
    {
        if (i % 2)
        {
            obs.push_back(1);
            obs.push_back(1);
            obs.push_back(0);
        }
        else
        {
            obs.push_back(100);
            obs.push_back(0);
            obs.push_back(0);
        }
    }
    std::vector<int*> vobs = {obs.data()};
    std::vector<double> hs = {0.0,1.0,2.0,10.0};
    int emask[] = {0, 1, 0};
    std::vector<int*> vemask = {emask};
    InferenceManager im(0, L, vobs, hs, emask, 5, {0}, 4 * 1e4 * 1e-8, 4 * 1e4 * 1e-8, 50);
    std::vector<std::vector<double> > params = {
        {0.2, 1.0, 2.0},
        {1.0, 1.0, 2.0},
        {1.0, 0.1, 0.1}
    };
    std::vector<std::pair<int, int> > deriv = { {1,0} };
    im.setParams<adouble>(params, deriv);
    im.Estep();
    adouble Q0 = im.Q(0.0)[0];
    Matrix<adouble> T = im.getTransition();
    Matrix<adouble> E = im.getEmission();
    params[1][0] += 1e-8;
    im.setParams<double>(params, deriv);
    double Q1 = im.Q(0.0)[0].value();
    Matrix<adouble> T2 = im.getTransition();
    Matrix<adouble> E2 = im.getEmission();
    std::cout << (Q1 - Q0.value()) * 1e8 << " " << Q0.derivatives() << std::endl << std::endl;
    std::cout << "[" << E2.template cast<double>() << "]" << std::endl;
    std::cout << (E2.template cast<double>() - E.template cast<double>()) * 1e8 << std::endl << std::endl;
    std::cout << E.unaryExpr([](adouble x) { return x.derivatives()(0); }).template cast<double>() << std::endl << std::endl;
    std::cout << (T2.template cast<double>() - T.template cast<double>()) * 1e8 << std::endl << std::endl;
    std::cout << T.unaryExpr([](adouble x) { return x.derivatives()(0); }).template cast<double>() << std::endl << std::endl;
}
*/
