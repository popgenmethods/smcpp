#include "inference_manager.h"

inline mpz_class multinomial(const std::vector<int> &ks)
{
    mpz_class num, den, tmp;
    int sum = 0;
    den = 1_mpz;
    for (auto k : ks) 
    {
        sum += k;
        mpz_fac_ui(tmp.get_mpz_t(), k);
        den *= tmp;
    }
    mpz_fac_ui(num.get_mpz_t(), sum);
    return num / den;
}

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

Matrix<int> make_two_mask(int m)
{
    Matrix<int> two_mask(3, m);
    two_mask.fill(0);
    two_mask.row(1).fill(1);
    return two_mask;
}


InferenceManager::InferenceManager(
            const int n, const std::vector<int> L,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const int* _emask,
            const int mask_freq,
            const double theta, const double rho, 
            const int block_size) :
    debug(false), forwardOnly(false), saveGamma(false),
    hidden_states(hidden_states),
    n(n), L(L),
    observations(observations), 
    emask(Eigen::Map<const Eigen::Matrix<int, 3, Eigen::Dynamic, Eigen::RowMajor>>(_emask, 3, n + 1)),
    two_mask(make_two_mask(n + 1)),
    mask_freq(mask_freq),
    theta(theta), rho(rho),
    block_size(block_size), 
    M(hidden_states.size() - 1), 
    csfs_d(n, M), csfs_ad(n, M)
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
        Eigen::Matrix<int, Eigen::Dynamic, 4> obs = 
            Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>::Map(observations[i], ell, 4);
        int max_n = obs.middleCols(1, 2).rowwise().sum().maxCoeff();
        if (max_n > n + 2 - 1)
            throw std::runtime_error("An observation has derived allele count greater than n + 1");
        PROGRESS("creating HMM");
        hmmptr h(new HMM(obs, n, block_size, &pi, &transition, mask_freq, this));
        hmms[i] = std::move(h);
    }
    // Collect all the block keys for recomputation later
    populate_block_prob_map();
}

void InferenceManager::populate_block_prob_map()
{
    Vector<adouble> tmp;
    mpz_class coef;
    for (auto &p : block_prob_map)
    {
        block_key key = p.first;
        bpm_keys.push_back(key);
        if (key.alt_block)
            nbs.insert(key.powers.begin()->first.nb);
    }
}

Matrix<double>& InferenceManager::subEmissionCoefs(int m)
{
    if (subEmissionCoefs_memo.count(m) == 0)
    {
        // FIXME: this direct sum representation is pretty wasteful
        Matrix<double> M(3 * (n + 1), 3 * (m + 1));
        M.setZero();
        for (int i = 0; i < n + 1; ++i)
            for (int j = 0; j < m + 1; ++j)
                M(i, j) = gsl_ran_hypergeometric_pdf(j, i, n - i, m);
        M.block(n + 1, m + 1, n + 1, m + 1) = M.block(0, 0, n + 1, m + 1);
        M.block(2 * (n + 1), 2 * (m + 1), n + 1, m + 1) = M.block(0, 0, n + 1, m + 1);
        subEmissionCoefs_memo[m] = M;
    }
    return subEmissionCoefs_memo[m];
}

template <typename T>
void InferenceManager::recompute_B(const PiecewiseExponentialRateFunction<T> &eta)
{
    PROGRESS("recompute B");
    Vector<adouble> two_probs[2] = {Vector<adouble>::Zero(M), Vector<adouble>::Zero(M)};
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n + 1; ++j)
            two_probs[i % 2] += emission.col((n + 1) * i + j);
    std::map<int, Matrix<adouble> > subemissions;
    PROGRESS("subemissions");
    for (int nb : nbs)
    {
        subemissions[nb] = emission.lazyProduct(subEmissionCoefs(nb));
        subemissions[nb].col(0) += subemissions[nb].rightCols<1>();
        subemissions[nb].rightCols<1>().fill(0);
        PROGRESS(std::endl << nb << std::endl << subemissions[nb].template cast<double>() << std::endl << std::endl);
    }
#pragma omp parallel for
    for (auto it = bpm_keys.begin(); it < bpm_keys.end(); ++it)
    {
        Eigen::Array<adouble, Eigen::Dynamic, 1> tmp(M), log_tmp(M);
        log_tmp.setZero();
        Vector<adouble> ob = Vector<adouble>::Zero(M);
        if (it->alt_block)
        {
            // Full SFS emission
            if (it->powers.size() != 1 or it->powers.begin()->second != 1)
                throw std::runtime_error("Alt blocks >1 not supported.");
            block_power bp = it->powers.begin()->first;
            Matrix<adouble> emission_nb = subemissions.at(bp.nb);
            if (bp.a == -1)
            {
                if (bp.nb == 0)
                    tmp.fill(eta.one);
                else    
                    tmp = (emission_nb.col(bp.b) + 
                            emission_nb.col((bp.nb + 1) + bp.b) + 
                            emission_nb.col(2 * (bp.nb + 1) + bp.b)
                          );
            }
            else
                tmp = emission_nb.col(bp.a * (bp.nb + 1) + bp.b);
        }
        else
        {
            // Reduced SFS emission
            std::vector<int> nobs(2, 0);
            for (auto &p : it->powers)
                if (p.first.a != -1)
                {
                    nobs[p.first.a % 2] += p.second;
                    log_tmp += two_probs[p.first.a % 2].array().log() * p.second;
                }
            log_tmp += log(multinomial(nobs).get_ui());
            tmp = exp(log_tmp);
        }
        if (tmp.maxCoeff() > 1.0 or tmp.minCoeff() <= 0.0)
        {
            std::cout << it->powers << std::endl;
            std::cout << tmp.template cast<double>().transpose() << std::endl;
            std::cout << tmp.maxCoeff() << std::endl;
            throw std::runtime_error("probability vector not in [0, 1]");
        }
        check_nan(tmp);
        block_prob_map.at(*it) = tmp.matrix();
    }
    PROGRESS_DONE();
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

std::vector<double> InferenceManager::randomCoalTimes(const ParameterVector params, double fac, const int size)
{
    PiecewiseExponentialRateFunction<double> eta(params, {}, hidden_states);
    std::vector<double> ret(size);
    std::mt19937 gen;
    for (int i = 0; i < size; ++i)
        ret[i] = eta.random_time(fac, 0., .99 * T_MAX, gen);
    return ret;
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
    transition = compute_transition<T>(eta, rho).template cast<adouble>();
    check_nan(transition);
    std::vector<Matrix<T> > sfss = sfs<T>(eta);
    Eigen::Matrix<T, 3, Eigen::Dynamic, Eigen::RowMajor> em_tmp(3, n + 1);
    for (int m = 0; m < M; ++m)
    {
        check_nan(sfss[m]);
        em_tmp = sfss[m];
        emission.row(m) = Matrix<T>::Map(em_tmp.data(), 1, 3 * (n + 1)).template cast<adouble>();
    }
    recompute_B(eta);
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

std::vector<Matrix<fbType>*> InferenceManager::getGammas()
{
    std::vector<Matrix<fbType>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm->gamma);
    }
    return ret;
}

std::pair<std::vector<Matrix<fbType>* >, std::vector<Matrix<fbType>* > > InferenceManager::getXisums()
{
    std::pair<std::vector<Matrix<fbType>* >, std::vector<Matrix<fbType>* > > ret;
    for (auto &hmm : hmms)
    {
        ret.first.push_back(&hmm->xisum);
        ret.second.push_back(&hmm->xisum_alt);
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

std::vector<block_key_vector> InferenceManager::getBlockKeys()
{
    std::vector<block_key_vector> ret;
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

std::vector<double> InferenceManager::loglik(double lambda)
{
    double reg = toDouble(regularizer);
    return parallel_select<double>([lambda, reg] (hmmptr &hmm) { return hmm->loglik() - lambda * reg; });
}

void InferenceManager::setParams_d(const ParameterVector params) 
{ 
    std::vector<std::pair<int, int>> d;
    setParams<double>(params, d);
}

void InferenceManager::setParams_ad(const ParameterVector params, 
        const std::vector<std::pair<int, int>> derivatives) 
{  
    setParams<adouble>(params, derivatives);
}

double InferenceManager::R(const ParameterVector params, double t)
{
    PiecewiseExponentialRateFunction<double> eta(params, std::vector<std::pair<int, int> >(), std::vector<double>());
    return (*eta.getR())(t);
}

double InferenceManager::getRegularizer() { return toDouble(regularizer); }

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
