#include "gsl/gsl_randist.h"

#include "inference_manager.h"

InferenceManager::InferenceManager(
            const int n, const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states,
            const std::vector<double> s) :
    debug(false), saveGamma(false), folded(false),
    hidden_states(hidden_states), s(s),
    n(n), 
    obs(map_obs(observations, obs_lengths)),
    M(hidden_states.size() - 1), 
    csfs(n, M),
    pi(M),
    nbs(fill_nbs()),
    subEmissionCoeffs(fill_subemissions()),
    targets(fill_targets()),
    tb(targets, &emission_probs),
    spanCutoff(1),
    ib{&pi, &tb, &emission_probs, &saveGamma, &spanCutoff},
    dirty({true, true, true})
{
    if (*std::min_element(hidden_states.begin(), hidden_states.end()) != 0.)
        throw std::runtime_error("first hidden interval should be [0, <something>)");
    pi = Vector<adouble>::Zero(M);
    transition = Matrix<adouble>::Zero(M, M);
    transition.setZero();
    emission = Matrix<adouble>::Zero(M, 3 * (n + 1));
    emission.setZero();
    hmms.resize(obs.size());

#pragma omp parallel for
    for (unsigned int i = 0; i < obs.size(); ++i)
    {
        int max_n = obs[i].middleCols(1, 2).rowwise().sum().maxCoeff();
        if (max_n > n + 2 - 1)
            throw std::runtime_error("Dataset did not validate: an observation has derived allele count greater than n + 1");
        if (obs[i](0, 0) > 1)
            throw std::runtime_error("Dataset did not validate: first observation must have span=1");
        DEBUG << "creating HMM";
        hmmptr h(new HMM(i, obs[i], &ib));
        hmms[i] = std::move(h);
    }
    // Collect all the block keys for recomputation later
    populate_emission_probs();
}

std::vector<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> > InferenceManager::map_obs(
        const std::vector<int*> &observations, const std::vector<int> &obs_lengths
        )
{
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor> > ret;
    for (unsigned int i = 0; i < observations.size(); ++i)
        ret.push_back(Eigen::Matrix<int, Eigen::Dynamic, 4, Eigen::RowMajor>::Map(observations[i], obs_lengths[i], 4));
    return ret;
};

std::set<std::pair<int, block_key> > InferenceManager::fill_targets()
{
    std::set<std::pair<int, block_key> > ret;
    for (auto ob : obs)
        for (int i = 0; i < ob.rows(); ++i)
            if (ob(i, 0) > 1)
                ret.insert({ob(i, 0), {ob(i, 1), ob(i, 2), ob(i, 3)}});
    return ret;
}

std::vector<int> InferenceManager::fill_nbs()
{
    std::set<int> s;
    for (auto ob : obs)
        for (int i = 0; i < ob.rows(); ++i)
            s.insert(ob(i, 3));
    return std::vector<int>(s.begin(), s.end());
}

void InferenceManager::recompute_initial_distribution()
{
    const PiecewiseExponentialRateFunction<adouble> eta = getEta();
    auto R = eta.getR();
    int M = eta.hidden_states.size() - 1;
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-(*R)(hidden_states[m])) - exp(-(*R)(hidden_states[m + 1]));
        assert(pi(m) > 0.0); 
        assert(pi(m) < 1.0); 
    }
    pi(M - 1) = exp(-(*R)(hidden_states[M - 1]));
    pi = pi.unaryExpr([] (const adouble &x) { if (x < 1e-20) return adouble(1e-20); return x; });
    pi /= pi.sum();
    check_nan(pi);
}

void InferenceManager::populate_emission_probs()
{
    Vector<adouble> tmp;
    block_key key;
    for (auto ob : obs)
        for (int i = 0; i < ob.rows(); ++i)
        {
            key = {ob(i, 1), ob(i, 2), ob(i, 3)}; 
            if (emission_probs.count(key) == 0)
            {
                emission_probs.insert({key, tmp});
                bpm_keys.push_back(key);
            }
        }
}

std::map<int, Matrix<double> > InferenceManager::fill_subemissions()
{
    std::map<int, Matrix<double> > ret;
    for (int m : nbs)
    {
        // FIXME: this direct sum representation is pretty wasteful
        Matrix<double> M(3 * (n + 1), 3 * (m + 1));
        M.setZero();
        for (int i = 0; i < n + 1; ++i)
            for (int j = 0; j < m + 1; ++j)
                M(i, j) = gsl_ran_hypergeometric_pdf(j, i, n - i, m);
        M.block(n + 1, m + 1, n + 1, m + 1) = M.block(0, 0, n + 1, m + 1);
        M.block(2 * (n + 1), 2 * (m + 1), n + 1, m + 1) = M.block(0, 0, n + 1, m + 1);
        ret.insert({m, M});
    }
    return ret;
}

void InferenceManager::recompute_emission_probs()
{
    Eigen::Matrix<adouble, 3, Eigen::Dynamic, Eigen::RowMajor> em_tmp(3, n + 1);
    std::vector<Matrix<adouble> > new_sfss = incorporate_theta(sfss, theta);
    for (int m = 0; m < M; ++m)
    {
        check_nan(new_sfss[m]);
        em_tmp = new_sfss[m];
        emission.row(m) = Matrix<adouble>::Map(em_tmp.data(), 1, 3 * (n + 1));
    }
    DEBUG << "recompute B";
    std::map<int, Matrix<adouble> > subemissions;
    DEBUG << "subemissions";
#pragma omp parallel for
    for (auto it = nbs.begin(); it < nbs.end(); ++it)
    {
        int nb = *it;
        Matrix<adouble> M = emission.lazyProduct(subEmissionCoeffs.at(nb));
        M.col(0) += M.rightCols(1);
        M.rightCols(1).fill(0.);
#pragma omp critical(subemissions)
        {
            subemissions[nb] = M;
        }
    }
    // subemissions[0] is computed more easily / accurately by direct method
    // than by marginalizing. (in particular, the derivatives)
    // std::cerr << "old sub[0]\n" << subemissions[0].template cast<double>() << std::endl;
    DEBUG << "done subemissions";
    subemissions[0] = Matrix<adouble>::Zero(M, 2);
    PiecewiseExponentialRateFunction<adouble> eta = getEta();
    std::vector<adouble> avg_ct = eta.average_coal_times();
    for (int m = 0; m < M; ++m)
    {
        subemissions[0](m, 1) = avg_ct[m];
        check_nan(subemissions[0](m, 1));
    }
    subemissions[0].col(1) *= 2. * theta;
    subemissions[0].col(0).fill(eta.one);
    subemissions[0].col(0) -= subemissions[0].col(1);
    // std::cerr << "new sub[0]\n" << subemissions[0].template cast<double>() << std::endl;
    DEBUG << "emission keys";
#pragma omp parallel for
    for (auto it = bpm_keys.begin(); it < bpm_keys.end(); ++it)
    {
        int a = (*it)[0], b = (*it)[1], nb = (*it)[2];
        std::set<std::array<int, 2> > ab {{a, b}};
        if (folded and nb > 0)
            ab.insert({a == -1 ? -1 : 2 - a, nb - b});
        Vector<adouble> tmp(M);
        tmp.fill(eta.zero);
        Matrix<adouble> emission_nb = subemissions.at(nb);
        for (std::array<int, 2> tup : ab)
        {
            a = tup[0]; b = tup[1];
            if (nb > 0)
            {
                if (a == -1)
                    tmp += (emission_nb.col(b) + 
                            emission_nb.col((nb + 1) + b) + 
                            emission_nb.col(2 * (nb + 1) + b));
                else
                    tmp += emission_nb.col(a * (nb + 1) + b);
            }
            else
            {
                if (a == -1)
                    tmp.fill(eta.one);
                else
                    tmp += emission_nb.col(a); // nb = 0 => b = 0 => a = a(nb + 1) + b
            }
        }
        if (tmp.maxCoeff() > 1.0 or tmp.minCoeff() <= 0.0)
        {
            std::cout << *it << std::endl;
            std::cout << tmp.template cast<double>().transpose() << std::endl;
            std::cout << tmp.maxCoeff() << std::endl;
            throw std::runtime_error("probability vector not in [0, 1]");
        }
        check_nan(tmp);
        emission_probs.at(*it) = tmp;
    }
    DEBUG << "recompute done";
}

PiecewiseExponentialRateFunction<adouble> InferenceManager::getEta()
{
    return PiecewiseExponentialRateFunction<adouble>(params, s, hidden_states);
}

void InferenceManager::setParams(const ParameterVector params)
{
    for (auto &pp : params)
        if (pp.size() != params[0].size())
            throw std::runtime_error("params must have matching sizes");
    this->params = params;
    dirty.params = true;
}

void InferenceManager::setRho(const double rho)
{
    this->rho = rho;
    dirty.rho = true;
}

void InferenceManager::setTheta(const double theta)
{
    this->theta = theta;
    dirty.theta = true;
}

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
    for (unsigned int i = 0; i < hmms.size(); ++i)
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

void InferenceManager::Estep(bool fbonly)
{
    DEBUG << "E step";
    do_dirty_work();
    parallel_do([fbonly] (hmmptr &hmm) { hmm->Estep(fbonly); });
}

void InferenceManager::do_dirty_work()
{
    // First set derivatives correctly. PiecewiseExponentialRateFunction
    // will already do this for all the params.
    PiecewiseExponentialRateFunction<adouble> eta = getEta();
    // Next figure out what changed and recompute accordingly.
    if (dirty.params)
    {
        recompute_initial_distribution();
        sfss = csfs.compute(eta);
    }
    if (dirty.theta or dirty.params)
        recompute_emission_probs();
    if (dirty.params or dirty.rho)
        transition = compute_transition(eta, rho);
    if (dirty.theta or dirty.params or dirty.rho)
        tb.update(transition);
    // restore pristine status
    dirty = {false, false, false};
}

std::vector<adouble> InferenceManager::Q(void)
{
    DEBUG << "InferenceManager::Q";
    do_dirty_work();
    std::vector<Vector<adouble> > ps = parallel_select<Vector<adouble> >([] (hmmptr &hmm) { return hmm->Q(); });
    adouble q1 = 0, q2 = 0, q3 = 0;
    for (int i = 0; i < ps.size(); ++i)
    {
        q1 += ps[i][0];
        q2 += ps[i][1];
        q3 += ps[i][2];
    }
    DEBUG1 << "\nq1:" << q1.value() << " [" << q1.derivatives().transpose() << "]\nq2:" 
            << q2.value() << " [" << q2.derivatives().transpose() << "]\nq3:" << q3.value()
            << " [" << q3.derivatives().transpose() << "]\n";
    return {q1, q2, q3};
}

std::vector<std::map<block_key, Vector<double> >*> InferenceManager::getGammaSums()
{
    std::vector<std::map<block_key, Vector<double> >*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm->gamma_sums);
    }
    return ret;
}

std::vector<Matrix<double>*> InferenceManager::getGammas()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm->gamma);
    }
    return ret;
}

std::vector<Matrix<double>*> InferenceManager::getXisums()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
    {
        ret.push_back(&hmm->xisum);
    }
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

std::map<block_key, Vector<adouble> >& InferenceManager::getEmissionProbs()
{
    return emission_probs;
}

std::vector<double> InferenceManager::loglik(void)
{
    return parallel_select<double>([] (hmmptr &hmm) { return hmm->loglik(); });
}

double InferenceManager::R(const ParameterVector params, double t)
{
    PiecewiseExponentialRateFunction<double> eta(params, s, std::vector<double>());
    return (*eta.getR())(t);
}

Matrix<adouble> sfs_cython(int n, const ParameterVector &p, std::vector<double> s, 
        double t1, double t2)
{ 
    PiecewiseExponentialRateFunction<adouble> eta(p, s, {t1, t2});
    ConditionedSFS<adouble> csfs(n - 2, 1);
    std::vector<Matrix<adouble> > v = csfs.compute(eta);
    return v[0];
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
