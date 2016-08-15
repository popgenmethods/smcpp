#include "inference_manager.h"
#include "transition.h"
#include "marginalize_sfs.h"
#include "jcsfs.h"

PiecewiseConstantRateFunction<adouble>* defaultEta(const std::vector<double> &hidden_states)
{
    std::vector<std::vector<adouble> > params;
    std::vector<adouble> p{adouble(1.0)};
    params.push_back(p);
    params.push_back(p);
    return new PiecewiseConstantRateFunction<adouble>(params, hidden_states);
}

InferenceManager::InferenceManager(
        const int npop,
        const int sfs_dim,
        const std::vector<int> obs_lengths,
        const std::vector<int*> observations,
        const std::vector<double> hidden_states,
        ConditionedSFS<adouble> *csfs) :
    saveGamma(false), folded(false),
    hidden_states(hidden_states),
    npop(npop),
    sfs_dim(sfs_dim),
    M(hidden_states.size() - 1),
    obs(map_obs(observations, obs_lengths)),
    csfs(csfs),
    hmms(obs.size()),
    pi(M),
    targets(fill_targets()),
    tb(targets, &emission_probs),
    ib{&pi, &tb, &emission_probs, &saveGamma},
    dirty({true, true, true}),
    eta(defaultEta(hidden_states))
{
    if (*std::min_element(hidden_states.begin(), hidden_states.end()) != 0.)
        throw std::runtime_error("first hidden interval should be [0, <something>)");
    pi = Vector<adouble>::Zero(M);
    transition = Matrix<adouble>::Zero(M, M);
    transition.setZero();
    InferenceBundle *ibp = &ib;
#pragma omp parallel for
    for (unsigned int i = 0; i < obs.size(); ++i)
    {
        DEBUG << "creating HMM";
        hmms[i].reset(new HMM(i, this->obs[i], ibp));
    }

    // Collect all the block keys for recomputation later
    populate_emission_probs();
}

void InferenceManager::recompute_initial_distribution()
{
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-(eta->R(hidden_states[m]))) - exp(-(eta->R(hidden_states[m + 1])));
        assert(pi(m) >= 0.0);
        assert(pi(m) <= 1.0);
    }
    pi(M - 1) = exp(-(eta->R(hidden_states[M - 1])));
    adouble small = eta->zero() + 1e-20;
    pi = pi.unaryExpr([small] (const adouble &x) { if (x < 1e-20) return small; return x; });
    CHECK_NAN(pi);
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
}

template <typename T>
std::vector<T> InferenceManager::parallel_select(std::function<T(hmmptr &)> lambda)
{
    std::vector<T> ret(hmms.size());
#pragma omp parallel for
    for (unsigned int i = 0; i < hmms.size(); ++i)
        ret[i] = lambda(hmms[i]);
    return ret;
}
template std::vector<double> InferenceManager::parallel_select(std::function<double(hmmptr &)>);
template std::vector<adouble> InferenceManager::parallel_select(std::function<adouble(hmmptr &)>);

void InferenceManager::Estep(bool fbonly)
{
    DEBUG << "E step";
    do_dirty_work();
    parallel_do([fbonly] (hmmptr &hmm) { hmm->Estep(fbonly); });
}

std::vector<adouble> InferenceManager::Q(void)
{
    DEBUG << "InferenceManager::Q";
    do_dirty_work();
    std::vector<Vector<adouble> > ps = parallel_select<Vector<adouble> >([] (hmmptr &hmm) { return hmm->Q(); });
    adouble q1 = 0, q2 = 0, q3 = 0;
    for (unsigned int i = 0; i < ps.size(); ++i)
    {
        q1 += ps[i][0];
        q2 += ps[i][1];
        q3 += ps[i][2];
    }
    DEBUG << "\nq1:" << q1.value() << " [" << q1.derivatives().transpose() << "]\nq2:"
        << q2.value() << " [" << q2.derivatives().transpose() << "]\nq3:" << q3.value()
        << " [" << q3.derivatives().transpose() << "]\n";
    return {q1, q2, q3};
}

std::vector<std::map<block_key, Vector<double> >*> InferenceManager::getGammaSums()
{
    std::vector<std::map<block_key, Vector<double> >*> ret;
    for (auto &hmm : hmms)
        ret.push_back(&hmm->gamma_sums);
    return ret;
}

std::vector<Matrix<double>*> InferenceManager::getGammas()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
        ret.push_back(&hmm->gamma);
    return ret;
}

std::vector<Matrix<double>*> InferenceManager::getXisums()
{
    std::vector<Matrix<double>*> ret;
    for (auto &hmm : hmms)
        ret.push_back(&hmm->xisum);
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

// Begin stuff for NPop inference manager
std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > 
    InferenceManager::map_obs(const std::vector<int*> &observations, const std::vector<int> &obs_lengths)
{
    std::vector<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > ret;
    for (unsigned int i = 0; i < observations.size(); ++i)
        ret.push_back(Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(
                    observations[i], obs_lengths[i], 1 + 3 * npop));
    return ret;
}

void InferenceManager::populate_emission_probs()
{
    Vector<adouble> tmp;
    for (auto ob : obs)
    {
        const int q = ob.cols() - 1;
        for (int i = 0; i < ob.rows(); ++i)
        {
            block_key key(ob.row(i).tail(q).transpose());
            if (emission_probs.count(key) == 0)
            {
                emission_probs.insert({key, tmp});
                bpm_keys.push_back(key);
            }
        }
    }
}

void InferenceManager::do_dirty_work()
{
    // Figure out what changed and recompute accordingly.
    if (dirty.eta)
    {
        recompute_initial_distribution();
        sfss = csfs->compute(*eta);
    }
    if (dirty.theta or dirty.eta)
        recompute_emission_probs();
    if (dirty.eta or dirty.rho)
        transition = compute_transition(*eta, rho);
    if (dirty.theta or dirty.eta or dirty.rho)
        tb.update(transition);
    // restore pristine status
    dirty = {false, false, false};
}

std::set<std::pair<int, block_key> > InferenceManager::fill_targets()
{
    std::set<std::pair<int, block_key> > ret;
    for (auto ob : obs)
    {
        const int q = ob.cols() - 1;
        for (int i = 0; i < ob.rows(); ++i)
            if (ob(i, 0) > 1)
                ret.insert({ob(i, 0), block_key(ob.row(i).tail(q).transpose())});
    }
    return ret;
}

void InferenceManager::setParams(const ParameterVector &params)
{
    eta.reset(new PiecewiseConstantRateFunction<adouble>(params, hidden_states));
    dirty.eta = true;
}

template <size_t P>
void NPopInferenceManager<P>::recompute_emission_probs()
{
    // Initialize emission matrix
    // Due to lack of good support for tensors, we store the emission
    // tensor in "flattened" matrix form. Note that this is actually
    // 1 larger along each axis than the true number, because the SFS
    // ranges in {0, 1, ..., n_pop_k}.
    emission = Matrix<adouble>::Zero(M, (na(0) + 1) * sfs_dim);
    emission.setZero();

    marginalize_sfs_a<P> ma;
    Eigen::Matrix<adouble, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> em_tmp(na(0) + 1, sfs_dim);
    std::vector<Matrix<adouble> > new_sfss = incorporate_theta(sfss, theta);
    for (int m = 0; m < M; ++m)
    {
        CHECK_NAN(new_sfss[m]);
        em_tmp = new_sfss[m];
        emission.row(m) = Matrix<adouble>::Map(em_tmp.data(), 1, (na(0) + 1) * sfs_dim);
    }
    
    DEBUG << "recompute B";
    Matrix<adouble> e2 = Matrix<adouble>::Zero(M, 2);
    std::vector<adouble> avg_ct = eta->average_coal_times();
    adouble small = eta->zero() + 1e-20;
    for (int m = 0; m < M; ++m)
    {
        if (std::isnan(avg_ct[m].value()))
        {
            // if the two lineages are separated by a split, their
            // average coalescence time within each interval before
            // (more recently than) the split is undefined. in this
            // case, assign very low probabilities to all such
            // observations.
            e2(m, 0) = small;
            e2(m, 1) = small;
        }
        else
        {
            e2(m, 1) = 2. * theta * avg_ct[m];
            e2(m, 0) = 1. - e2(m, 1);
        }
        // CHECK_NAN(e2(m, 1));
    }
    const adouble zero = eta->zero();
    const adouble one = zero + 1.;
#pragma omp parallel for
    for (auto it = bpm_keys.begin(); it < bpm_keys.end(); ++it)
    {
        // std::set<block_key> keys;
        // keys.insert(key);
        /*
        if (this->folded)
        {
            Vector<int> new_key(it->size());
            for (size_t p = 0; p < P; ++p)
            {
                int a = key(3 * p);
                int b = key(3 * p + 1);
                int nb = key(3 * p + 2);
                new_key(1 + 2 * p) = nb - b;
                new_key(2 + 2 * p) = nb;
            }
            keys.emplace(new_key);
        }
        */
        block_key k = *it;
        Vector<adouble> tmp(M);
        tmp.fill(zero);
        bool reduced = true;
        FixedVector<int, P> a, b, nb;
        for (unsigned int p = 0; p < P; ++p)
        {
            a(p) = k(3 * p);
            b(p) = k(1 + 3 * p);
            nb(p) = k(2 + 3 * p);
            reduced &= nb(p) == 0;
        }
        if (reduced and (a.isConstant(-1) or (a.minCoeff() >= 0)))
        {
            if (a.isConstant(-1))
                tmp.fill(one);
            else // if (a.minCoeff() >= 0)
                tmp = e2.col(a.sum() % 2);
        }
        else
            tmp = ma(emission, n, a, na, b, nb);
        if (tmp.maxCoeff() > 1.0 or tmp.minCoeff() <= 0.0)
        {
            std::cout << k << std::endl;
            std::cout << tmp.template cast<double>().transpose() << std::endl;
            std::cout << tmp.maxCoeff() << std::endl;
            throw std::runtime_error("probability vector not in [0, 1]");
        }
        CHECK_NAN(tmp);
        this->emission_probs.at(k) = tmp;
    }
    DEBUG << "recompute done";
}

Matrix<adouble> sfs_cython(const int n, const ParameterVector p, 
        const double t1, const double t2, bool below_only)
{
    std::vector<double> hs{t1, t2};
    OnePopConditionedSFS<adouble> csfs(n);
    std::vector<Matrix<adouble> > v;
    PiecewiseConstantRateFunction<adouble> eta(p, hs);
    if (below_only)
        v = csfs.compute_below(eta);
    else
        v = csfs.compute(eta);
    return v[0];
}

OnePopInferenceManager::OnePopInferenceManager(
            const int n,
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states) :
        NPopInferenceManager(
                FixedVector<int, 1>::Constant(n),
                FixedVector<int, 1>::Constant(2),
                obs_lengths, observations, hidden_states, 
                new OnePopConditionedSFS<adouble>(n)) {}

TwoPopInferenceManager::TwoPopInferenceManager(
            const int n1, const int n2,
            const int a1, const int a2,
            const std::vector<int> obs_lengths,
            const std::vector<int*> observations,
            const std::vector<double> hidden_states) :
        NPopInferenceManager(
                (FixedVector<int, 2>() << n1, n2).finished(),
                (FixedVector<int, 2>() << a1, a2).finished(),
                obs_lengths, observations, hidden_states, 
                new JointCSFS<adouble>(n1, n2, a1, a2, hidden_states)),
        a1(a1), a2(a2)
{
    if (not ((a1 == 2 and a2 == 0) or 
             (a1 == 1 and a2 == 1)))
        throw std::runtime_error("configuration not supported");
}

void TwoPopInferenceManager::setParams(const ParameterVector &params1, const ParameterVector &params2, const double split)
{
    if (a1 == 1 and a2 == 1)
    {
        ParameterVector paramsSplit = shiftParams(params1, split);
        // before split, prevent all coalescence
        adouble inf = paramsSplit[0][0];
        inf *= 0.;
        inf.value() = INFINITY;
        paramsSplit[0].emplace(paramsSplit[0].begin(), inf);
        paramsSplit[1].emplace(paramsSplit[1].begin(), split);
        InferenceManager::setParams(paramsSplit);
    }
    else
        InferenceManager::setParams(params1);
    dynamic_cast<JointCSFS<adouble>*>(csfs.get())->pre_compute(params1, params2, split);
}

template class NPopInferenceManager<1>;
template class NPopInferenceManager<2>;
