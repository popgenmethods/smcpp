#include "inference_manager.h"

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
        const ConditionedSFS<adouble> *csfs) :
    saveGamma(false), folded(false),
    npop(npop),
    sfs_dim(sfs_dim),
    hidden_states(hidden_states),
    obs(map_obs(observations, obs_lengths)),
    M(hidden_states.size() - 1),
    csfs(csfs),
    pi(M),
    targets(fill_targets()),
    tb(targets, &emission_probs),
    ib{&pi, &tb, &emission_probs, &saveGamma},
    dirty({true, true, true}),
    eta(defaultEta(hidden_states)),
    tp(ThreadPool::getInstance())
{
    if (*std::min_element(hidden_states.begin(), hidden_states.end()) != 0.)
        throw std::runtime_error("first hidden interval should be [0, <something>)");
    pi = Vector<adouble>::Zero(M);
    transition = Matrix<adouble>::Zero(M, M);
    transition.setZero();
    // Due to lack of good support for tensors, we store the emission
    // tensor in "flattened" matrix form. Note that this is actually
    // 1 larger along each axis than the true number, because the SFS
    // ranges in {0, 1, ..., n_pop_k}.
    emission = Matrix<adouble>::Zero(M, 3 * sfs_dim);
    emission.setZero();
    hmms.resize(obs.size());

    ThreadPool &tp = ThreadPool::getInstance();
    InferenceBundle *ibp = &ib;
    for (unsigned int i = 0; i < obs.size(); ++i)
    {
        // Move all validation to Python
        /*
           int max_n = obs[i].middleCols(1, 2).rowwise().sum().maxCoeff();
           if (max_n > n + 2 - 1)
           throw std::runtime_error("Dataset did not validate: an observation has derived allele count greater than n + 1");
           if (obs[i](0, 0) > 1)
           throw std::runtime_error("Dataset did not validate: first observation must have span=1");
           */
        tp.enqueue([i, ibp, this] 
        {
            DEBUG << "creating HMM";
            hmmptr h(new HMM(i, this->obs[i], ibp));
            this->hmms[i] = std::move(h);
        });
    }
    // Collect all the block keys for recomputation later
    populate_emission_probs();
}

void InferenceManager::recompute_initial_distribution()
{
    int M = eta->hidden_states.size() - 1;
    for (int m = 0; m < M - 1; ++m)
    {
        pi(m) = exp(-(eta->R(hidden_states[m]))) - exp(-(eta->R(hidden_states[m + 1])));
        assert(pi(m) > 0.0);
        assert(pi(m) < 1.0);
    }
    pi(M - 1) = exp(-(eta->R(hidden_states[M - 1])));
    pi = pi.unaryExpr([] (const adouble &x) { if (x < 1e-20) return adouble(1e-20); return x; });
    pi /= pi.sum();
    check_nan(pi);
}

/*
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
*/


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
    std::vector<std::future<void>> results;
    for (auto &hmmptr : hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, std::ref(hmmptr))));
    for (auto &res : results)
        res.wait();
}

template <typename T>
std::vector<T> InferenceManager::parallel_select(std::function<T(hmmptr &)> lambda)
{
    std::vector<std::future<T>> results;
    for (auto &hmmptr : hmms)
        results.emplace_back(tp.enqueue(std::bind(lambda, std::ref(hmmptr))));
    std::vector<T> ret;
    for (auto &res : results)
        ret.push_back(res.get());
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
                    observations[i], obs_lengths[i], 2 + 2 * npop));
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
    Eigen::Matrix<adouble, 3, Eigen::Dynamic, Eigen::RowMajor> em_tmp(3, sfs_dim);
    std::vector<Matrix<adouble> > new_sfss = incorporate_theta(sfss, theta);
    for (int m = 0; m < M; ++m)
    {
        check_nan(new_sfss[m]);
        em_tmp = new_sfss[m];
        emission.row(m) = Matrix<adouble>::Map(em_tmp.data(), 1, 3 * sfs_dim);
    }
    
    DEBUG << "recompute B";
    Matrix<adouble> e2 = Matrix<adouble>::Zero(M, 2);
    std::vector<adouble> avg_ct = eta->average_coal_times();
    for (int m = 0; m < M; ++m)
    {
        e2(m, 1) = avg_ct[m];
        check_nan(e2(m, 1));
    }
    e2.col(1) *= 2. * theta;
    e2.col(0).setOnes();
    e2.col(0) -= e2.col(1);
    std::vector<std::future<void> > results;
    for (auto it = bpm_keys.begin(); it < bpm_keys.end(); ++it)
        results.emplace_back(tp.enqueue([this, it, e2]
        {
            block_key key = *it;
            std::set<block_key> keys;
            keys.insert(key);
            if (this->folded)
            {
                Vector<int> new_key(it->size());
                new_key(0) = (key(0) == -1) ? -1 : 2 - key(0);
                for (size_t p = 0; p < P; ++p)
                {
                    int b = key(1 + 2 * p), nb = key(2 + 2 * p);
                    new_key(1 + 2 * p) = nb - b;
                    new_key(2 + 2 * p) = nb;
                }
                keys.emplace(new_key);
            }
            Vector<adouble> tmp(M);
            tmp.setZero();
            for (block_key k : keys)
            {
                int a = k(0);
                bool reduced = true;
                FixedVector<int, P> b, nb;
                for (int p = 0; p < P; ++p)
                {
                    b(p) = k(1 + 2 * p);
                    nb(p) = k(2 + 2 * p);
                    reduced &= nb(p) == 0;
                }
                if (reduced)
                {
                    if (a == -1)
                        tmp.setOnes();
                    else
                        tmp = e2.col(a % 2);
                }
                else
                {
                    if (a == -1)
                        for (int i = 0; i < 3; ++i)
                            tmp += marginalize_sfs<P>()(emission.middleCols(i * sfs_dim, sfs_dim), n, b, nb);
                    else
                        tmp = marginalize_sfs<P>()(emission.middleCols(a * sfs_dim, sfs_dim), n, b, nb);
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
            this->emission_probs.at(*it) = tmp;
        }));
    for (auto &&result : results) 
        result.wait();
    DEBUG << "recompute done";
}

Matrix<adouble> sfs_cython(const int n, const ParameterVector p, 
        const double t1, const double t2, bool below_only)
{
    std::vector<double> hs{t1, t2};
    OnePopConditionedSFS<adouble> csfs(n - 2, 1);
    std::vector<Matrix<adouble> > v;
    PiecewiseConstantRateFunction<adouble> eta(p, hs);
    if (below_only)
        v = csfs.compute_below(eta);
    else
        v = csfs.compute(eta);
    return v[0];
}

template class NPopInferenceManager<1>;
template class NPopInferenceManager<2>;
