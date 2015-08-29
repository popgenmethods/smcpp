#include "hmm.h"

long num_blocks(int total_loci, int block_size, int mask_freq, int mask_offset)
{
    long denom = (mask_freq - 1) * block_size + 1;
    long base = total_loci / denom;
    base *= mask_freq;
    long remain = total_loci % denom;
    bool first = true;
    while (remain > 0)
    {
        if (first)
        {
            first = false;
            remain--;
        }
        else
            remain -= block_size;
        base++;
    }
    return base;
}

HMM::HMM(Eigen::Matrix<int, Eigen::Dynamic, 2> obs, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission, const Matrix<adouble> *emission_mask,
        const int mask_freq, const int mask_offset) : 
    obs(obs), block_size(block_size), 
    pi(pi), transition(transition), emission(emission), emission_mask(emission_mask),
    mask_freq(mask_freq), mask_offset(mask_offset),
    M(pi->rows()), Ltot(num_blocks(obs.col(0).sum(), block_size, mask_freq, mask_offset)),
    Bptr(Ltot), logBptr(Ltot), alpha_hat(M, Ltot), beta_hat(M, Ltot), gamma(M, Ltot), xisum(M, M), c(Ltot) 
{ 
    prepare_B();
}

void HMM::prepare_B()
{
    PROGRESS("preparing B");
    std::map<int, int> powers;
    Vector<adouble> tmp(M);
    const Matrix<adouble> *em_ptr;
    bool alt_block, alt_block_next;
    int current_block_size = 1;
    unsigned long int R, ob, i = 0, block = 0, tobs = 0;
    std::pair<bool, std::map<int, int>> key;
    unsigned long int L = obs.col(0).sum();
    for (unsigned long int ell = 0; ell < obs.rows(); ++ell)
    {
        R = obs(ell, 0);
        ob = obs(ell, 1);
        for (int r = 0; r < R; ++r)
        {
            powers[ob]++;
            if (tobs > L)
                throw std::domain_error("what?");
            tobs++;
            if (++i == current_block_size or (r == R - 1 and ell == obs.rows() - 1))
            {
                i = 0;
                alt_block = (block + mask_offset) % mask_freq == 0;
                alt_block_next = (block + mask_offset) % mask_freq == mask_freq - 1;
                em_ptr = alt_block ? emission : emission_mask;
                key = {alt_block, powers};
                if (block_prob_map.count(key) == 0)
                {
                    tmp.setOnes();
                    for (auto &p : powers)
                        tmp = tmp.cwiseProduct(em_ptr->col(p.first).array().pow(p.second).matrix());
                    block_prob_map[key] = {tmp, tmp.array().log()};
                }
                Bptr[block] = &block_prob_map[key].first;
                logBptr[block++] = &block_prob_map[key].second;
                block_prob_counts[key]++;
                current_block_size = (alt_block_next) ? 1 : block_size;
                powers.clear();
            }
        }
    }
    for (auto &p : block_prob_map)
        reverse_map[&block_prob_map[p.first].second] = p.first;
    PROGRESS_DONE();
}

double HMM::loglik()
{
    double ret = c.array().log().sum();
    domain_error(ret);
    return ret;
}

void HMM::domain_error(double ret)
{
    if (std::isinf(ret) or std::isnan(ret))
    {
        std::cout << pi->template cast<double>() << std::endl << std::endl;
        std::cout << transition->template cast<double>() << std::endl << std::endl;
        std::cout << emission->template cast<double>() << std::endl << std::endl;
        throw std::domain_error("badness encountered");
    }
}

/*
template <typename T>
std::vector<int>& HMM<T>::viterbi(void)
{
    // Compute logs of transition and emission matrices 
    // Do not bother to record derivatives since we don't use them for Viterbi algorithm
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    Eigen::MatrixXd log_transition = transition.template cast<double>().array().log();
    Eigen::MatrixXd log_emission = emission.template cast<double>().array().log();
    std::vector<double> V(M), V1(M);
    std::vector<std::vector<int>> path(M), newpath(M);
    Eigen::MatrixXd pid = pi.template cast<double>();
    std::vector<int> zeros(M, 0);
    double p, p2, lemit;
    int st;
    for (int m = 0; m < M; ++m)
    {
        V[m] = log(pid(m)) + log_emission(m, emission_index(n, obs(0,1), obs(0,2)));
        path[m] = zeros;
        path[m][m]++;
    }
    for (int ell = 0; ell < L; ++ell)
    {
        int R = obs(ell, 0);
        if (ell == 0)
            R--;
        for (int r = 0; r < R; ++r)
        {
            for (int m = 0; m < M; ++m)
            {
                lemit = log_emission(m, emission_index(n, obs(ell, 1), obs(ell, 2)));
                p = -INFINITY;
                st = 0;
                for (int m2 = 0; m2 < M; ++m2)
                {
                    p2 = V[m2] + log_transition(m2, m) + lemit;
                    if (p2 > p)
                    {
                        p = p2;
                        st = m2;
                    }
                }
                V1[m] = p;
                newpath[m] = path[st];
                newpath[m][m]++;
            }
            path = newpath;
            V = V1;
        }
    }
    p = -INFINITY;
    st = 0;
    for (int m = 0; m < M; ++m)
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
*/

void HMM::recompute_B(void)
{
    PROGRESS("recompute B");
    const Matrix<adouble> *em_ptr;
    Vector<adouble> tmp(M);
    bool alt_block;
    double mult = 1.0;
    for (auto &bp_pair : block_prob_map)
    {
        alt_block = bp_pair.first.first;
        em_ptr = alt_block ? emission : emission_mask;
        std::map<int, int> power = bp_pair.first.second;
        tmp.setOnes();
        // mult = alt_block ? 1000.0 : 1.0;
        for (auto &pow : power)
            tmp = tmp.cwiseProduct(em_ptr->col(pow.first).array().pow(pow.second).matrix());
        block_prob_map[bp_pair.first] = {tmp, mult * tmp.array().log()};
    }
    PROGRESS_DONE();
}

void HMM::forward_backward(void)
{
    PROGRESS("forward backward");
    Matrix<double> tt = transition->template cast<double>();
    Matrix<double> ttpow = tt.pow(block_size);
    alpha_hat.col(0) = pi->template cast<double>().cwiseProduct(Bptr[0]->template cast<double>());
	c(0) = alpha_hat.col(0).sum();
    alpha_hat.col(0) /= c(0);
    for (int ell = 1; ell < Ltot; ++ell)
    {
        // alpha_hat.col(ell) = bt.col(ell).asDiagonal() * (((ell + mask_offset) % mask_freq == 0) ? tt : ttpow) * alpha_hat.col(ell - 1);
        alpha_hat.col(ell) = Bptr[ell]->template cast<double>().asDiagonal() * 
            (((ell + mask_offset) % mask_freq == 0) ? tt : ttpow).transpose() * alpha_hat.col(ell - 1);
        c(ell) = alpha_hat.col(ell).sum();
        if (std::isnan(toDouble(c(ell))))
            throw std::domain_error("something went wrong in forward algorithm");
        alpha_hat.col(ell) /= c(ell);
    }
    beta_hat.col(Ltot - 1) = Vector<double>::Ones(M);
    for (int ell = Ltot - 2; ell >= 0; --ell)
        beta_hat.col(ell) = (((ell + 1 + mask_offset) % mask_freq == 0) ? tt : ttpow) * 
            Bptr[ell + 1]->template cast<double>().asDiagonal() * beta_hat.col(ell + 1) / c(ell + 1);
    PROGRESS_DONE();
}


void HMM::Estep(void)
{
    PROGRESS("E step");
    forward_backward();
	gamma = alpha_hat.cwiseProduct(beta_hat);
    Matrix<double> gs = gamma.colwise().sum();
    xisum = Matrix<double>::Zero(M, M);
    for (int ell = 1; ell < Ltot; ++ell)
        xisum = xisum + alpha_hat.col(ell - 1) * Bptr[ell]->template cast<double>().
            cwiseProduct(beta_hat.col(ell)).transpose() / c(ell);
    Matrix<double> tr = transition->template cast<double>();
    xisum = xisum.cwiseProduct(tr);
    PROGRESS_DONE();
}

adouble HMM::Q(void)
{
    PROGRESS("HMM::Q");
    Eigen::Array<adouble, Eigen::Dynamic, Eigen::Dynamic> gam = gamma.template cast<adouble>().array();
    Eigen::Array<adouble, Eigen::Dynamic, Eigen::Dynamic> xis = xisum.template cast<adouble>().array();
    adouble ret = (gam.col(0) * pi->array().log()).sum();
    std::map<const decltype(logBptr)::value_type, int> counts;
    for (int ell = 0; ell < Ltot; ++ell)
    {
        ret += (gam.col(ell) * (*logBptr[ell])).sum();
        counts[logBptr[ell]]++;
        domain_error(toDouble(ret));
    }
    ret += (xis * transition->array().log()).sum();
    PROGRESS_DONE();
    std::vector<decltype(counts)::value_type*> a;
    for (auto &p : counts)
        a.push_back(&p);
    std::sort(a.begin(), a.end(), 
            [] (const decltype(counts)::value_type *a, const decltype(counts)::value_type *b)
            { return a->second > b->second; });
    /*
    for (auto aa : a)
    {
        if (aa->second < 100)
            continue;
        std::cout << "count: " << aa->second << std::endl;
        std::cout << reverse_map[aa->first] << std::endl;
        std::cout << aa->first->template cast<double>().transpose() << std::endl << std::endl;
    }
    */
    return ret;
}
