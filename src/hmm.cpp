#include "hmm.h"

long num_blocks(int total_loci, int block_size, int alt_block_size, int mask_freq)
{
    long base = mask_freq * std::floor(total_loci / (double)((mask_freq - 1) * block_size + alt_block_size));
    int remain = total_loci % ((mask_freq - 1) * block_size + alt_block_size);
    if (remain == 0)
        return base;
    else if (remain <= alt_block_size)
        return base + 1;
    return base + 1 + std::ceil((remain - alt_block_size) / (double)block_size);
}

HMM::HMM(const Matrix<int> &obs, int n, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const int mask_freq, InferenceManager* im) :
    n(n), block_size(block_size), alt_block_size(1),
    pi(pi), transition(transition),
    mask_freq(mask_freq),
    M(pi->rows()), 
    Ltot(num_blocks(obs.col(0).sum(), block_size, alt_block_size, mask_freq)),
    Bptr(Ltot),
    xisum(M, M), 
    xisum_alt(M, M),
    c(Ltot),
    im(im)
{ 
    prepare_B(obs);
}

bool HMM::is_alt_block(int block) 
{
    return block % mask_freq == 0;
}

void HMM::prepare_B(const Matrix<int> &obs)
{
    PROGRESS("preparing B");
    decltype(block_key::powers) powers;
    Vector<adouble> tmp(M);
    const Matrix<adouble> *em_ptr;
    unsigned long int R, i = 0, block = 0, tobs = 0;
    block_key key;
    unsigned long int L = obs.col(0).sum();
    block_power ob;
    int current_block_size = (is_alt_block(block)) ? alt_block_size : block_size;
    for (unsigned int ell = 0; ell < obs.rows(); ++ell)
    {
        R = obs(ell, 0);
        ob = {obs(ell, 1), obs(ell, 2), obs(ell, 3)};
        for (unsigned int r = 0; r < R; ++r)
        {
            powers[ob]++;
            if (tobs > L)
                throw std::domain_error("what?");
            tobs++;
            if (++i == current_block_size or (r == R - 1 and ell == obs.rows() - 1))
            {
                i = 0;
                key.alt_block = is_alt_block(block);
                key.powers = powers;
                block_keys.emplace_back(key.alt_block, std::vector<std::pair<block_power, int> >(key.powers.begin(), key.powers.end()));
#pragma omp critical(map_insert)
                {
                    if (im->block_prob_map.count(key) == 0)
                        im->block_prob_map.insert({key, tmp});
                }
                try
                {
                    Bptr[block] = &(im->block_prob_map.at(key));
                    bmap[Bptr[block]] = key;
                }
                catch (std::out_of_range)
                {
                    std::cout << block << " " << key.alt_block << " " << key.powers << " " << im->block_prob_map.count(key) << std::endl;
                    throw;
                }
                block++;
                powers.clear();
                current_block_size = (is_alt_block(block)) ? alt_block_size : block_size;
            }
        }
    }
    PROGRESS_DONE();
}

double HMM::loglik()
{
    double ret = c.array().log().sum();
    return ret;
}

void HMM::domain_error(double ret)
{
    if (std::isinf(ret) or std::isnan(ret))
    {
        std::cout << pi->template cast<double>() << std::endl << std::endl;
        std::cout << transition->template cast<double>() << std::endl << std::endl;
        throw std::domain_error("badness encountered");
    }
}


void HMM::forward_only(void)
{
    PROGRESS("forward only");
    Matrix<double> tt = transition->template cast<double>();
    Matrix<fbType> ttpow = tt.pow(block_size).template cast<fbType>();
    Matrix<fbType> delta = ttpow * Bptr[Ltot - 1]->template cast<fbType>().asDiagonal();
    Vector<fbType> d = delta.rowwise().sum().cwiseInverse();
    delta = d.asDiagonal() * delta;
    fbType ll = -log(d(0)); // line 8
    // Lines 9 - 11
    std::map<Vector<adouble>*, Matrix<fbType> > a;
    a[Bptr[Ltot - 1]] = delta;
    // Lines 12 - 13
    std::vector<Matrix<fbType> > b(M, Matrix<fbType>::Zero(M, M));
    Matrix<fbType> delta1 = delta, q1, q2, ttB, B;
    for (int ell = Ltot - 2; ell >= 1; --ell) // Line 15
    {
        B = Bptr[ell]->template cast<fbType>().asDiagonal();
        ttB = ttpow * B;
        q1 = ttB * ttpow;
        q2 = delta.cwiseQuotient(ttpow);
        delta = ttB.cwiseQuotient(q1 * q2.transpose());
        ll += log(ttpow(0, 0) * B(0, 0) / delta(0, 0));
        for (auto &p : a)
            a[p.first] = delta * p.second;
        if (a.count(Bptr[ell]) == 0)
            a[Bptr[ell]] = Matrix<fbType>::Zero(M, M);
        a[Bptr[ell]] += delta;
        for (int m = 0; m < M; ++m)
            b[m] = delta * b[m] + delta * delta1.col(m).asDiagonal();
        delta1 = delta;
    }
    B = Bptr[0]->template cast<fbType>().asDiagonal();
    q1 = pi->template cast<fbType>().transpose() * B * ttpow;
    q2 = delta.cwiseQuotient(ttpow);
    delta = (pi->template cast<fbType>().transpose() * B).cwiseQuotient((q2 * q1).transpose());
    for (auto &p : a)
        gamma_sums[p.first] = delta * p.second;
    gamma_sums[Bptr[0]] += delta;
    gamma0 = delta;
    xisum.setZero();
    xisum_alt.setZero();
    for (int u = 0; u < M; ++u)
        for (int v = 0; v < M; ++v)
        {
            for (int h = 0; h < M; ++h)
                xisum(u, v) += delta(h) * b[v](h, u);
            xisum(u, v) += delta(u) * delta1(u, v);
        }
    ll += log(pi->template cast<fbType>()(0) * B(0, 0) / delta(0, 0));
    PROGRESS_DONE();
}

void HMM::forward_backward(void)
{
    PROGRESS("forward backward");
    alpha_hat = Matrix<fbType>::Zero(M, Ltot);
    Matrix<double> tt = transition->template cast<double>();
    Matrix<double> ttpow = tt.pow(block_size).template cast<double>();
    Matrix<double> ttalt = tt.pow(alt_block_size).template cast<double>();
    Matrix<double> *tr;
    Vector<double> tmp(M);
    tmp = pi->template cast<double>().cwiseProduct(Bptr[0]->template cast<double>());
	c(0) = tmp.sum();
    alpha_hat.col(0) = (tmp / c(0)).template cast<fbType>();
    for (int ell = 1; ell < Ltot; ++ell)
    {
        tr = (is_alt_block(ell - 1)) ? &ttalt : &ttpow;
        tmp = Bptr[ell]->template cast<double>().asDiagonal() * tr->transpose() * alpha_hat.col(ell - 1).template cast<double>();
        c(ell) = tmp.sum();
        alpha_hat.col(ell) = (tmp / c(ell)).template cast<fbType>();
        if (std::isnan(toDouble(c(ell))) or std::isinf(toDouble(c(ell))) or c(ell) <= 0.)
        {
            std::cout << tmp.transpose() << std::endl << std::endl;
            std::cout << Bptr[ell]->template cast<double>().transpose() << std::endl;
            std::cout << ell << std::endl;
            std::cout << alpha_hat.col(ell - 2).transpose() << std::endl << std::endl;
            std::cout << alpha_hat.col(ell - 1).transpose() << std::endl << std::endl;
            std::cout << alpha_hat.col(ell).transpose() << std::endl << std::endl;
            std::cout << c(ell - 2) << " " << c(ell - 1) << " " << c(ell) << std::endl;
            throw std::domain_error("something went wrong in forward algorithm");
        }
    }
    Vector<fbType> beta = Vector<fbType>::Ones(M), g;
    gamma_sums.clear();
    gamma_sums[Bptr[Ltot - 1]] = alpha_hat.col(Ltot - 1);
    // Transitions
    xisum.setZero();
    xisum_alt.setZero();
    Matrix<double> tmpmat = Bptr[Ltot - 1]->template cast<double>().asDiagonal();
    tmpmat /= c(Ltot - 1);
    xisum += alpha_hat.col(Ltot - 2) * beta.transpose() * tmpmat.template cast<fbType>();
    if (im->saveGamma)
    {
        gamma = Matrix<fbType>::Zero(M, Ltot);
        gamma.col(Ltot - 1) = alpha_hat.col(Ltot - 1);
    }
    Matrix<fbType> *xir;
    for (int ell = Ltot - 2; ell >= 0; --ell)
    {
        tr = is_alt_block(ell) ? &ttalt : &ttpow;
        tmpmat = (*tr) * Bptr[ell + 1]->template cast<double>().asDiagonal();
        tmpmat /= c(ell + 1);
        beta = tmpmat.template cast<fbType>() * beta;
        if (gamma_sums.count(Bptr[ell]) == 0)
            gamma_sums.insert({Bptr[ell], Vector<fbType>::Zero(M)});
        g = alpha_hat.col(ell).cwiseProduct(beta);
        gamma_sums.at(Bptr[ell]) += g;
        if (im->saveGamma)
            gamma.col(ell) = g;
        if (ell > 0)
        {
            tmpmat = Bptr[ell]->template cast<double>().asDiagonal();
            tmpmat /= c(ell);
            xir = (is_alt_block(ell - 1)) ? &xisum_alt : &xisum;
            *xir += alpha_hat.col(ell - 1) * beta.transpose() * tmpmat.template cast<fbType>();
            check_nan(*xir);
        }
    }
    xisum = xisum.cwiseProduct(ttpow.template cast<fbType>());
    xisum_alt = xisum_alt.cwiseProduct(ttalt.template cast<fbType>());
    check_nan(xisum);
    check_nan(xisum_alt);
    gamma0 = g;
    PROGRESS_DONE();
}


void HMM::Estep(void)
{
    if (im->forwardOnly)
        forward_only();
    else
        forward_backward();
}

Matrix<adouble> mymatpow(const Matrix<adouble> M, int p)
{
    if (p == 1)
        return M;
    if (p % 2 == 0) 
    {
        Matrix<adouble> M2 = mymatpow(M, p / 2);
        return M2 * M2;
    }
    Matrix<adouble> M2 = mymatpow(M, (p - 1) / 2);
    return M * M2 * M2;
}

adouble HMM::Q(void)
{
    PROGRESS("HMM::Q");
    adouble q1, q2, q3;
    q1 = (gamma0.array().template cast<adouble>() * pi->array().log()).sum();
    q2 = 0.0;
    for (auto &p : gamma_sums)
        q2 += (p.first->array().log() * p.second.array().template cast<adouble>()).sum();
    Matrix<adouble> ttpow = mymatpow(*transition, block_size).array().log().matrix();
    Matrix<adouble> ttalt = mymatpow(*transition, alt_block_size).array().log().matrix();
    check_nan(ttalt);
    check_nan(ttpow);
    q3 = xisum.template cast<adouble>().cwiseProduct(ttpow).sum();
    q3 += xisum_alt.template cast<adouble>().cwiseProduct(ttalt).sum();
    check_nan(q1);
    check_nan(q2);
    check_nan(q3);
    PROGRESS("q1:" << q1 << " [" << q1.derivatives().transpose() << "]\nq2:" 
            << q2 << " [" << q2.derivatives().transpose() << "]\nq3:" << q3 
            << " [" << q3.derivatives().transpose() << "]\n");
    return q1 + q2 + q3;
}
