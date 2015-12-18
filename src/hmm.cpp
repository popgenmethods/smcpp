#include "hmm.h"

HMM::HMM(const Matrix<int> &obs, int n, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission,
        const int mask_freq, InferenceManager* im) :
    n(n), block_size(block_size), 
    pi(pi), transition(transition), emission(emission),
    mask_freq(mask_freq),
    M(pi->rows()), 
    Ltot(std::ceil(obs.col(0).sum() / (double)block_size)),
    Bptr(Ltot),
    xisum(M, M), c(Ltot),
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
    std::map<std::pair<int, int>, int> powers;
    Vector<adouble> tmp(M);
    const Matrix<adouble> *em_ptr;
    bool alt_block;
    unsigned long int R, i = 0, block = 0, tobs = 0;
    block_key key;
    unsigned long int L = obs.col(0).sum();
    std::pair<int, int> ob;
    for (unsigned int ell = 0; ell < obs.rows(); ++ell)
    {
        R = obs(ell, 0);
        ob = {obs(ell, 1), obs(ell, 2)};
        for (unsigned int r = 0; r < R; ++r)
        {
            alt_block = is_alt_block(block);
            if (! alt_block or (alt_block and i == 0))
                // For alt block, take only the first observation
                powers[ob]++;
            if (tobs > L)
                throw std::domain_error("what?");
            tobs++;
            if (++i == block_size or (r == R - 1 and ell == obs.rows() - 1))
            {
                i = 0;
                key.alt_block = alt_block;
                key.powers = powers;
                block_keys.emplace_back(key.alt_block, key.powers);
                im->bpm_lock.lock();
                if (im->block_prob_map.count(key) == 0)
                    im->block_prob_map[key] = tmp;
                im->bpm_lock.unlock();
                Bptr[block] = &(im->block_prob_map[key]);
                block++;
                powers.clear();
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
        std::cout << emission->template cast<double>() << std::endl << std::endl;
        throw std::domain_error("badness encountered");
    }
}


void HMM::forward_only(void)
{
    PROGRESS("forward only");
    Matrix<double> tt = transition->template cast<double>();
    Matrix<float> ttpow = tt.pow(block_size).template cast<float>();
    Matrix<float> delta = ttpow * Bptr[Ltot - 1]->template cast<float>().asDiagonal();
    Vector<float> d = delta.rowwise().sum().cwiseInverse();
    delta = d.asDiagonal() * delta;
    float ll = -log(d(0)); // line 8
    // Lines 9 - 11
    std::map<Vector<adouble>*, Matrix<float> > a;
    a[Bptr[Ltot - 1]] = delta;
    // Lines 12 - 13
    std::vector<Matrix<float> > b(M, Matrix<float>::Zero(M, M));
    Matrix<float> delta1 = delta, q1, q2, ttB, B;
    for (int ell = Ltot - 2; ell >= 1; --ell) // Line 15
    {
        B = Bptr[ell]->template cast<float>().asDiagonal();
        ttB = ttpow * B;
        q1 = ttB * ttpow;
        q2 = delta.cwiseQuotient(ttpow);
        delta = ttB.cwiseQuotient(q1 * q2.transpose());
        ll += log(ttpow(0, 0) * B(0, 0) / delta(0, 0));
        for (auto &p : a)
            a[p.first] = delta * p.second;
        if (a.count(Bptr[ell]) == 0)
            a[Bptr[ell]] = Matrix<float>::Zero(M, M);
        a[Bptr[ell]] += delta;
        for (int m = 0; m < M; ++m)
            b[m] = delta * b[m] + delta * delta1.col(m).asDiagonal();
        delta1 = delta;
    }
    B = Bptr[0]->template cast<float>().asDiagonal();
    q1 = pi->template cast<float>().transpose() * B * ttpow;
    q2 = delta.cwiseQuotient(ttpow);
    delta = (pi->template cast<float>().transpose() * B).cwiseQuotient((q2 * q1).transpose());
    for (auto &p : a)
        gamma_sums[p.first] = delta * p.second;
    gamma_sums[Bptr[0]] += delta;
    gamma0 = delta;
    xisum.setZero();
    for (int u = 0; u < M; ++u)
        for (int v = 0; v < M; ++v)
        {
            for (int h = 0; h < M; ++h)
                xisum(u, v) += delta(h) * b[v](h, u);
            xisum(u, v) += delta(u) * delta1(u, v);
        }
    ll += log(pi->template cast<float>()(0) * B(0, 0) / delta(0, 0));
    PROGRESS_DONE();
}

void HMM::forward_backward(void)
{
    PROGRESS("forward backward");
    alpha_hat = Matrix<float>::Zero(M, Ltot);
    Matrix<double> tt = transition->template cast<double>();
    Matrix<double> ttpow = tt.pow(block_size).template cast<double>();
    Vector<double> tmp(M);
    Matrix<double> tmpmat;
    tmp = pi->template cast<double>().cwiseProduct(Bptr[0]->template cast<double>());
	c(0) = tmp.sum();
    alpha_hat.col(0) = (tmp / c(0)).template cast<float>();
    for (int ell = 1; ell < Ltot; ++ell)
    {
        tmp = Bptr[ell]->template cast<double>().cwiseProduct(ttpow.transpose() * alpha_hat.col(ell - 1).template cast<double>());
        c(ell) = tmp.sum();
        alpha_hat.col(ell) = (tmp / c(ell)).template cast<float>();
        if (std::isnan(toDouble(c(ell))))
            throw std::domain_error("something went wrong in forward algorithm");
    }
    Vector<float> beta = Vector<float>::Ones(M), g;
    gamma_sums.clear();
    gamma_sums[Bptr[Ltot - 1]] = alpha_hat.col(Ltot - 1);
    xisum.setZero();
    tmpmat = Bptr[Ltot - 1]->template cast<double>().asDiagonal();
    tmpmat /= c(Ltot - 1);
    xisum += alpha_hat.col(Ltot - 2) * beta.transpose() * tmpmat.template cast<float>();
    if (im->saveGamma)
    {
        gamma = Matrix<float>::Zero(M, Ltot);
        gamma.col(Ltot - 1) = alpha_hat.col(Ltot - 1);
    }
    for (int ell = Ltot - 2; ell >= 0; --ell)
    {
        tmpmat = ttpow * Bptr[ell + 1]->template cast<double>().asDiagonal();
        tmpmat /= c(ell + 1);
        beta = tmpmat.template cast<float>() * beta;
        if (gamma_sums.count(Bptr[ell]) == 0)
            gamma_sums[Bptr[ell]] = Vector<float>::Zero(M);
        g = alpha_hat.col(ell).cwiseProduct(beta);
        gamma_sums[Bptr[ell]] += g;
        if (im->saveGamma)
            gamma.col(ell) = g;
        if (ell > 0)
        {
            tmpmat = Bptr[ell]->template cast<double>().asDiagonal();
            tmpmat /= c(ell);
            xisum += alpha_hat.col(ell - 1) * beta.transpose() * tmpmat.template cast<float>();
        }
    }
    xisum = xisum.cwiseProduct(ttpow.template cast<float>());
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
    q3 = xisum.template cast<adouble>().cwiseProduct(ttpow).sum();
    PROGRESS_DONE();
    check_nan(q1);
    check_nan(q2);
    check_nan(q3);
    PROGRESS("q1:" << q1 << " [" << q1.derivatives().transpose() << "]\nq2:" 
            << q2 << " [" << q2.derivatives().transpose() << "]\nq3:" << q3 
            << " [" << q3.derivatives().transpose() << "]\n");
    return q1 + q2 + q3;
}
