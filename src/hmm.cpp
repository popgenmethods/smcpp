#include "hmm.h"

Matrix<int> make_two_mask(int m)
{
    Matrix<int> two_mask(3, m);
    two_mask.fill(0);
    two_mask.row(1).fill(1);
    return two_mask;
}

HMM::HMM(const Matrix<int> &obs, int n, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission, const Matrix<int> emission_mask,
        const int mask_freq) :
    n(n), block_size(block_size), 
    pi(pi), transition(transition), emission(emission), emission_mask(emission_mask),
    two_mask(make_two_mask(emission_mask.cols())),
    mask_freq(mask_freq),
    M(pi->rows()), 
    Ltot(std::ceil(obs.col(0).sum() / block_size)),
    Bptr(Ltot),
    xisum(M, M), c(Ltot) 
{ 
    prepare_B(obs);
}

bool HMM::is_alt_block(int block) 
{
    return block % mask_freq == 0;
}

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
    mpz_class coef;
    for (unsigned int ell = 0; ell < obs.rows(); ++ell)
    {
        R = obs(ell, 0);
        ob = {obs(ell, 1), obs(ell, 2)};
        for (unsigned int r = 0; r < R; ++r)
        {
            alt_block = is_alt_block(block);
            if (! alt_block or (alt_block and i == 0))
                powers[ob]++;
            if (tobs > L)
                throw std::domain_error("what?");
            tobs++;
            if (++i == block_size or (r == R - 1 and ell == obs.rows() - 1))
            {
                i = 0;
                key.alt_block = alt_block;
                key.powers = powers;
                if (block_prob_map.count(key) == 0)
                {
                    tmp.setOnes();
                    block_prob_map[key] = tmp;
                    std::array<std::map<std::set<int>, int>, 4> classes;
                    std::vector<int> ctot(4, 0);
                    const Matrix<int> &emask = alt_block ? emission_mask : two_mask;
                    int ai;
                    for (auto &p : powers)
                    {
                        std::set<int> s;
                        int a = p.first.first, b = p.first.second;
                        if (a >= 0 and b >= 0)
                        {
                            ai = 0;
                            s = {emask(a, b)};
                        }
                        else if (a >= 0)
                        {
                            for (int bb = 0; bb < n + 1; ++bb)
                                s.insert(emask(a, bb));
                            ai = 1;
                        }
                        else if (b >= 0)
                        {
                            // a is missing => sum along cols
                            s = {emask(0, b), emask(1, b), emask(2, b)};
                            ai = 2;
                        }
                        else
                        {
                           ai = 3;
                        }
                        if (classes[ai].count(s) == 0)
                            classes[ai].emplace(s, 0);
                        classes[ai][s] += p.second;
                        ctot[ai] += p.second;
                    }
                    coef = multinomial(ctot);
                    for (int j = 0; j < 4; ++j)
                    {
                        std::vector<int> values;
                        for (auto &kv : classes[j])
                            values.push_back(kv.second);
                        coef *= multinomial(values);
                    }
                    comb_coeffs[key] = coef.get_ui();
                }
                Bptr[block] = &block_prob_map[key];
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

void HMM::recompute_B(void)
{
    PROGRESS("recompute B");
    std::map<int, Vector<adouble> > mask_probs, two_probs;
    int em;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < n + 1; ++j)
        {
            em = emission_mask(i, j);
            if (mask_probs.count(em) == 0)
                mask_probs[em] = Vector<adouble>::Zero(M);
            mask_probs[em] += emission->col((n + 1) * i + j);
            if (two_probs.count(i % 2) == 0)
                two_probs[i % 2] = Vector<adouble>::Zero(M);
            two_probs[i % 2] += emission->col((n + 1) * i + j);
        }
    for (auto &bpm : block_prob_map)
    {
        Eigen::Array<adouble, Eigen::Dynamic, 1> tmp = Eigen::Array<adouble, Eigen::Dynamic, 1>::Ones(M), tmp2;
        Eigen::Array<adouble, Eigen::Dynamic, 1> log_tmp = Eigen::Array<adouble, Eigen::Dynamic, 1>::Zero(M);
        const Matrix<int> &emask = bpm.first.alt_block ? emission_mask : two_mask;
        std::map<int, Vector<adouble> > &prbs = bpm.first.alt_block ? mask_probs : two_probs;
        Vector<adouble> ob;
        for (auto &p : bpm.first.powers)
        {
            ob = Vector<adouble>::Zero(M);
            int a = p.first.first, b = p.first.second;
            if (a == -1)
            {
                if (b == -1)
                    // Double missing!
                    continue;
                else
                {
                    for (int x : std::set<int>{emask(0, b), emask(1, b), emask(2, b)})
                        ob += prbs[x];
                }
            }
            else
            {
                if (b == -1)
                {
                    std::set<int> bbs;
                    for (int bb = 0; bb < emask.cols(); ++bb)
                        bbs.insert(emask(a, bb));
                    for (int x : bbs)
                        ob += prbs[x];
                }
                else
                    ob = prbs[emask(a, b)];
            }
            log_tmp += ob.array().log() * p.second;
        }
        log_tmp += log(comb_coeffs[bpm.first]);
        tmp = exp(log_tmp);
        if (tmp.maxCoeff() > 1.0 or tmp.minCoeff() < 0.0)
            throw std::runtime_error("probability vector not in [0, 1]");
        check_nan(tmp);
        block_prob_map[bpm.first] = tmp.matrix();
    }
    PROGRESS_DONE();
}

/*
void HMM::linear_backward(void)
{
    Matrix<double> tt = transition->template cast<double>();
    Matrix<double> ttpow = tt.pow(block_size);
    delta_star = ttpow * Bptr[Ltot - 1].template cast<double>.asDiagonal();
    d = delta_star.rowwise().sum();
    delta = delta_star.rowwise() / d;
    ell = log(d(0)); // line 8
    // Lines 9 - 11
    std::map<Vector<adouble>*, Matrix<double> > a;
    a[Bptr[Ltot - 1]] = delta;
    // Lines 12 - 13
    std::vector<Matrix<double> > b(M, Matrix<double>::Zero(M, M));

    Matrix<double> Tr1 = is_alt_block(Ltot - 1) ? ttalt : ttpow, Tr2;
    Matrix<double> q1, q2, B;
    for (int ell = Ltot - 2; ell >= 0; ++ell) // Line 15
    {
        B = Bptr[ell]->template cast<double>().asDiagonal();
        Tr2 = Tr1;
        Tr1 = is_alt_block(ell) ? ttalt : ttpow;
        q1 = Tr1 * B * Tr2; // Line 18
        q2 = delta / Tr1;
        q3 = q1 * q2.transpose();
        d1 = Tr1 * B;
        delta = d1.cwiseQuotient(q3);
    }
}
*/

void HMM::forward_backward(void)
{
    PROGRESS("forward backward");
    alpha_hat = Matrix<float>::Zero(M, Ltot);
    Matrix<double> tt = transition->template cast<double>();
    Matrix<float> ttpow = tt.pow(block_size).template cast<float>();
    alpha_hat.col(0) = pi->template cast<float>().cwiseProduct(Bptr[0]->template cast<float>());
	c(0) = alpha_hat.col(0).sum();
    alpha_hat.col(0) /= c(0);
    for (int ell = 1; ell < Ltot; ++ell)
    {
        alpha_hat.col(ell) = Bptr[ell]->template cast<float>().cwiseProduct(ttpow.transpose() * alpha_hat.col(ell - 1));
        c(ell) = alpha_hat.col(ell).sum();
        if (std::isnan(toDouble(c(ell))))
            throw std::domain_error("something went wrong in forward algorithm");
        alpha_hat.col(ell) /= c(ell);
    }
    Vector<float> beta = Vector<float>::Ones(M), gamma;
    gamma_sums.clear();
    gamma_sums[Bptr[Ltot - 1]] = alpha_hat.col(Ltot - 1);
    xisum.setZero();
    xisum += alpha_hat.col(Ltot - 2) * beta.transpose() * 
        Bptr[Ltot - 1]->template cast<float>().asDiagonal() / c(Ltot - 1);
    Vector<float> gtot(M);
    gtot.setZero();
    for (int ell = Ltot - 2; ell >= 0; --ell)
    {
        beta = ttpow * (Bptr[ell + 1]->template cast<float>().cwiseProduct(beta)) / c(ell + 1);
        if (gamma_sums.count(Bptr[ell]) == 0)
            gamma_sums[Bptr[ell]] = Vector<float>::Zero(M);
        gamma = alpha_hat.col(ell).cwiseProduct(beta);
        gamma_sums[Bptr[ell]] += gamma;
        gtot += gamma;
        if (ell > 0)
            xisum += alpha_hat.col(ell - 1) * beta.transpose() * Bptr[ell]->template cast<float>().asDiagonal() / c(ell);
    }
    xisum = xisum.cwiseProduct(ttpow);
    gamma0 = gamma;
    PROGRESS(std::endl << "xisums and stuff" << Ltot << " " << xisum.sum() << " " << gtot.sum() << std::endl);
    PROGRESS_DONE();
}


void HMM::Estep(void)
{
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
