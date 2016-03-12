#include "hmm.h"

HMM::HMM(const int hmm_num, const Matrix<int> &obs, const InferenceBundle* ib) : 
    hmm_num(hmm_num), obs(obs), ib(ib), M(ib->pi->rows()), L(obs.rows()), alpha_hat(M, L), xisum(M, M), c(L)
{}

double HMM::loglik()
{
    double ret = c.array().log().sum();
    check_nan(ret);
    return ret;
}

void HMM::domain_error(double ret)
{
    if (std::isinf(ret) or std::isnan(ret))
    {
        std::cout << ib->pi->template cast<double>() << std::endl << std::endl;
        std::cout << ib->tb->Td << std::endl << std::endl;
        throw std::domain_error("badness encountered");
    }
}

void HMM::Estep(bool fbOnly)
{
    TransitionBundle *tb = ib->tb;
    alpha_hat = Matrix<double>::Zero(M, L);
    if (*(ib->saveGamma))
        gamma = Matrix<double>::Zero(M, L);
    Matrix<double> T = tb->Td;
    gamma_sums.clear();
    const Vector<double> z = Vector<double>::Zero(M);
    gamma_sums.emplace(ob_key(0), z);
    alpha_hat.col(0) = ib->pi->template cast<double>().cwiseProduct(ib->emission_probs->at(ob_key(0)).template cast<double>());
	c(0) = alpha_hat.col(0).sum();
    alpha_hat.col(0) /= c(0);
    block_key key;
    Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
    DEBUG("forward algorithm (HMM #" << hmm_num << ")");
    for (int ell = 1; ell < L; ++ell)
    {
        key = ob_key(ell);
        gamma_sums.emplace(key, z);
        B = ib->emission_probs->at(key).template cast<double>().asDiagonal();
        int span = obs(ell, 0);
        if (tb->eigensystems.count(key) > 0)
        {
            eigensystem es = tb->eigensystems.at(key);
            // alpha_hat.col(ell) = (es.P * (es.d.array().pow(span).matrix().asDiagonal() * 
            //             (es.Pinv * alpha_hat.col(ell - 1).template cast<std::complex<double> >()))).real();
            alpha_hat.col(ell) = (es.P_r * (es.d_r.array().pow(span).matrix().asDiagonal() * 
                        (es.Pinv_r * alpha_hat.col(ell - 1))));
        }
        else
        {
            Matrix<double> M = (B * T.transpose()).pow(span);
            alpha_hat.col(ell) = M * alpha_hat.col(ell - 1);
        }
        c(ell) = alpha_hat.col(ell).sum();
        alpha_hat.col(ell) /= c(ell);
    }
    Vector<double> beta = Vector<double>::Ones(M), v(M), alpha(M);
    xisum.setZero();
    Matrix<double> Q(M, M), Qt(M, M), xis(M, M);
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> atmp(M, *(ib->spanCutoff)), btmp(M, *(ib->spanCutoff));
    DEBUG("backward algorithm (HMM #" << hmm_num << ")");
    for (int ell = L - 1; ell > 0; --ell)
    {
        v.setZero();
        int span = obs(ell, 0);
        key = ob_key(ell);
        B = ib->emission_probs->at(ob_key(ell)).template cast<double>().asDiagonal();
        if (tb->span_Qs.count({span, key}) > 0 and span > *(ib->spanCutoff))
        {
            eigensystem es = tb->eigensystems.at(key);
            if (not fbOnly)
            {
                Q = es.Pinv_r * alpha_hat.col(ell - 1) * beta.transpose() * es.P_r;
                Q = Q.cwiseProduct(tb->span_Qs.at({span, key}).real());
                v = (es.P_r * es.d_r.asDiagonal() * Q * es.Pinv_r).diagonal() / c(ell);
                xisum += (es.P_r * Q * es.Pinv_r) * B / c(ell);
            }
            beta = (es.Pinv_r.transpose() * (es.d_r.array().pow(span).matrix().asDiagonal() * 
                        (es.P_r.transpose() * beta)));
        }
        else
        {
            if (span == 1)
            {
                v = alpha_hat.col(ell).cwiseProduct(beta);
                if (not fbOnly)
                    xisum += alpha_hat.col(ell - 1) * beta.transpose() * B / c(ell);
                beta = T * (B * beta);
            }
            else
            {
                Q = B * T.transpose();
                alpha = alpha_hat.col(ell - 1);
                for (int i = 0; i < span; ++i)
                {
                    alpha = Q * alpha;
                    atmp.col(i) = alpha;
                    btmp.col(span - i - 1) = beta;
                    beta = Q.transpose() * beta;
                }
                if (not fbOnly)
                {
                    v = (atmp.leftCols(span) * btmp.leftCols(span)).rowwise().sum().matrix() / c(ell);
                    xis = alpha_hat.col(ell - 1) * btmp.col(0).matrix().transpose();
                    for (int i = 1; i < span; ++i)
                        xis += atmp.col(i - 1).matrix() * btmp.col(i).transpose().matrix();
                    xisum += xis * B / c(ell);
                }
            }
        }
        beta /= c(ell);
        if (not fbOnly)
            gamma_sums.at(key) += v;
        if (*(ib->saveGamma))
            gamma.col(ell) = v;
    }
    gamma0 = alpha_hat.col(0).cwiseProduct(beta);
    gamma_sums.at(ob_key(0)) += gamma0;
    if (*(ib->saveGamma))
        gamma.col(0) = gamma0;
    if (not fbOnly)
        xisum = xisum.cwiseProduct(T);
    PROGRESS_DONE();
}

adouble HMM::Q(void)
{
    DEBUG("HMM::Q");
    adouble q1, q2, q3;
    q1 = 0.0;
    Vector<adouble> pi = *(ib->pi);
    for (int m = 0; m < M; ++m)
        q1 += log(pi(m)) * gamma0(m);

    q2 = 0.0;
    for (auto &p : gamma_sums)
    {
        Vector<adouble> ep = ib->emission_probs->at(p.first);
        for (int m = 0; m < M; ++m)
            q2 += p.second(m) * log(ep(m));
    }

    // Am getting a weird memory-related bug here when I do this all in
    // one line. Something related to threading and CRTP, maybe. Split
    // things up a bit to fix.
    q3 = 0.0;
    Matrix<adouble> T = ib->tb->T;
    for (int m1 = 0; m1 < M; ++m1)
        for (int m2 = 0; m2 < M; ++m2)
            q3 += log(T(m1, m2)) * xisum(m1, m2);

    check_nan(q1);
    check_nan(q2);
    check_nan(q3);
    DEBUG("q1:" << q1 << " [" << q1.derivatives().transpose() << "]\nq2:" 
            << q2 << " [" << q2.derivatives().transpose() << "]\nq3:" << q3 
            << " [" << q3.derivatives().transpose() << "]\n");
    return q1 + q2 + q3;
}
