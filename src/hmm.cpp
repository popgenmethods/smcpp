#include <unsupported/Eigen/MatrixFunctions>

#include "common.h"
#include "inference_manager.h"
#include "inference_bundle.h"
#include "hmm.h"

HMM::HMM(const int hmm_num,
         const Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> > &obs,
         const InferenceBundle* ib) :
    hmm_num(hmm_num), obs(obs), ib(ib), M(ib->pi->rows()), L(obs.rows()), ll(0.),
    alpha_hat(M, L + 1), xisum(M, M), gamma(M, 1), log_c(L + 1)
    // Gamma has one column because gamma.col(0) will be set to calculate
    // the initial distribution term
{
    gamma.setZero();
    gamma_sums.clear();
    Vector<double> uniform = ib->pi->template cast<double>();
    for (int ell = 0; ell < L; ++ell)
    {
        int span = obs(ell, 0);
        block_key key = ob_key(ell);
        if (gamma_sums.count(key) == 0)
            gamma_sums.emplace(key, Vector<double>::Zero(M));
        gamma_sums.at(key) += span * uniform;
    }
    xisum.setZero();
}

double HMM::loglik()
{
    return ll;
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
    if (*(ib->saveGamma))
        gamma = Matrix<double>::Zero(M, L + 1);
    Matrix<double> T = tb->Td;
    gamma_sums.clear();
    const Vector<double> z = Vector<double>::Zero(M);
    gamma_sums.emplace(ob_key(0), z);
    Eigen::DiagonalMatrix<double, Eigen::Dynamic, Eigen::Dynamic> B;
    DEBUG1 << "forward algorithm (HMM #" << hmm_num << ")";
    Vector<double> a;
    int prog = (int)((double)L * 0.1);
    ll = 0.;
    alpha_hat.col(0) = ib->pi->template cast<double>().template cast<float>();
    log_c(0) = 0.;
    for (int ell = 1; ell < L + 1; ++ell)
    {
        if (ell == prog)
        {
            DEBUG1 << "hmm " << hmm_num << ": " << (int)(100. * (double)ell / (double)L) << "%";
            prog += (int)((double)L * 0.1);
        }
        block_key key = ob_key(ell - 1);
        gamma_sums.emplace(key, z);
        B = ib->emission_probs->at(key).template cast<double>().asDiagonal();
        int span = obs(ell - 1, 0);
        if (span > 1 and tb->eigensystems.count(key) > 0)
        {
            const eigensystem es = tb->eigensystems.at(key);
            a = (es.P_r * (es.d_r_scaled.array().pow(span).matrix().asDiagonal() *
                        (es.Pinv_r * alpha_hat.col(ell - 1).template cast<double>())));
            double s = a.sum();
            a /= s;
            log_c(ell) = std::log(s) + span * std::log(es.scale);
            alpha_hat.col(ell) = a.template cast<float>();
        }
        else
        {
            // if (span != 1) throw std::runtime_error("span != 1");
            Matrix<double> M = (B * T.transpose()).pow(span);
            alpha_hat.col(ell) = M.template cast<float>() * alpha_hat.col(ell - 1);
            double s = alpha_hat.col(ell).sum();
            log_c(ell) = std::log(s);
            alpha_hat.col(ell) /= s;
        }
        CHECK_NAN(alpha_hat.col(ell));
        alpha_hat.col(ell) = alpha_hat.col(ell).unaryExpr(
                [] (const float &x) { if (x < 1e-10f) return 1e-10f; return x; }
            );
        ll += log_c(ell);
    }
    Vector<double> beta = Vector<double>::Ones(M), v(M), alpha(M), log_beta;
    xisum.setZero();
    Matrix<double> Q_r(M, M), sq(M, M), xis(M, M), xis1(M, M), tmp;
    double p, vM, log_C, log_p;
    DEBUG1 << "backward algorithm (HMM #" << hmm_num << ")";
    for (int ell = L; ell > 0; --ell)
    {
        v.setZero();
        int span = obs(ell - 1, 0);
        block_key key = ob_key(ell - 1);
        B = ib->emission_probs->at(key).template cast<double>().asDiagonal();
        if (span > 1 and tb->eigensystems.count(key) > 0)
        {
            const eigensystem es = tb->eigensystems.at(key);
            log_p = std::log(es.scale) * (span - 1);
            {
                Q_r = es.Pinv_r * (alpha_hat.col(ell - 1).template cast<double>() * beta.transpose()) * es.P_r;
                sq = tb->span_Qs.at({span,key});
                Q_r = Q_r.cwiseProduct(sq);
                v = ((es.P_r * es.d_r.asDiagonal() * Q_r * es.Pinv_r).diagonal().array().abs().log() - 
                        log_c(ell) + std::log(es.scale) * (span - 1));
                vM = v.maxCoeff();
                v = v.array() - vM;
                log_C = std::log(span) - vM - std::log(v.array().exp().sum());
                v = (v.array() + vM + log_C).exp();
                xis = ((es.P_r * Q_r * es.Pinv_r * B).array().abs().log() - log_c(ell) + log_p + log_C).exp();
                log_beta = ((es.Pinv_r.transpose() * (es.d_r_scaled.array().pow(span).matrix().asDiagonal() *
                            (es.P_r.transpose() * beta))).array().log() + log_p + log_C + std::log(es.scale));
                vM = log_beta.maxCoeff();
                log_beta = log_beta.array() - vM;
                beta = log_beta.array().exp();
            }
        }
        else
        {
            if (span != 1)
                throw std::runtime_error("span");
            v = alpha_hat.col(ell).template cast<double>().cwiseProduct(beta);
            p = v.sum();
            v /= p;
            xis = alpha_hat.col(ell - 1).template cast<double>() * beta.transpose() * B / exp(log_c(ell));
            xis /= p;
            beta = T * (B * beta);
        }
        xisum += xis;
        beta /= beta.sum();
        CHECK_NAN(xisum);
        CHECK_NAN(v);
        CHECK_NAN(beta);
        gamma_sums.at(key) += v;
        if (*(ib->saveGamma))
            gamma.col(ell) = v;
    }
    gamma.col(0) = alpha_hat.col(0).template cast<double>().cwiseProduct(beta);
    xisum = xisum.cwiseProduct(T);
    xisum = xisum.unaryExpr([] (const double &x) { if (x < 1e-20) return 1e-20; return x; });
}

Vector<adouble> HMM::Q(void)
{
    DEBUG1 << "HMM::Q";
    Vector<adouble> ret(4);
    ret.setZero();
    Vector<adouble> pi = *(ib->pi);
    ret(0) = (pi.array().log() * gamma.col(0).array().template cast<adouble_base_type>()).sum();
    std::vector<adouble> gss[2];
    for (auto &p : gamma_sums)
    {
        int i = (int)(p.first.nb() > 0);
        std::vector<adouble> &gs = gss[i];
        Vector<adouble> ep = ib->emission_probs->at(p.first);
        Vector<adouble> c = ep.array().log().matrix().cwiseProduct(p.second);
        if (ep.minCoeff() <= 0.0)
        {
            WARNING << "zeros detected in emission probability, key=" << p.first;
            gs.clear();
            gs[0] = adouble(-INFINITY);
            break;
        }
        gs.insert(std::end(gs), c.data(), c.data() + M);
    }
    ret(1) = doubly_compensated_summation(gss[0]);
    ret(2) = doubly_compensated_summation(gss[1]);
    Matrix<adouble> T = ib->tb->T;
    Matrix<adouble> log_T = T.array().log().matrix();
    CHECK_NAN(log_T);
    Matrix<adouble> prod = log_T.cwiseProduct(xisum.template cast<adouble_base_type>());
    std::vector<adouble> es(prod.data(), prod.data() + M * M);
    ret(3) = doubly_compensated_summation(es);
    CHECK_NAN(ret);
    DEBUG1
        << "ret0:" << toDouble(ret(0))
        << " ret1:" << toDouble(ret(1))
        << " ret2:" << toDouble(ret(2))
        << " ret3:" << toDouble(ret(3));
    return ret;
}
