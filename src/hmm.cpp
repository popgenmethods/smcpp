#include "hmm.h"
#include <csignal>

HMM::HMM(Eigen::Matrix<int, Eigen::Dynamic, 2> obs, const int block_size,
        const Vector<adouble> *pi, const Matrix<adouble> *transition, 
        const Matrix<adouble> *emission) : 
    obs(obs), block_size(block_size), 
    pi(pi), transition(transition), emission(emission),
    M(pi->rows()), Ltot(ceil(obs.col(0).sum() / block_size)), 
    B(M, Ltot), alpha_hat(M, Ltot), beta_hat(M, Ltot), gamma(M, Ltot), xisum(M, M), c(Ltot) 
{ }

double HMM::loglik()
{
    double ret = c.array().log().sum();
    domain_error(ret);
    return ret;
}

void HMM::domain_error(double ret)
{
    if (isinf(ret) or isnan(ret))
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
    B.setOnes();
    int block = 0;
    int i = 0, ob, R;
    std::map<int, int> powers;
    for (int ell = 0; ell < obs.rows(); ++ell)
    {
        R = obs(ell, 0);
        ob = obs(ell, 1);
        for (int r = 0; r < R; ++r)
        {
            powers[ob]++;
            if (++i == block_size)
            {
                i = 0;
                for (auto &p : powers)
                    B.col(block) = B.col(block).cwiseProduct(emission->col(p.first).array().pow(p.second).matrix());
                powers.clear();
                block++;
            }
        }
    }
}

void HMM::forward_backward(void)
{
    Matrix<double> tt = transition->transpose().template cast<double>();
    Matrix<double> bt = B.template cast<double>();
    alpha_hat.col(0) = pi->template cast<double>().cwiseProduct(bt.col(0));
	c(0) = alpha_hat.col(0).sum();
    alpha_hat.col(0) /= c(0);
    for (int ell = 1; ell < Ltot; ++ell)
    {
        alpha_hat.col(ell) = bt.col(ell).asDiagonal() * tt * alpha_hat.col(ell - 1);
        c(ell) = alpha_hat.col(ell).sum();
        if (isnan(toDouble(c(ell))))
            throw std::domain_error("something went wrong in forward algorithm");
        alpha_hat.col(ell) /= c(ell);
    }

    beta_hat.col(Ltot - 1) = Vector<double>::Ones(M);
    Matrix<double> tr = transition->template cast<double>();
    for (int ell = Ltot - 2; ell >= 0; --ell)
        beta_hat.col(ell) = tr * bt.col(ell + 1).asDiagonal() * beta_hat.col(ell + 1) / c(ell + 1);
}


void HMM::Estep(void)
{
    forward_backward();
	gamma = alpha_hat.cwiseProduct(beta_hat);
    xisum = Matrix<double>::Zero(M, M);
    for (int ell = 1; ell < Ltot; ++ell)
        xisum = xisum + alpha_hat.col(ell - 1) * B.col(ell).template cast<double>().
            cwiseProduct(beta_hat.col(ell)).transpose() / c(ell);
    Matrix<double> tr = transition->template cast<double>();
    xisum = xisum.cwiseProduct(tr);
}

adouble HMM::Q(void)
{
    Eigen::Array<adouble, Eigen::Dynamic, Eigen::Dynamic> gam = gamma.template cast<adouble>().array();
    Eigen::Array<adouble, Eigen::Dynamic, Eigen::Dynamic> xis = xisum.template cast<adouble>().array();
    adouble r1 = (gam.col(0) * pi->array().log()).sum();
    adouble r2 = (gam * B.array().log()).sum();
    adouble r3 = (xis * transition->array().log()).sum();
    adouble ret = r1 + r2 + r3;
    Matrix<adouble> Bl = B.array().log().matrix();
    /*
    for (int i = 0; i < Bl.rows(); ++i)
        for (int j = 0; j < Bl.cols(); ++j)
        {
            Vector<double> bd = Bl(i, j).derivatives();
            for (int k = 0; k < bd.rows(); ++k)
            {
                double x = bd(k);
                if (isnan(x) or isinf(x))
                {
                    std::cout << i << "," << j << " " << bd.transpose() << std::endl;
                    std::cout << B.col(j).transpose().template cast<double>() << std::endl;
                }
            }
        }
    */
    domain_error(toDouble(ret));
    return ret;
}
