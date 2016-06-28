#ifndef TRANSITION_BUNDLE_H
#define TRANSITION_BUNDLE_H

#include "common.h"

struct eigensystem
{
    Eigen::MatrixXcd P, Pinv;
    Eigen::VectorXcd d;
    double scale;
    Eigen::VectorXcd d_scaled;
    Matrix<double> P_r, Pinv_r;
    Vector<double> d_r;
    Vector<double> d_r_scaled;
    bool cplx;
    eigensystem(const Eigen::EigenSolver<Matrix<double> > &es) :
        P(es.eigenvectors()), Pinv(P.inverse()), d(es.eigenvalues()),
        P_r(P.real()), Pinv_r(Pinv.real()), 
        scale(d.cwiseAbs().maxCoeff()), d_scaled(d / scale),
        d_r(d.real()), d_r_scaled(d_r / scale),
        cplx(d.imag().cwiseAbs().maxCoeff() > 0)
    {
        DEBUG << "max imag: " << d.imag().cwiseAbs().maxCoeff();
    }
};

class TransitionBundle
{
    public:
    TransitionBundle(const std::set<std::pair<int, block_key> > &targets_s,
            const std::map<block_key, Vector<adouble> >* emission_probs) : 
        targets(targets_s.begin(), targets_s.end()),
        emission_probs(emission_probs) {}

    void update(const Matrix<adouble> &new_T)
    {
        T = new_T;
        Td = T.template cast<double>();
        const int M = T.rows();
        eigensystems.clear();
        span_Qs.clear();
#pragma omp parallel for
        for (auto it = targets.begin(); it < targets.end(); ++it)
        {
            Matrix<double> tmp;
            int span = it->first;
            block_key key = it->second;
#pragma omp critical(checkEigensystem)
            {
                if (eigensystems.count(key) == 0)
                {
                    tmp = emission_probs->at(key).template cast<double>().asDiagonal() * Td.transpose();
                    Eigen::EigenSolver<Matrix<double> > es(tmp);
                    eigensystems.emplace(key, es);
                }
            }
            eigensystem eig = eigensystems.at(key);
            Matrix<std::complex<double> > Q(M, M);
            for (int a = 0; a < M; ++a)
                for (int b = 0; b < M; ++b)
                    if (a == b)
                        Q(a, b) = std::pow(eig.d_scaled(a), span - 1) * (double)span;
                    else
                        Q(a, b) = (std::pow(eig.d_scaled(a), span) - std::pow(eig.d_scaled(b), span)) / 
                            (eig.d_scaled(a) - eig.d_scaled(b));
#pragma omp critical(spanQinsert)
            span_Qs.emplace(*it, Q);
        }
    }
    Matrix<adouble> T;
    Matrix<double> Td;
    Eigen::VectorXcd d;
    Eigen::MatrixXcd P, Pinv;
    // std::map<std::pair<int, block_key>, Matrix<double> > span_Qs;
    std::map<std::pair<int, block_key>, Matrix<std::complex<double> > > span_Qs;
    std::map<block_key, eigensystem> eigensystems;

    private:
    const std::vector<std::pair<int, block_key> > targets;
    const std::map<block_key, Vector<adouble> >* emission_probs;
};

#endif

