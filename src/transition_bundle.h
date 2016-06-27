#ifndef TRANSITION_BUNDLE_H
#define TRANSITION_BUNDLE_H

#include "common.h"

struct eigensystem
{
    Eigen::MatrixXcd P, Pinv;
    Eigen::VectorXcd d;
    Matrix<double> P_r, Pinv_r;
    Vector<double> d_r;
    double scale;
    Vector<double> d_r_scaled;
    eigensystem(const Eigen::EigenSolver<Matrix<double> > &es) :
        P(es.eigenvectors()), Pinv(P.inverse()), d(es.eigenvalues()),
        P_r(P.real()), Pinv_r(Pinv.real()), d_r(d.real()),
        scale(d_r.maxCoeff()), d_r_scaled(d_r / scale)
    {
        double i1 = Pinv.imag().cwiseAbs().maxCoeff();
        double i2 = P.imag().cwiseAbs().maxCoeff();
        if (i1 > 1e-10 or i2 > 1e-10)
            WARNING << "Non-negligible imaginary part of eigendecomposition: i1=" << i1 << " i2=" << i2;
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
            tmp = Matrix<double>::Zero(M, M);
            for (int a = 0; a < M; ++a)
                for (int b = 0; b < M; ++b)
                    tmp(a, b) = (a == b) ? (double)span * std::pow(eig.d_r_scaled(a), span - 1) : 
                        (std::pow(eig.d_r_scaled(a), span) - std::pow(eig.d_r_scaled(b), span)) / 
                        (eig.d_r_scaled(a) - eig.d_r_scaled(b));
#pragma omp critical(spanQinsert)
            span_Qs.emplace(*it, tmp);
        }
    }
    Matrix<adouble> T;
    Matrix<double> Td;
    Eigen::VectorXcd d;
    Eigen::MatrixXcd P, Pinv;
    std::map<std::pair<int, block_key>, Matrix<double> > span_Qs;
    std::map<block_key, eigensystem> eigensystems;

    private:
    const std::vector<std::pair<int, block_key> > targets;
    const std::map<block_key, Vector<adouble> >* emission_probs;
};

#endif

