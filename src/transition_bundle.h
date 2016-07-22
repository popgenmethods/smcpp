#ifndef TRANSITION_BUNDLE_H
#define TRANSITION_BUNDLE_H

#include "common.h"
#include "block_key.h"
#include "ThreadPool.h"

struct eigensystem
{
    Eigen::MatrixXcd P, Pinv;
    Eigen::VectorXcd d;
    Matrix<double> P_r, Pinv_r;
    double scale;
    Eigen::VectorXcd d_scaled;
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
    TransitionBundle(
            const std::set<std::pair<int, block_key> > &targets_s,
            const std::map<block_key, Vector<adouble> >* emission_probs) : 
        targets(targets_s.begin(), targets_s.end()),
        emission_probs(emission_probs), tp(ThreadPool::getInstance()) {}

    void update(const Matrix<adouble> &new_T);
    Matrix<adouble> T;
    Matrix<double> Td;
    Eigen::VectorXcd d;
    Eigen::MatrixXcd P, Pinv;
    std::map<std::pair<int, block_key>, Matrix<std::complex<double> > > span_Qs;
    std::map<block_key, eigensystem> eigensystems;

    private:
    const std::vector<std::pair<int, block_key> > targets;
    const std::map<block_key, Vector<adouble> >* emission_probs;
    ThreadPool &tp;
};

#endif

