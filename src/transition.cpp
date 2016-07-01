#include "transition.h"

template <typename T>
Matrix<T> HJTransition<T>::matrix_exp(T c_rho, T c_eta)
{
    T sq = sqrt(4 * c_eta * c_eta + c_rho * c_rho);
    T s = sinh(0.5 * sq);
    T c = cosh(0.5 * sq);
    T e = exp(-c_eta - c_rho / 2.);
    Matrix<T> Q(3, 3);
    Q(0, 0) = e * (c + (2 * c_eta - c_rho) * s / sq);
    Q(0, 1) = 2 * e * c_rho * s / sq;
    Q(0, 2) = 1. - Q(0, 0) - Q(0, 1);
    Q(1, 0) = 2 * e * c_eta * s / sq;
    Q(1, 1) = e * (c - (2 * c_eta - c_rho) * s / sq);
    Q(1, 2) = 1. - Q(1, 0) - Q(1, 1);
    Q(2, 0) = 0;
    Q(2, 1) = 0;
    Q(2, 2) = 1;
    check_nan(Q);
    return Q;
}

template <typename T>
void HJTransition<T>::compute(void)
{
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    std::vector<Matrix<T> > expms;
    std::vector<Matrix<T> > expm_prods;
    expms.push_back(Matrix<T>::Identity(3, 3));
    expm_prods.push_back(Matrix<T>::Identity(3, 3));
    for (int i = 1; i < eta->ts.size(); ++i)
    {
        if (std::isinf(toDouble(eta->ts[i])))
        {
            Matrix<T> Q(3,3);
            Q << 
                0, 0, 1,
                0, 0, 1,
                0, 0, 1;
            expms.push_back(Q);
        }
        else
        {
            T delta = eta->ts[i] - eta->ts[i - 1];
            T c_rho = delta * this->rho;
            T c_eta = eta->Rrng[i] - eta->Rrng[i - 1];
            expms.push_back(matrix_exp(c_rho, c_eta));
        }
        expm_prods.push_back(expm_prods.back() * expms.back());
    }
    this->Phi.setZero();
    for (int j = 1; j < this->M; ++j)
    {
        for (int k = 1; k < j; ++k)
            this->Phi(j - 1, k - 1) = expm_prods[eta->hs_indices[k]](0, 2) - expm_prods[eta->hs_indices[k - 1]](0, 2);
        if (j == this->M - 1)
            this->Phi(j - 1, j - 1) = 1. - expm_prods[eta->hs_indices[j - 1]](0, 2);
        else 
        {
            this->Phi(j - 1, j - 1) = expm_prods[eta->hs_indices[j]](0, 0);
            Matrix<T> A = Matrix<T>::Identity(3, 3);
            for (int ell = eta->hs_indices[j - 1]; ell < eta->hs_indices[j]; ++ell)
                A = A * expms[ell];
            this->Phi(j - 1, j - 1) += expm_prods[eta->hs_indices[j - 1]](0, 0) * A(0, 2);
            this->Phi(j - 1, j - 1) += expm_prods[eta->hs_indices[j - 1]](0, 1) * A(1, 2);
        }
        for (int k = j + 1; k < this->M; ++k)
        {
            T p_coal = exp(-(eta->Rrng[eta->hs_indices[k - 1]] - eta->Rrng[eta->hs_indices[j]]));
            if (k < this->M - 1)
                p_coal *= -expm1(-(eta->Rrng[eta->hs_indices[k]] - eta->Rrng[eta->hs_indices[k - 1]]));
            this->Phi(j - 1, k - 1) = expm_prods[eta->hs_indices[j]](0, 1) * p_coal;
        }
    }
    T thresh = 1e-20 * eta->one;
    this->Phi = this->Phi.unaryExpr([thresh] (const T &x) { if (x < thresh) return thresh; return x; });
    check_nan(this->Phi);
}

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &eta, const double rho)
{
    DEBUG << "computing transition";
    Matrix<T> ret = HJTransition<T>(eta, rho).matrix();
    DEBUG << "done computing transition";
    return ret;
}

template Matrix<double> compute_transition(const PiecewiseExponentialRateFunction<double> &eta, const double rho);
template Matrix<adouble> compute_transition(const PiecewiseExponentialRateFunction<adouble> &eta, const double rho);
