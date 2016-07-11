#include "transition.h"

template <typename T, typename U>
Matrix<T> matrix_exp(U c_rho, T c_eta)
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
HJTransition<T>::HJTransition(const PiecewiseConstantRateFunction<T> &eta, const double rho) : Transition<T>(eta, rho)
{
    std::vector<T> avg_coal_times = eta.average_coal_times();
    std::vector<Matrix<T> > expms;
    std::vector<Matrix<T> > expm_prods;
    expms.push_back(Matrix<T>::Identity(3, 3));
    expm_prods.push_back(Matrix<T>::Identity(3, 3));
    for (int i = 1; i < eta.ts.size(); ++i)
    {
        if (std::isinf(eta.ts[i]))
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
            double delta = eta.ts[i] - eta.ts[i - 1];
            double c_rho = delta * this->rho;
            T c_eta = eta.Rrng[i] - eta.Rrng[i - 1];
            expms.push_back(matrix_exp(c_rho, c_eta));
        }
        expm_prods.push_back(expm_prods.back() * expms.back());
    }

    std::vector<int> avc_ip;
    for (T x : avg_coal_times)
        avc_ip.push_back(insertion_point(toDouble(x), eta.ts, 0, eta.ts.size()));

    this->Phi.setZero();
    for (int j = 1; j < this->M; ++j)
    {
        // subdiagonal
        for (int k = 1; k < j; ++k)
            this->Phi(j - 1, k - 1) = expm_prods[eta.hs_indices[k]](0, 2) - expm_prods[eta.hs_indices[k - 1]](0, 2);
        // diagonal element
        // this is an approximation
        Matrix<T> A = Matrix<T>::Identity(3, 3);
        for (int ell = eta.hs_indices[j - 1]; ell < avc_ip[j - 1]; ++ell)
            A = A * expms[ell];
        T delta = avg_coal_times[j - 1] - eta.ts[avc_ip[j - 1]];
        T c_rho = delta * this->rho;
        T c_eta = eta.R(avg_coal_times[j - 1]) - eta.Rrng[avc_ip[j - 1]];
        A = A * matrix_exp(c_rho, c_eta);
        Matrix<T> B = expm_prods[eta.hs_indices[j - 1]] * A;
        this->Phi(j - 1, j - 1) = B(0, 0);
        this->Phi(j - 1, j - 1) += expm_prods[eta.hs_indices[j - 1]](0, 0) * A(0, 2);
        this->Phi(j - 1, j - 1) += expm_prods[eta.hs_indices[j - 1]](0, 1) * A(1, 2);
        // superdiagonal
        for (int k = j + 1; k < this->M; ++k)
        {
            T p_coal = exp(-(eta.Rrng[eta.hs_indices[k - 1]] - eta.Rrng[eta.hs_indices[j]]));
            if (k < this->M - 1)
                p_coal *= -expm1(-(eta.Rrng[eta.hs_indices[k]] - eta.Rrng[eta.hs_indices[k - 1]]));
            this->Phi(j - 1, k - 1) = expm_prods[eta.hs_indices[j]](0, 1) * p_coal;
        }
    }
    T thresh(1e-20);
    this->Phi = this->Phi.unaryExpr([thresh] (const T &x) { if (x < thresh) return thresh; return x; });
    check_nan(this->Phi);
}

template <typename T>
Matrix<T> compute_transition(const PiecewiseConstantRateFunction<T> &eta, const double rho)
{
    DEBUG << "computing transition";
    Matrix<T> ret = HJTransition<T>(eta, rho).matrix();
    DEBUG << "done computing transition";
    return ret;
}

template Matrix<double> compute_transition(const PiecewiseConstantRateFunction<double> &eta, const double rho);
template Matrix<adouble> compute_transition(const PiecewiseConstantRateFunction<adouble> &eta, const double rho);
