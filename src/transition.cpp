#include "transition.h"

namespace hj_matrix_exp {
    const double A_rho_data[] = {
        -1, 1, 0,
         0, 0, 0,
         0, 0, 0};
    static Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A_rho(A_rho_data);

    const double A_eta_data[] = {
        0, 0, 0,
        1, -2, 1,
        0, 0, 0};
    static Eigen::Matrix<double, 3, 3, Eigen::RowMajor> A_eta(A_eta_data);

    template <typename T>
    Matrix<T> transition_exp(T c_rho, T c_eta);

    template <>
    Matrix<double> transition_exp(double c_rho, double c_eta)
    {
        Matrix<double> M = c_rho * A_rho + c_eta * A_eta;
        return M.exp();
    }

    struct expm_functor
    {
        typedef double Scalar;
        typedef Eigen::Matrix<Scalar, 1, 2> InputType;
        typedef Eigen::Matrix<Scalar, 9, 1> ValueType;
        typedef Eigen::Matrix<Scalar, 9, 2> JacobianType;

        static const int InputsAtCompileTime = 1;
        static const int ValuesAtCompileTime = 9;

        static int values() { return 9; }

        expm_functor() {}
        int operator()(const InputType &x, ValueType &f) const
        {
            Eigen::Matrix<double, 3, 3, Eigen::ColMajor> M = transition_exp(x(0), x(1));
            f = Eigen::Matrix<double, 9, 1, Eigen::ColMajor>::Map(M.data(), 9, 1);
            return 0;
        }
    };

    template <>
    Matrix<adouble> transition_exp(adouble c_rho, adouble c_eta)
    {
        // Compute derivative dependence on c_eta by numerical differentiation
        expm_functor f;
        Eigen::NumericalDiff<expm_functor> numDiff(f);
        Eigen::Matrix<double, 9, 2> df;
        Eigen::Matrix<double, 1, 2> meta;
        meta(0, 0) = c_rho.value();
        meta(0, 1) = c_eta.value();
        numDiff.df(meta, df);
        Matrix<adouble> ret = transition_exp(c_rho.value(), c_eta.value()).cast<adouble>();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                ret(i, j).derivatives() = c_rho.derivatives() * df(3 * j + i, 0) + 
                    c_eta.derivatives() * df(3 * j + i, 1);
        return ret;
    }

};

template <typename T>
Matrix<T> HJTransition<T>::matrix_exp(T c_rho, T c_eta)
{
    return hj_matrix_exp::transition_exp(c_rho, c_eta);
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
        T delta = eta->ts[i] - eta->ts[i - 1];
        T c_rho = delta * this->rho;
        T c_eta = eta->Rrng[i] - eta->Rrng[i - 1];
        expms.push_back(matrix_exp(c_rho, c_eta));
        expm_prods.push_back(expm_prods.back() * expms.back());
    }
    this->Phi.setZero();
    for (int j = 1; j < this->M; ++j)
    {
        for (int k = 1; k < j; ++k)
            this->Phi(j - 1, k - 1) = expm_prods[eta->hs_indices[k]](0, 2) - expm_prods[eta->hs_indices[k - 1]](0, 2);
        if (j == this->M - 1)
            this->Phi(j - 1, j - 1) = 1. - expm_prods[eta->hs_indices[j]](0, 2);
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
    // T thresh = 1e-20 * eta->one;
    // this->Phi = this->Phi.unaryExpr([thresh] (const T &x) { if (x < thresh) return thresh; return x; });
    // check_nan(this->Phi);
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
