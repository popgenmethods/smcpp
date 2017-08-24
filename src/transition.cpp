#include <unsupported/Eigen/MPRealSupport>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/NumericalDiff>
#include "mpreal.h"

#include "transition.h"
#include "piecewise_constant_rate_function.h"

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
            {
                ret(i, j).derivatives().resize(2); 
                ret(i, j).derivatives()(0) = df(3 * j + i, 0);
                ret(i, j).derivatives()(1) = df(3 * j + i, 1);
            }
        return ret;
    }

};

template <typename T>
struct mpfr_promote;

template <>
struct mpfr_promote<adouble>
{
    typedef Eigen::AutoDiffScalar<Eigen::Matrix<mpfr::mpreal, Eigen::Dynamic, 1> > type;
    static type cast(const adouble &x)
    {
        return type(mpfr::mpreal(x.value()), x.derivatives().template cast<mpfr::mpreal>());
    }
    static Matrix<adouble> back_cast(const Matrix<type> &x)
    {
        Matrix<adouble> ret(x.rows(), x.cols());
        for (int i = 0; i < x.rows(); ++i)
            for (int j = 0; j < x.cols(); ++j)
                ret(i, j) = adouble((double)x(i, j).value(), x(i, j).derivatives().template cast<double>());
        return ret;
    }
};

template <>
struct mpfr_promote<double>
{
    typedef mpfr::mpreal type;
    static type cast(const double &x)
    {
        return mpfr::mpreal(x);
    }
    static Matrix<double> back_cast(const Matrix<type> &x)
    {
        return x.template cast<double>();
    }
};

template <typename T>
Matrix<T> matrix_exp(T c_rho, T c_eta)
{
    // return hj_matrix_exp::transition_exp(c_rho, c_eta);
    T sq = sqrt(4 * c_eta * c_eta + c_rho * c_rho);
    T s = sinh(0.5 * sq) / sq;
    T c = cosh(0.5 * sq);
    T e = exp(-c_eta - c_rho / 2.);
    Matrix<T> Q(3, 3);
    Q(0, 0) = e * (c + (2 * c_eta - c_rho) * s);
    Q(0, 1) = 2 * e * c_rho * s;
    Q(0, 2) = 1. - Q(0, 0) - Q(0, 1);
    Q(1, 0) = 2 * e * c_eta * s;
    Q(1, 1) = e * (c - (2 * c_eta - c_rho) * s);
    Q(1, 2) = 1. - Q(1, 0) - Q(1, 1);
    Q.row(2).setZero();
    Q(2, 2) = T(1.);
    return Q;
}


template <typename T>
void HJTransition<T>::compute_expms()
{
    typedef typename mpfr_promote<T>::type U;
    mpfr::mpreal::set_default_prec(256);
    const std::vector<double> ts = this->eta.getTs();
    const std::vector<int> hs_indices = this->eta.getHsIndices();
    const std::vector<T> Rrng = this->eta.getRrng();
    const std::vector<T> ada = this->eta.getAda();
    std::vector<Matrix<U> > expm_U(ts.size(), Matrix<U>::Identity(3, 3));
    std::vector<Matrix<U> > expm_prods_U(ts.size(), Matrix<U>::Identity(3, 3));
    for (int i = hs_indices[0] + 1; i < (int)ts.size(); ++i)
    {
        if (std::isinf(ts[i]))
        {
            Matrix<U> Q(3,3);
            Q.setZero();
            Q.col(2).setOnes();
            expm_U.push_back(Q);
        }
        else
        {
            double delta = ts[i] - ts[i - 1];
            U c_eta = mpfr_promote<T>::cast(ada[i - 1] * delta);
            U c_rho = 0. * c_eta;
            c_rho += delta;
            c_rho *= this->rho;
            expm_U.at(i) = matrix_exp(c_rho, c_eta);
        }
        expm_prods_U.at(i) = expm_prods_U.at(i - 1) * expm_U.at(i);
    }
    for (int i = 0; i < (int)ts.size(); ++i)
    {
        expms.push_back(mpfr_promote<T>::back_cast(expm_U.at(i)));
        expm_prods.push_back(mpfr_promote<T>::back_cast(expm_prods_U.at(i)));
    }
}

template <typename T>
HJTransition<T>::HJTransition(const PiecewiseConstantRateFunction<T> &eta, const double rho) : 
    Transition<T>(eta, rho) 
{
    const std::vector<double> ts = eta.getTs();
    const std::vector<int> hs_indices = eta.getHsIndices();
    const std::vector<T> Rrng = eta.getRrng();
    const std::vector<T> ada = eta.getAda();
    const std::vector<T> avg_coal_times = eta.average_coal_times();

    // Compute expm matrices in higher precision.
    compute_expms();
    // Prevent issues with mulithreaded access to members causing
    // changes in derivative coherence.
    std::vector<Matrix<T> > const& expm_prods_const = expm_prods;

    std::vector<int> avc_ip;
    for (T x : avg_coal_times)
    {
        int ip = std::distance(ts.begin(), std::upper_bound(ts.begin(), ts.end(), toDouble(x))) - 1;
        avc_ip.push_back(ip);
    }
    Vector<T> expm_diff(this->M - 2);
    for (int k = 1; k < this->M - 1; ++k)
        expm_diff(k - 1) = expm_prods.at(hs_indices.at(k))(0, 2) - 
                expm_prods.at(hs_indices.at(k - 1))(0, 2);
    this->Phi.fill(eta.zero());
#pragma omp parallel for
    for (int j = 1; j < this->M; ++j)
    {
        // Important:
        // adouble is not thread safe. Calling a - b where a and b are
        // adoubles actually modifies a and b to have coherent derivatives.
        // This means that different threads must make copies of any variable
        // which is indexed by k.
        this->Phi.row(j - 1).head(j - 1) = expm_diff.head(j - 1);
        const int S = 1;
        for (int s = 0; s < S; ++s)
        {
            // sample a random coalescence time
            T rct = avg_coal_times[j - 1]; // eta.random_time(1.0, hidden_states[j - 1], hidden_states[j], gen);
            int rct_ip = avc_ip[j - 1]; // std::distance(ts.begin(), std::upper_bound(ts.begin(), ts.end(), toDouble(rct))) - 1;
            Matrix<T> A = Matrix<T>::Identity(3, 3);
            for (int ell = hs_indices.at(j - 1); ell < rct_ip; ++ell)
            {
                A = A * expms.at(ell);
            }
            T delta = rct - ts[rct_ip];
            T c_eta = ada[rct_ip] * delta;
            T c_rho = 0. * c_eta;
            c_rho += delta * this->rho;
            A = A * matrix_exp(c_rho, c_eta);
            Matrix<T> B = expm_prods[hs_indices[j - 1]] * A; // this gets us up to avg_coal_time
            T Rj = c_eta;
            for (int jj = rct_ip + 1; jj < j; ++jj)
                Rj += ada[jj] * (ts[jj + 1] - ts[jj]);
            T p_float = B(0, 1) * exp(-Rj);
            // superdiagonal
            for (int k = j + 1; k < this->M; ++k)
            {
                T Rk1 = Rrng[hs_indices[k - 1]];
                T Rk = Rrng[hs_indices[k]];
                T p_coal = exp(-(Rk1 - Rj));
                if (k < this->M - 1)
                    p_coal *= -expm1(-(Rk - Rk1));
                this->Phi(j - 1, k - 1) += p_float * p_coal / S;
            }
        }
        this->Phi(j - 1, j - 1) = 0.;
        T s = this->Phi.row(j - 1).sum();
        this->Phi(j - 1, j - 1) = 1. - s;

    }
    T small = eta.zero() + 1e-20;
    this->Phi = this->Phi.unaryExpr([small] (const T &x) { if (x < 1e-20) return small; return x; });
    CHECK_NAN(this->Phi);
    const double beta = 1e-5;
    T p2 = eta.zero() + beta / this->M;
    Matrix<T> Phi2(this->M, this->M);
    Phi2.fill(p2);
    this->Phi *= (1 - beta);
    this->Phi += Phi2;
}

template <typename T>
Matrix<T> compute_transition(const PiecewiseConstantRateFunction<T> &eta, const double rho)
{
    DEBUG1 << "computing transition";
    Matrix<T> ret = HJTransition<T>(eta, rho).matrix();
    DEBUG1 << "done computing transition";
    return ret;
}

template Matrix<double> compute_transition(const PiecewiseConstantRateFunction<double> &eta, const double rho);
template Matrix<adouble> compute_transition(const PiecewiseConstantRateFunction<adouble> &eta, const double rho);
