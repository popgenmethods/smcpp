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
    Q.row(2).setZero();
    Q(2, 2) = T(1.);
    return Q;
}


/*
template <>
Matrix<adouble> matrix_exp_correct_derivatives(adouble c_rho, adouble c_eta)
{
    Matrix<adouble> M = matrix_exp(c_rho, c_eta);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            M(i, j).derivatives() = 
                (M(i, j).derivatives()(1) * c_eta.derivatives()).eval();
    return M;
}


template <typename T>
Matrix<T> stable_matmul(const Matrix<T> &A, const Matrix<T> &B)
{
    assert(A.cols() == B.rows());
    Matrix<T> ret(A.rows(), B.cols());
    std::vector<T> gs(A.cols());
    for (int i = 0; i < A.rows(); ++i)
        for (int j = 0; j < B.cols(); ++j)
        {
            for (int k = 0; k < A.cols(); ++k)
                gs[k] = A(i, k) * B(k, j);
            ret(i, j) = doubly_compensated_summation(gs);
        }
    return ret;
}
*/

template <typename T>
void HJTransition<T>::compute_expms()
{
    typedef typename mpfr_promote<T>::type U;
    mpfr::mpreal::set_default_prec(256);
    std::vector<Matrix<U> > expm_U, expm_prods_U;
    expm_U.push_back(Matrix<U>::Identity(3, 3));
    expm_prods_U.push_back(Matrix<U>::Identity(3, 3));
    const std::vector<double> ts = this->eta.getTs();
    const std::vector<int> hs_indices = this->eta.getHsIndices();
    const std::vector<T> Rrng = this->eta.getRrng();
    for (int i = 1; i < (int)ts.size(); ++i)
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
            U c_eta = mpfr_promote<T>::cast(Rrng[i]);
            c_eta -= mpfr_promote<T>::cast(Rrng[i - 1]);
            U c_rho = 0. * c_eta;
            c_rho += ts[i];
            c_rho -= ts[i - 1];
            c_rho *= this->rho;
            expm_U.push_back(matrix_exp(c_rho, c_eta));
        }
        expm_prods_U.push_back(expm_prods_U.back() * expm_U.back());
    }
    for (int i = 0; i < (int)ts.size(); ++i)
    {
        expms.push_back(mpfr_promote<T>::back_cast(expm_U.at(i)));
        expm_prods.push_back(mpfr_promote<T>::back_cast(expm_prods_U.at(i)));
    }
}

/*
template <typename T>
std::vector<Matrix<T> > HJTransition<T>::compute_expm_prods()
{
    std::vector<Matrix<T> > ret;
    ret.push_back(Matrix<T>::Identity(3, 3));
    for (int i = 1; i < this->eta.getTs().size(); ++i)
        ret.push_back(expm_prods.back() * expms.at(i));
    return ret;
}

template <>
std::vector<Matrix<adouble> > HJTransition<adouble>::compute_expm_prods()
{
    std::vector<Matrix<adouble> > ret;
    const std::vector<double> ts = this->eta.getTs();
    const std::vector<adouble> Rrng = this->eta.getRrng();
    ret.push_back(Matrix<adouble>::Identity(3, 3));
    Matrix<double> jac(ts.size(), eta.getNder());
    jac.setZero();
    for (int i = 1; i < (int)ts.size() - 1; ++i)
    {
        adouble c_eta = Rrng[i] - Rrng[i - 1];
        jac.row(i) = c_eta.derivatives().transpose();
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
            {
                Vector<double> &d = expms.at(i)(j, k).derivatives();
                double dx = d(1);
                d.resize(ts.size());
                d.setZero();
                d(i) = dx;
            }
        ret.push_back(ret.back() * expms.at(i));
    }
    ret.push_back(ret.back() * expms.back());
    // Now clean up all derivatives
    jac.transposeInPlace();
    std::array<std::reference_wrapper<std::vector<Matrix<adouble> > >, 2> v{ret, expms};
    for (int a = 0; a < 2; ++a)
        for (int t = 1; t < (int)ts.size(); ++t)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                {
                    if (t < (int)ts.size() - a)
                    {
                        Matrix<double> d = v[a].get().at(t)(i, j).derivatives();
                        Matrix<double> d2 = stable_matmul(jac, d);
                        v[a].get().at(t)(i, j).derivatives() = d2;
                    }
                }
    return ret;
}
*/

template <typename T>
HJTransition<T>::HJTransition(const PiecewiseConstantRateFunction<T> &eta, const double rho) : 
    Transition<T>(eta, rho) 
{
    const std::vector<double> ts = eta.getTs();
    const std::vector<int> hs_indices = eta.getHsIndices();
    const std::vector<T> Rrng = eta.getRrng();

    // Compute expm matrices in higher precision.
    compute_expms();
    // Prevent issues with mulithreaded access to members causing
    // changes in derivativen coherence.
    std::vector<Matrix<T> > const& expm_prods_const = expm_prods;

    std::vector<T> avg_coal_times = eta.average_coal_times();
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
        // subdiagonal
        // for (int k = 1; k < j; ++k)
        //     this->Phi(j - 1, k - 1) = expm_prods.at(hs_indices.at(k))(0, 2) - 
        //         expm_prods.at(hs_indices.at(k - 1))(0, 2);
        // diagonal element
        // this is an approximation
        // Matrix<T> A = Matrix<T>::Identity(3, 3);
        // for (int ell = hs_indices.at(j - 1); ell < avc_ip.at(j - 1); ++ell)
        //     A = A * expms.at(ell);
        // T delta = avg_coal_times[j - 1] - ts[avc_ip[j - 1]];
        // T c_eta = eta.R(avg_coal_times[j - 1]) - Rrng[avc_ip[j - 1]];
        // T c_rho = 0. * c_eta;
        // c_rho += delta * this->rho;
        // A = A * matrix_exp(c_rho, c_eta);
        // Matrix<T> B = expm_prods[hs_indices[j - 1]] * A;
        // this->Phi(j - 1, j - 1) = B(0, 0);
        // this->Phi(j - 1, j - 1) += expm_prods[hs_indices[j - 1]](0, 0) * A(0, 2);
        // this->Phi(j - 1, j - 1) += expm_prods[hs_indices[j - 1]](0, 1) * A(1, 2);
        // superdiagonal
        for (int k = j + 1; k < this->M; ++k)
        {
            T Rk1 = Rrng[hs_indices[k - 1]];
            T Rk = Rrng[hs_indices[k]];
            T Rj = Rrng[hs_indices[j]];
            T p_coal = exp(-(Rk1 - Rj));
            if (k < this->M - 1)
                p_coal *= -expm1(-(Rk - Rk1));
            this->Phi(j - 1, k - 1) = expm_prods_const[hs_indices[j]](0, 1) * p_coal;
        }
        this->Phi(j - 1, j - 1) = 0.;
        T s = this->Phi.row(j - 1).sum();
        this->Phi(j - 1, j - 1) = 1. - s;

    }
    T small = eta.zero() + 1e-10;
    this->Phi = this->Phi.unaryExpr([small] (const T &x) { if (x < 1e-10) return small; return x; });
    CHECK_NAN(this->Phi);
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
