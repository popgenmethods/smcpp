#include "transition.h"

const double A_rho_data[] = {
    -1, 1, 0, 0,
     0, -.5, .5, 0,
     0, 0, 0, 0,
     0, 0, 0, 0};
static Eigen::Matrix<double, 4, 4, Eigen::RowMajor> A_rho(A_rho_data);

const double A_eta_data[] = {
    0, 0, 0, 0,
    1, -2, 0, 1,
    0, 4, -5, 1,
    0, 0, 0, 0};
static Eigen::Matrix<double, 4, 4, Eigen::RowMajor> A_eta(A_eta_data);

Matrix<double> transition_exp(double c_rho, double c_eta)
{
    Matrix<double> M = c_rho * A_rho + c_eta * A_eta;
    return M.exp();
}

struct expm_functor
{
    typedef double Scalar;
    typedef Eigen::Matrix<Scalar, 1, 1> InputType;
    typedef Eigen::Matrix<Scalar, 16, 1> ValueType;
    typedef Eigen::Matrix<Scalar, 16, 1> JacobianType;

    static const int InputsAtCompileTime = 1;
    static const int ValuesAtCompileTime = 16;

    static int values() { return 16; }

    expm_functor(double c_rho) : c_rho(c_rho) {}
    int operator()(const InputType &x, ValueType &f) const
    {
        Eigen::Matrix<double, 4, 4, Eigen::ColMajor> M = transition_exp(c_rho, x(0,0));
        f = Eigen::Matrix<double, 16, 1, Eigen::ColMajor>::Map(M.data(), 16, 1);
        return 0;
    }
    double c_rho; 
};

Matrix<adouble> transition_exp(double c_rho, adouble c_eta)
{
    // Compute derivative dependence on c_eta by numerical differentiation
    expm_functor f(c_rho);
    Eigen::NumericalDiff<expm_functor> numDiff(f);
    Eigen::Matrix<double, 16, 1> df;
    Eigen::Matrix<double, 1, 1> meta;
    meta(0, 0) = c_eta.value();
    numDiff.df(meta, df);
    Matrix<adouble> ret = transition_exp(c_rho, c_eta.value()).cast<adouble>();
    Eigen::Matrix<double, 4, 4, Eigen::ColMajor> ddf = Eigen::Matrix<double, 4, 4, Eigen::ColMajor>::Map(df.data(), 4, 4);
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j)
            ret(i, j).derivatives() = c_eta.derivatives() * ddf(i, j);
    return ret;
}

template <typename T>
Transition<T>::Transition(const PiecewiseExponentialRateFunction<T> &eta, const std::vector<double> &hidden_states, double rho) :
    eta(&eta), _hs(hidden_states), rho(rho), M(hidden_states.size()), I(4, 4), Phi(M - 1, M - 1)
{
    I.setIdentity();
    Phi.setZero();
    compute();
}

template <typename T>
void Transition<T>::compute(void)
{
    auto R = eta->getR();
    T r, p_coal;
    for (int j = 1; j < M; ++j)
        for (int k = 1; k < M; ++k)
        {
            if (k < j)
            {
                r = expm(0, k)(0, 3) - expm(0, k - 1)(0, 3);
            }
            else if (k == j && j == M - 1)
            {
                r = 1. - expm(0, k - 1)(0, 3);
            }
            else if (k == j)
            {
                r = expm(0, k)(0, 0);
                for (int i = 0; i < 3; ++i)
                    r += expm(0, k - 1)(0, i) * expm(k - 1, k)(i, 3);
            }
            else
            {
                p_coal = exp(-((*R)(_hs[k - 1]) - (*R)(_hs[j])));
                if (k < M - 1)
                {
                    // Else d[k] = +inf, coalescence in [d[k-1], +oo) is assured.
                    p_coal *= -expm1(-((*R)(_hs[k]) - (*R)(_hs[k - 1])));
                }
                r = (expm(0, j)(0, 1) + expm(0, j)(0, 2)) * p_coal;
            }
            Phi(j - 1, k - 1) = dmax(r, 1e-16);
        }
    // Normalize because small errors might throw off the fwd-backward algorithm later on
    // Phi = Phi.array().colwise() / Phi.array().rowwise().sum();
}

template <typename T>
Matrix<T>& Transition<T>::matrix(void) { return Phi; }

/*
template <typename T>
void Transition<T>::store_results(double* outtrans, double* outjac)
{
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _trans(outtrans, M - 1, M - 1);
    _trans = Phi.cast<double>();
    int m = 0;
    for (int i = 0; i < M - 1; ++i)
        for (int j = 0; j < M - 1; ++j)
        {
            Eigen::VectorXd d = Phi(i, j).derivatives();
            // printf("i:%i j:%i dtransition/da[0]:%f\n", i, j, d(0));
            for (int k = 0; k < d.rows(); ++k)
                outjac[m++] = d(k);
        }
}
*/

template <typename T>
Matrix<T> Transition<T>::expm(int i, int j)
{
    auto R = eta->getR();
    std::pair<int, int> key = {i, j};
    if (_expm_memo.count(key) == 0)
    {
        double c_rho;
        T c_eta;
        Matrix<T> ret(M, M);
        if (i == j)
            ret = I;
        else
        {
            c_rho = rho * (_hs[j] - _hs[i]);
            c_eta = (*R)(_hs[j]) - (*R)(_hs[i]);
            /*
            AdMatrix A = c_rho * A_rho.cast<adouble>() + c_eta * A_eta.cast<adouble>();
            Eigen::HouseholderQR<AdMatrix> qr(A);
            AdMatrix Q = qr.householderQ();
            AdMatrix R = qr.matrixQR().block(0,0,A.cols(),A.cols()).triangularView<Eigen::Upper>();
            std::cout << Q.cast<double>() << std::endl;
            std::cout << R.cast<double>() << std::endl;
            std::cout << (Q * R).cast<double>() << std::endl;
            std::cout << A.cast<double>() << std::endl;
            */
            ret = transition_exp(c_rho, c_eta);
        }
        _expm_memo[key] = ret;
    }
    return _expm_memo[key];
}

void store_transition(const Matrix<double> &trans, double* outtrans)
{
    int M = trans.cols();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> _outtrans(outtrans, M, M);
    _outtrans = trans;
}

void store_transition(const Matrix<adouble> &trans, double* outtrans, double* outjac)
{
    store_transition(trans.cast<double>(), outtrans);
    int M = trans.cols();
    int num_derivatives = trans(0,0).derivatives().rows();
    Eigen::VectorXd d;
    int m = 0;
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < M; ++j)
        {
            d = trans(i, j).derivatives();
            assert(d.rows() == num_derivatives);
            for (int k = 0; k < num_derivatives; ++k)
                outjac[m++] = d(k);
        }
}

void cython_calculate_transition(const std::vector<std::vector<double>> params,
        const std::vector<double> hidden_states, double rho, double* outtrans)
{
    RATE_FUNCTION<double> eta(params);
    Matrix<double> trans = compute_transition(eta, hidden_states, rho);
    store_transition(trans, outtrans);
}

void cython_calculate_transition_jac(const std::vector<std::vector<double>> params,
        const std::vector<double> hidden_states, double rho, double* outtrans, double* outjac)
{
    RATE_FUNCTION<adouble> eta(params);
    Matrix<adouble> trans = compute_transition(eta, hidden_states, rho);
    store_transition(trans, outtrans, outjac);
}


template class Transition<double>;
template class Transition<adouble>;

