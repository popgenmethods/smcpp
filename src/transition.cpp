#include "transition.h"

namespace smc_prime_transition
{
    int depth0 = 8;
    template <typename T>
    struct trans_integrand_helper
    {
        const int i, j;
        const PiecewiseExponentialRateFunction<T> *eta;
        const double left, right;
        const T log_denom;
    };

    template <typename T>
    struct p_intg_helper
    {
        const PiecewiseExponentialRateFunction<T>* eta;
        const double rho, left, right;
        const T log_denom;
    };

    template <typename T>
    T p_integrand(const double x, p_intg_helper<T> *pih)
    {
        double v, jac;
        if (pih->right < INFINITY)
        {
            v = x * pih->right + (1. - x) * pih->left;
            jac = pih->right - pih->left;
        }
        else
        {
            v = pih->left + x / (1. - x);
            jac = 1. / (1. - x) / (1. - x);
        }
        if (v == INFINITY)
            return 0.;
        T ret = pih->eta->eta(v) * exp(-(pih->eta->R(v) - pih->log_denom + pih->rho * v)) * jac;
        check_nan(ret);
        return ret;
    }

    template <typename T>
    T trans_integrand(const double x, trans_integrand_helper<T> *tih)
    {
        const int i = tih->i, j = tih->j; 
        double h, jac;
        if (tih->right < INFINITY)
        {
            h = x * tih->right + (1. - x) * tih->left;
            jac = tih->right - tih->left;
        }
        else
        {
            h = tih->left + x / (1. - x);
            jac = 1. / (1. - x) / (1. - x);
        }
        const PiecewiseExponentialRateFunction<T> *eta = tih->eta;
        const std::vector<double> &t = eta->hidden_states;
        if (h == 0)
            return eta->eta(0) * (exp(-eta->R(t[j - 1])) - exp(-eta->R(t[j])));
        T Rh = eta->R(h);
        T eRh = exp(-(Rh + tih->log_denom));
        T ret = eta->zero, tmp;
        // f1
        T htj_min = dmin(h, t[j]);
        T htj_max = dmax(h, t[j]);
        if (h < t[j])
        {
            tmp = eta->R_integral(h, -2 * Rh, 2) * (exp(-(eta->R(dmax(h, t[j - 1])) + tih->log_denom)) - 
                    exp(-(eta->R(t[j]) + tih->log_denom)));
            check_negative(tmp);
            ret += tmp;
            check_nan(ret);
        }
        // f2
        if (h > t[j - 1])
        {
            T r1 = eta->R_integral(t[j - 1], -2 * eta->R(t[j - 1]) - Rh - tih->log_denom, 2);
            T r2 = eta->R_integral(htj_min, -2 * eta->R(htj_min) - Rh - tih->log_denom, 2);
            T r3 = eRh * (htj_min - t[j - 1]); 
            tmp = r1 - r2 + r3;
            check_negative(tmp);
            ret += 0.5 * tmp;
            check_nan(ret);
        }
        if (i == j)
        {
            tmp = 0.5 * (eRh * h - eta->R_integral(h, -3 * Rh - tih->log_denom, 2));
            check_negative(tmp);
            ret += tmp;
            check_nan(ret);
        }
        // ret = f1 + f2 + f3;
        ret *= eta->eta(h) / h;
        ret *= jac;
        check_nan(ret);
        if (ret < 0)
            throw std::domain_error("negative value of positive integral");
        return ret;
    }
};

template <typename T>
T SMCPrimeTransition<T>::P_no_recomb(const int i)
{
    using namespace smc_prime_transition;
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    std::vector<double> t = eta->hidden_states;
    T log_denom = eta->R(t[i - 1]);
    p_intg_helper<T> h = {eta, 2. * this->rho, t[i - 1], t[i], log_denom};
    T ret, more_denom;
    if (t[i] < INFINITY)
        more_denom = -expm1(-(eta->R(t[i]) - eta->R(t[i - 1])));
    else
        more_denom = eta->one;
    check_nan(more_denom);
    int depth = 1024;
    double tol = 1e-10;
    if (false) //(rho < 1e-3)
    {
        ret = eta->R_integral(t[i]) - eta->R_integral(t[i - 1]);
        ret -= t[i] * exp(-eta->R(t[i])) - t[i - 1] * exp(-eta->R(t[i - 1]));
        ret *= this->rho;
        ret /= exp(-eta->R(t[i - 1])) - exp(-eta->R(t[i]));
        ret = 1. - ret;
    } 
    else
    {
        // slow.
        do {
            ret = adaptiveSimpsons(std::function<T(const double, p_intg_helper<T>*)>(p_integrand<T>), &h, 0., 1., tol, depth);
            check_nan(ret);
            ret /= more_denom;
            check_nan(ret);
            depth *= 2;
            if (depth > 2048)
                PROGRESS("P_nr_recomb at " << depth << " nodes");
        } while (ret >= 1);
    }
    // Alternative approach for rho << 1
    check_nan(ret);
    check_negative(ret);
    check_negative(1. - ret);
    return ret;
}


template <typename T>
T SMCPrimeTransition<T>::trans(int i, int j)
{
    using namespace smc_prime_transition;
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    const std::vector<double> t = eta->hidden_states;
    // T log_denom = eta->R(t[i - 1]);
    T log_denom = eta->zero;
    trans_integrand_helper<T> tih = {i, j, eta, t[i - 1], t[i], log_denom};
    int depth = 256;
    double tol = 1e-10;
    T ret;
    T denom = exp(-eta->R(t[i - 1]));
    if (t[i] < INFINITY)
        denom -= exp(-eta->R(t[i]));
    ret = adaptiveSimpsons(std::function<T(const double, trans_integrand_helper<T>*)>(trans_integrand<T>),
            &tih, 0., 1., tol, depth);
    ret /= denom;
    check_negative(ret);
    check_nan(ret);
    if (ret > 1 or ret < 0)
    {
        std::cout << ret << " " << denom << std::endl;
        throw std::domain_error("ret is not a probability");
    }
    return ret;
}

template <typename T>
void SMCPrimeTransition<T>::compute(void)
{
    Matrix<double> rt(this->M - 1, this->M - 1);
    PROGRESS("transition");
#pragma omp parallel for
    for (int i = 1; i < this->M; ++i)
    {
        T pnr = P_no_recomb(i);
        for (int j = 1; j < this->M; ++j)
        {
            T tr = trans(i, j);
            rt(i - 1, j - 1) = toDouble(tr);
            this->Phi(i - 1, j - 1) = (1. - pnr) * tr;
            if (i == j)
                this->Phi(i - 1, j - 1) += pnr;
            check_nan(this->Phi(i - 1, j - 1));
            check_negative(this->Phi(i - 1, j - 1));
            // if (Phi(i - 1, j - 1) < 1e-20)
            // {
            //     std::cout << "really small phi: " << i << " " << j << " " << Phi(i - 1, j - 1) << std::endl;
            //     Phi(i - 1, j - 1) = 1e-20;
            // }
        }
    }
}

namespace hj_transition
{
    const double A_rho_data[] = {
        -2, 2, 0, 0,
         0, -1, 1, 0,
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
};

template <typename T>
void HJTransition<T>::compute(void)
{
    using namespace hj_transition;
    const PiecewiseExponentialRateFunction<T>* eta = this->eta;
    auto R = eta->getR();
    T r, p_coal;
    for (int j = 1; j < this->M; ++j)
        for (int k = 1; k < this->M; ++k)
        {
            if (k < j)
            {
                r = expm(0, k)(0, 3) - expm(0, k - 1)(0, 3);
            }
            else if (k == j && eta->hidden_states[j] == INFINITY)
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
                p_coal = exp(-((*R)(eta->hidden_states[k - 1]) - (*R)(eta->hidden_states[j])));
                if (eta->hidden_states[k] < INFINITY)
                {
                    // Else d[k] = +inf, coalescence in [d[k-1], +oo) is assured.
                    p_coal *= -expm1(-((*R)(eta->hidden_states[k]) - (*R)(eta->hidden_states[k - 1])));
                }
                double c_rho = this->rho * eta->hidden_states[j];
                T c_eta = (*R)(eta->hidden_states[j]);
                Matrix<T> em = transition_exp(c_rho, c_eta);
                r = (em(0, 1) + em(0, 2)) * p_coal;
            }
            this->Phi(j - 1, k - 1) = r;
            if (this->Phi(j - 1, k - 1) < 1e-16)
                std::cout << "phi is tiny" << j << " " << k << " " << this->Phi(j - 1, k - 1) << std::endl;
            // Phi(j - 1, k - 1) = dmax(r, 1e-16);
        }
}

template <typename T>
void HJTransition<T>::compute_hs_midpoints()
{
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    hs_midpoints.clear();
    for (int m = 1; m < this->M; ++m)
    {
        // int_t[m - 1]^t[m] t * eta(t) exp(-R(t)) dt = 
        // int_t[m - 1]^t[m] exp(-R(t)) dt - t * exp(-R(t)) |_t[m-1]^t[m]
        T tm1 = eta->hidden_states[m - 1];
        T tm = eta->hidden_states[m];
        T R0 = eta->R_integral(tm1);
        T R1 = eta->R_integral(tm);
        hs_midpoints.push_back((R1 - R0) - (tm * exp(-eta->R(tm)) - tm1 * exp(-eta->R(tm1))));
        hs_midpoints.back() /= exp(-eta->R(tm1)) - exp(-eta->R(tm));
    }
}

template <typename T>
Matrix<T> HJTransition<T>::expm(int i, int j)
{
    using namespace hj_transition;
    const PiecewiseExponentialRateFunction<T> *eta = this->eta;
    auto R = eta->getR();
    const int M = this->M;
    std::pair<int, int> key = {i, j};
    if (_expm_memo.count(key) == 0)
    {
        double c_rho;
        T c_eta;
        Matrix<T> ret(M, M);
        if (i == j)
            ret = Matrix<T>::Identity(M, M);
        else
        {
            c_rho = this->rho * (eta->hidden_states[j] - eta->hidden_states[i]);
            c_eta = (*R)(eta->hidden_states[j]) - (*R)(eta->hidden_states[i]);
            ret = transition_exp(c_rho, c_eta);
        }
        _expm_memo[key] = ret;
    }
    return _expm_memo[key];
}

bool useHJ = true;
void setHJ(bool b) { useHJ = b; }

template <typename T>
Matrix<T> compute_transition(const PiecewiseExponentialRateFunction<T> &eta, const double rho)
{
    if (useHJ)
        return HJTransition<T>(eta, rho).matrix();
    else
        return SMCPrimeTransition<T>(eta, rho).matrix();
}

template Matrix<double> compute_transition(const PiecewiseExponentialRateFunction<double> &eta, const double rho);
template Matrix<adouble> compute_transition(const PiecewiseExponentialRateFunction<adouble> &eta, const double rho);
