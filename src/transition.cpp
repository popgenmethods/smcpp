#include "transition.h"

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
T Transition<T>::P_no_recomb(const int i)
{
    std::vector<double> t = eta->hidden_states;
    T log_denom = eta->R(t[i - 1]);
    p_intg_helper<T> h = {eta, 2. * rho, t[i - 1], t[i], log_denom};
    T ret, more_denom;
    if (t[i] < INFINITY)
        more_denom = -expm1(-(eta->R(t[i]) - eta->R(t[i - 1])));
    else
        more_denom = eta->one;
    check_nan(more_denom);
    int depth = 128;
    double tol = 1e-10;
    do {
        ret = adaptiveSimpsons(std::function<T(const double, p_intg_helper<T>*)>(p_integrand<T>), &h, 0., 1., tol, depth);
        check_nan(ret);
        ret /= more_denom;
        check_nan(ret);
        depth *= 2;
        if (depth > 256)
            PROGRESS("P_nr_recomb at " << depth << " nodes");
    } while (ret >= 1);
    check_nan(ret);
    check_negative(ret);
    check_negative(1. - ret);
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
        tmp = eta->R_integral(h, -2 * Rh) * (exp(-(eta->R(dmax(h, t[j - 1])) + tih->log_denom)) - 
                exp(-(eta->R(t[j]) + tih->log_denom)));
        check_negative(tmp);
        ret += tmp;
        check_nan(ret);
    }
    // f2
    if (h > t[j - 1])
    {
        T r1 = eta->R_integral(t[j - 1], -2 * eta->R(t[j - 1]) - Rh - tih->log_denom);
        T r2 = eta->R_integral(htj_min, -2 * eta->R(htj_min) - Rh - tih->log_denom);
        T r3 = eRh * (htj_min - t[j - 1]); 
        tmp = r1 - r2 + r3;
        check_negative(tmp);
        ret += 0.5 * tmp;
        check_nan(ret);
    }
    if (i == j)
    {
        tmp = 0.5 * (eRh * h - eta->R_integral(h, -3 * Rh - tih->log_denom));
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

template <typename T>
T Transition<T>::trans(int i, int j)
{
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
        throw std::domain_error("ret is not a probability");
    return ret;
}


template <typename T>
Transition<T>::Transition(const PiecewiseExponentialRateFunction<T> &eta, const double rho) :
    eta(&eta), M(eta.hidden_states.size()), Phi(M - 1, M - 1), rho(rho)
{
    Phi.setZero();
    compute();
}

template <typename T>
void Transition<T>::compute(void)
{
    Matrix<double> rt(M - 1, M - 1);
    PROGRESS("transition");
#pragma omp parallel for
    for (int i = 1; i < M; ++i)
    {
        T pnr = P_no_recomb(i);
        for (int j = 1; j < M; ++j)
        {
            T tr = trans(i, j);
            rt(i - 1, j - 1) = toDouble(tr);
            Phi(i - 1, j - 1) = (1. - pnr) * tr;
            if (i == j)
                Phi(i - 1, j - 1) += pnr;
            check_nan(Phi(i - 1, j - 1));
            check_negative(Phi(i - 1, j - 1));
            // if (Phi(i - 1, j - 1) < 1e-20)
            // {
            //     std::cout << "really small phi: " << i << " " << j << " " << Phi(i - 1, j - 1) << std::endl;
            //     Phi(i - 1, j - 1) = 1e-20;
            // }
        }
    }
    // std::cout << "rt\n" << rt << std::endl;
}

template <typename T>
Matrix<T>& Transition<T>::matrix(void) { return Phi; }

template class Transition<double>;
template class Transition<adouble>;

int transition_main(int argc, char** argv)
{
    std::vector<std::vector<double> > params = {
        {0.5, 2.0, 1.0},
        {5.0, 0.2, 1.0},
        {0.2, 1.0, 1.0}
    };
    std::vector<double> hs = {0.0, 0.5, 1.0, 2.0, 20.0};
    std::vector<std::pair<int, int> > deriv = { {0,0} };
    double rho = 4 * 1e4 * 1e-9;
    PiecewiseExponentialRateFunction<adouble> eta(params, deriv, hs);
    params[0][0] += 1e-8;
    PiecewiseExponentialRateFunction<double> eta2(params, deriv, hs);
    Transition<adouble> T(eta, rho);
    Transition<double> T2(eta2, rho);
    Matrix<adouble> M = T.matrix();
    std::cout << M.template cast<double>() << std::endl << std::endl;
    std::cout << M.unaryExpr([=](adouble x) { 
            if (x.derivatives().size() == 0) return 0.; 
            return x.derivatives()(0); }).template cast<double>() << std::endl << std::endl;
    Matrix<double> M2 = T2.matrix();
    std::cout << (M2 - M.template cast<double>()) * 1e8 << std::endl << std::endl;
}
