#include "piecewise_exponential_rate_function.h"

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params,
        const std::vector<double> hidden_states) : 
    PiecewiseExponentialRateFunction(params, std::vector<std::pair<int, int>>(), hidden_states) {}

std::vector<std::pair<int, int>> derivatives_from_params(const std::vector<std::vector<double>> params)
{
    std::vector<std::pair<int, int>> ret;
    for (size_t i = 0; i < params.size(); ++i)
        for (size_t j = 0; j < params[0].size(); ++j)
            ret.emplace_back(i, j);
    return ret;
} 

template <>
PiecewiseExponentialRateFunction<adouble>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params, 
        const std::vector<double> hidden_states) :
    PiecewiseExponentialRateFunction(params, derivatives_from_params(params), hidden_states) {}

template <>
adouble PiecewiseExponentialRateFunction<adouble>::init_derivative(double x)
{
    return adouble(x, Vector<double>::Zero(derivatives.size()));
}

template <>
double PiecewiseExponentialRateFunction<double>::init_derivative(double x)
{
    return x;
}

template <typename T>
inline void vec_insert(std::vector<T> &v, const int pos, const T &x)
{
    v.insert(v.begin() + pos, x);
}

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params, 
        const std::vector<std::pair<int, int>> derivatives,
        const std::vector<double> hidden_states) :
    params(params),
    derivatives(derivatives), K(params[0].size()), ada(params[0].begin(), params[0].end()), 
    adb(params[1].begin(), params[1].end()), ads(params[2].begin(), params[2].end()),
    ts(K + 1), Rrng(K), _reg(0.0), 
    zero(init_derivative(0.0)), one(init_derivative(1.0)),
    hidden_states(hidden_states)
{
    // Final piece is required to be flat.
    T adatmp;
    ts[0] = 0;
    Rrng[0] = 0;
    // These constant values need to have compatible derivative shape
    // with the calculated values.
    initialize_derivatives();
    for (int k = 0; k < K; ++k)
    {
        adb[k] = ada[k];
        ada[k] = 1. / ada[k];
        adb[k] = 1. / adb[k];
        adb[k] *= 0.0;
        ts[k + 1] = ts[k] + ads[k];
        // adb[k] = 0.0 * (log(adb[k]) - log(ada[k])) / (ts[k + 1] - ts[k]);
    }
    adb[K - 1] *= 0.0;
    ts[K] *= INFINITY;

    int ip;
    for (double h : hidden_states)
    {
        ip = insertion_point(h, ts, 0, ts.size());
        if (ts[ip] == h)
            hs_indices.push_back(ip);
        else
        {
            vec_insert<T>(ts, ip + 1, (T)h);
            vec_insert<T>(ada, ip + 1, ada[ip]);
            vec_insert<T>(adb, ip + 1, adb[ip]);
            hs_indices.push_back(ip + 1);
        }
    }
    K = ada.size();
    Rrng.resize(K + 1);
    compute_antiderivative();

    eta.reset(new PExpEvaluator<T>(ada, adb, ts, Rrng));
    R.reset(new PExpIntegralEvaluator<T>(ada, adb, ts, Rrng));
    Rinv.reset(new PExpInverseIntegralEvaluator<T>(ada, adb, ts, Rrng));

    // Compute a TV-like regularizer
    T x = 0.0;
    T xmax = ts.rbegin()[1] * 1.1;
    T step = (xmax - x) / 1000;
    std::vector<T> xs, ys;
    while (x < xmax)
    {
        xs.push_back(x);
        x += step;
    }
    if (xs.size())
    {
        ys = (*eta)(xs);
        for (int i = 1; i < ys.size(); ++i)
            _reg += myabs(ys[i] - ys[i - 1]);
    }
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::initialize_derivatives(void) {}

static int nd;

template <>
void PiecewiseExponentialRateFunction<adouble>::initialize_derivatives(void)
{
    nd = derivatives.size();
    Eigen::VectorXd z = Eigen::VectorXd::Zero(nd);
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(nd, nd);
    for (int k = 0; k < K; ++k)
    {
        ada[k].derivatives() = z;
        adb[k].derivatives() = z;
        ads[k].derivatives() = z;
    }
    std::vector<adouble>* dl[3] = {&ada, &adb, &ads};
    int d = 0;
    for (auto p : derivatives)
        (*dl[p.first])[p.second].derivatives() = I.col(d++);
    ts[0].derivatives() = z;
    Rrng[0].derivatives() = z;
}

mpreal_wrapper<double> convert(double d) { return mpfr::mpreal(d); }
mpreal_wrapper<adouble> convert(adouble d) { return mpreal_wrapper<adouble>(d.value(), d.derivatives().template cast<mpfr::mpreal>()); }

template <typename T>
constexpr mpreal_wrapper<T> _single_integral_helper(const int lam, 
        const mpreal_wrapper<T> &_tsm, const mpreal_wrapper<T> &_tsm1, 
        const mpreal_wrapper<T> &_ada, const mpreal_wrapper<T> &_Rrng)
{
    return -exp(-(lam + 1) * _Rrng) * expm1(-(lam + 1) * _ada * (_tsm1 - _tsm)) / (lam + 1);
}

template <typename T>
Vector<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::single_integrals(
        const mp_prec_t prec, const int lam) const
{
    mpfr::mpreal::set_default_prec(prec);
    mpreal_wrapper<T> _Rrng, _ada, _tsm, _tsm1; 
    Vector<mpreal_wrapper<T> > single_integrals(K);
    for (int m = 0; m < K; ++m)
    {
        _Rrng = convert(Rrng[m]);
        _ada = convert(ada[m]);
        _tsm = convert(ts[m]);
        _tsm1 = convert(ts[m + 1]);
        single_integrals(m) = _single_integral_helper<T>(lam, _tsm, _tsm1, _ada, _Rrng);
    }
    return single_integrals;
}

template <typename T>
inline mpreal_wrapper<T> _double_integral_helper(const int rate, const int lam, 
        const mpreal_wrapper<T> &_tsm, const mpreal_wrapper<T> &_tsm1, 
        const mpreal_wrapper<T> &_ada, const mpreal_wrapper<T> &_Rrng)
{
    const int l1 = lam + 1;
    const int l1r = l1 + rate;
    mpreal_wrapper<T> diff = _tsm1 - _tsm;
    if (rate == 0)
    {
        mpreal_wrapper<T> e1 = exp(-l1 * _ada * diff);
        if (e1 == 0)
            return exp(-l1 * _Rrng) / (l1 * l1 * _ada);
        else
            return exp(-l1 * _Rrng) * (1 - exp(-l1 * _ada * diff) * (1 + l1 * _ada * diff)) / (l1 * l1 * _ada);
    }
    /*
    return (
            exp(-l1r * _Rrng) * expm1(-l1r * _ada * (_tsm1 - _tsm)) / l1r -
            exp(-l1r * _Rrng) * expm1(-l1 * (_tsm1 - _tsm)) / l1
           ) / (rate * _ada);
           */
    return exp(-l1r * _Rrng) * (expm1(-l1r * _ada * diff) / l1r - expm1(-l1 * _ada * diff) / l1) / (rate * _ada);
}

template <typename T>
Matrix<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::double_integrals(
        const int n, const mp_prec_t prec, const int lam, bool below) const
{
    mpfr::mpreal::set_default_prec(prec);
    long int rate;
    mpreal_wrapper<T> _Rrng, _ada, _tsm, _tsm1;
    Matrix<mpreal_wrapper<T> > double_integrals(K, n - 1);
    double_integrals.setZero();
    //
    // \int_0^t_k alpha(tau) exp(-R(tau)) \int_0^\tau exp(-rate * R(t)) dt
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=0}^{m-2} \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_t[m]^tau exp(-rate * (a[m](t - t[m]) + Rrng[m])))]
    //
    for (int m = 0; m < K; ++m)
    {
        _Rrng = convert(Rrng[m]);
        _ada = convert(ada[m]);
        _tsm = convert(ts[m]);
        _tsm1 = convert(ts[m + 1]);
        for (int j = 2; j < n + 1; ++j)
        {
            rate = j * (j - 1) / 2 - (int)below;
            double_integrals(m, j - 2) = _double_integral_helper<T>(rate, lam, _tsm, _tsm1, _ada, _Rrng);
        }
    }
    return double_integrals;
}

template <typename T>
inline mpreal_wrapper<T> _inner_integral_helper(const int rate, 
        const mpreal_wrapper<T> &_tsm, const mpreal_wrapper<T> &_tsm1, 
        const mpreal_wrapper<T> &_ada, const mpreal_wrapper<T> &_Rrng)
{
    if (rate == 0)
        return _tsm1 - _tsm;
    return -exp(-rate * _Rrng) * expm1(-rate * _ada * (_tsm1 - _tsm)) / (rate * _ada);
}

template <typename T>
Matrix<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::inner_integrals(const int n, const mp_prec_t prec, bool below) const
{
    mpfr::mpreal::set_default_prec(prec);
    long int rate;
    mpreal_wrapper<T> _Rrng, _Rrng1, _ada, _tsm, _tsm1, diff, _hs, _adadiff;
    Matrix<mpreal_wrapper<T> > inner_integrals(K, n - 1);
    inner_integrals.setZero();
    //
    // \int_0^t_k alpha(tau) exp(-R(tau)) \int_0^\tau exp(-rate * R(t)) dt
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=0}^{m-2} \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_t[m]^tau exp(-rate * (a[m](t - t[m]) + Rrng[m])))]
    //
    for (int m = 0; m < K; ++m)
    {
        _Rrng = convert(Rrng[m]);
        _ada = convert(ada[m]);
        _tsm = convert(ts[m]);
        _tsm1 = convert(ts[m + 1]);
        for (int j = 2; j < n + 1; ++j)
        {
            rate = j * (j - 1) / 2 - (int)below;
            inner_integrals(m, j - 2) = _inner_integral_helper<T>(rate, _tsm, _tsm1, _ada, _Rrng);
        }
    }
    return inner_integrals;
}

template <typename T>
Matrix<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::mpfr_tjj_double_integral_above(
        const int n, const mp_prec_t prec, long lam) const
{
    mpfr::mpreal::set_default_prec(prec);
    Matrix<mpreal_wrapper<T> > inner_int = inner_integrals(n + 1, prec, false);
    Matrix<mpreal_wrapper<T> > double_int = double_integrals(n + 1, prec, lam, false);
    Vector<mpreal_wrapper<T> > single_int = single_integrals(prec, lam);
    // \int_0^t_k alpha(tau) exp(-(1 + lam) R(tau)) \int_\tau^\inf exp(-rate * R(t)) dt
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(1 + lam) * (a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=m+1}^K \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_tau^t[m+1] exp(-rate * (a[m](t - t[m]) + Rrng[m])))]
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(1 + lam) * (a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=m+1}^K \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_t[m]^t[m+1] exp(-rate * (a[m](t - t[m]) + Rrng[m])) - \int_t[m]^\tau exp(-rate * (a[m]*(t - t[m]) + Rrng[m])))]
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(1 + lam) * (a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=m}^K \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) - 
    //          \int_t[m]^\tau exp(-rate * (a[m]*(t - t[m]) + Rrng[m])))]
    std::vector<mpreal_wrapper<T> > cs;
    // stably compute reverse cumulative sum
    for (int j = 2; j < n + 2; ++j)
    {
        cs.clear();
        for (int m = K - 1; m > -1; --m)
        {
            cs.push_back(inner_int(m, j - 2));
            inner_int(m, j - 2) = mpreal_wrapper_type<T>::fsum(cs);
        }
    }
    // ts_integrals[m] = \int_ts[m]^ts[m+1] \int_\tau^\infty
    Matrix<mpreal_wrapper<T> > ts_integrals = single_int.asDiagonal() * inner_int - double_int;
    // Now calculate with hidden state integration limits
    size_t H = hidden_states.size();
    Matrix<mpreal_wrapper<T> > ret(H - 1, n);
    Matrix<mpreal_wrapper<T> > last = ts_integrals.topRows(hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < hs_indices.size(); ++h)
    {
        next = ts_integrals.topRows(hs_indices[h]).colwise().sum();
        ret.row(h - 1) = next - last;
        last = next;
    }
    return ret;
}

template <typename T>
Matrix<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::mpfr_tjj_double_integral_below(
        const int n, const mp_prec_t prec) const
{
    mpfr::mpreal::set_default_prec(prec);
    Matrix<mpreal_wrapper<T> > inner_int = inner_integrals(n + 2, prec, true);
    Matrix<mpreal_wrapper<T> > double_int = double_integrals(n + 2, prec, 0, true);
    Vector<mpreal_wrapper<T> > single_int = single_integrals(prec, 0);
    // \int_0^t_k alpha(tau) exp(-(1 + lam) R(tau)) \int_\tau^\inf exp(-rate * R(t)) dt
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(1 + lam) * (a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=m+1}^K \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_tau^t[m+1] exp(-rate * (a[m](t - t[m]) + Rrng[m])))]
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(1 + lam) * (a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=m+1}^K \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_t[m]^t[m+1] exp(-rate * (a[m](t - t[m]) + Rrng[m])) - \int_t[m]^\tau exp(-rate * (a[m]*(t - t[m]) + Rrng[m])))]
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(1 + lam) * (a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=m}^K \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) - 
    //          \int_t[m]^\tau exp(-rate * (a[m]*(t - t[m]) + Rrng[m])))]
    Matrix<mpreal_wrapper<T> > ts_integrals, _cumsum(K, n + 1);
    _cumsum.setZero();
    std::vector<mpreal_wrapper<T> > cs;
    // stably compute cumulative sum
    for (int j = 2; j < n + 3; ++j)
    {
        cs.clear();
        for (int m = 1; m < K; ++m)
        {
            cs.push_back(inner_int(m - 1, j - 2));
            _cumsum(m, j - 2) = mpreal_wrapper_type<T>::fsum(cs);
        }
    }
    // ts_integrals[m] = \int_ts[m]^ts[m+1] \int_\tau^\infty
    inner_int = _cumsum;
    ts_integrals = single_int.asDiagonal() * inner_int + double_int;
    // Now calculate with hidden state integration limits
    size_t H = hidden_states.size();
    Matrix<mpreal_wrapper<T> > ret(H - 1, n + 1);
    Matrix<mpreal_wrapper<T> > last = ts_integrals.topRows(hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < hs_indices.size(); ++h)
    {
        next = ts_integrals.topRows(hs_indices[h]).colwise().sum();
        ret.row(h - 1) = next - last;
        last = next;
    }
    return ret;
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::print_debug() const
{
    std::vector<std::pair<std::string, std::vector<T>>> arys = 
    {{"ada", ada}, {"adb", adb}, {"ads", ads}, {"ts", ts}, {"Rrng", Rrng}};
    std::cout << std::endl;
    for (auto p : arys)
    {
        std::cout << p.first << std::endl;
        for (adouble x : p.second)
            std::cout << x.value() << "::" << x.derivatives().transpose() << std::endl;
        std::cout << std::endl;
    }
    std::cout << "reg: " << toDouble(_reg) << std::endl << std::endl;
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::compute_antiderivative()
{
    for (int k = 0; k < K; ++k)
    {
        if (adb[k] == 0.0)
            Rrng[k + 1] = Rrng[k] + ada[k] * (ts[k + 1] - ts[k]);
        else
            Rrng[k + 1] = Rrng[k] + (ada[k] / adb[k]) * expm1(adb[k] * (ts[k + 1] - ts[k]));
    }
}

template class PiecewiseExponentialRateFunction<double>;
template class PiecewiseExponentialRateFunction<adouble>;
