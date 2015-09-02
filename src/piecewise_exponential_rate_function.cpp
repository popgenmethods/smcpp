#include "piecewise_exponential_rate_function.h"

constexpr long nC2(int n) { return n * (n - 1) / 2; }

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
PiecewiseExponentialRateFunction<adouble>::PiecewiseExponentialRateFunction(
        const std::vector<std::vector<double>> params, 
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
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(
        const std::vector<std::vector<double>> params, 
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
    ts[K] = INFINITY;

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
    _reg = 0.0;
    /*
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
    */
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

template <typename T>
mpreal_wrapper<T> convert(const T &x) { return mpreal_wrapper<T>(x); }

template <>
mpreal_wrapper<adouble> convert(const adouble &d) 
{ 
    return mpreal_wrapper<adouble>(d.value(), d.derivatives().template cast<mpfr::mpreal>()); 
}

template <typename T>
inline T _single_double_integral_below(const int rate,
        const T &_tsm, const T &_tsm1, const T &_ada_m, const T &_Rrng_m,
        const T &_tsk, const T &_tsk1, const T &_ada_k, const T &_Rrng_k)
{
    if (rate == 0)
    {
        if (_tsm1 == INFINITY)
            return exp(-_Rrng_m) * (_tsk1 - _tsk);
        return -exp(-_Rrng_m) * expm1(-_ada_m * (_tsm1 - _tsm)) * (_tsk1 - _tsk);
    }
    T e1, e2;
    if (_tsk1 == INFINITY) e1 = T(-1); else e1 = expm1(-rate * _ada_k * (_tsk1 - _tsk));
    if (_tsm1 == INFINITY) e2 = T(-1); else e2 = expm1(-_ada_m * (_tsm1 - _tsm));
    return exp(-rate * _Rrng_k - _Rrng_m) * e1 * e2 / rate / _ada_k;
}

template <typename T>
inline T _single_double_integral_above(const int rate, const int l1,
        const T &_tsm, const T &_tsm1, const T &_ada_m, const T &_Rrng_m,
        const T &_tsk, const T &_tsk1, const T &_ada_k, const T &_Rrng_k)
{
    int lam = l1 + 1;
    if (rate == 0)
        return -exp(-lam * _Rrng_m) * exp(-lam * _ada_m * _tsm1) * expm1(-lam * _ada_m * (_tsm1 - _tsm)) * (_tsm1 - _tsm) / lam;
    if (lam == rate)
    {
        if (_tsk1 == INFINITY)
            return exp(-lam * _Rrng_k) * _ada_m * (_tsm1 - _tsm) / _ada_k / lam;    
        return -exp(-lam * _Rrng_k) * expm1(-lam * _ada_k * (_tsk1 - _tsk)) * _ada_m * (_tsm1 - _tsm) / _ada_k / lam;    
    }
    T e = -(lam - rate) * _ada_m * (_tsm1 - _tsm);
    if (e <= 200)
    {
        if (_tsk1 == INFINITY)
            return -exp(-rate * _Rrng_k - (lam - rate) * _Rrng_m) * expm1(e) / rate / (lam - rate) / _ada_k;
        return exp(-rate * _Rrng_k - (lam - rate) * _Rrng_m) * expm1(-rate * _ada_k * (_tsk1 - _tsk)) *
            expm1(-(lam - rate) * _ada_m * (_tsm1 - _tsm)) / rate / (lam - rate) / _ada_k;
    }
    if (_tsk1 == INFINITY)
        return exp((-rate * _Rrng_k - (lam - rate) * _Rrng_m) + e - log(rate) - log(rate - lam) - log(_ada_k));
    return exp(
            (-rate * _Rrng_k - (lam - rate) * _Rrng_m) + 
            log(expm1(-rate * _ada_k * (_tsk1 - _tsk)) / (lam - rate)) + 
            e - log(rate) - log(_ada_k));
}


template <typename U>
inline U _double_integral_below_helper(const int rate, const U &_tsm, const U &_tsm1, const U &_ada, const U &_Rrng)
{
    const int l1r = 1 + rate;
    U z = _tsm - _tsm;
    const U l1rinv = 1 / (z + U(l1r));
    U diff = _tsm1 - _tsm;
    U _adadiff = _ada * diff;
    if (rate == 0)
    {
        U e1 = exp(-_adadiff);
        if (e1 == 0)
            return exp(-_Rrng) / _ada;
        else
            return exp(-_Rrng) * (1 - exp(-_adadiff) * (1 + _adadiff)) / _ada;
    }
    if (_tsm1 == INFINITY)
        return exp(-l1r * _Rrng) * (1 - l1rinv) / (rate * _ada);
    return exp(-l1r * _Rrng) * (expm1(-l1r * _adadiff) * l1rinv - expm1(-_adadiff)) / (rate * _ada);
}

template <typename U>
inline U _double_integral_above_helper(const int rate, const int lam, const U &_tsm, const U &_tsm1, const U &_ada, const U &_Rrng)
{
    U diff = _tsm1 - _tsm;
    U adadiff = _ada * diff;
    long l1 = lam + 1;
    if (rate == 0)
        return exp(-l1 * _Rrng) * (expm1(-l1 * adadiff) + l1 * adadiff) / l1 / l1 / _ada;
    if (l1 == rate)
    {
        if (_tsm1 == INFINITY)
            return exp(-rate * _Rrng) / rate / rate / _ada;
        return exp(-rate * _Rrng) * (1 - exp(-rate * adadiff) * (1 + rate * adadiff)) / rate / rate / _ada;
    }
    if (_tsm1 == INFINITY)
        return exp(-l1 * _Rrng) / l1 / rate / _ada;
    return -exp(-l1 * _Rrng) * (expm1(-l1 * adadiff) / l1 + (exp(-rate * adadiff) - exp(-l1 * adadiff)) / (l1 - rate)) / rate / _ada;
}

template <typename T>
Matrix<T> PiecewiseExponentialRateFunction<T>::tjj_all_above(const int n, 
        const MatrixXq &X0, const MatrixXq &Uinv_mp0, const MatrixXq &X2, 
        const MatrixXq &Uinv_mp2) const
{
    Matrix<T> Ch(n + 1, n), ret(3, n + 1), T_subtend;
    for (int j = 2; j < n + 3; ++j)
    {
        long lam = nC2(j) - 1;
        Ch.row(j - 2) = tjj_double_integral_above(n, lam).row(0);
    }
    ret.setZero();
    T_subtend = ((X0.template cast<T>().cwiseProduct(Ch.transpose()).colwise().sum()) * Uinv_mp0.template cast<T>());
    ret.block(0, 1, 1, n) += T_subtend.template cast<T>();
    T_subtend = ((X2.template cast<T>().cwiseProduct(Ch.colwise().reverse().transpose()).colwise().sum()) * Uinv_mp2.template cast<T>());
    ret.block(2, 0, 1, n) += T_subtend.template cast<T>();
    return ret;
}

template <typename T>
Matrix<T> PiecewiseExponentialRateFunction<T>::tjj_double_integral_above(const int n, long lam) const
{
    Matrix<T> ts_integrals(K, n);
    for (int m = 0; m < K; ++m)
    {
        for (int j = 2; j < n + 2; ++j)
        {
            long rate = nC2(j);
            ts_integrals(m, j - 2) = _double_integral_above_helper<T>(rate, lam, ts[m], ts[m + 1], ada[m], Rrng[m]);
            for (int k = m + 1; k < K; ++k)
                ts_integrals(m, j - 2) += _single_double_integral_above(rate, lam,
                        ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], ada[k], Rrng[k]);

        }
    }
    // Now calculate with hidden state integration limits
    size_t H = hidden_states.size();
    Matrix<T> ret(H - 1, n);
    Matrix<T> last = ts_integrals.topRows(hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < hs_indices.size(); ++h)
    {
        next = ts_integrals.topRows(hs_indices[h]).colwise().sum();
        ret.row(h - 1) = next - last;
        last = next;
    }
    return ret;
}

template <typename T>
Matrix<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::tjj_double_integral_below(
        const int n, const mp_prec_t prec) const
{
    mpfr::mpreal::set_default_prec(prec);
    Matrix<mpreal_wrapper<T> > ts_integrals(K, n + 1);
    std::vector<mpreal_wrapper<T> > cs;
    for (int m = 0; m < K; ++m)
    {
        for (int j = 2; j < n + 3; ++j)
        {
            long rate = nC2(j) - 1;
            cs.clear();
            cs.push_back(_double_integral_below_helper<mpreal_wrapper<T> >(rate, convert(ts[m]), convert(ts[m + 1]), convert(ada[m]), convert(Rrng[m])));
            for (int k = 0; k < m; ++k)
                cs.push_back(_single_double_integral_below<mpreal_wrapper<T> >(rate, 
                            convert(ts[m]), convert(ts[m + 1]), 
                            convert(ada[m]), convert(Rrng[m]), convert(ts[k]), convert(ts[k + 1]), 
                            convert(ada[k]), convert(Rrng[k])));
            ts_integrals(m, j - 2) = mpreal_wrapper_type<T>::fsum(cs);
        }
    }
    // Now calculate with hidden state integration limits
    size_t H = hidden_states.size();
    Matrix<mpreal_wrapper<T> > ret(H - 1, n + 1);
    Matrix<mpreal_wrapper<T> > last = ts_integrals.topRows(hs_indices[0]).colwise().sum(), next;
    mpreal_wrapper<T> h1, h2;
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
