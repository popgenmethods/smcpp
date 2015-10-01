#include "piecewise_exponential_rate_function.h"

// Private conversion helpers
template <typename T, typename U>
struct convert
{
    static T run(const U& x, const mp_prec_t prec) { return T(x, prec); }
};

template <>
struct convert<mpreal_wrapper_type<adouble>::type, adouble>
{
    static mpreal_wrapper_type<adouble>::type run(const adouble &x, const mp_prec_t prec)
    {
        Vector<mpfr::mpreal> d(x.derivatives().rows());
        for (int i = 0; i < d.rows(); ++i)
            d(i) = mpfr::mpreal(x.derivatives()(i), prec);
        return mpreal_wrapper_type<adouble>::type(mpfr::mpreal(x.value(), prec), d);
    }
};

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
    // ts[K] = INFINITY;
    ts[K] = T_MAX;

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

    _eta.reset(new PExpEvaluator<T>(ada, adb, ts, Rrng));
    _R.reset(new PExpIntegralEvaluator<T>(ada, adb, ts, Rrng));
    _Rinv.reset(new PExpInverseIntegralEvaluator<T>(ada, adb, ts, Rrng));
    for (int k = 1; k < K; ++k)
        _reg += abs(ada[k] - ada[k - 1]);
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

template <typename T, typename U>
inline T _single_double_integral_below(mp_prec_t prec, const int rate,
        const U &tsm, const U &tsm1, const U &ada_m, const U &Rrng_m,
        const U &tsk, const U &tsk1, const U &ada_k, const U &Rrng_k)
{
    T _tsm = convert<T,U>::run(tsm, prec), 
      _tsm1 = convert<T,U>::run(tsm1, prec),
      _ada_m = convert<T,U>::run(ada_m, prec),
      _Rrng_m = convert<T,U>::run(Rrng_m, prec), 
      _tsk = convert<T,U>::run(tsk, prec), 
      _tsk1 = convert<T,U>::run(tsk1, prec),
      _ada_k = convert<T,U>::run(ada_k, prec),
      _Rrng_k = convert<T,U>::run(Rrng_k, prec);
    if (rate == 0)
    {
        if (tsm1 == INFINITY)
            return exp(-_Rrng_m) * (_tsk1 - _tsk);
        return -exp(-_Rrng_m) * expm1(-_ada_m * (_tsm1 - _tsm)) * (_tsk1 - _tsk);
    }
    T one = convert<T, U>::run(1.0, prec);
    T e1, e2;
    if (tsk1 == INFINITY) e1 = -one; else e1 = expm1(-rate * _ada_k * (_tsk1 - _tsk));
    if (tsm1 == INFINITY) e2 = -one; else e2 = expm1(-_ada_m * (_tsm1 - _tsm));
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

template <typename T, typename U>
inline T _double_integral_below_helper(mp_prec_t prec, const int rate, const U &tsm, const U &tsm1, const U &ada, const U &Rrng)
{
    T _tsm = convert<T,U>::run(tsm, prec), 
      _tsm1 = convert<T,U>::run(tsm1, prec), 
      _ada = convert<T,U>::run(ada, prec), 
      _Rrng = convert<T,U>::run(Rrng, prec);
    const int l1r = 1 + rate;
    T z = _tsm - _tsm;
    const T l1rinv = 1 / (z + l1r);
    T diff = _tsm1 - _tsm;
    T _adadiff = _ada * diff;
    if (rate == 0)
    {
        T e1 = exp(-_adadiff);
        if (tsm1 == INFINITY)
            return exp(-_Rrng) / _ada;
        else
            return exp(-_Rrng) * (1 - exp(-_adadiff) * (1 + _adadiff)) / _ada;
    }
    if (tsm1 == INFINITY)
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
void PiecewiseExponentialRateFunction<T>::tjj_double_integral_above(const int n, long jj, std::vector<Matrix<T> > &C) const
{
    long lam = nC2(jj) - 1;
    Matrix<T> ts_integrals(K, n);
    for (int m = 0; m < K; ++m)
    {
        for (int j = 2; j < n + 2; ++j)
        {
            long rate = nC2(j);
            ts_integrals(m, j - 2) = _double_integral_above_helper<T>(rate, lam, ts[m], ts[m + 1], ada[m], Rrng[m]);
            check_nan(ts_integrals(m, j - 2));
            for (int k = m + 1; k < K; ++k)
            {
                ts_integrals(m, j - 2) += _single_double_integral_above(rate, lam,
                        ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], ada[k], Rrng[k]);
                check_nan(ts_integrals(m, j - 2));
            }

        }
    }
    // Now calculate with hidden state integration limits
    size_t H = hidden_states.size();
    Matrix<T> ret(H - 1, n);
    Matrix<T> last = ts_integrals.topRows(hs_indices[0]).colwise().sum(), next;
    for (int h = 1; h < hs_indices.size(); ++h)
    {
        next = ts_integrals.topRows(hs_indices[h]).colwise().sum();
        C[h - 1].row(jj - 2) = next - last;
        last = next;
    }
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::tjj_double_integral_below(
        const int n, const mp_prec_t prec, const int m, Matrix<mpreal_wrapper<T> > &tgt) const
{
    Vector<mpreal_wrapper<T> > ts_integrals(n + 1);
    std::vector<mpreal_wrapper<T> > cs;
    for (int j = 2; j < n + 3; ++j)
    {
        long rate = nC2(j) - 1;
        cs.clear();
        mp_prec_t wprec = prec + 10;
        /*
        MPInterval mpi(0.0, prec);
        mpi = _double_integral_below_helper<MPInterval, T>(wprec, rate, ts[m], ts[m + 1], ada[m], Rrng[m]);
        while (-mpi.delta().get_exp() < prec)
        {
            wprec += 10;
            mpi = _double_integral_below_helper<MPInterval, T>(wprec, rate, ts[m], ts[m + 1], ada[m], Rrng[m]);
            PROGRESS("prec miss:" << j << " " << wprec);
        }
        mpfr::mpreal::set_default_rnd(MPFR_RNDN);
        */
        cs.push_back(_double_integral_below_helper<mpreal_wrapper<T>, T>(wprec, rate, ts[m], ts[m + 1], ada[m], Rrng[m]));
        check_nan(cs.back());
        for (int k = 0; k < m; ++k)
        {
            wprec = prec + 10;
            /*
            mpi = _single_double_integral_below<MPInterval, T>(wprec, rate, 
                    ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], 
                    ada[k], Rrng[k]);
            while (-mpi.delta().get_exp() < prec)
            {
                wprec += 10;
                mpi = _single_double_integral_below<MPInterval, T>(wprec, rate, 
                        ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], 
                        ada[k], Rrng[k]);
                PROGRESS("prec miss:" << j << " " << k << " " << wprec);
            }
            */
            cs.push_back(_single_double_integral_below<mpreal_wrapper<T>, T>(wprec, rate, 
                        ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], 
                        ada[k], Rrng[k]));
            check_nan(cs.back());
        }
        ts_integrals(j - 2) = mpreal_wrapper_type<T>::fsum(cs);
    }
    tgt.row(m) = ts_integrals.transpose();
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

template <typename T>
T PiecewiseExponentialRateFunction<T>::R_integral(const T x, const T v) const
{
    int ip = insertion_point(x, ts, 0, ts.size());
    T ret = zero, Ra = R(v), r, tmp;
    for (int i = 0; i < ip + 1; ++i)
    {
        if (ts[i + 1] == INFINITY)
            tmp = x;
        else
            tmp = dmin(x, ts[i + 1]);
        r = exp(2 * (Rrng[i] + (tmp - ts[i]) * ada[i] - Ra));
        r -= exp(2 * (Rrng[i] - Ra));
        ret += r / (2. * ada[i]);
    }
    return ret;
}

template class PiecewiseExponentialRateFunction<double>;
template class PiecewiseExponentialRateFunction<adouble>;
