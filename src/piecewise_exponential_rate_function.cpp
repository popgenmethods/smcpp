#include "piecewise_exponential_rate_function.h"

using Eigen::expintei;
using eint::expintei;

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
        ada[k] = 1. / ada[k];
        adb[k] = 1. / adb[k];
        ts[k + 1] = ts[k] + ads[k];
        adb[k] = (log(adb[k]) - log(ada[k])) / (ts[k + 1] - ts[k]);
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
            if (adb[ip] == 0)
            {
                vec_insert<T>(ada, ip + 1, ada[ip]);
                vec_insert<T>(adb, ip + 1, adb[ip]);
            }
            else
            {
                vec_insert<T>(ada, ip + 1, ada[ip] * exp(adb[ip] * ((T)h - ts[ip])));
                vec_insert<T>(adb, ip + 1, (log(ada[ip] / ada[ip + 1]) + adb[ip] * (ts[ip + 2] - ts[ip])) / (ts[ip + 2] - (T)h));
            }
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
    last *= one;
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
        mp_prec_t wprec = prec + 20;
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
                        ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], ada[k], Rrng[k]));
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
    Rrng[0] = zero;
    for (int k = 0; k < K; ++k)
    {
        if (adb[k] == 0.0)
            Rrng[k + 1] = Rrng[k] + ada[k] * (ts[k + 1] - ts[k]);
        else
            Rrng[k + 1] = Rrng[k] + (ada[k] / adb[k]) * expm1(adb[k] * (ts[k + 1] - ts[k]));
    }
}

template <typename T>
T PiecewiseExponentialRateFunction<T>::R_integral(const T x, const T y) const
{
    // int_0^x exp(-2 * R(t)) dt
    int ip = insertion_point(x, ts, 0, ts.size());
    T ret = zero, tmp, r;
    for (int i = 0; i < ip + 1; ++i)
    {
        tmp = dmin(x, ts[i + 1]) - ts[i];
        if (adb[i] == 0)
        {
            r = exp(2 * Rrng[i] + y) * expm1(2 * tmp * ada[i]);
            r /= 2. * ada[i];
        }
        else
        {
            T adab = ada[i] / adb[i];
            T c1 = 2 * adab * exp(adb[i] * tmp);
            T c2 = 2 * adab;
            T r1 = expintei(c1);
            T r2 = expintei(c2);
            r = r1 - r2;
            r *= exp(2 * (Rrng[i] - adab) + y) / adb[i];
        }
        ret += r;
    }
    return ret;
}

template class PiecewiseExponentialRateFunction<double>;
template class PiecewiseExponentialRateFunction<adouble>;

namespace newstuff
{
    template <typename T>
    T eint_helper(T x, T* r)
    {
        return exp(x + *r) / x;
    }

    template <typename T>
    T eintdiff(const T a, const T b, T r)
    {
        /*
        Workspace<T> Work(128, 128);
        Function<T, T> F(eint_helper<T>, &r);
        T result, abserr;
        int status;
        try 
        {
            status = Work.qag(F, a, b, (T)eps, (T)0, result, abserr);
        }
        catch (const char* reason) 
        {
            std::cerr << reason << std::endl;
            return result;
        }
        return result;
        */
        // = e(r) * (eint(b) - eint(a));
        // = -\int_a^b exp(t+r) / t dt
        if (a > b)
            return -eintdiff(b, a, r);
        return exp(r) * (expintei(b) - expintei(a));
        //
        // std::function<T(const T, const T*)> f(eint_helper<T>);
        // return adaptiveSimpsons<T>(f, &r, a, b, eps, 20);
    }

    template <typename T>
    inline T _single_integral(const int rate, const T &tsm, const T &tsm1, const T &ada, const T &adb, const T &Rrng)
    {
        // = int_ts[m]^ts[m+1] exp(-rate * R(t)) dt
        const int c = rate;
        if (rate == 0)
            return tsm1 - tsm;
        if (adb == 0.)
        {
            T ret = exp(-c * Rrng);
            if (tsm1 < INFINITY)
                ret *= -expm1(-c * ada * (tsm1 - tsm));
            return ret / ada / c;
        }
        T e1 = -c * exp(adb * (tsm1 - tsm)) * ada / adb;
        T e2 = -c * ada / adb;
        T e3 =  c * (ada / adb - Rrng);
        return eintdiff(e1, e2, e3);
    }


    template <typename T>
    inline T _double_integral_below_helper_ei(const int rate, const T &tsm, const T &tsm1, 
            const T &ada, const T &adb, const T &Rrng)
    {
        // We needn't cover the tsm1==INFINITY case here as the last piece is assumed flat (i.e., adb=0).
        long c = rate;
        T eadb = exp(adb * (tsm1 - tsm));
        T adadb = ada / adb;
        if (c == 0)
        {
            T a1 = -adadb;
            T b1 = -eadb * adadb;
            T cons1 = -b1;
            T int1 = eintdiff(a1, b1, cons1);
            return exp((eadb - 1.) * adadb - Rrng) * (int1 + adb * (tsm1 - tsm)) / adb;
        }
        T cons1 = (2 + c) * adadb;
        T cons2 = adadb * (2 + c + eadb);
        T a1 = -c * adadb * eadb;
        T b1 = -c * adadb;
        T int1 = eintdiff(a1, b1, cons1);
        T a2 = -(c + 1) * adadb;
        T b2 = -(c + 1) * adadb * eadb;
        T int2 = eintdiff(a2, b2, cons2);
        T cons3 = exp(-(ada * (1 + eadb) / adb + (1 + c) * Rrng));
        return cons3 * (int1 + int2) / adb;
    }

    template <typename T>
    inline T _double_integral_above_helper_ei(const int rate, const int lam, const T &tsm, const T &tsm1, 
            const T &ada, const T &adb, const T &Rrng)
    {
        long d = lam;
        long c = rate;
        T eadb = exp(adb * (tsm1 - tsm));
        T cons1 = ada * c / adb;
        T a1 = -cons1 * eadb;
        T b1 = -cons1;
        T c1 = cons1 - d * Rrng;
        T ed1 = eintdiff(a1, b1, c1, 1e-16);
        if (c != d)
        {
            T cons2 = ada * d / adb;
            T a2 = -cons2;
            T b2 = -cons2 * eadb;
            T c2 = cons2 - d * Rrng;
            return (ed1 + eintdiff(a2, b2, c2, 1e-16)) / adb / (c - d);
        }
        return (exp(-d * Rrng) * (-adb * expm1(-ada / adb * d * expm1(adb * (tsm1 - tsm)))) + ada * d * ed1) / (adb * adb * d);
    }

    template <typename U>
    inline U _double_integral_above_helper(const int rate, const int lam, const U &_tsm, const U &_tsm1, const U &_ada, const U &_Rrng)
    {
        U diff = _tsm1 - _tsm;
        U adadiff = _ada * diff;
        long l1 = lam;
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
    // void PiecewiseExponentialRateFunction<T>::tjj_double_integral_above(const int n, long jj, std::vector<Matrix<T> > &C) const
    void tjj_double_integral_above(const int n, long jj, std::vector<Matrix<T> > &C, const PiecewiseExponentialRateFunction<T> &eta) 
    {
        long lam = nC2(jj);
        auto K = eta.K;
        auto Rrng = eta.Rrng, ada = eta.ada, adb = eta.adb, ts = eta.ts;
        auto hidden_states = eta.hidden_states;
        auto hs_indices = eta.hs_indices;
        auto zero = eta.zero;
        auto one = eta.one;
        Matrix<T> ts_integrals(K, n);
        ts_integrals.fill(zero);
        std::vector<T> single_integrals;
        T e1, e2;
        for (int m = 0; m < K; ++m)
        {
            e1 = exp(-Rrng[m]);
            if (m < K - 1)
                e1 -= exp(-Rrng[m + 1]);
            single_integrals.push_back(e1);
        }

        for (int m = 0; m < K; ++m)
        {
            for (int j = 2; j < n + 2; ++j)
            {
                long rate = nC2(j);
                if (adb[m] == 0)
                    ts_integrals(m, j - 2) = _double_integral_above_helper<T>(rate, lam, ts[m], ts[m + 1], ada[m], Rrng[m]);
                else
                    ts_integrals(m, j - 2) = _double_integral_above_helper_ei<T>(rate, lam, ts[m], ts[m + 1], ada[m], adb[m], Rrng[m]);
                check_nan(ts_integrals(m, j - 2));
                T tmp = zero;
                for (int k = m + 1; k < K; ++k)
                {
                    /*
                    ts_integrals(m, j - 2) += _single_double_integral_above(rate, lam,
                            ts[m], ts[m + 1], ada[m], Rrng[m], ts[k], ts[k + 1], ada[k], Rrng[k]);
                    */
                    tmp += _single_integral(rate, ts[k], ts[k + 1], ada[k], adb[k], Rrng[k]);
                }
                if (m + 1 < K)
                {
                    if (lam != rate)
                        ts_integrals(m, j - 2) += single_integrals[m] * tmp / (lam - rate);
                    else
                        ts_integrals(m, j - 2) += single_integrals[m] * (Rrng[m + 1] - Rrng[m]);
                }
                check_nan(ts_integrals(m, j - 2));
            }
        }
        // Now calculate with hidden state integration limits
        size_t H = hidden_states.size();
        Matrix<T> ret(H - 1, n);
        Matrix<T> last = ts_integrals.topRows(hs_indices[0]).colwise().sum(), next;
        last *= one;
        for (int h = 1; h < hs_indices.size(); ++h)
        {
            next = ts_integrals.topRows(hs_indices[h]).colwise().sum();
            C[h - 1].row(jj - 2) = next - last;
            last = next;
        }
    }

    template <typename T>
    // void PiecewiseExponentialRateFunction<T>::tjj_double_integral_below(
    void tjj_double_integral_below(const int n, const mp_prec_t prec, const int m, Matrix<mpreal_wrapper<T> > &tgt, const PiecewiseExponentialRateFunction<T> &eta)
    {
        Vector<mpreal_wrapper<T> > ts_integrals(n + 1);
        std::vector<mpreal_wrapper<T> > cs;
        auto K = eta.K;
        auto Rrng = eta.Rrng, ada = eta.ada, adb = eta.adb, ts = eta.ts;
        auto hidden_states = eta.hidden_states;
        auto hs_indices = eta.hs_indices;
        auto zero = eta.zero;
        auto one = eta.one;
        std::vector<T> single_integrals;
        T e1, e2;
        mp_prec_t wprec = prec + 10;
        mpreal_wrapper<T> 
            _tsm = convert<mpreal_wrapper<T>, T >::run(ts[m], wprec), 
            _tsm1 = convert<mpreal_wrapper<T>, T >::run(ts[m + 1], wprec), 
            _ada = convert<mpreal_wrapper<T>, T >::run(ada[m], wprec), 
            _adb = convert<mpreal_wrapper<T>, T >::run(adb[m], wprec), 
            _Rrng = convert<mpreal_wrapper<T>, T >::run(Rrng[m], wprec),
            _Rrng1 = convert<mpreal_wrapper<T>, T >::run(Rrng[m + 1], wprec);
        mpreal_wrapper<T> si = exp(-_Rrng);
        if (m < K - 1)
            si -= exp(-_Rrng1);
        for (int j = 2; j < n + 3; ++j)
        {
            long rate = nC2(j) - 1;
            cs.clear();
            if (adb[m] == 0.)
                ts_integrals(j - 2) = _double_integral_below_helper<mpreal_wrapper<T>, T>(wprec, rate, ts[m], ts[m + 1], ada[m], Rrng[m]);
            else
            {
                ts_integrals(j - 2) = _double_integral_below_helper_ei(rate, _tsm, _tsm1, _ada, _adb, _Rrng);
            }
            for (int k = 0; k < m; ++k)
            {
                wprec = prec + 10;
                mpreal_wrapper<T> 
                    _tsk = convert<mpreal_wrapper<T>, T >::run(ts[k], wprec), 
                    _tsk1 = convert<mpreal_wrapper<T>, T >::run(ts[k + 1], wprec), 
                    _ada = convert<mpreal_wrapper<T>, T >::run(ada[k], wprec), 
                    _adb = convert<mpreal_wrapper<T>, T >::run(adb[k], wprec), 
                    _Rrng = convert<mpreal_wrapper<T>, T >::run(Rrng[k], wprec);
                cs.push_back(_single_integral(rate, _tsk, _tsk1, _ada, _adb, _Rrng));
                check_nan(cs.back());
            }
            if (m > 0)
            {
                mpreal_wrapper<T> tmp = mpreal_wrapper_type<T>::fsum(cs);
                ts_integrals(j - 2) += si * tmp;
            }
        }
        tgt.row(m) = ts_integrals.transpose();
    }
}

int main1(int argc, char** argv)
{
    std::vector<std::vector<double> > params = {
        {0.5, 1.0, 2.0},
        {0.5, 1.5, 2.0},
        {0.1, 0.1, 0.1}
    };
    std::vector<double> hs = {0.0, .02, 0.1, 1.0, 2.0, 10., 25.};
    std::vector<std::pair<int, int> > deriv = { {0,0} };
    PiecewiseExponentialRateFunction<adouble> eta(params, deriv, hs);
    int n = 5;
    Matrix<mpreal_wrapper<adouble> > ts_integrals(eta.K, n + 1); 
    Matrix<mpreal_wrapper<adouble> > ts_integrals1(eta.K, n + 1); 
    eta.print_debug();
    for (int m = 0; m < eta.K; ++m)
    {
        eta.tjj_double_integral_below(n, 60, m, ts_integrals);
        newstuff::tjj_double_integral_below(n, 60, m, ts_integrals1, eta);
    }
    // eta.tjj_double_integral_below(n, 60, 1, ts_integrals);
    // newstuff::tjj_double_integral_below(n, 60, 1, ts_integrals1, eta);
    std::cout << ts_integrals.template cast<adouble>().template cast<double>() << std::endl << std::endl;
    std::cout << ts_integrals1.template cast<adouble>().template cast<double>() << std::endl << std::endl;
    mpreal_wrapper<adouble> t1 = _single_double_integral_below<mpreal_wrapper<adouble>,adouble>(53, 0, 
            eta.ts[1], eta.ts[2], eta.ada[1], eta.Rrng[1],
            eta.ts[0], eta.ts[1], eta.ada[0], eta.Rrng[0]);
    mpreal_wrapper<adouble> 
        _ts0 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.ts[0], 53), 
        _ts1 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.ts[1], 53), 
        _ts2 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.ts[2], 53), 
        _ada0 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.ada[0], 53), 
        _ada1 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.ada[1], 53), 
        _adb0 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.adb[0], 53),
        _adb1 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.adb[1], 53),
        _Rrng0 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.Rrng[0], 53),
        _Rrng1 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.Rrng[1], 53),
        _Rrng2 = convert<mpreal_wrapper<adouble>, adouble>::run(eta.Rrng[2], 53);
    mpreal_wrapper<adouble> t2 = newstuff::_single_integral(0, _ts0, _ts1, _ada0, _adb0, _Rrng0);
    t2 *= exp(-_Rrng1) - exp(-_Rrng2);
    // mpreal_wrapper<adouble> t2 = newstuff::_double_integral_above_helper_ei(10, 12, _ts1, _ts2, _ada1, _adb1, _Rrng1);
    // mpreal_wrapper<adouble> t3 = newstuff::_double_integral_above_helper_ei(12, 12, _ts1, _ts2, _ada1, _adb1, _Rrng1);
    std::cout << t1.value() << std::endl;
    std::cout << t2.value() << std::endl;
//     eta.print_debug();
  //   std::cout << newstuff::eintdiff(1.0, 2.0, 0.0, 1e-8) << std::endl;
    // std::cout << newstuff::eintdiff(-10.0, -9.0, 0.0, 1e-8) << std::endl;
}
