#include "piecewise_exponential_rate_function.h"

constexpr long nC2(int n) { return n * (n - 1) / 2; }

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params,
        const std::vector<double> hidden_states) : 
    PiecewiseExponentialRateFunction(params, std::vector<std::pair<int, int>>(), hidden_states) 
{ }

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
    derivatives(derivatives), 
    zero(init_derivative(0.0)), one(init_derivative(1.0)),
    K(params[0].size()), ada(params[0].begin(), params[0].end()), 
    adb(params[1].begin(), params[1].end()), ads(params[2].begin(), params[2].end()),
    ts(K + 1), Rrng(K), 
    _reg(zero),
    hidden_states(hidden_states),
    tmax(std::accumulate(params[2].begin(), params[2].end() - 1, 0.0))
{
    for (auto &pp : params)
        if (pp.size() != params[0].size())
            throw std::runtime_error("all params must have same size");
    // Final piece is required to be flat.
    ts[0] = zero;
    Rrng[0] = zero;
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
    adb[K - 1] = zero;
    ts[K] = T_MAX;

    std::vector<T> new_ada, new_adb, new_ts;
    new_ts.push_back(zero);
    ada.insert(ada.begin(), ada.at(0));
    adb.insert(adb.begin(), zero);
    ts.insert(ts.begin() + 1, 0.01 * ts.at(1));
    new_ada.push_back(ada[0]);
    new_adb.push_back(zero);
    new_ts.push_back(ts[1]);
    const int D = 20;
    for (int k = 1; k < K + 1; ++k)
    {
        if (adb[k] == 0.)
        {
            new_ada.push_back(ada[k]);
            new_adb.push_back(adb[k]);
            new_ts.push_back(ts[k + 1]);
        }
        else
        {
            T log_t = log(ts[k]);
            T log_t1 = log(ts[k + 1]);
            T delta = (log_t1 - log_t) / (double)D;
            for (int i = 0; i < D; ++i)
            {
                log_t += delta;
                T t = exp(log_t);
                new_ada.push_back(ada[k] * exp(adb[k] * (t - ts[k])));
                new_adb.push_back(zero);
                new_ts.push_back(t);
            }
        }
    }
    K = new_ada.size();
    new_ts[K] = T_MAX;
    if (false)
    {
        std::cout << "ada: " << ada << std::endl;
        std::cout << "adb: " << adb << std::endl;
        std::cout << "ts: " << ts << std::endl;
        std::cout << "new_ada: " << new_ada << std::endl;
        std::cout << "new_adb: " << new_adb << std::endl;
        std::cout << "new_ts: " << new_ts << std::endl;
    }
    ada = new_ada;
    adb = new_adb;
    ts = new_ts;

    int ip;
    for (double h : hidden_states)
    {
        ip = insertion_point(h, ts, 0, ts.size());
        if (myabs(ts[ip] - h) < 1e-8)
        // if (ts[ip] == h)
            hs_indices.push_back(ip);
        else if (ip + 1 < ts.size() and myabs(ts[ip + 1] - h) < 1e-8)
            hs_indices.push_back(ip + 1);
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
                throw std::runtime_error("should have b=0");
                vec_insert<T>(ada, ip + 1, ada[ip] * exp(adb[ip] * (one * h - ts[ip])));
                vec_insert<T>(adb, ip + 1, (log(ada[ip] / ada[ip + 1]) + adb[ip] * (ts[ip + 2] - ts[ip])) / (ts[ip + 2] - (T)h));
            }
            check_nan(ada[ip + 1]);
            check_nan(adb[ip + 1]);
            check_nan(ts[ip + 1]);
            hs_indices.push_back(ip + 1);
        }
    }
    K = ada.size();
    for (int k = 0; k < K; ++k)
        if (myabs(adb[k]) < 1e-2)
            adb[k] = zero;
    Rrng.resize(K + 1);
    compute_antiderivative();

    _eta.reset(new PExpEvaluator<T>(ada, adb, ts, Rrng));
    _R.reset(new PExpIntegralEvaluator<T>(ada, adb, ts, Rrng));
    _Rinv.reset(new PExpInverseIntegralEvaluator<T>(ada, adb, ts, Rrng));
    // lastly
    compute_regularizer();
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::compute_regularizer()
{
    _reg = zero;
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::initialize_derivatives(void) {}

template <>
void PiecewiseExponentialRateFunction<adouble>::initialize_derivatives(void)
{
    int nd = derivatives.size();
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
        if (p.first < 3)
            (*dl[p.first])[p.second].derivatives() = I.col(d++);
    ts[0].derivatives() = z;
    Rrng[0].derivatives() = z;
}

template <typename T>
inline T _double_integral_below_helper(const int rate, const T &tsm, const T &tsm1, const T &ada, 
        const T &Rrng, const T &log_denom)
{
    const int l1r = 1 + rate;
    T _tsm = tsm, _tsm1 = tsm1, _ada = ada, _Rrng = Rrng; // don't ask
    T z = _tsm - _tsm;
    const T l1rinv = 1 / (z + l1r);
    T diff = _tsm1 - _tsm;
    T _adadiff = _ada * diff;
    if (rate == 0)
    {
        if (tsm1 == INFINITY)
            return exp(-_Rrng - log_denom) / _ada;
        else
            return exp(-_Rrng - log_denom) * (1 - exp(-_adadiff) * (1 + _adadiff)) / _ada;
    }
    if (tsm1 == INFINITY)
        return exp(-l1r * _Rrng - log_denom) * (1 - l1rinv) / (rate * _ada);
    return exp(-l1r * _Rrng - log_denom) * (expm1(-l1r * _adadiff) * l1rinv - expm1(-_adadiff)) / (rate * _ada);
}

template <typename U>
inline U _double_integral_above_helper(const int rate, const int lam, const U &_tsm, 
        const U &_tsm1, const U &_ada, const U &_Rrng, const U &log_coef)
{
    U diff = _tsm1 - _tsm;
    U adadiff = _ada * diff;
    long l1 = lam + 1;
    if (rate == 0)
        return exp(-l1 * _Rrng + log_coef) * (expm1(-l1 * adadiff) + l1 * adadiff) / l1 / l1 / _ada;
    if (l1 == rate)
    {
        if (_tsm1 == INFINITY)
            return exp(-rate * _Rrng + log_coef) / rate / rate / _ada;
        return exp(-rate * _Rrng + log_coef) * (1 - exp(-rate * adadiff) * (1 + rate * adadiff)) / rate / rate / _ada;
    }
    if (_tsm1 == INFINITY)
        return exp(-l1 * _Rrng + log_coef) / l1 / rate / _ada;
    return -exp(-l1 * _Rrng + log_coef) * (expm1(-l1 * adadiff) / l1 + (exp(-rate * adadiff) - exp(-l1 * adadiff)) / (l1 - rate)) / rate / _ada;
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
T PiecewiseExponentialRateFunction<T>::R_integral(const double a, const double b) const
{
    // int_a^b exp(-R(t)) dt
    T Ta = one * a, Tb = one * b;
    int ip_a = insertion_point(Ta, ts, 0, ts.size());
    int ip_b = insertion_point(Tb, ts, 0, ts.size());
    T ret = zero, r;
    for (int i = ip_a; i < ip_b + 1; ++i)
    {
        T left = dmax(Ta, ts[i]);
        T right = dmin(Tb, ts[i + 1]);
        T diff = right - left;
        if (adb[i] == 0)
        {
            T Rleft = R(left);
            r = -exp(-Rleft) * expm1(-diff * ada[i]);
            r /= ada[i];
        }
        else
        {
            T adab = ada[i] / adb[i];
            T c1 = -adab * exp(adb[i] * (left - ts[i]));
            T c2 = -adab * exp(adb[i] * (right - ts[i]));
            T c3 = adab - Rrng[i];
            r = eintdiff(c1, c2, c3) / adb[i];
            check_negative(r);
            check_nan(r);
        }
        // if (r > 100.)
        //     throw std::domain_error("what");
        check_negative(r);
        check_nan(r);
        ret += r;
    }
    return ret;
}


template <typename T>
inline T _single_integral(const int rate, const T &tsm, const T &tsm1, const T &ada, const T &adb, const T &Rrng, const T &log_coef)
{
    // = int_ts[m]^ts[m+1] exp(-rate * R(t)) dt
    const int c = rate;
    if (rate == 0)
        return exp(log_coef) * (tsm1 - tsm);
    if (adb == 0.)
    {
        T ret = exp(-c * Rrng + log_coef);
        if (tsm1 < INFINITY)
            ret *= -expm1(-c * ada * (tsm1 - tsm));
        ret /= ada * c;
        check_negative(ret);
        return ret;
    }
    T e1 = -c * ada / adb;
    T e2 = -c * exp(adb * (tsm1 - tsm)) * ada / adb;
    T e3 =  c * (ada / adb - Rrng) + log_coef;
    T ret = eintdiff(e1, e2, e3) / adb;
    check_nan(ret);
    check_negative(ret);
    return ret;
}

template <typename T>
inline T _double_integral_below_helper_ei(const int rate, const T &tsm, const T &tsm1, 
        const T &ada, const T &adb, const T &Rrng, const T &log_denom)
{
    // We needn't cover the tsm1==INFINITY case here as the last piece is assumed flat (i.e., adb=0).
    long c = rate;
    T eadb = exp(adb * (tsm1 - tsm));
    T adadb = ada / adb;
    if (c == 0)
    {
        T a1 = -adadb;
        T b1 = -eadb * adadb;
        T cons1 = adadb - Rrng - log_denom;
        T int1 = eintdiff(a1, b1, cons1);
        int1 /= adb;
        int1 += exp(adadb * (1. - eadb) - Rrng - log_denom) * (tsm - tsm1);
        check_negative(int1);
        check_nan(int1);
        return int1;
    }
    T cons1 = (2 + c) * adadb;
    T cons2 = adadb * (2 + c + eadb);
    T a1 = -c * adadb * eadb;
    T b1 = -c * adadb;
    T int1 = eintdiff(a1, b1, cons1);
    T a2 = -(c + 1) * adadb;
    T b2 = -(c + 1) * adadb * eadb;
    T int2 = eintdiff(a2, b2, cons2);
    T cons3 = exp(-(ada * (1 + eadb) / adb + (1 + c) * Rrng) - log_denom);
    T ret = cons3 * (int1 + int2) / adb;
    check_negative(ret);
    check_nan(ret);
    return ret;
}

template <typename T>
inline T _double_integral_above_helper_ei(const int rate, const int lam, const T &tsm, const T &tsm1, 
        const T &ada, const T &adb, const T &Rrng, const T &log_coef)
{
    long d = lam + 1;
    long c = rate;
    T eadb = exp(adb * (tsm1 - tsm));
    T cons1 = ada * c / adb;
    T a1 = -cons1 * eadb;
    T b1 = -cons1;
    T c1 = cons1 - d * Rrng + log_coef;
    T ed1 = eintdiff(a1, b1, c1);
    if (c != d)
    {
        T cons2 = ada * d / adb;
        T a2 = -cons2;
        T b2 = -cons2 * eadb;
        T c2 = cons2 - d * Rrng;
        return (ed1 + eintdiff(a2, b2, c2)) / adb / (c - d);
    }
    T ret = (exp(-d * Rrng + log_coef) * (-adb * expm1(-ada / adb * d * expm1(adb * (tsm1 - tsm)))) + ada * d * ed1) / (adb * adb * d);
    check_negative(ret);
    check_nan(ret);
    return ret;
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::tjj_double_integral_above(const int n, long jj, std::vector<Matrix<T> > &C) const
{
    T tmp;
    long lam = nC2(jj) - 1;
    // Now calculate with hidden state integration limits
    for (unsigned int h = 0; h < hs_indices.size() - 1; ++h)
    {
        C[h].row(jj - 2).setZero();
        T log_denom = -Rrng[hs_indices[h]] + log(-expm1(-(Rrng[hs_indices[h + 1]] - Rrng[hs_indices[h]])));
        for (int m = hs_indices[h]; m < hs_indices[h + 1]; ++m)
        {
            for (int j = 2; j < n + 2; ++j)
            {
                long rate = nC2(j);
                if (adb[m] == 0)
                    tmp = _double_integral_above_helper<T>(rate, lam, ts[m], ts[m + 1], ada[m], Rrng[m], -log_denom);
                else
                    tmp = _double_integral_above_helper_ei<T>(rate, lam, ts[m], ts[m + 1], ada[m], adb[m], Rrng[m], -log_denom);
                try 
                {
                    check_nan(C[h](jj - 2, j - 2));
                    check_negative(C[h](jj - 2, j - 2));
                } catch (std::runtime_error)
                {
                    CRITICAL(m << " " << rate << " " << lam << " " << ts[m] << " " << ts[m + 1] << " " << ada[m] << " " << adb[m] << " " << Rrng[m] << " " << log_denom);
                    throw;
                }
                C[h](jj - 2, j - 2) += tmp;
                T log_coef = -log_denom, fac;
                long rp = lam + 1 - rate;
                if (rp == 0)
                    fac = Rrng[m + 1] - Rrng[m];
                else
                {
                    if (rp < 0)
                    {
                        if (-rp * (Rrng[m + 1] - Rrng[m]) > 20)
                        {
                            log_coef += -rp * Rrng[m + 1];
                            fac = -one / rp;
                        }
                        else
                        {
                            log_coef += -rp * Rrng[m];
                            fac = -expm1(-rp * (Rrng[m + 1] - Rrng[m])) / rp;
                        }
                    }
                    else
                    {
                        if (-rp * (Rrng[m] - Rrng[m + 1]) > 20)
                        {
                            log_coef += -rp * Rrng[m];
                            fac = one / rp;
                        }
                        else
                        {
                            log_coef += -rp * Rrng[m + 1];
                            fac = expm1(-rp * (Rrng[m] - Rrng[m + 1])) / rp;
                        }
                    }
                }
                for (int k = m + 1; k < K; ++k)
                {
                    T si = _single_integral(rate, ts[k], ts[k + 1], ada[k], adb[k], Rrng[k], log_coef) * fac;
                    C[h](jj - 2, j - 2) += si;
                    check_nan(C[h](jj - 2, j - 2));
                    check_negative(C[h](jj - 2, j - 2));
                }
                check_nan(C[h](jj - 2, j - 2));
                check_negative(C[h](jj - 2, j - 2));
            }
        }
    }
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::tjj_double_integral_below(
        const int n, const int h, Matrix<T> &tgt) const
{
    T log_denom = -Rrng[hs_indices[h]] + log(-expm1(-(Rrng[hs_indices[h + 1]] - Rrng[hs_indices[h]])));
    for (int m = hs_indices[h]; m < hs_indices[h + 1]; ++m)
    {
        Vector<T> ts_integrals(n + 1);
        T log_coef = -Rrng[m];
        T fac = one;
        if (m < K - 1)
            fac = -expm1(-(Rrng[m + 1] - Rrng[m]));
        for (int j = 2; j < n + 3; ++j)
        {
            long rate = nC2(j) - 1;
            if (adb[m] == 0.)
                ts_integrals(j - 2) = _double_integral_below_helper<T>(rate, ts[m], ts[m + 1], ada[m], Rrng[m], -log_denom);
            else
            {
                throw std::runtime_error("b!=0 has been disabled");
                // ts_integrals(j - 2) = _double_integral_below_helper_ei(rate, ts[m], ts[m + 1], ada[m], adb[m], Rrng[m] - log_denom);
            }
            for (int k = 0; k < m; ++k)
            {
                T _c = log_coef - log_denom;
                ts_integrals(j - 2) += fac * _single_integral(rate, ts[k], ts[k + 1], ada[k], adb[k], Rrng[k], _c);
            }
            check_negative(ts_integrals(j - 2));
        }
        tgt.row(h) += ts_integrals.transpose();
    }
}

template <typename T>
T exp1_conditional(T a, T b, std::mt19937 &gen)
{
    // If X ~ Exp(1),
    // P(X < x | a <= X <= b) = (e^-a - e^-x) / (e^-a - e^-b)
    // so P^-1(y) = -log(e^-a - (e^-a - e^-b) * y)
    //            = -log(e^-a(1 - (1 - e^-(b-a)) * y)
    //            = a - log(1 - (1 - e^-(b-a)) * y)
    //            = a - log(1 + expm1(-(b-a)) * y)
    double unif = std::uniform_real_distribution<double>{0.0, 1.0}(gen);
    if (std::isinf(toDouble(b)))
        return a - log1p(-unif);
    else
        return a - log1p(expm1(-(b - a)) * unif);
}

template <typename T>
T PiecewiseExponentialRateFunction<T>::random_time(const double fac, const T &a, const T &b, std::mt19937 &gen) const
{
    T Rb;
    if (b == INFINITY)
        Rb = R(0.99 * T_MAX);
    else
        Rb = R(b);
    return (*getRinv())(exp1_conditional(R(a), Rb, gen) / fac);
}


template <typename T>
T PiecewiseExponentialRateFunction<T>::random_time(const T &a, const T &b, std::mt19937 &gen) const
{
    return random_time(1.0, a, b, gen);
}


template class PiecewiseExponentialRateFunction<double>;
template class PiecewiseExponentialRateFunction<adouble>;
