#include "piecewise_constant_rate_function.h"

constexpr long nC2(int n) { return n * (n - 1) / 2; }

template <typename T>
inline T _conv(const adouble x);

template <typename T>
inline std::vector<T> _vconv(const std::vector<adouble> v);

template <>
inline double _conv(const adouble x) { return x.value(); }

template <>
inline adouble _conv(const adouble x) { return x; }

template <>
inline std::vector<adouble> _vconv(const std::vector<adouble> v) { return v; }

template <>
inline std::vector<double> _vconv(const std::vector<adouble> v) 
{ 
    std::vector<double> ret; 
    for (adouble x : v)
        ret.push_back(x.value());
    return ret;
}

template <typename T>
PiecewiseConstantRateFunction<T>::PiecewiseConstantRateFunction(
        const std::vector<std::vector<adouble>> params, 
        const std::vector<double> hidden_states) :
    params(params),
    nder(params[0][0].derivatives().size()),
    K(params[0].size()), 
    ada(_vconv<T>(params[0])),
    s(_vconv<double>(params[1])),
    ts(K + 1), Rrng(K), 
    hidden_states(hidden_states)
{
    for (auto &pp : params)
        if (pp.size() != params[0].size())
            throw std::runtime_error("all params must have same size");
    // Final piece is required to be flat.
    ts[0] = 0.;
    Rrng[0] = 0.;
    // These constant values need to have compatible derivative shape
    // with the calculated values.
    // Fix last piece to be constant
    for (int k = 0; k < K; ++k)
    {
        ada[k] = 1. / ada[k];
        ts[k + 1] = ts[k] + s[k];
    }
    ts[K] = INFINITY;

    for (double h : hidden_states)
    {
        if (std::isinf(h))
        {
            hs_indices.push_back(ts.size() - 1);
            continue;
        }
        std::vector<double>::iterator ti = std::upper_bound(ts.begin(), ts.end(), h) - 1;
        int ip = ti - ts.begin();
        if (std::abs(*ti - h) < 1e-8)
        // if (ts[ip] == h)
            hs_indices.push_back(ip);
        else if (ti + 1 < ts.end() and std::abs(*(ti + 1) - h) < 1e-8)
            hs_indices.push_back(ip + 1);
        else
        {
            ts.insert(ti + 1, h);
            ada.insert(ada.begin() + ip + 1, ada[ip]);
            CHECK_NAN(ada[ip + 1]);
            CHECK_NAN(ts[ip + 1]);
            hs_indices.push_back(ip + 1);
        }
    }
    K = ada.size();
    Rrng.resize(K + 1);
    compute_antiderivative();
}

template <typename T>
inline T _double_integral_below_helper(const int rate, const double &tsm, const double &tsm1, const T &ada, 
        const T &Rrng, const T &log_denom)
{
    if (ada == 0)
        return 0.;
    const int l1r = 1 + rate;
    const double l1rinv = 1. / (double)l1r;
    double diff = tsm1 - tsm;
    T adadiff = ada * diff;
    if (rate == 0)
    {
        if (tsm1 == INFINITY)
            return exp(-Rrng - log_denom) / ada;
        else
            return exp(-Rrng - log_denom) * (1. - exp(-adadiff) * (1. + adadiff)) / ada;
    }
    if (tsm1 == INFINITY)
        return exp(-l1r * Rrng - log_denom) * (1. - l1rinv) / (rate * ada);
    return exp(-l1r * Rrng - log_denom) * (expm1(-l1r * adadiff) * l1rinv - expm1(-adadiff)) / (rate * ada);
}

template <typename T>
inline T _double_integral_above_helper(const int rate, const int lam, const double tsm, 
        const double tsm1, const T ada, const T Rrng, const T log_coef)
{
    if (ada == 0)
        return 0.;
    double diff = tsm1 - tsm;
    T adadiff = ada * diff;
    long l1 = lam + 1;
    if (rate == 0)
    {
        return exp(-l1 * Rrng + log_coef) * (expm1(-l1 * adadiff) + l1 * adadiff) / l1 / l1 / ada;
    }
    if (l1 == rate)
    {
        if (tsm1 == INFINITY)
            return exp(-rate * Rrng + log_coef) / rate / rate / ada;
        return exp(-rate * Rrng + log_coef) * (1 - exp(-rate * adadiff) * (1 + rate * adadiff)) / rate / rate / ada;
    }
    if (tsm1 == INFINITY)
        return exp(-l1 * Rrng + log_coef) / l1 / rate / ada;
    // return -exp(-l1 * _Rrng + log_coef) * (expm1(-l1 * adadiff) / l1 + (exp(-rate * adadiff) - exp(-l1 * adadiff)) / (l1 - rate)) / rate / _ada;
    if (rate < l1)
        return -exp(-l1 * Rrng + log_coef) * (expm1(-l1 * adadiff) / l1 + (exp(-rate * adadiff) * -expm1(-(l1 - rate) * adadiff) / (l1 - rate))) / rate / ada;
    else
        return -exp(-l1 * Rrng + log_coef) * 
            (
             expm1(-l1 * adadiff) / l1 + 
             (exp(-l1 * adadiff) * expm1(-(rate - l1) * adadiff) / (l1 - rate))
            ) / rate / ada;
}

template <typename T>
void PiecewiseConstantRateFunction<T>::print_debug() const
{
    std::vector<std::pair<std::string, std::vector<T> > > arys = {{"ada", ada}, {"Rrng", Rrng}};
    for (auto p : arys)
    {
        CRITICAL << p.first << "\n";
        for (adouble x : p.second)
        {
            CRITICAL << x.value()
                     << "::" << x.derivatives().transpose() 
                     << "\n";
        }
    }
}

template <typename T>
void PiecewiseConstantRateFunction<T>::compute_antiderivative()
{
    Rrng[0] = 0.;
    for (int k = 0; k < K; ++k)
        Rrng[k + 1] = Rrng[k] + ada[k] * (ts[k + 1] - ts[k]);
}

template <typename T>
T PiecewiseConstantRateFunction<T>::R_integral(const double a, const double b, const T log_denom) const
{
    // int_a^b exp(-R(t)) dt
    int ip_a = std::upper_bound(ts.begin(), ts.end(), a) - 1 - ts.begin();
    int ip_b = std::upper_bound(ts.begin(), ts.end(), b) - 1 - ts.begin();
    // If b == inf the comparison is always false so a special case is needed.
    ip_b = std::isinf(b) ? ts.size() - 2 : ip_b;
    T ret = 0., r, Rleft;
    double left, right, diff;
    for (int i = ip_a; i < ip_b + 1; ++i)
    {
        left = std::max(a, ts[i]);
        right = std::min(b, ts[i + 1]);
        diff = right - left;
        Rleft = R(left);
        r = exp(-(Rleft + log_denom));
        if (!std::isinf(toDouble(diff)))
            r *= -expm1(-diff * ada[i]);
        r /= ada[i];
        CHECK_NAN_OR_NEGATIVE(r);
        ret += r;
    }
    return ret;
}


template <typename T>
inline T _single_integral(const int rate, const double &tsm, const double &tsm1, 
        const T &ada, const T &Rrng, const T &log_coef)
{
    // = int_ts[m]^ts[m+1] exp(-rate * R(t)) dt
    const int c = rate;
    if (rate == 0)
        return exp(log_coef) * (tsm1 - tsm);
    T ret = exp(-c * Rrng + log_coef);
    if (tsm1 < INFINITY)
        ret *= -expm1(-c * ada * (tsm1 - tsm));
    ret /= ada * c;
    CHECK_NAN_OR_NEGATIVE(ret);
    return ret;
}

template <typename T>
void PiecewiseConstantRateFunction<T>::tjj_double_integral_above(const int n, long jj, std::vector<Matrix<T> > &C) const
{
    T tmp;
    long lam = nC2(jj) - 1;
    // Now calculate with hidden state integration limits
    for (unsigned int h = 0; h < hs_indices.size() - 1; ++h)
    {
        C[h].row(jj - 2).setZero();
        T log_denom = -Rrng[hs_indices[h]];
        if (Rrng[hs_indices[h + 1]] != INFINITY)
            log_denom += log(-expm1(-(Rrng[hs_indices[h + 1]] - Rrng[hs_indices[h]])));
        for (int m = hs_indices[h]; m < hs_indices[h + 1]; ++m)
        {
            for (int j = 2; j < n + 2; ++j)
            {
                long rate = nC2(j);
                tmp = _double_integral_above_helper<T>(rate, lam, ts[m], ts[m + 1], ada[m], Rrng[m], -log_denom);
                try 
                {
                    CHECK_NAN_OR_NEGATIVE(tmp);
                    CHECK_NAN_OR_NEGATIVE(C[h](jj - 2, j - 2));
                } 
                catch (std::runtime_error)
                {
                    CRITICAL << "nan detected:\n j=" << j << "m=" << m << " rate=" << rate 
                             << " lam=" << lam << " ts[m]=" << ts[m] << " ts[m + 1]=" 
                             << ts[m + 1] << " ada[m]=" << ada[m] << " Rrng[m]=" << Rrng[m] 
                             << " log_denom=" << log_denom;
                    CRITICAL << "tmp=" << tmp;
                    CRITICAL << "C[h](jj - 2, j - 2)= " << C[h](jj - 2, j - 2);
                    CRITICAL << "h=" << h;
                    CRITICAL << "I am eta: ";
                    print_debug();
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
                            fac = -1. / rp;
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
                            fac = 1. / rp;
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
                    T si = _single_integral(rate, ts[k], ts[k + 1], ada[k], Rrng[k], log_coef) * fac;
                    C[h](jj - 2, j - 2) += si;
                    CHECK_NAN_OR_NEGATIVE(C[h](jj - 2, j - 2));
                }
                CHECK_NAN_OR_NEGATIVE(C[h](jj - 2, j - 2));
            }
        }
    }
}

template <typename T>
void PiecewiseConstantRateFunction<T>::tjj_double_integral_below(
        const int n, const int h, Matrix<T> &tgt) const
{
    DEBUG << "in tjj_double_integral_below";
    T log_denom = -Rrng[hs_indices[h]];
    if (Rrng[hs_indices[h + 1]] != INFINITY)
        log_denom += log(-expm1(-(Rrng[hs_indices[h + 1]] - Rrng[hs_indices[h]])));
    for (int m = hs_indices[h]; m < hs_indices[h + 1]; ++m)
    {
        Vector<T> ts_integrals(n + 1);
        T log_coef = -Rrng[m];
        T fac = 1.;
        if (m < K - 1)
            fac = -expm1(-(Rrng[m + 1] - Rrng[m]));
        for (int j = 2; j < n + 3; ++j)
        {
            long rate = nC2(j) - 1;
            ts_integrals(j - 2) = _double_integral_below_helper<T>(rate, ts[m], ts[m + 1], ada[m], Rrng[m], log_denom);
            for (int k = 0; k < m; ++k)
            {
                T _c = log_coef - log_denom;
                ts_integrals(j - 2) += fac * _single_integral(rate, ts[k], ts[k + 1], ada[k], Rrng[k], _c);
            }
            CHECK_NAN_OR_NEGATIVE(ts_integrals(j - 2));
        }
        tgt.row(h) += ts_integrals.transpose();
    }
    DEBUG << "exiting tjj_double_integral_below";
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
T PiecewiseConstantRateFunction<T>::random_time(const double a, const double b, const long long seed) const
{
    std::mt19937 gen(seed);
    return random_time(1., a, b, gen);
}

template <typename T>
T PiecewiseConstantRateFunction<T>::random_time(const double fac, const double a, const double b, std::mt19937 &gen) const
{
    T Rb;
    if (b == INFINITY)
        Rb = INFINITY;
    else
        Rb = R(b);
    return Rinv(exp1_conditional(R(a), Rb, gen) / fac);
}


template <typename T>
std::vector<T> PiecewiseConstantRateFunction<T>::average_coal_times() const
{
    std::vector<T> ret;
    for (int i = 1; i < hidden_states.size(); ++i)
    {
        // discretize by expected coalescent time within each hidden state
        // e_coal = \int_t0^t1 t eta(t) exp(-R(t)) 
        //        = t0 e^(-R(t0)) - t1 e^(-R(t1)) + \int
        T log_denom = -Rrng[hs_indices[i - 1]];
        bool inf = std::isinf(toDouble(ts[hs_indices[i]]));
        if (!inf)
           log_denom += log(-expm1(-(Rrng[hs_indices[i]] - Rrng[hs_indices[i - 1]])));
        T x = hidden_states[i - 1] * exp(-((Rrng[hs_indices[i - 1]]) + log_denom)) +
            R_integral(ts[hs_indices[i - 1]], ts[hs_indices[i]], log_denom);
        if (!inf)
            x -= hidden_states[i] * exp(-((Rrng[hs_indices[i]]) + log_denom));
        ret.push_back(x);
        if (ret.back() > hidden_states[i] or ret.back() < hidden_states[i - 1])
            throw std::runtime_error("erroneous average coalescence time");
    }
    return ret;
}

template <typename T>
T PiecewiseConstantRateFunction<T>::R(const T t) const
{
    std::vector<double>::const_iterator ti = std::upper_bound(ts.begin(), ts.end(), toDouble(t)) - 1;
    int ip = ti - ts.begin();
    return Rrng[ip] + ada[ip] * (t - *ti);
}


template <typename T>
T PiecewiseConstantRateFunction<T>::Rinv(const T y) const
{
    typename std::vector<T>::const_iterator R = std::upper_bound(Rrng.begin(), Rrng.end(), y) - 1;
    int ip = R - Rrng.begin();
    return (y - *R) / ada[ip] + ts[ip];
}

template class PiecewiseConstantRateFunction<double>;
template class PiecewiseConstantRateFunction<adouble>;
