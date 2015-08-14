#include "piecewise_exponential_rate_function.h"

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params) : 
    PiecewiseExponentialRateFunction(params, std::vector<std::pair<int, int>>()) {}

std::vector<std::pair<int, int>> derivatives_from_params(const std::vector<std::vector<double>> params)
{
    std::vector<std::pair<int, int>> ret;
    for (size_t i = 0; i < params.size(); ++i)
        for (size_t j = 0; j < params[0].size(); ++j)
            ret.emplace_back(i, j);
    return ret;
} 

template <>
PiecewiseExponentialRateFunction<adouble>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params) : 
    PiecewiseExponentialRateFunction(params, derivatives_from_params(params)) {}

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
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params, 
        const std::vector<std::pair<int, int>> derivatives) :
    params(params),
    derivatives(derivatives), K(params[0].size()), ada(params[0].begin(), params[0].end()), 
    adb(params[1].begin(), params[1].end()), ads(params[2].begin(), params[2].end()),
    ts(K + 1), Rrng(K), _reg(0.0), 
    zero(init_derivative(0.0)), one(init_derivative(1.0))
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
mpreal_wrapper<T> PiecewiseExponentialRateFunction<T>::mpfr_tjj_integral(
    const long rate, T t1, T t2, T offset, mp_prec_t prec) const
{
    mpfr::mpreal::set_default_prec(prec);
    T right;
    mpreal_wrapper<T> ret;
    // Compute \int_0^(t2-t1) exp(-rate * \int_t1^t eta(s) ds) dt
    //  = \int_0^(t2-t1) exp(-rate * (R(t) - R(t1))) dt
    int start = 0;
    if (t1 > 0)
        start = insertion_point(t1, ts, 0, K);
    int end = K;
    if (! std::isinf(toDouble(t2)))
        end = insertion_point(t2, ts, 0, K) + 1;
    ret = mpfr::mpreal("0");
    mpreal_wrapper<T> _rate, _Rrng, _offset, _ada, _ts, _right, c1, c2, c3;
    for (int m = start; m < end; ++m)
    {
        // c = Rrng[m] - offset;
        // left = tsm;
        // if (t1 > tsm)
            // left = mpreal_wrapper<T>(t1);
        right = ts[m + 1];
        if (m == K - 1 or t2 < ts[m + 1])
            right = t2;
        _Rrng = convert(Rrng[m]);
        _offset = convert(offset);
        _ada = convert(ada[m]);
        _ts = convert(ts[m]);
        _right = convert(right);
        c1 = rate * (_Rrng - _offset);
        c2 = rate * _ada * (_ts - _right);
        c3 = _ada * rate;
        ret -= (exp(-c1) * expm1(c2)) / c3;
    }
    return ret;
} 


template <typename T>
Matrix<mpreal_wrapper<T> > PiecewiseExponentialRateFunction<T>::mpfr_tjj_double_integral(const int n, const std::vector<double> hidden_states, const mp_prec_t prec) const
{
    mpfr::mpreal::set_default_prec(prec);
    long int rate;
    mpreal_wrapper<T> _Rrng, _Rrng1, _ada, _ts, _tsm1, diff, _hs, _adadiff;
    Matrix<mpreal_wrapper<T> > inner_integrals(K, n + 1), double_integrals(K, n + 1);
    Vector<mpreal_wrapper<T> > single_integrals(K);
    inner_integrals.setZero();
    double_integrals.setZero();
    //
    // \int_0^t_k alpha(tau) exp(-R(tau)) \int_0^\tau exp(-rate * R(t)) dt
    //    = \sum_{m=0}^{k-1} \int_{t_m}^{t_{m+1}} a[m] * exp(-(a[m](t - t[m]) + Rrng[m])) * 
    //      [(\sum_{ell=0}^{m-2} \int_t[ell]^t[ell+1] exp(-rate * (a[ell](t - t[ell]) + Rrng[ell]))) + 
    //      (\int_t[m]^tau exp(-rate * (a[m](t - t[m]) + Rrng[m])))]
    //
    /*
    std::vector<mpreal_wrapper<T> > mpRrng(K + 1), mpts(K + 1);
    std::transform(Rrng.begin(), Rrng.end(), mpRrng.begin(), convert);
    std::transform(ts.begin(), ts.end(), mpts.begin(), convert);
    */
    for (int m = 0; m < K; ++m)
    {
        _Rrng = convert(Rrng[m]);
        _Rrng1 = convert(Rrng[m + 1]);
        _ada = convert(ada[m]);
        _ts = convert(ts[m]);
        _tsm1 = convert(ts[m + 1]);
        diff = _tsm1 - _ts;
        _adadiff = _ada * diff;
        if (m + 1 < K)
        {
            inner_integrals(m + 1, 0) = _tsm1;
            // double_integrals(m, 0) = exp(-_Rrng) * (1 - exp(-_adadiff) * (1 + _adadiff)) / _ada;
            double_integrals(m, 0) = -exp(-_Rrng) * (expm1(-_adadiff) + _adadiff * exp(-_adadiff)) / _ada;
            // single_integrals(m) = exp(-_Rrng) - exp(-_Rrng1);
            single_integrals(m) = -exp(-_Rrng) * expm1(-_adadiff);
        }
        else
        {
            double_integrals(m, 0) = exp(-_Rrng) / _ada;
            single_integrals(m) = exp(-_Rrng);
        }
        for (int j = 3; j < n + 3; ++j)
        {
            rate = j * (j - 1) / 2 - 1;
            // \int_0^t[m] exp(-rate * R(t)) dt
            if (m + 1 < K)
            {
                // inner_integrals(m + 1, j - 2) = (exp(-rate * _Rrng) - exp(-rate * _Rrng1)) / _ada / rate + 
                    // inner_integrals(m, j - 2);
                inner_integrals(m + 1, j - 2) = -(exp(-rate * _Rrng) * expm1(-rate * _adadiff)) / _ada / rate + 
                    inner_integrals(m, j - 2);
                double_integrals(m, j - 2) = exp(-(rate + 1) * _Rrng1) + exp(-(rate + 1) * _Rrng) * rate;
            } 
            else 
            {
                double_integrals(m, j - 2) = exp(-(rate + 1) * _Rrng) * rate;
            }
            double_integrals(m, j - 2) /= rate + 1;
            if (m + 1 < K)
                double_integrals(m, j - 2) -= exp(-(rate + 1) * _Rrng - _ada * diff);
            double_integrals(m, j - 2) /= rate * _ada;
        }
    }
    // \int_ts[m]^ts[m+1] eta(tau) * exp(-R(tau)) * \int_0^ts[m] exp(-rate * R(t))
    Matrix<mpreal_wrapper<T> > ts_integrals = double_integrals + single_integrals.asDiagonal() * inner_integrals;
    std::vector<double> hs(hidden_states.begin(), hidden_states.end());
    if (hidden_states[0] > 0.0)
        hs.insert(hs.begin(), 0.0);
    if (! isinf(hidden_states.back()))
        hs.push_back(INFINITY);
    size_t H = hs.size();
    Matrix<mpreal_wrapper<T> > tmp(1, n + 1), prev_int(1, n + 1), new_int(1, n + 1), ret(H - 1, n + 1);
    mpreal_wrapper<T> _Rcur, _Rlast;
    // Make sure it has the correct # of derivatives
    _Rlast = convert(zero);
    prev_int.fill(_Rlast);
    for (int h = 1; h < H - 1; ++h)
    {
        int hi = insertion_point((T)hs[h], ts, 0, K);
        _Rrng = convert(Rrng[hi]);
        _Rrng1 = convert(Rrng[hi + 1]);
        _ada = convert(ada[hi]);
        _ts = convert(ts[hi]);
        _hs = convert(hs[h]);
        diff = _hs - _ts;
        _adadiff = _ada * diff;
        // Integrals
        new_int = ts_integrals.topRows(hi).colwise().sum();
        // new_int += inner_integrals.row(hi) * (exp(-_Rrng) - exp(-(_ada * diff + _Rrng)));
        new_int += inner_integrals.row(hi) * -(exp(-_Rrng) * expm1(-_adadiff));
        // new_int(0) += exp(-_Rrng) * (1 - exp(-_ada * diff) * (1 + _ada * diff)) / _ada;
        new_int(0) += exp(-_Rrng) * -(expm1(-_adadiff) + exp(-_adadiff) * _adadiff) / _ada;
        for (int j = 3; j < n + 3; ++j)
        {
            rate = j * (j - 1) / 2 - 1;
            // \int_0^t[m] exp(-rate * R(t)) dt
            // tmp(j - 2) = exp(-(rate + 1) * (_ada * diff + _Rrng)) + exp(-(rate + 1) * _Rrng) * rate;
            tmp(j - 2) = exp(-(rate + 1) * (_adadiff + _Rrng)) + exp(-(rate + 1) * _Rrng) * rate;
            tmp(j - 2) /= rate + 1;
            tmp(j - 2) -= exp(-(rate + 1) * _Rrng - _ada * diff);
            tmp(j - 2) /= rate * _ada;
        }
        new_int += tmp;
        ret.row(h - 1) = new_int - prev_int;
        // Conditional coalescence distribution
        // ret.row(h - 1) /= exp(-_Rlast) - exp(-_Rcur);
        ret.row(h - 1) /= exp(-_Rrng) * -expm1(-_adadiff);
        prev_int = new_int;
    }
    ret.bottomRows(1) = (ts_integrals.colwise().sum() - prev_int) / exp(-_Rrng);
    if (hidden_states[0] > 0.0)
        ret = ret.bottomRows(ret.rows() - 1).eval();
    if (! isinf(hidden_states.back()))
        ret = ret.topRows(ret.rows() - 1).eval();
    return ret;
}

template <typename T>
T PiecewiseExponentialRateFunction<T>::tjj_integral(double rate, T t1, T t2, T offset) const
{
    // Compute \int_0^(t2-t1) exp(-rate * \int_t1^t eta(s) ds) dt
    //  = \int_0^(t2-t1) exp(-rate * (R(t) - R(t1))) dt
    int start = 0;
    if (t1 > 0)
        start = insertion_point(t1, ts, 0, K);
    int end = K;
    if (! std::isinf(toDouble(t2)))
        end = insertion_point(t2, ts, 0, K) + 1;
    T ret = this->zero, left, right;
    for (int m = start; m < end; ++m)
    {
        T c = Rrng[m] - offset;
        left = dmax(t1, ts[m]);
        if (std::isinf(toDouble(t2)))
            right = ts[m + 1];
        else if (m == K - 1)
            right = t2;
        else
            right = dmin(t2, ts[m + 1]);
        if (std::isinf(toDouble(right)))
        {
            // here we assume that adb[k] == 0 (flat last piece)
            // R(t) = Rrng[K - 1] + ada[K - 1](t - ts[K - 1])
            ret += exp(-rate * (ada[m] * (left - ts[m]) + c)) / ada[m] / rate;
        } 
        else if (adb[m] == 0.0)
        {
            /* ret += exp(-rate * (ada[m] * (right - ts[m]) + c)) * expm1(-rate * (ada[m] * (left - right))) / ada[m] / rate; */
            ret -= (exp(-rate * (ada[m] * (right - ts[m]) + c)) - 
                    exp(-rate * (ada[m] * (left - ts[m]) + c))) / ada[m] / rate;
        }
        else
        {
            T r, ei1, ei2;
            T c1 = -ada[m] / adb[m] * rate;
            T args[2] = {exp(adb[m] * (left - ts[m])), exp(adb[m] * (right - ts[m]))};
            T out[2];
            for (int i = 0; i < 2; ++i)
            {
                T argi = c1 * args[i];
                if (myabs(argi) > 45)
                {
                    out[i] = 1.0;
                    double fac = 1.0;
                    T outer = exp(c1 * (args[i] - 1.0) - rate * c) / argi;
                    for (int i = 1; i < 10; ++i)
                    {
                        fac *= i;
                        ei1 += fac / argi;
                        argi *= argi;
                    }
                    out[i] *= outer;
                } else
                {
                    // out[i] = exponentialintegralei(argi) * exp(-c1 - rate * c);
                }
            }
            // ei1 = exponentialintegralei(arg1)
            // ei2 = exponentialintegralei(arg2)
            // ret += (ei2 - ei1) * exp(rate * (ada[m] / adb[m] - c)) / adb[m];
            ret += (out[1] - out[0]) / adb[m];
            if (std::isnan(toDouble(ret)))
            {
                std::cout << "NaN in tjj_integral after Ei: " << toDouble(out[1]) << " " << toDouble(out[0]) << std::endl;
                throw std::domain_error("ack");
            }
        }
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
    for (int k = 0; k < K - 1; ++k)
    {
        if (adb[k] == 0.0)
            Rrng[k + 1] = Rrng[k] + ada[k] * (ts[k + 1] - ts[k]);
        else
            Rrng[k + 1] = Rrng[k] + (ada[k] / adb[k]) * expm1(adb[k] * (ts[k + 1] - ts[k]));
    }
    // insertion_point() assumes that the last entry is always +oo.
    Rrng.push_back(INFINITY);
}

template class PiecewiseExponentialRateFunction<double>;
template class PiecewiseExponentialRateFunction<adouble>;
