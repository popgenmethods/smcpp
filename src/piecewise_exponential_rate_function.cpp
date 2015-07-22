#include "piecewise_exponential_rate_function.h"

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> params) : 
    PiecewiseExponentialRateFunction(params, std::vector<std::pair<int, int>>()) {}

std::vector<std::pair<int, int>> derivatives_from_params(const std::vector<std::vector<double>> params)
{
    std::vector<std::pair<int, int>> ret;
    for (int i = 0; i < params.size(); ++i)
        for (int j = 0; j < params[0].size(); ++j)
            ret.emplace_back(i, j);
    // std::cout << "building in all derivatives" << std::endl;
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
        ada[k] = 1. / ada[k];
        adb[k] = 1. / adb[k];
        ts[k + 1] = ts[k] + ads[k];
        adb[k] = (log(adb[k]) - log(ada[k])) / (ts[k + 1] - ts[k]);
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
        (*dl[p.first])[p.second].derivatives() = I.col(d++);
    ts[0].derivatives() = z;
    Rrng[0].derivatives() = z;
}

template <typename T>
mpfr::mpreal PiecewiseExponentialRateFunction<T>::mpfr_tjj_integral(double rate, T t1, T t2, T offset) const
{
    // Compute \int_0^(t2-t1) exp(-rate * \int_t1^t eta(s) ds) dt
    //  = \int_0^(t2-t1) exp(-rate * (R(t) - R(t1))) dt
    int start = 0;
    if (t1 > 0)
        start = insertion_point(t1, ts, 0, K);
    int end = K;
    if (! std::isinf(toDouble(t2)))
        end = insertion_point(t2, ts, 0, K) + 1;
    mpfr::mpreal ret = 0.0, left, right;
    for (int m = start; m < end; ++m)
    {
        double R = toDouble(Rrng[m]);
        double c = R - toDouble(offset);
        double a = toDouble(ada[m]);
        double b = toDouble(adb[m]);
        double tsm = toDouble(ts[m]);
        double tsm1 = toDouble(ts[m + 1]);
        left = tsm;
        if (t1 > tsm)
            left = toDouble(t1);
        right = tsm1;
        if (m == K - 1 or t2 < tsm1)
            right = toDouble(t2);
        if (mpfr::isinf(right))
        {
            // here we assume that adb[k] == 0 (flat last piece)
            // R(t) = Rrng[K - 1] + ada[K - 1](t - ts[K - 1])
            ret += exp(-rate * (a * (left - tsm) + c)) / a / rate;
        } 
        else if (b == 0.0)
        {
            ret -= (exp(-rate * (a * (right - tsm) + c)) - 
                    exp(-rate * (a * (left - tsm) + c))) / a / rate;
        }
        else
        {
            mpfr::mpreal c1 = -a / b * rate;
            mpfr::mpreal arg1 = c1 * exp(b * (left - tsm));
            mpfr::mpreal arg2 = c1 * exp(b * (right - tsm));
            mpfr::mpreal ei1 = exponentialintegralei(arg1);
            mpfr::mpreal ei2 = exponentialintegralei(arg2);
            ret += (ei2 - ei1) * exp(rate * (a / b - c)) / b;
        }
    }
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
                    out[i] = exponentialintegralei(argi) * exp(-c1 - rate * c);
                }
                // std::cout << i << " argsi=" << args[i] << ";c1=" << c1 << ";rate=" << rate << ";c=" << c << ";out=" << out[i] << std::endl;
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
