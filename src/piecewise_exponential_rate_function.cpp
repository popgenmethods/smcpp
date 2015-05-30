#include "piecewise_exponential_rate_function.h"

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> &params) :
    RateFunction<T>(params), K(params[0].size()), ada(params[0].begin(), params[0].end()), 
    adb(params[1].begin(), params[1].end()), ads(params[2].begin(), params[2].end()),
    ts(K + 1), Rrng(K)
{
    // Final piece is required to be flat.
    ts[0] = 0;
    Rrng[0] = 0;
    // These constant values need to have compatible derivative shape
    // with the calculated values.
    initialize_derivatives();
    for (int k = 0; k < K; ++k)
    {
        ts[k + 1] = ts[k] + ads[k];
        adb[k] = (log(ada[k]) - log(adb[k])) / (ts[k + 1] - ts[k]);
    }
    adb[K - 1] = 0.0;
    ts[K] = INFINITY;
    compute_antiderivative();

    R.reset(new PExpIntegralEvaluator<T>(ada, adb, ts, Rrng));
    Rinv.reset(new PExpInverseIntegralEvaluator<T>(ada, adb, ts, Rrng));

    // Compute a TV-like regularizer
    PExpEvaluator<T> eta(ada, adb, ts, Rrng);
    T x = 0.0;
    T xmax = ts.rbegin()[1] * 1.1;
    T step = (xmax - x) / 1000;
    std::vector<T> xs, ys;
    while (x < xmax)
    {
        xs.push_back(x);
        x += step;
    }
    ys = eta(xs);
    _reg = 0.0;
    for (int i = 1; i < ys.size(); ++i)
        _reg += myabs(ys[i] - ys[i - 1]);
    print_debug();
}

template <typename T>
void PiecewiseExponentialRateFunction<T>::initialize_derivatives(void) {}

template <>
void PiecewiseExponentialRateFunction<adouble>::initialize_derivatives(void)
{
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3 * K, 3 * K);
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(3 * K);
    for (int k = 0; k < K; ++k)
    {
        ada[k].derivatives() = I.row(k);
        adb[k].derivatives() = I.row(K + k);
        ads[k].derivatives() = I.row(2 * K + k);
    }
    ts[0].derivatives() = zero;
    Rrng[0].derivatives() = zero;
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
            std::cout << x.value() << " ";
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
