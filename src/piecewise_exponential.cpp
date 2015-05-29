#include "piecewise_exponential.h"

template <typename T>
PiecewiseExponentialRateFunction<T>::PiecewiseExponentialRateFunction(const std::vector<std::vector<double>> &params) :
    RateFunction<T>(params), ada(params[0].begin(), params[0].end()), 
    adb(params[1].begin(), params[1].end()), ads(params[2].begin(), params[2].end()),
    ts(K), Rrng(K), K(ada.size())
{
    // Final piece is required to be flat.
    adb[K - 1] = 0.0;
    ts[0] = 0;
    Rrng[0] = 0;
    // These constant values need to have compatible derivative shape
    // with the calculated values.
    initialize_derivatives();

    for (int k = 1; k < K; ++k)
    {
        // ts[k] = ts[k - 1] + .1 + exp(adlogs[k]);
        ts[k] = ts[k - 1] + ads[k];
        adb[k - 1] = (log(ada[k - 1]) - log(adb[k - 1])) / (ts[k] - ts[k - 1]);
    }
    compute_antiderivative();
    R.reset(new PExpIntegralEvaluator<T>(ada, adb, ts, Rrng));
    Rinv.reset(new PExpIntegralEvaluator<T>(ada, adb, ts, Rrng));
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
    for (auto p : arys)
    {
        std::cout << p.first << std::endl;
        for (adouble x : p.second)
            std::cout << x.value() << " ";
        std::cout << std::endl << std::endl;
    }
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
}

template class PiecewiseExponentialRateFunction<double>;
template class PiecewiseExponentialRateFunction<adouble>;
