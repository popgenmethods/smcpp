#include "piecewise_exponential.h"

template class PiecewiseExponential<double>;
template class PiecewiseExponential<adouble>;

template <typename T>
PiecewiseExponential<T>::PiecewiseExponential(
        std::vector<double> logu, std::vector<double> logv, std::vector<double> logs) :
    K(logu.size()), logu(logu), logv(logv), logs(logs), adlogu(K), adlogv(K), adlogs(K),
    ada(K), adb(K), ts(K), Rrng(K)
{
    // First, set correct derivative dependences
    for (int k = 0; k < K; ++k)
    {
        adlogu[k] = logu[k];
        adlogv[k] = logv[k];
        adlogs[k] = logs[k];
    }
    // These constant values need to have compatible derivative shape
    // with the calculated values.
    initialize_derivatives();

    ts[0] = 0;
    Rrng[0] = 0;
    for (int k = 1; k < K; ++k)
    {
        // ts[k] = ts[k - 1] + .1 + exp(adlogs[k]);
        ts[k] = ts[k - 1] + abs(adlogs[k]);
    }
    ada[0] = adlogu[0];
    for (int k = 1; k < K; ++k)
    {
        // ts[k] *= 20.0 / ts[K - 1];
        // ts[k] = ts[k - 1] + 0.1 + exp(adlogs[k]);
        // ada[k] = N_lower + exp(adlogu[k]);
        ada[k] = adlogu[k];
        adb[k - 1] = adlogv[k - 1];
        // adb[k - 1] = log((N_lower + exp(adlogv[k - 1])) / ada[k - 1]) / (ts[k] - ts[k - 1]);
        // adb[k - 1] = (adlogv[k - 1] - adlogu[k - 1]) / (ts[k] - ts[k - 1]);
        // adb[k - 1] = log(adlogv[k - 1] / adlogu[k - 1]) / (ts[k] - ts[k - 1]);
        // (adlogv[k - 1] - adlogu[k - 1]) / (ts[k] - ts[k - 1]);
        // Purposely leave adb[K - 1] undefined here since we require
        // the last piece to be flat.
    }
    adb[K - 1] = 0.0;
    compute_antiderivative();
}

template <>
void PiecewiseExponential<double>::initialize_derivatives(void) {}

template <>
void PiecewiseExponential<adouble>::initialize_derivatives(void)
{
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(3 * K, 3 * K);
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(3 * K);
    for (int k = 0; k < K; ++k)
    {
        adlogu[k].derivatives() = I.row(k);
        adlogv[k].derivatives() = I.row(K + k);
        adlogs[k].derivatives() = I.row(2 * K + k);
    }
    ts[0].derivatives() = zero;
    Rrng[0].derivatives() = zero;
}

template <typename T>
int PiecewiseExponential<T>::num_derivatives(void)
{
    return 3 * K;
}

template <typename T>
T PiecewiseExponential<T>::R(T t) const
{
    int ip = insertion_point(t, ts, 0, K);
    if (adb[ip] == 0.0)
        return Rrng[ip] + ada[ip] * (t - ts[ip]);
    else
        return ada[ip] / adb[ip] * expm1(adb[ip] * (t - ts[ip])) + Rrng[ip];
}


template <typename T>
double PiecewiseExponential<T>::double_R(double t) const
{
    if (isinf(t))
        return INFINITY;
    return toDouble(R(t));
}

template <typename T>
T PiecewiseExponential<T>::inverse_rate(T y, T t, double coalescence_rate) const
{
    if (isinf(toDouble(y)))
        return INFINITY;
    // Return x such that rate * \int_t^{t + x} eta(s) ds = y
    T Rt0 = y / coalescence_rate + R(t);
    // Enforce constant last period
    // FIXME: this is a potential source of bugs if Rt0 < Rrng[K - 1] only by
    // a very small amount.
    int ip = insertion_point(Rt0, Rrng, 0, K);
    // return log((Rt0 - Rc[ip]) / Ra[ip]) / Rb[ip] + ts[ip] - t;
    // log((Rt0 - Rc[ip]) / Ra[ip]) = log( Rt0 / Ra[ip] + (Ra[ip] - Rng[ip]) / Ra[ip]
    //                              = log((Rt0 - Rrng[ip]) / Ra[ip] + 1)
    if (adb[ip] == 0.0) 
        return (Rt0 - Rrng[ip]) / ada[ip] + ts[ip] - t;
    else
        return log1p((Rt0 - Rrng[ip]) * adb[ip] / ada[ip]) / adb[ip] + ts[ip] - t;
}

// Don't overload this: keeps leading to problems with the derivatives()
// getting blown away at various points.
template <typename T>
double PiecewiseExponential<T>::double_inverse_rate(double y, double t, double coalescence_rate) const
{
    return toDouble(inverse_rate(y, (T)t, coalescence_rate));
}

template <typename T>
void PiecewiseExponential<T>::print_debug() const
{
    std::vector<std::pair<std::string, std::vector<T>>> arys = 
    {{"adlogu", adlogu}, {"adlogv", adlogv}, {"adlogs", adlogs}, 
        {"ada", ada}, {"adb", adb}, {"ts", ts}, {"Rrng", Rrng}};
    for (auto p : arys)
    {
        std::cout << p.first << std::endl;
        for (adouble x : p.second)
            std::cout << x.value() << " ";
        std::cout << std::endl << std::endl;
    }
}

template <typename T>
void PiecewiseExponential<T>::compute_antiderivative()
{
    for (int k = 0; k < K - 1; ++k)
    {
        if (adb[k] == 0.0)
            Rrng[k + 1] = Rrng[k] + ada[k] * (ts[k + 1] - ts[k]);
        else
            Rrng[k + 1] = Rrng[k] + (ada[k] / adb[k]) * expm1(adb[k] * (ts[k + 1] - ts[k]));
    }
}

template <typename T>
std::vector<std::vector<T>> PiecewiseExponential<T>::ad_vars(void) const
{
    return {adlogu, adlogv, adlogs};
}
