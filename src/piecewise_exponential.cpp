#include "piecewise_exponential.h"

template <typename T>
PiecewiseExponential<T>::PiecewiseExponential(
        std::vector<double> a, std::vector<double> b, std::vector<double> s) :
    K(a.size()), a(a), b(b), s(s), ada(K), adb(K), ads(K), ts(K), Rrng(K)
{
    for (int k = 0; k < K; ++k)
    {
        ada[k] = a[k];
        adb[k] = b[k];
        ads[k] = s[k];
    }
    // Final piece is required to be flat.
    adb[K - 1] = 0.0;
    ts[0] = 0;
    Rrng[0] = 0;
    one = 1.0;
    // These constant values need to have compatible derivative shape
    // with the calculated values.
    initialize_derivatives();

    for (int k = 1; k < K; ++k)
    {
        // ts[k] = ts[k - 1] + .1 + exp(adlogs[k]);
        ts[k] = ts[k - 1] + ads[k];
    }
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
        ada[k].derivatives() = I.row(k);
        adb[k].derivatives() = I.row(K + k);
        ads[k].derivatives() = I.row(2 * K + k);
    }
    ts[0].derivatives() = zero;
    Rrng[0].derivatives() = zero;
    one.derivatives() = zero;
}

template <typename T>
int PiecewiseExponential<T>::num_derivatives(void)
{
    return 3 * K;
}

template <typename T>
T PiecewiseExponential<T>::R(T t) const
{
    t *= one;
    if (isinf(toDouble(t)))
        return INFINITY;
    int ip = insertion_point(t, ts, 0, K);
    if (adb[ip] == 0.0)
        return Rrng[ip] + ada[ip] * (t - ts[ip]);
    else
        return ada[ip] / adb[ip] * expm1(adb[ip] * (t - ts[ip])) + Rrng[ip];
}

template <typename T>
double PiecewiseExponential<T>::double_R(double t) const
{
    return toDouble(R(t));
}

template <typename T>
T PiecewiseExponential<T>::inverse_rate(T y, T t, double coalescence_rate) const
{
    y *= one;
    t *= one;
    if (isinf(toDouble(y)))
        return INFINITY;
    // Return x such that rate * \int_t^{t + x} eta(s) ds = y
    T Rt0 = y / coalescence_rate + R(t);
    // Enforce constant last period
    // FIXME: this is a potential source of bugs if Rt0 < Rrng[K - 1] only by
    // a very small amount.
    int ip = insertion_point(Rt0, Rrng, 0, K);
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
    return {ada, adb, ads};
}

template class PiecewiseExponential<double>;
template class PiecewiseExponential<adouble>;

