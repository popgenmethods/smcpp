#include "piecewise_exponential.h"

PiecewiseExponential::PiecewiseExponential(std::vector<double> logu, std::vector<double> logv, std::vector<double> logs) :
    K(logu.size()), logu(logu), logv(logv), logs(logs), adlogu(K), adlogv(K), adlogs(K),
    ada(K), adb(K), ts(K), Ra(K), Rb(K), Rc(K), Rrng(K)
{
    // First, set correct derivative dependences
    auto I = Eigen::MatrixXd::Identity(3 * K, 3 * K);
    for (int k = 0; k < K; ++k)
    {
        adlogu[k] = logu[k];
        adlogu[k].derivatives() = I.row(k);
        adlogv[k] = logv[k];
        adlogv[k].derivatives() = I.row(K + k);
        adlogs[k] = logs[k];
        adlogs[k].derivatives() = I.row(2 * K + k);
    }
    ts[0] = 0;
    ada[0] = exp(logu[0]);
    for (int k = 1; k < K; ++k)
    {
        ts[k] = ts[k - 1] + exp(adlogs[k]);
        ada[k] = exp(logu[k]);
        adb[k - 1] = (logv[k - 1] - logu[k - 1]) / (ts[k] - ts[k - 1]);
        // Purposely leave adb[K - 1] undefined here since we require
        // the last piece to be flat.
    }
    compute_antiderivative();
}

int PiecewiseExponential::num_derivatives(void)
{
    return 3 * K;
}

adouble PiecewiseExponential::R(adouble t) const
{
    // Require flat final piece
    if (t >= ts[K - 1])
        return Rrng[K - 1] + ada[K - 1] * (t - ts[K - 1]);
    int ip = insertion_point(t, ts, 0, K);
    return Ra[ip] * exp(Rb[ip] * (t - ts[ip])) + Rc[ip];
}

double PiecewiseExponential::double_R(double t) const
{
    if (isinf(t))
        return INFINITY;
    return R(t).value();
}
// y and coalescence rate will never depend (continuously) on model parameters
// so we do not pass them as adouble. 
adouble PiecewiseExponential::inverse_rate(double y, adouble t, double coalescence_rate) const
{
    if (isinf(y))
        return INFINITY;
    // Return x such that rate * \int_t^{t + x} eta(s) ds = y
    adouble Rt0 = y / coalescence_rate;
    if (t > 0)
        Rt0 += R(t);
    // Enforce constant last period
    if (Rt0 > Rrng[K - 1])
        return (Rt0 - Rrng[K - 1]) / ada[K - 1] + ts[K - 1] - t;
    int ip = insertion_point(Rt0, Rrng, 0, K);
    // return log((Rt0 - Rc[ip]) / Ra[ip]) / Rb[ip] + ts[ip] - t;
    // log((Rt0 - Rc[ip]) / Ra[ip]) = log( Rt0 / Ra[ip] + (Ra[ip] - Rng[ip]) / Ra[ip]
    //                              = log((Rt0 - Rrng[ip]) / Ra[ip] + 1)
    return Eigen::log1p((Rt0 - Rrng[ip]) / Ra[ip]) / Rb[ip] + ts[ip] - t;
}

// Don't overload this: keeps leading to problems with the derivatives()
// getting blown away at various points.
double PiecewiseExponential::double_inverse_rate(double y, double t, double coalescence_rate) const
{
    return inverse_rate(y, (adouble)t, coalescence_rate).value();
}

void PiecewiseExponential::print_debug() const
{
    std::vector<std::pair<std::string, std::vector<adouble>>> arys = 
    {{"ada", ada}, {"adb", adb}, {"ts", ts}, {"Ra", Ra}, 
        {"Rb", Rb}, {"Rc", Rc}, {"Rrng", Rrng}};
    for (auto p : arys)
    {
        std::cout << p.first << std::endl;
        for (adouble x : p.second)
            std::cout << x.value() << " ";
        std::cout << std::endl << std::endl;
    }
}

void PiecewiseExponential::compute_antiderivative()
{
    Rrng[0] = 0;

    for (int k = 0; k < K - 1; ++k)
    {
        if (adb[k] == 0.0)
            throw std::domain_error("b cannot be zero");
        Ra[k] = ada[k] / adb[k];
        Rc[k] = -Ra[k] + Rrng[k];
        Rb[k] = adb[k];
        Rrng[k + 1] = Rrng[k] + Ra[k] * expm1(Rb[k] * (ts[k + 1] - ts[k]));
    }
}

std::vector<std::vector<adouble>> PiecewiseExponential::ad_vars(void) const
{
    return {adlogu, adlogv, adlogs};
}
