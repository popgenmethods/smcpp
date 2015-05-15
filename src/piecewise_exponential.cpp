#include "piecewise_exponential.h"

PiecewiseExponential::PiecewiseExponential(const std::vector<double> &sqrt_a, 
        const std::vector<double> &b, const std::vector<double> &sqrt_s) :
    _K(sqrt_a.size()),
    sqrt_a(sqrt_a), b(b), sqrt_s(sqrt_s),
    adsqrt_a(_K), adb(_K), adsqrt_s(_K), adasq(_K), ts(_K), Ra(_K), Rb(_K), Rc(_K), Rrng(_K)
{
    // First, set correct derivative dependences
    auto I = Eigen::MatrixXd::Identity(3 * _K, 3 * _K);
    for (int k = 0; k < _K; ++k)
    {
        adsqrt_a[k] = sqrt_a[k];
        adsqrt_a[k].derivatives() = I.row(k);
        adb[k] = b[k];
        adb[k].derivatives() = I.row(_K + k);
        adsqrt_s[k] = sqrt_s[k];
        adsqrt_s[k].derivatives() = I.row(2 * _K + k);
    }
    ts[0] = 0;
    adasq[0] = pow(adsqrt_a[0], 2);
    for (int k = 1; k < _K; ++k)
    {
        ts[k] = ts[k - 1] + pow(adsqrt_s[k], 2);
        adasq[k] = pow(adsqrt_a[k], 2);
    }
    _compute_antiderivative();
}

int PiecewiseExponential::num_derivatives(void)
{
    return 3 * _K;
}

int PiecewiseExponential::K(void) const
{
    return _K;
}

adouble PiecewiseExponential::R(adouble t) const
{
    int ip = insertion_point(t, ts, 0, _K);
    // std::cout << "insertion point: " << ip << std::endl;
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
    int ip = insertion_point(Rt0, Rrng, 0, _K);
    if (Rb[ip] == 0.0)
        throw std::domain_error("b cannot be zero");
    adouble ret = log((Rt0 - Rc[ip]) / Ra[ip]) / Rb[ip] + ts[ip] - t;
    /*
    if (y == 1.25)
    {
        std::vector<double> ra(_K), rb(_K), rc(_K), tt(_K), rr(_K);
        std::transform(Ra.begin(), Ra.end(), ra.begin(), [](adouble x){return x.value();});
        std::transform(Rb.begin(), Rb.end(), rb.begin(), [](adouble x){return x.value();});
        std::transform(Rc.begin(), Rc.end(), rc.begin(), [](adouble x){return x.value();});
        std::transform(Rrng.begin(), Rrng.end(), rr.begin(), [](adouble x){return x.value();});
        std::transform(ts.begin(), ts.end(), tt.begin(), [](adouble x){return x.value();});
        std::cout << std::endl << "Ra " << ra << std::endl;
        std::cout << "Rb " << rb << std::endl;
        std::cout << "Rc " << rc << std::endl;
        std::cout << "Rrng " << rr << std::endl;
        std::cout << "ts " << tt << std::endl;
        std::cout << t << std::endl;
        std::cout << Rt0 << std::endl;
        std::cout << Ra[ip] << std::endl;
        std::cout << Rb[ip] << std::endl;
        std::cout << Rc[ip] << std::endl;
        std::cout << ts[ip] << std::endl;
        std::cout << ret.value() << std::endl;
    }
    assert(!isnan(ret.value()));
    */
    return ret;
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
    {{"adasq", adasq}, {"adb", adb}, {"ts", ts}, {"Ra", Ra}, 
        {"Rb", Rb}, {"Rc", Rc}, {"Rrng", Rrng}};
    for (auto p : arys)
    {
        std::cout << p.first << std::endl;
        for (adouble x : p.second)
            std::cout << x.value() << " ";
        std::cout << std::endl << std::endl;
    }
}

void PiecewiseExponential::_compute_antiderivative()
{
    Rrng[0] = 0;
    for (int k = 0; k < _K; ++k)
    {
        if (adb[k] == 0.0)
            throw std::domain_error("b cannot be zero");
        Ra[k] = adasq[k] / adb[k];
        Rc[k] = -Ra[k] + Rrng[k];
        Rb[k] = adb[k];
        if (k < _K - 1)
            Rrng[k + 1] = Rrng[k] + Ra[k] * expm1(Rb[k] * (ts[k + 1] - ts[k]));
    }
}
