#ifndef PIECEWISE_EXPONENTIAL_H
#define PIECEWISE_EXPONENTIAL_H

#include <Eigen/Dense>
#include "common.h"

class PiecewiseExponential
{
    public:
    PiecewiseExponential(const std::vector<double> &sqrt_a, const std::vector<double> &b, const std::vector<double> &sqrt_s) :
        _K(sqrt_a.size()), adasq(_K), adb(_K), ts(_K), Ra(_K), Rb(_K), Rc(_K), Rrng(_K)
    {
        // First, set correct derivative dependences
        auto I = Eigen::MatrixXd::Identity(3 * _K, 3 * _K);
        std::vector<adouble> adsqrt_a(_K), adsqrt_s(_K);
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

    int num_derivatives(void)
    {
        return 3 * _K;
    }

    int K(void)
    {
        return _K;
    }

    adouble R(adouble t)
    {
        ip = insertion_point(t, ts, 0, _K);
        // std::cout << "insertion point: " << ip << std::endl;
        return Ra[ip] * exp(Rb[ip] * (t - ts[ip])) + Rc[ip];
    }

    // y and coalescence rate will never depend (continuously) on model parameters
    // so we do not pass them as adouble. 
    adouble inverse_rate(double y, adouble t, double coalescence_rate)
    {
        if (isinf(y))
            return INFINITY;
        // Return x such that rate * \int_t^{t + x} eta(s) ds = y
        adouble Rt0 = y / coalescence_rate;
        if (t > 0)
            Rt0 += R(t);
        ip = insertion_point(Rt0, Rrng, 0, _K);
        if (Rb[ip] == 0.0)
            throw std::domain_error("b cannot be zero");
        return log((Rt0 - Rc[ip]) / Ra[ip]) / Rb[ip] + ts[ip] - t;
    }

    // Don't overload this: keeps leading to problems with the derivatives()
    // getting blown away at various points.
    double double_inverse_rate(double y, double t, double coalescence_rate)
    {
        return inverse_rate(y, (adouble)t, coalescence_rate).value();
    }

    void print_debug()
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

    private:
    int _K, ip;
    std::vector<adouble> adasq, adb, ts, Ra, Rb, Rc, Rrng;

    int insertion_point(adouble x, const std::vector<adouble>& ary, int first, int last)
    {
        int mid;
        while(first + 1 < last)
        {
            mid = (int)((first + last) / 2);
            if (ary[mid] > x)
                last = mid;
            else    
                first = mid;
        }
        return first;
    }

    void _compute_antiderivative()
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

};

#endif
