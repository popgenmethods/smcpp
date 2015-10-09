#ifndef EXPONENTIAL_INTEGRALS_H
#define EXPONENTIAL_INTEGRALS_H

#include <iostream>
#include <gmpxx.h>
#include "mpreal.h"
#include "common.h"
#include "mpreal_support.h"
#include "gauss_legendre.h"

mpfr::mpreal mpf_ei(const mpfr::mpreal&, const mp_prec_t);
mpfr::mpreal mpf_ei(const mpfr::mpreal&, bool, const mp_prec_t);

#ifdef EINTDIFF_QUAD
template <typename T>
T eint_helper(T x, void* r)
{
    return exp(x + *(T*)r) / x;
}

template <typename T>
struct eintdiff
{
    static T run(const T a, const T b, const T r)
    {
        // = e(r) * (eint(b) - eint(a));
        // = -\int_a^b exp(t+r) / t dt
        if (a > b)
            return -run(b, a, r);
        // T ret = exp(r) * (expintei(b) - expintei(a));
        // check_nan(ret);
        // return mpreal_wrapper_type<T>::convertBack(ret);
        // std::function<T(const T, T*)> f(eint_helper<T>);
        // return adaptiveSimpsons<T>(f, &r, a, b, 1e-8, 20);
        T ret = gauss_legendre(64, eint_helper<T>, (void*)&r, a, b);
        check_nan(ret);
        return ret;
    }
};

#else
template <typename T>
T expintei(const T&, const T&);

template <typename T>
struct eintdiff
{
    static T run(const T &a, const T &b, const T &r)
    {
        T ret = expintei(b, r) - expintei(a, r);
        check_nan(ret);
        return ret;
    }
};

template <typename T>
struct eintdiff<Eigen::AutoDiffScalar<T> >
{
    static Eigen::AutoDiffScalar<T> run(const Eigen::AutoDiffScalar<T> &a, 
            const Eigen::AutoDiffScalar<T> &b, const Eigen::AutoDiffScalar<T> &c)
    {
        Eigen::AutoDiffScalar<T> ret;
        ret.value() = eintdiff<typename Eigen::AutoDiffScalar<T>::Scalar>::run(a.value(), b.value(), c.value());
        // ret.derivatives() = exp(c.value()) * (exp(b.value()) / b.value() * b.derivatives() - exp(a.value()) / a.value() * a.derivatives());
        ret.derivatives() = exp(b.value() + c.value()) / b.value() * b.derivatives() - exp(a.value() + c.value()) / a.value() * a.derivatives();
        ret.derivatives() += c.derivatives() * ret.value();
        return ret;
    }
};
#endif

#endif
