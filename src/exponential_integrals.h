#ifndef EXPONENTIAL_INTEGRALS_H
#define EXPONENTIAL_INTEGRALS_H

#include <iostream>
#include <gmpxx.h>
#include "mpreal.h"
#include "common.h"
#include "mpreal_support.h"
#include "gauss_legendre.h"

mpfr::mpreal mpf_ei(const mpfr::mpreal&, const mp_prec_t);
mpfr::mpreal mpf_ei(const mpfr::mpreal&, const mpfr::mpreal&, const mp_prec_t);

template <typename T>
T expintei(const T&, const T&);

template <typename T>
T eint_helper(T x, void* r)
{
    return exp(x + *(T*)r) / x;
}

#define EINTDIFF_QUAD 0
#ifdef EINTDIFF_QUAD
template <typename T>
T eintdiff(const T a, const T b, T r)
{
    // = e(r) * (eint(b) - eint(a));
    // = -\int_a^b exp(t+r) / t dt
    if (a > b)
        return -eintdiff(b, a, r);
    // T ret = exp(r) * (expintei(b) - expintei(a));
    // check_nan(ret);
    // return mpreal_wrapper_type<T>::convertBack(ret);
    // std::function<T(const T, T*)> f(eint_helper<T>);
    // return adaptiveSimpsons<T>(f, &r, a, b, 1e-8, 20);
    T ret = gauss_legendre(64, eint_helper<T>, (void*)&r, a, b);
    check_nan(ret);
    return ret;
}
#else
template <typename T>
T eintdiff(const T &a, const T &b, const T &r)
{
    return expintei(b, r) - expintei(a, r);
}
#endif


#endif
