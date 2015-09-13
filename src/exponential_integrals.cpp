#include <iostream>
#include <gmpxx.h>
#include "mpreal.h"

// This is copied almost verbatim from the Python mpmath.libmp module.
// The author of that code is Fredrik Johansson (https://github.com/fredrik-johansson)
// Any mistakes are my fault.

inline mpz_class to_fixed(mpfr::mpreal x, mp_prec_t wp)
{
    mp_exp_t e;
    mpz_class ret;
    mpfr::mpreal man = mpfr::frexp(x, &e);
    mp_prec_t offset = wp + e;
    if (offset >= 0) 
        man <<= offset;
    else 
        man >>= -offset;
    mpfr_get_z(ret.get_mpz_t(), man.mpfr_srcptr(), mpfr::mpreal::get_default_rnd());
    return ret;
}


mpz_class ei_taylor(mpz_class x, mp_prec_t prec)
{
    mpz_class s = x, t = x;
    int k = 2;
    while (t)
    {
        t = ((t * x) >> prec) / k;
        s += t / k;
        k++;
    }
    return s;
}


mpz_class ei_asymptotic(mpz_class x, mp_prec_t prec)
{
    mpz_class one = 1_mpz << prec;
    mpz_class t = ((one << prec) / x);
    x = t;
    mpz_class s = one + x;
    int k = 2;
    while (t)
    {
        t = (k * t * x) >> prec;
        s += t;
        k++;
    }
    return s;
}

mpfr::mpreal mpf_ei(mpfr::mpreal x, const mp_prec_t prec)
{
    mp_exp_t expo;
    mpfr::mpreal man = mpfr::frexp(x, &expo);
    mp_prec_t bc = x.getPrecision();
    mp_prec_t xmag = bc + expo;
    mp_prec_t wp = prec + 20;
    mpfr::mpreal::set_default_prec(wp);
    bool can_use_asymp = xmag > wp;
    mpfr::mpreal xabsint, v;
    mpz_class tmp;
    mpfr::mpreal xprec(x);
    xprec.setPrecision(wp);
    if (not can_use_asymp)
    {
        if (expo >= 0)
            xabsint = man << expo;
        else
            xabsint = man >> (-expo);
        can_use_asymp = xabsint > int(wp * 0.693) + 10;
    }
    if (can_use_asymp)
    {
        v = (xmag > wp) ? "1" : mpfr::ldexp(mpfr::mpreal(ei_asymptotic(to_fixed(x, wp), wp).get_mpz_t(), wp), -wp);
        v *= exp(xprec) / xprec;
    }
    else
    {
        wp += 2 * (int)(abs(x).toLong());
        mpz_class u = to_fixed(x, wp);
        mpz_class u2 = ei_taylor(u, wp) + to_fixed(mpfr::const_euler(wp), wp);
        mpfr::mpreal t1 = mpfr::ldexp(mpfr::mpreal(u2.get_mpz_t(), wp), -wp);
        mpfr::mpreal t2 = log(abs(xprec));
        v = t1 + t2;
    }
    return v;
}

int main(int argc, char** argv)
{
    int prec = atoi(argv[2]);
    mpfr::mpreal::set_default_prec(prec);
    mpfr::mpreal x(argv[1]);
    mpfr::mpreal ei = mpf_ei(x, prec);
    std::cout << ei.toString() << std::endl;
}
