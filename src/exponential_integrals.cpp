#include "exponential_integrals.h"

template <>
mpfr::mpreal expintei(const mpfr::mpreal &x, const mpfr::mpreal &y)
{
    return mpf_ei(x, y, x.getPrecision());
}

/*
template <>
adouble expintei(const adouble&)
{
    throw std::domain_error("what");
}

template <>
mpreal_wrapper<adouble> expintei(const mpreal_wrapper<adouble> &)
{
    throw std::domain_error("what");
}
*/

template <>
double expintei(const double &x, const double &y)
{
    mpfr::mpreal xd(x), yd(y);
    xd.setPrecision(53);
    yd.setPrecision(53);
    return expintei(xd, yd).toDouble();
}

#ifndef EINTDIFF_QUAD
template <>
double eintdiff(const double &a, const double &b, const double &r)
{
    mpfr::mpreal ma(a, 70), mb(b, 70), mr(r, 70);
    return eintdiff(ma, mb, mr).toDouble();
}

template <typename T>
Eigen::AutoDiffScalar<T> eintdiff(const Eigen::AutoDiffScalar<T> a, 
        const Eigen::AutoDiffScalar<T> b, const Eigen::AutoDiffScalar<T> c)
{
    Eigen::AutoDiffScalar<T> ret;
    ret.value() = eintdiff(a.value(), b.value(), c.value());
    ret.derivatives() = exp(c.value()) * (exp(b.value()) / b.value() * b.derivatives() - exp(a.value()) / a.value() * a.derivatives());
    ret.derivatives() += c.derivatives() * ret.value();
    return ret;
}
#endif


// This is copied almost verbatim from the Python mpmath.libmp module.
//
// The author of that code is Fredrik Johansson (https://github.com/fredrik-johansson)
// Any mistakes are my fault.

inline mpz_class to_fixed(mpfr::mpreal x, mp_prec_t wp)
{
    mp_exp_t e;
    mpz_class ret;
    mpfr::mpreal man = mpfr::frexp(x, &e);
    mp_prec_t offset = wp + e;
    man *= sgn(x);
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

mpfr::mpreal mpf_ei(const mpfr::mpreal &x, const mp_prec_t prec)
{
    return mpf_ei(x, 0, prec);
}

mpfr::mpreal mpf_ei(const mpfr::mpreal &_x, const mpfr::mpreal &log_coef, const mp_prec_t prec)
{
    mpfr::mpreal x = _x;
    /*
    if (x > 0)
    {
        x.setPrecision(prec);
        return mpfr::eint(x);
    }
    */
    mp_exp_t expo;
    mpfr::mpreal man = mpfr::frexp(x, &expo);
    mp_prec_t bc = x.getPrecision();
    mp_prec_t xmag = bc + expo;
    mp_prec_t wp = prec + 20;
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
        v *= exp(xprec + log_coef) / xprec;
    }
    else
    {
        wp += 2 * (int)(abs(x).toLong());
        mpz_class u = to_fixed(x, wp);
        mpz_class u2 = ei_taylor(u, wp) + to_fixed(mpfr::const_euler(wp), wp);
        mpfr::mpreal t1 = mpfr::ldexp(mpfr::mpreal(u2.get_mpz_t(), wp), -wp);
        mpfr::mpreal t2 = log(abs(xprec));
        v = t1 + t2;
        v *= exp(log_coef);
    }
    return v;
}

/*
#include "gsl/gsl_sf_expint.h"
int main(int argc, char** argv)
{
    for (int i=-10; i < 11; ++i)
    {
        double x = pow(2.5, i);
        std::cout << mpf_ei(x, (mp_prec_t)53).toDouble() << " " << gsl_sf_expint_Ei(x) << std::endl;
    }
}
*/
