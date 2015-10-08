#include "exponential_integrals.h"

#ifndef EINTDIFF_QUAD
template <>
mpfr::mpreal expintei(const mpfr::mpreal &x, const mpfr::mpreal &y)
{
    return mpf_ei(x, true, x.getPrecision()) * exp(x + y);
}

#include "gsl/gsl_sf_expint.h"
template <>
double expintei(const double &x, const double &y)
{
    gsl_sf_result res;
    gsl_sf_expint_Ei_scaled_e(x, &res);
    return exp(x + y) * res.val;
    mpfr::mpreal xd(x), yd(y);
    xd.setPrecision(53);
    yd.setPrecision(53);
    return expintei(xd, yd).toDouble();
}

template <>
double eintdiff<double>::run(const double &a, const double &b, const double &r)
{
    // mpfr::mpreal ma(a, 70), mb(b, 70), mr(r, 70);
    // double ret = eintdiff<mpfr::mpreal>::run(ma, mb, mr).toDouble();
    double ret = expintei(b, r) - expintei(a, r);
    // std::cout << "eintdiff: " << ret << " " << exp(r) * (gsl_sf_expint_Ei(b) - gsl_sf_expint_Ei(a)) << std::endl;
    return ret;
}

#endif


// This is copied almost verbatim from the Python mpmath.libmp module.
//
// The author of that code is Fredrik Johansson (https://github.com/fredrik-johansson)
// Any mistakes are my fault.

struct libmp_repr
{
    libmp_repr(const mpfr::mpreal &x) : 
        man(), expo(mpfr_get_z_exp(man.get_mpz_t(), x.mpfr_srcptr())),
        bc(mpz_sizeinbase(man.get_mpz_t(), 2))
    {
        sgn = mpz_sgn(man.get_mpz_t());
        man *= sgn;
        sgn = (sgn == -1) ? 1 : 0;
    }
    mpz_class man;
    mp_prec_t expo;
    int sgn;
    mp_prec_t bc;
};

inline mpz_class to_fixed(const libmp_repr &x, const mp_prec_t prec)
{
    mp_prec_t offset = prec + x.expo;
    if (x.sgn)
        if (offset >= 0) return (-x.man) << offset;
        else return (-x.man) >> (-offset);
    else
        if (offset >= 0) return x.man << offset;
        else return x.man >> (-offset);
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
    return mpf_ei(x, false, prec);
}

mpfr::mpreal mpf_ei(const mpfr::mpreal &_x, bool scaled, const mp_prec_t prec)
{
    libmp_repr x(_x);
    mpfr::mpreal v("0", prec);
    mp_prec_t xmag = x.expo + x.bc;
    mp_prec_t wp = prec + 20;
    bool can_use_asymp = xmag > wp;
    if (not can_use_asymp)
    {
        mpz_class xabsint;
        if (x.expo >= 0)
            xabsint = x.man << x.expo;
        else
            xabsint = x.man >> (-x.expo);
        can_use_asymp = xabsint > int(wp * 0.693) + 10;
    }
    mpfr::mpreal xp(_x);
    xp.setPrecision(prec);
    if (can_use_asymp)
    {
        if (xmag > wp)
            v = "1";
        else
        {
            mpz_class eia = ei_asymptotic(to_fixed(x, wp), wp);
            mpfr_set_z_2exp(v.mpfr_ptr(), eia.get_mpz_t(), -wp, MPFR_RNDN);
        }
        if (not scaled)
            v *= exp(xp);
        v /= xp;
    }
    else
    {
        wp += 2 * (int)(abs(_x).toLong());
        mpz_class u = to_fixed(x, wp);
        mpz_class eul = to_fixed(libmp_repr(mpfr::const_euler(wp)), wp);
        mpz_class u2 = ei_taylor(u, wp);
        u2 += eul;
        mpfr::mpreal t1("0", wp); mpfr_set_z_2exp(t1.mpfr_ptr(), u2.get_mpz_t(), -wp, MPFR_RNDN);
        xp.setPrecision(wp);
        mpfr::mpreal t2 = log(abs(xp));
        v = t1 + t2;
        if (scaled)
            v *= exp(-xp);
    }
    v.setPrecision(prec);
    return v;
}

#include "gsl/gsl_sf_expint.h"
int eint_main(int argc, char** argv)
{
    /*
    double ada = 0.9996804464437794, adb = -0.031297501072304855, Rrng = 0,
           y = -0.035036331483659006, tmp = 0.011684646627566872;
    double adab = ada / adb;
    double c1 = 2 * adab * exp(adb * tmp);
    double c2 = 2 * adab;
    double c3 = 2 * (Rrng - adab) + y;
    std::cout << c1 << " " << mpf_ei(c1, (mp_prec_t)53) << " " << gsl_sf_expint_Ei(c1) << std::endl;
    std::cout << c1 << " " << mpf_ei(c2, (mp_prec_t)53) << " " << gsl_sf_expint_Ei(c2) << std::endl;
    */
    /*
    Vector<double> d(3);
    d(0) = 1;
    d(1) = 0;
    d(2) = 0;
    adouble x(1.0, d);
    d(0) = 0; d(1) = 1;
    adouble y(2.0, d);
    d(1) = 0; d(2) = 1;
    adouble z(3.0, d);
    adouble d1 = eintdiff<adouble>::run(x, y, z);
    double dx = eintdiff<double>::run(x.value() + 1e-8, y.value(), z.value());
    double dy = eintdiff<double>::run(x.value(), y.value() + 1e-8, z.value());
    double dz = eintdiff<double>::run(x.value(), y.value(), z.value() + 1e-8);
    std::cout << d1.derivatives() << std::endl;
    std::cout << (dx - d1.value()) * 1e8 << std::endl;
    std::cout << (dy - d1.value()) * 1e8 << std::endl;
    std::cout << (dz - d1.value()) * 1e8 << std::endl;
    */
    double a(-64.347021769700607621);
    double b(-64.019322340869338830);
    double c(61.0133954162841680852);
    adouble ad(a, 3, 0);
    adouble bd(b, 3, 1);
    adouble cd(c, 3, 2);
    mpreal_wrapper<adouble> mad = mpreal_wrapper_type<adouble>::convert(ad);
    mpfr::mpreal aa("-64.0193223408693388304", 63);
    mad.value() = aa;
    mpreal_wrapper<adouble> mbd = mpreal_wrapper_type<adouble>::convert(bd);
    mbd.value().setPrecision(63);
    mbd.value() = "-64.3470217697006076213";
    mpreal_wrapper<adouble> mcd = mpreal_wrapper_type<adouble>::convert(cd);
    mcd.value().setPrecision(63);
    mcd.value() = "61.0133954162841680852";
    std::cout << eintdiff<double>::run(b, a, c) << " " << eintdiff<mpfr::mpreal>::run(b, a, c) << std::endl ;
    std::cout << eintdiff<mpreal_wrapper<adouble> >::run(mbd, mad, mcd) << std::endl;
}
