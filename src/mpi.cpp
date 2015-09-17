#include "mpi.h"

mpfr::mpreal MPInterval::delta()
{
    mpfr::mpreal ret;
    ret.setPrecision(b.getPrecision());
    mpfr_sub(ret.mpfr_ptr(), b.mpfr_srcptr(), a.mpfr_srcptr(), MPFR_RNDU);
    return ret;
}

mpfr::mpreal MPInterval::mid()
{
    return (a + b) / 2.0;
}

std::ostream& operator<<(std::ostream& os, const MPInterval &x)
{
    os << "[" << x.a << "," << x.b << "]";
    return os;
}

MPInterval operator+(const MPInterval &x, const int y) { return y + x; }

MPInterval operator+(const int x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    a.setPrecision(y.a.getPrecision());
    b.setPrecision(y.b.getPrecision());
    mpfr_add_si(a.mpfr_ptr(), y.a.mpfr_srcptr(), x, MPFR_RNDD);
    mpfr_add_si(b.mpfr_ptr(), y.b.mpfr_srcptr(), x, MPFR_RNDU);
    return MPInterval(a, b);
}

MPInterval operator+(const MPInterval &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    mpfr_add(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
    mpfr_add(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
    return MPInterval(a, b);
}

MPInterval operator-(const MPInterval &x)
{
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    mpfr_neg(a.mpfr_ptr(), x.b.mpfr_srcptr(), MPFR_RNDD);
    mpfr_neg(b.mpfr_ptr(), x.a.mpfr_srcptr(), MPFR_RNDU);
    return MPInterval(a, b);
}

MPInterval operator-(const int x, const MPInterval &y)
{
    // x.a - y.b <= x - y <= x.b - y.a
    MPInterval ny = -y;
    return x + ny;
}

MPInterval operator-(const MPInterval &x, const MPInterval &y)
{
    // x.a - y.b <= x - y <= x.b - y.a
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    mpfr_sub(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDD);
    mpfr_sub(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
    return MPInterval(a, b);
}

MPInterval operator*(const MPInterval &x, const double &y)
{
    return y * x;
}

MPInterval operator*(const double x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    a.setPrecision(y.a.getPrecision());
    b.setPrecision(y.b.getPrecision());
    mpfr_mul_d(a.mpfr_ptr(), y.a.mpfr_srcptr(), x, MPFR_RNDD);
    mpfr_mul_d(b.mpfr_ptr(), y.b.mpfr_srcptr(), x, MPFR_RNDU);
    return MPInterval(a, b);
}

MPInterval operator*(const mpfr::mpreal &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    a.setPrecision(y.a.getPrecision());
    b.setPrecision(y.b.getPrecision());
    mpfr_mul(a.mpfr_ptr(), x.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
    mpfr_mul(b.mpfr_ptr(), x.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
    return MPInterval(a, b);
}

MPInterval operator*(const MPInterval &x, const mpfr::mpreal &y)
{
    return y * x;
}

MPInterval operator*(const MPInterval &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    int sas = sgn(x.a), sbs = sgn(x.b), tas = sgn(y.a), tbs = sgn(y.b);
    if (sas >= 0)
    {
        if (tas >= 0)
        {
            mpfr_mul(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
            mpfr_mul(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
            /*
            down(); a = sa * ta;
            up(); b = sb * tb;
            */
        } 
        else if (tbs <= 0)
        {
            mpfr_mul(a.mpfr_ptr(), x.b.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
            mpfr_mul(b.mpfr_ptr(), x.a.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
            /*
            down(); a = sb * ta;
            up(); b = sa * tb;
            */
        }
        else
        {
            mpfr_mul(a.mpfr_ptr(), x.b.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
            mpfr_mul(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
            /*
            down(); a = sb * ta;
            up(); b = sb * tb;
            */
        }
    } 
    else if (sbs <= 0)
    {
        if (tas >= 0)
        {
            mpfr_mul(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDD);
            mpfr_mul(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDU);
            /*
            down(); a = sa * tb;
            up(); b = sb * ta;
            */
        } 
        else if (tbs <= 0)
        {
            mpfr_mul(a.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDD);
            mpfr_mul(b.mpfr_ptr(), x.a.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDU);
            /*
            down(); a = sb * tb;
            up(); b = sa * ta;
            */
        }
        else
        {
            mpfr_mul(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDD);
            mpfr_mul(b.mpfr_ptr(), x.a.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDU);
            /*
            down(); a = sa * tb;
            up(); b = sa * ta;
            */
        }
    }
    else
    {   
        auto cases = {x.a * y.a, x.a * y.b, x.b * y.a, x.b * y.b};
        a = std::min(cases);
        b = std::max(cases);
    }
    return MPInterval(a, b);
}

MPInterval operator/(const MPInterval &x, const int y)
{
    return x / MPInterval(y, x.a.getPrecision());
}

MPInterval operator/(const int x, const MPInterval &y)
{
    return MPInterval(x, y.a.getPrecision()) / y;
}

MPInterval operator/(const MPInterval &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    int sas = sgn(x.a), sbs = sgn(x.b), tas = sgn(y.a), tbs = sgn(y.b);
    mpfr::mpreal sa = x.a, sb = x.b, ta = y.a, tb = y.b; 
    MPInterval R(mpfr::const_infinity(-1, a.getPrecision()), mpfr::const_infinity(1, a.getPrecision()));
    mpfr::mpreal z("0", a.getPrecision());
    MPInterval zero(z);
    if (sas == 0 and sbs == 0)
    {
        if ((tas < 0 and tbs > 0) or (tas == 0 or tbs == 0))
            return R;
        return zero;
    }
    if (tas < 0 and tbs > 0)
        return R;
    if (tas < 0)
        return (-x) / (-y);
    if (tas == 0)
    {
        if (sas < 0 and sbs > 0)
            return R;
        if (tas == tbs)
            return R;
        if (sas >= 0)
        {
            mpfr_div(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDD);
            b = mpfr::const_infinity(1);
        }
        if (sbs <= 0)
        {
            a = mpfr::const_infinity(-1);
            mpfr_div(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
        }
    }
    else
    {
        if (sas >= 0)
        {
            mpfr_div(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDD);
            mpfr_div(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDU);
            /*
            down();
            a = sa / tb;
            up();
            b = sb / ta;
            */
        }
        else if (sbs <= 0)
        {
            mpfr_div(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
            mpfr_div(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
            /*
            down();
            a = sa / ta;
            up();
            b = sb / tb;
            */
        }
        else
        {
            mpfr_div(a.mpfr_ptr(), x.a.mpfr_srcptr(), y.a.mpfr_srcptr(), MPFR_RNDD);
            mpfr_div(b.mpfr_ptr(), x.b.mpfr_srcptr(), y.b.mpfr_srcptr(), MPFR_RNDU);
            /*
            down();
            a = sa / ta;
            up();
            b = sb / tb;
            */
        }
    }  
    return MPInterval(a, b);
}

MPInterval expm1(const MPInterval &x)
{
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    mpfr_expm1(a.mpfr_ptr(), x.a.mpfr_srcptr(), MPFR_RNDD);
    mpfr_expm1(b.mpfr_ptr(), x.b.mpfr_srcptr(), MPFR_RNDU);
    return MPInterval(a, b);
}

MPInterval exp(const MPInterval &x)
{
    mpfr::mpreal a, b;
    a.setPrecision(x.a.getPrecision());
    b.setPrecision(x.b.getPrecision());
    mpfr_exp(a.mpfr_ptr(), x.a.mpfr_srcptr(), MPFR_RNDD);
    mpfr_exp(b.mpfr_ptr(), x.b.mpfr_srcptr(), MPFR_RNDU);
    return MPInterval(a, b);
}

/*
int main(int argc, char** argv)
{
    mpfr::mpreal::set_default_prec(atoi(argv[1]));
    MPInterval a(1.1);
    MPInterval c(1.2);
    MPInterval b = exp(-a);
    MPInterval d = b * c / (1 + b + c * 3.0 + 2);
    std::cout << d << " " << d.delta() << std::endl;
}
*/

