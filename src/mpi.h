#include <iostream>

#include "mpreal.h"

inline void up() { mpfr::mpreal::set_default_rnd(MPFR_RNDU); }
inline void down() { mpfr::mpreal::set_default_rnd(MPFR_RNDD); }

class MPInterval
{
    public:
    MPInterval(const mpfr::mpreal &x) : a(x), b(x) {}
    MPInterval(const mpfr::mpreal &a, const mpfr::mpreal &b) : a(a), b(b) {}

    mpfr::mpreal delta() 
    {
        mpfr::mpreal::set_default_rnd(GMP_RNDU);
        return b - a;
    }

    const mpfr_srcptr pa() const { return a.mpfr_srcptr(); }
    const mpfr_srcptr pb() const { return b.mpfr_srcptr(); }

    friend MPInterval operator+(const int, const MPInterval &);
    friend MPInterval operator+(const MPInterval&, const MPInterval &);
    friend MPInterval operator-(const MPInterval &, const MPInterval &);
    friend MPInterval operator*(const MPInterval &, const MPInterval &);
    friend MPInterval operator/(const MPInterval &, const MPInterval &);
    friend MPInterval operator*(const mpfr::mpreal &, const MPInterval &);
    friend MPInterval operator*(const double &, const MPInterval &);
    friend std::ostream& operator<<(std::ostream&, const MPInterval&); 
    friend MPInterval exp(const MPInterval &x);

    private:
    mpfr::mpreal a, b;
};

std::ostream& operator<<(std::ostream& os, const MPInterval &x)
{
    os << "[" << x.a << "," << x.b << "]";
    return os;
}

MPInterval operator+(const MPInterval &x, const int y) { return y + x; }

MPInterval operator+(const int x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    down();
    a = x + y.a;
    up();
    b = x + y.b;
    return MPInterval(a, b);
}

MPInterval operator+(const MPInterval &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    down();
    a = x.a + y.a;
    up();
    b = x.b + y.b;
    return MPInterval(a, b);
}

MPInterval operator-(const MPInterval &x)
{
    mpfr::mpreal a, b;
    mpfr_neg(a.mpfr_ptr(), x.pb(), GMP_RNDD);
    mpfr_neg(b.mpfr_ptr(), x.pa(), GMP_RNDU);
    return MPInterval(a, b);
}

MPInterval operator-(const MPInterval &x, const MPInterval &y)
{
    // x.a - y.b <= x - y <= x.b - y.a
    mpfr::mpreal a, b;
    mpfr_sub(a.mpfr_ptr(), x.pa(), y.pb(), GMP_RNDD);
    mpfr_sub(b.mpfr_ptr(), x.pb(), y.pa(), GMP_RNDU);
    return MPInterval(a, b);
}

MPInterval operator*(const double &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    down();
    a = x * y.a;
    up();
    b = x * y.b;
    return MPInterval(a, b);
}

MPInterval operator*(const mpfr::mpreal &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    down();
    a = x * y.a;
    up();
    b = x * y.b;
    return MPInterval(a, b);
}

MPInterval operator*(const MPInterval &x, const mpfr::mpreal &y)
{
    return y * x;
}

MPInterval operator*(const MPInterval &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    int sas = sgn(x.a), sbs = sgn(x.b), tas = sgn(y.a), tbs = sgn(y.b);
    mpfr::mpreal sa = x.a, sb = x.b, ta = y.a, tb = y.b; 
    if (sas >= 0)
    {
        if (tas >= 0)
        {
            down(); a = sa * ta;
            up(); b = sb * tb;
        } 
        else if (tbs <= 0)
        {
            down(); a = sb * ta;
            up(); b = sa * tb;
        }
        else
        {
            down(); a = sb * ta;
            up(); b = sb * tb;
        }
    } 
    else if (sbs <= 0)
    {
        if (tas >= 0)
        {
            down(); a = sa * tb;
            up(); b = sb * ta;
        } 
        else if (tbs <= 0)
        {
            down(); a = sb * tb;
            up(); b = sa * ta;
        }
        else
        {
            down(); a = sa * tb;
            up(); b = sa * ta;
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

MPInterval operator/(const MPInterval &x, const MPInterval &y)
{
    mpfr::mpreal a, b;
    int sas = sgn(x.a), sbs = sgn(x.b), tas = sgn(y.a), tbs = sgn(y.b);
    mpfr::mpreal sa = x.a, sb = x.b, ta = y.a, tb = y.b; 
    MPInterval R(-INFINITY, INFINITY);
    if (sas == 0 and sbs == 0)
    {
        if ((tas < 0 and tbs > 0) or (tas == 0 or tbs == 0))
            return R;
        return MPInterval(0.0);
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
            down();
            a = sa / tb;
            b = INFINITY;
        }
        if (sbs <= 0)
        {
            a = -INFINITY;
            up();
            b = sb / tb;
        }
    }
    else
    {
        if (sas >= 0)
        {
            down();
            a = sa / tb;
            up();
            b = sb / ta;
        }
        else if (sbs <= 0)
        {
            down();
            a = sa / ta;
            up();
            b = sb / tb;
        }
        else
        {
            down();
            a = sa / ta;
            up();
            b = sb / tb;
        }
    }  
    return MPInterval(a, b);
}

inline MPInterval exp(const MPInterval &x)
{
    return MPInterval(exp(x.a, MPFR_RNDD), exp(x.b, MPFR_RNDU));    
}

int main(int argc, char** argv)
{
    mpfr::mpreal::set_default_prec(atoi(argv[1]));
    MPInterval a(1.1);
    MPInterval c(1.2);
    MPInterval b = exp(-a);
    MPInterval d = b * c / (1 + b + c * 3.0 + 2);
    std::cout << d << " " << d.delta() << std::endl;
}
