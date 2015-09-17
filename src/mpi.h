#ifndef MPI_H
#define MPI_H

#include <iostream>

#include "common.h"
#include "mpreal.h"

class MPInterval
{
    public:
    MPInterval(const mpfr::mpreal &x) : a(x), b(x) {}
    MPInterval(const double x, const mp_prec_t prec) : a(x, prec), b(x, prec) {}
    MPInterval(const mpfr::mpreal &a, const mpfr::mpreal &b) : a(a), b(b) {}
    MPInterval() : MPInterval(0.0) {}
    MPInterval(const adouble &x, const mp_prec_t prec) : MPInterval(mpfr::mpreal(x.value(), prec)) {}

    mpfr::mpreal delta();
    mpfr::mpreal mid();

    friend MPInterval operator+(const int, const MPInterval &);
    friend MPInterval operator+(const MPInterval &, const int);
    friend MPInterval operator+(const MPInterval&, const MPInterval &);
    friend MPInterval operator-(const MPInterval &, const MPInterval &);
    friend MPInterval operator-(const int, const MPInterval &);
    friend MPInterval operator-(const MPInterval &);
    friend MPInterval operator*(const MPInterval &, const MPInterval &);
    friend MPInterval operator/(const MPInterval &, const MPInterval &);
    friend MPInterval operator/(const int, const MPInterval &);
    friend MPInterval operator/(const MPInterval &, const int);
    friend MPInterval operator*(const mpfr::mpreal &, const MPInterval &);
    friend MPInterval operator*(const double, const MPInterval &);
    friend std::ostream& operator<<(std::ostream&, const MPInterval&); 
    friend MPInterval exp(const MPInterval &x);
    friend MPInterval expm1(const MPInterval &x);

    private:
    mpfr::mpreal a, b;
};

#endif
