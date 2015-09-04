#ifndef MPQ_SUPPORT_H
#define MPQ_SUPPORT_H

#include <gmpxx.h>
#include <Eigen/Core>

#include "common.h"
#include "mpreal_support.h"

inline double log(const mpq_class &x) { return log(mpfr::mpreal(x.get_mpq_t())).toDouble(); }

namespace Eigen {
    template<> 
    struct NumTraits<mpq_class> : NumTraits<long long> // permits to get the epsilon, dummy_precision, lowest, highest functions
        {
            typedef mpq_class Real;
            // typedef mpfr::mpreal NonInteger;
            typedef mpq_class Nested;
            enum {
                IsComplex = 0,
                IsInteger = 0,
                IsSigned = 1,
                RequireInitialization = 1,
                ReadCost = 1,
                AddCost = 3,
                MulCost = 3
            };
        };
    namespace internal
    {
        template <>
        struct cast_impl<mpq_class, mpreal_wrapper<adouble> >
        {
            static inline mpreal_wrapper<adouble> run(const mpq_class &x)
            {
                return mpreal_wrapper<adouble>(mpfr::mpreal(x.get_mpq_t()));
            }
        };
        template <>
        struct cast_impl<mpq_class, mpfr::mpreal>
        {
            static inline mpfr::mpreal run(const mpq_class &x)
            {
                return mpfr::mpreal(x.get_mpq_t());
            }
        };
        template <>
        struct cast_impl<mpq_class, double>
        {
            static inline double run(const mpq_class &x)
            {
                return mpq_get_d(x.get_mpq_t());
            }
        };
        template <>
            struct scalar_product_traits<mpreal_wrapper<adouble>, mpq_class>
            {
                const static int Defined = 1;
                typedef mpreal_wrapper<adouble> ReturnType;
            };
        template <>
            struct scalar_product_traits<mpreal_wrapper<double>, mpq_class>
            {
                const static int Defined = 1;
                typedef mpreal_wrapper<double> ReturnType;
            };
        template <>
            struct cast_impl<mpq_class, adouble>
            {
                static inline adouble run(const mpq_class &x)
                {
                    return adouble(x.get_d());
                }
            };
    }
}

typedef Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
typedef Eigen::Matrix<mpq_class, Eigen::Dynamic, 1> VectorXq;

inline const mpfr::mpreal operator*(const mpfr::mpreal& a, const mpq_class& b)
{
    mpfr::mpreal x(0, mpfr_get_prec(a.mpfr_ptr()));
    mpfr_mul_q(x.mpfr_ptr(), a.mpfr_ptr(), b.get_mpq_t(), mpfr::mpreal::get_default_rnd());
    return x;
}

inline const mpreal_wrapper<adouble> operator*(const mpreal_wrapper<adouble>& a, const mpq_class& b)
{
    VectorXq v(a.derivatives().rows());
    v.fill(b);
    return mpreal_wrapper<adouble>(a.value() * b, a.derivatives().cwiseProduct(v));
}

#endif
