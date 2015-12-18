#ifndef MPQ_SUPPORT_H
#define MPQ_SUPPORT_H

#include <gmpxx.h>
#include <Eigen/Core>

#include "common.h"

#define MPQ_CONSTRUCT(v, a, b) \
    mpq_class v(a, b);\
    if (a == 0) v = 0_mpq;

namespace Eigen {
    template<> 
    struct NumTraits<mpq_class> : NumTraits<long long> // permits to get the epsilon, dummy_precision, lowest, highest functions
        {
            typedef mpq_class Real;
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
            struct scalar_product_traits<double, mpq_class>
            {
                typedef double ReturnType;
            };
        template <>
            struct scalar_product_traits<adouble, mpq_class>
            {
                typedef adouble ReturnType;
            };
        template <>
            struct cast_impl<mpq_class, adouble>
            {
                static inline adouble run(const mpq_class &x)
                {
                    return adouble(mpq_get_d(x.get_mpq_t()));
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
    }
}

typedef Eigen::Matrix<mpq_class, Eigen::Dynamic, Eigen::Dynamic> MatrixXq;
typedef Eigen::Matrix<mpq_class, Eigen::Dynamic, 1> VectorXq;

#endif
