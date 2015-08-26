#ifndef MPQ_SUPPORT_H
#define MPQ_SUPPORT_H

#include <gmpxx.h>
#include <mpreal.h>
#include <Eigen/Core>

#include "common.h"

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
        struct cast_impl<mpq_class, double>
        {
            static inline double run(const mpq_class &x)
            {
                return mpq_get_d(x.get_mpq_t());
            }
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
#endif
