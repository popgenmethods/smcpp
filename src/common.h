#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <Eigen/Dense>

#include "prettyprint.hpp"

#define AUTODIFF 1

#ifdef NDEBUG
#define _DEBUG(x)
#else
#define _DEBUG(x) x
#endif

#ifdef AUTODIFF
#include <unsupported/Eigen/AutoDiff>
typedef Eigen::AutoDiffScalar<Eigen::VectorXd> adouble;
namespace Eigen {
    // Allow for casting of adouble matrices to double
    namespace internal 
    {
        template <>
            struct cast_impl<adouble, double>
            {
                static inline double run(const adouble &x)
                {
                    return x.value();
                }
            };
    }

// Copied from Eigen's AutoDiffScalar.h
#define EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(FUNC,CODE) \
  template<typename DerType> \
  inline const Eigen::AutoDiffScalar<Eigen::CwiseUnaryOp<Eigen::internal::scalar_multiple_op<typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > \
  FUNC(const Eigen::AutoDiffScalar<DerType>& x) { \
    using namespace Eigen; \
    typedef typename Eigen::internal::traits<typename Eigen::internal::remove_all<DerType>::type>::Scalar Scalar; \
    typedef AutoDiffScalar<CwiseUnaryOp<Eigen::internal::scalar_multiple_op<Scalar>, const typename Eigen::internal::remove_all<DerType>::type> > ReturnType; \
    CODE; \
  }

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(expm1,
  Scalar expm1x = std::expm1(x.value());
  Scalar expx = std::exp(x.value());
  return ReturnType(expm1x,x.derivatives() * expx);
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log1p,
  Scalar log1px = std::log1p(x.value());
  return ReturnType(log1px, x.derivatives() * (Scalar(1) / (Scalar(1) + x.value())));
)

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY
};
 

#else

typedef double adouble;

#endif

typedef Eigen::Matrix<adouble, Eigen::Dynamic, Eigen::Dynamic> AdMatrix;
typedef Eigen::Matrix<adouble, Eigen::Dynamic, 1> AdVector;

typedef struct AdMatrixWrapper {
    AdMatrix mat;
} AdMatrixWrapper;


template <typename T>
inline int insertion_point(const T x, const std::vector<T>& ary, int first, int last)
{
    int mid;
    while(first + 1 < last)
    {
        mid = (int)((first + last) / 2);
        if (ary[mid] > x)
            last = mid;
        else    
            first = mid;
    }
    return first;
}

#endif
