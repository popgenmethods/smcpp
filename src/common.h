#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <Eigen/Dense>

#include "prettyprint.hpp"

#define AUTODIFF 1
#define RATE_FUNCTION PiecewiseExponentialRateFunction

#ifdef NDEBUG
#define _DEBUG(x)
#else
#define _DEBUG(x) x
#endif

template <typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;


#ifdef AUTODIFF
#include <unsupported/Eigen/AutoDiff>
typedef Eigen::AutoDiffScalar<Eigen::VectorXd> adouble;
inline double toDouble(const adouble &a) { return a.value(); }
inline double toDouble(const double &d) { return d; }
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

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(atan,
  Scalar atanx = std::atan(x.value());
  return ReturnType(atanx, x.derivatives() * (Scalar(1) / (Scalar(1) + x.value() * x.value())));
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

inline double myabs(double a) { return std::abs(a); }
inline adouble myabs(adouble a) { return Eigen::abs(a); }

#endif
