#ifndef COMMON_H
#define COMMON_H

#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <Eigen/Dense>

#include "prettyprint.hpp"

#define AUTODIFF 1
#define EIGEN_NO_AUTOMATIC_RESIZING 1
#define RATE_FUNCTION PiecewiseExponentialRateFunction

#ifdef NDEBUG
#define _DEBUG(x)
#else
#define _DEBUG(x) x
#endif

template <typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
// For cython
typedef Matrix<double> DoubleMatrix;
typedef Vector<double> DoubleVector;

template <typename _Scalar, int NX=Eigen::Dynamic, int NY=Eigen::Dynamic>
struct Functor
{
    typedef _Scalar Scalar;
    enum {
        InputsAtCompileTime = NX,
        ValuesAtCompileTime = NY
    };
    typedef Eigen::Matrix<Scalar,InputsAtCompileTime,1> InputType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,1> ValueType;
    typedef Eigen::Matrix<Scalar,ValuesAtCompileTime,InputsAtCompileTime> JacobianType;

    const int m_inputs, m_values;

    Functor() : m_inputs(InputsAtCompileTime), m_values(ValuesAtCompileTime) {}
    Functor(int inputs, int values) : m_inputs(inputs), m_values(values) {}

    int inputs() const { return m_inputs; }
    int values() const { return m_values; }

    // you should define that in the subclass :
    // //  void operator() (const InputType& x, ValueType* v, JacobianType* _j=0) const;
};

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

inline adouble pow(const adouble &x, const adouble &y)
{
    return pow(x, y.value());
}

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
inline void init_eigen() { Eigen::initParallel(); }
inline void fill_jacobian(const adouble &ll, double* outjac)
{
    Eigen::VectorXd d = ll.derivatives();
    Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>, Eigen::RowMajor> _jac(outjac, d.rows());
    _jac = d;
}
inline void store_matrix(Matrix<double> *M, double* out)
{
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(out, M->rows(), M->cols()) = *M;
}

inline double dmin(double a, double b) { return std::min(a, b); }
inline double dmax(double a, double b) { return std::max(a, b); }

inline adouble dmin(adouble a, adouble b)
{
    return (a + b - myabs(a - b)) / 2;
}

inline adouble dmax(adouble a, adouble b)
{
    return (a + b + myabs(a - b)) / 2;
}

#endif
