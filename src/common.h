#ifndef COMMON_H
#define COMMON_H

#include <mutex>
#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "mpreal.h"
#include <cmath>

#include "prettyprint.hpp"

#define AUTODIFF 1
#define EIGEN_NO_AUTOMATIC_RESIZING 1
#define RATE_FUNCTION PiecewiseExponentialRateFunction

#ifdef NDEBUG
#define _DEBUG(x)
#else
#define _DEBUG(x) x
#endif


#if 0
extern std::mutex mtx;
#define PROGRESS(x) mtx.lock(); std::cout << __FILE__ << ":" << __func__ << "... " << std::flush; mtx.unlock();
#define PROGRESS_DONE() mtx.lock(); std::cout << "done." << std::endl << std::flush; mtx.unlock();
#else
#define PROGRESS(x)
#define PROGRESS_DONE()
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

using mpfr::exp;
using mpfr::expm1;
using std::exp;
using std::expm1;
using mpfr::log1p;
using std::log1p;

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(expm1,
  Scalar expm1x = expm1(x.value());
  Scalar expx = exp(x.value());
  return ReturnType(expm1x, x.derivatives() * expx);
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(ceil,
  Scalar ceil = std::ceil(x.value());
  return ReturnType(ceil, x.derivatives() * Scalar(0));
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(log1p,
  Scalar log1px = std::log1p(x.value());
  return ReturnType(log1px, x.derivatives() * (Scalar(1) / (Scalar(1) + x.value())));
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

template <typename T, typename U>
inline int insertion_point(const T x, const std::vector<U>& ary, int first, int last)
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
inline mpfr::mpreal myabs(mpfr::mpreal a) { return mpfr::abs(a); }

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
inline void store_matrix(Matrix<adouble> *M, double* out)
{
    Matrix<double> MM = M->template cast<double>();
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Map(out, MM.rows(), MM.cols()) = MM;
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

inline void check_for_nans(Vector<double> x) 
{
    for (int i = 0; i < x.rows(); ++i)
        if (std::isnan(x(i)))
            throw std::domain_error("got nans in x");
}

inline void check_for_nans(Vector<adouble> x) 
{
    Vector<double> vd = x.template cast<double>();
    check_for_nans(vd);
    for (int i = 0; i < x.rows(); ++i)
        check_for_nans(x(i).derivatives());
}

#endif
