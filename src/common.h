#ifndef COMMON_H
#define COMMON_H

#include <mutex>
#include <iostream>
#include <vector>
#include <random>
#include <array>
#include <cmath>

// #define EIGEN_USE_MKL_ALL
#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

#include "prettyprint.hpp"

#define EIGEN_NO_AUTOMATIC_RESIZING 1

// Maximum time (in coalescent units) considered for all integrals and hidden states.
// Technically this is not necessary -- the method works with T_MAX=infinity -- but
// the derivatives and integrals seem a bit more accurate when everything is finite.
//

typedef std::array<int, 3> block_key;
const double T_MAX = 50.0;

template <typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;

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

#include <unsupported/Eigen/AutoDiff>

typedef Eigen::AutoDiffScalar<Eigen::VectorXd> adouble;
inline double toDouble(const adouble &a) { return a.value(); }
inline double toDouble(const double &d) { return d; }

namespace Eigen {
    // Allow for casting of adouble matrices to double
    namespace internal 
    {
        template <>
            struct cast_impl<adouble, float>
            {
                static inline double run(const adouble &x)
                {
                    return static_cast<float>(x.value());
                }
            };
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

using std::exp;
using std::expm1;
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

};

template <typename T, typename U>
inline unsigned int insertion_point(const T x, const std::vector<U>& ary, unsigned int first, unsigned int last)
{
    unsigned int mid;
    while(first + 1 < last)
    {
        mid = (unsigned int)((first + last) / 2);
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

void store_matrix(const Matrix<double> &M, double* out);
void store_matrix(const Matrix<adouble> &M, double* out);
void store_matrix(const Matrix<adouble> &M, double *out, double *jac);

template <typename T>
inline T dmin(const T a, const T b) { if (a > b) return b; return a; }
template <typename T>
inline T dmax(const T a, const T b) { if (a > b) return a; return b; }

void crash_backtrace(const char*, const int);

#define check_nan(X) { try { _check_nan(X); } catch (std::runtime_error e) { crash_backtrace(__FILE__, __LINE__); throw; } }

inline void _check_nan(const double x) { if (std::isnan(x) or std::isinf(x)) throw std::runtime_error("nan/inf detected"); }

template <typename T>
void _check_nan(const Eigen::AutoDiffScalar<T> &x);

template <typename Derived>
void _check_nan(const Eigen::DenseBase<Derived> &M)
{
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            try 
            {
                _check_nan(M.coeff(i, j));
            }
            catch (std::runtime_error)
            {
                std::cout << i << " " << j << " " << M.coeff(i, j) << std::endl;
                throw;
            }
        }
}

template <typename T>
void _check_nan(const Eigen::AutoDiffScalar<T> &x)
{
    _check_nan(x.value());
    _check_nan(x.derivatives());
}

template <typename T>
void _check_nan(const Vector<T> &x) 
{ 
    for (int i = 0; i < x.rows(); ++i) 
        _check_nan(x(i));
}

#define check_negative(X) { try { _check_negative(X); } catch (std::runtime_error e) { std::cout << __FILE__ << ":" << __LINE__ << std::endl; throw; } }

template <typename T>
void _check_negative(const T x)
{
    if (x < -1e-16)
        throw std::runtime_error("negative x");
}

void init_logger_cb(void(*)(const char*, const char*, const char*));
void call_logger(const char*, const char*, const char*);
struct Logger
{
    static void(*logger_cb)(const char*, const char*, const char*);
    Logger(const char* name, const char* level) : name(name), level(level) {}
    const char* name;
    const char* level;
    std::stringstream stream;
    template <typename T>
    Logger &operator<<(const T &data)
    {
        stream << data;
        return *this;
    }
    void flush()
    {
        call_logger(name, level, stream.str().c_str());
    }
};

#define DEBUG(x) (Logger(__FILE__, "DEBUG") << x).flush()
#define INFO(x) (Logger(__FILE__, "INFO") << x).flush()
#define WARN(x) (Logger(__FILE__, "WARN") << x).flush()

#endif
