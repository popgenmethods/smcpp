#ifndef COMMON_H
#define COMMON_H

#include <mutex>
#include <iostream>
#include <vector>
#include <random>
#include <array>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/AutoDiff>

#include "prettyprint.hpp"
#include "hash.h"

template <typename T> using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
template <typename T> using Vector = Eigen::Matrix<T, Eigen::Dynamic, 1>;
template <typename T, size_t P> using FixedVector = Eigen::Matrix<T, P, 1>;

typedef Eigen::AutoDiffScalar<Eigen::VectorXd> adouble;
typedef std::vector<std::vector<adouble>> ParameterVector;

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
using std::cosh;
using std::sinh;

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

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(sinh,
  Scalar sinhx = std::sinh(x.value());
  Scalar coshx = std::cosh(x.value());
  return ReturnType(sinhx, x.derivatives() * coshx);
)

EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY(cosh,
  Scalar sinhx = std::sinh(x.value());
  Scalar coshx = std::cosh(x.value());
  return ReturnType(coshx, x.derivatives() * sinhx);
)

#undef EIGEN_AUTODIFF_DECLARE_GLOBAL_UNARY

};

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

void crash_backtrace(const char*, const int);

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
    Logger(const char* name, int line, const char* level) : name(name), line(line), level(level) {}
    const char* name;
    const int line;
    const char* level;
    std::stringstream stream;
    template <typename T>
    Logger &operator<<(const T &data)
    {
        (stream) << data;
        return *this;
    }
    ~Logger()
    {
        std::ostringstream oss;
        oss << name << ":" << line;
        call_logger(oss.str().c_str(), level, stream.str().c_str());
    }
};

#define DEBUG Logger(__FILE__, __LINE__, "DEBUG")
#define DEBUG1 Logger(__FILE__, __LINE__, "DEBUG1")
#define INFO Logger(__FILE__, __LINE__, "INFO")
#define WARNING Logger(__FILE__, __LINE__, "WARNING")
#define CRITICAL Logger(__FILE__, __LINE__, "CRITICAL")

inline void check_nan(const double x) { if (std::isnan(x) or std::isinf(x)) throw std::runtime_error("nan/inf detected"); }

template <typename T>
void check_nan(const Eigen::AutoDiffScalar<T> &x);

template <typename Derived>
void check_nan(const Eigen::DenseBase<Derived> &M)
{
    for (int i = 0; i < M.rows(); ++i)
        for (int j = 0; j < M.cols(); ++j)
        {
            try 
            {
                check_nan(M.coeff(i, j));
            }
            catch (std::runtime_error)
            {
                CRITICAL << M.rows() << " " << M.cols() << " " << i << " " << j << " " << M.coeff(i, j);
                crash_backtrace("", -1);
                throw;
            }
        }
}

template <typename T>
void check_nan(const Eigen::AutoDiffScalar<T> &x)
{
    check_nan(x.value());
    check_nan(x.derivatives());
}

inline adouble double_vec_to_adouble(const double &x, const std::vector<double> &dx)
{
    adouble ret;
    Vector<double> vdx(dx.size());
    for (unsigned int i = 0; i < dx.size(); ++i)
        vdx(i) = dx[i];
    return adouble(x, vdx);
}

#endif
